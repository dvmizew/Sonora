import contextlib
import dataclasses
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import acoustid
import ftfy
import httpx
from musicbrainzngs import MusicBrainzError

from sonora.audio.art import process_album_cover_art, process_artist_artwork
from sonora.audio.bpm import calculate_bpm
from sonora.audio.cuesheet import read_cuesheet_content
from sonora.audio.metadata import read_track_metadata, write_track_metadata
from sonora.audio.replaygain import calculate_album_replaygain
from sonora.core.logger import (
    LOG,
    create_progress,
    interactive_pause_listener,
    wait_if_paused,
)
from sonora.core.models import TrackInfo
from sonora.core.state import get_library_state
from sonora.core.utils import (
    clean_disambiguation,
    clean_title,
    deduplicate_title_features,
    find_audio_files,
    group_files_by_parent,
    is_valid_uuid,
    normalize_date,
    normalize_genre,
    normalize_str,
    resolve_artist_name,
)
from sonora.services.acoustid import lookup_acoustid
from sonora.services.deezer import (
    fetch_deezer_album_details,
    fetch_deezer_track_details,
)
from sonora.services.discogs import search_discogs_release
from sonora.services.genius import fetch_genius_song_details
from sonora.services.itunes import (
    fetch_itunes_album_details,
    fetch_itunes_track_metadata,
)
from sonora.services.lastfm import fetch_lastfm_tags
from sonora.services.lyrics import process_track_lyrics
from sonora.services.musicbrainz import (
    fetch_album_track_mbids,
    fetch_musicbrainz_recording_details,
    fetch_musicbrainz_release_details,
    fetch_track_mbid,
    search_musicbrainz_release,
)
from sonora.services.theaudiodb import (
    fetch_theaudiodb_track_details,
)

LAST_TAGGING_FAILURES: list[dict[str, str]] = []
LAST_TAGGED_TRACKS: list[TrackInfo] = []
_NETWORK_EXCEPTIONS = (
    MusicBrainzError,
    acoustid.AcoustidError,
    acoustid.WebServiceError,
    httpx.HTTPError,
    OSError,
    ValueError,
    RuntimeError,
    TypeError,
    KeyError,
    AttributeError,
    TimeoutError,
)


_SKIP_DIFF_FIELDS: frozenset[str] = frozenset(
    {
        "file_path",
        "lyrics",
        "synced_lyrics",
        "sample_rate",
        "bitrate",
        "channels",
        "is_lossless",
        "art_width",
        "art_height",
    }
)


def get_last_tagging_failures() -> list[dict[str, str]]:
    """Return failures recorded during the latest tagging run."""
    return list(LAST_TAGGING_FAILURES)


def get_last_tagged_tracks() -> list[TrackInfo]:
    """Return successfully tagged tracks recorded during the latest tagging run."""
    return list(LAST_TAGGED_TRACKS)


def _apply_mapping(
    track_info: TrackInfo,
    data: dict[str, Any],
    field_map: dict[str, str],
) -> None:
    for src_key, target_attr in field_map.items():
        val = data.get(src_key)
        if val is None:
            continue
        val_str = str(val).strip()
        if not val_str or val_str.lower() in ("none", "null"):
            continue

        if target_attr == "genre":
            val_str = normalize_genre(val_str) or val_str
        elif target_attr == "date":
            val_str = normalize_date(val_str) or val_str

        if not getattr(track_info, target_attr):
            setattr(track_info, target_attr, val_str)


def _enrich_acoustid(
    track_info: TrackInfo,
    file_path: Path,
    acoustid_api_key: str | None,
    album_track_mbids: dict[int, str] | None = None,
    force: bool = False,
) -> None:
    if not acoustid_api_key:
        return
    # If the album match already resolved an authoritative MBID for this track, skip expensive audio fingerprinting
    album_mbids = album_track_mbids or {}
    if (
        track_info.track_number
        and track_info.track_number in album_mbids
        and is_valid_uuid(album_mbids[track_info.track_number])
        and not force
    ):
        return
    if is_valid_uuid(track_info.musicbrainz_trackid) and not force:
        return
    try:
        acoustid_mbid = lookup_acoustid(
            file_path,
            api_key=acoustid_api_key,
            expected_artist=track_info.artist,
            expected_title=track_info.title,
        )
        if is_valid_uuid(acoustid_mbid):
            track_info.musicbrainz_trackid = acoustid_mbid
            LOG.info(f"   ∟ 🎯 [acoustid] Matched MBID: {acoustid_mbid[:8]}...")
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"AcoustID lookup failed for {track_info.title}: {error}")


def _enrich_musicbrainz(
    track_info: TrackInfo,
    album_mbid: str | None,
    album_track_mbids: dict[int, str] | None,
    album_mb_release_details: dict[str, object] | None,
    force: bool = False,
) -> None:
    try:
        album_mbids = album_track_mbids or {}
        if (
            (not is_valid_uuid(track_info.musicbrainz_trackid) or force)
            and track_info.track_number
            and track_info.track_number in album_mbids
        ):
            candidate_mbid = album_mbids[track_info.track_number]
            if is_valid_uuid(candidate_mbid):
                track_info.musicbrainz_trackid = candidate_mbid
                LOG.info(
                    f"   ∟ 🏷️ [MusicBrainz Album Match] Found MBID: {candidate_mbid[:8]}..."
                )
        elif not is_valid_uuid(track_info.musicbrainz_trackid) or force:
            mbid = fetch_track_mbid(track_info.artist, track_info.title)
            if is_valid_uuid(mbid):
                track_info.musicbrainz_trackid = mbid
                LOG.info(f"   ∟ 🏷️ [MusicBrainz] Found MBID: {mbid[:8]}...")

        if not is_valid_uuid(track_info.musicbrainz_albumid):
            if is_valid_uuid(album_mbid):
                track_info.musicbrainz_albumid = str(album_mbid)
            else:
                search_artist = (
                    track_info.album_artist
                    if track_info.album_artist
                    else track_info.artist
                )
                release = search_musicbrainz_release(search_artist, track_info.album)
                if release:
                    musicbrainz_id = release.get("id")
                    if is_valid_uuid(str(musicbrainz_id)):
                        track_info.musicbrainz_albumid = str(musicbrainz_id)
                    if not track_info.date:
                        date_str = release.get("date")
                        if isinstance(date_str, str) and len(date_str) >= 4:
                            track_info.date = normalize_date(date_str)

        # Batch lookup from pre-fetched album release details (zero network calls)
        mb_rec = None
        if album_mb_release_details:
            tracks_by_pos = album_mb_release_details.get("tracks_by_position", {})
            tracks_by_id = album_mb_release_details.get("tracks_by_mbid", {})
            if (
                isinstance(tracks_by_id, dict)
                and track_info.musicbrainz_trackid in tracks_by_id
            ):
                mb_rec = tracks_by_id[track_info.musicbrainz_trackid]
            elif (
                isinstance(tracks_by_pos, dict)
                and track_info.track_number in tracks_by_pos
            ):
                mb_rec = tracks_by_pos[track_info.track_number]

        if not mb_rec and is_valid_uuid(track_info.musicbrainz_trackid):
            mb_rec = fetch_musicbrainz_recording_details(track_info.musicbrainz_trackid)

        if mb_rec and isinstance(mb_rec, dict):
            _apply_mapping(
                track_info,
                mb_rec,
                {
                    "isrc": "isrc",
                    "disambiguation": "disambiguation",
                    "composer": "composer",
                    "lyricist": "lyricist",
                    "producers": "producers",
                    "remixer": "remixer",
                    "musicbrainz_workid": "musicbrainz_workid",
                },
            )

        if is_valid_uuid(track_info.musicbrainz_albumid):
            mb_rel = (
                album_mb_release_details
                if (
                    album_mb_release_details
                    and album_mbid
                    and track_info.musicbrainz_albumid == str(album_mbid)
                )
                else fetch_musicbrainz_release_details(track_info.musicbrainz_albumid)
            )
            if mb_rel:
                _apply_mapping(
                    track_info,
                    mb_rel,
                    {
                        "barcode": "barcode",
                        "release_country": "release_country",
                        "release_status": "release_status",
                        "release_type": "release_type",
                        "musicbrainz_releasegroupid": "musicbrainz_releasegroupid",
                        "label": "label",
                        "catalog_number": "catalog_number",
                        "media": "media",
                        "language": "language",
                        "script": "script",
                        "artist_sort": "artist_sort",
                    },
                )
                if mb_rel.get("total_tracks") and not track_info.total_tracks:
                    track_info.total_tracks = int(str(mb_rel["total_tracks"]))
                if mb_rel.get("total_discs") and not track_info.total_discs:
                    track_info.total_discs = int(str(mb_rel["total_discs"]))
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"MusicBrainz enrichment failed for {track_info.title}: {error}")


def _enrich_itunes(
    track_info: TrackInfo,
    album_itunes_details: dict[str, Any] | None = None,
) -> None:
    try:
        data = None
        if album_itunes_details:
            t_by_num = album_itunes_details.get("tracks_by_number", {})
            t_by_title = album_itunes_details.get("tracks_by_title", {})
            if isinstance(t_by_num, dict) and track_info.track_number in t_by_num:
                data = t_by_num[track_info.track_number]
            elif (
                isinstance(t_by_title, dict)
                and normalize_str(track_info.title) in t_by_title
            ):
                data = t_by_title[normalize_str(track_info.title)]

        if not data and (not track_info.genre or not track_info.advisory):
            data = fetch_itunes_track_metadata(track_info.artist, track_info.title)

        if data and isinstance(data, dict):
            _apply_mapping(
                track_info,
                data,
                {
                    "genre": "genre",
                    "date": "date",
                    "advisory": "advisory",
                    "copyright": "copyright",
                    "itunes_trackid": "itunes_trackid",
                    "itunes_collectionid": "itunes_collectionid",
                    "itunes_artistid": "itunes_artistid",
                    "release_country": "release_country",
                },
            )
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"iTunes enrichment failed for {track_info.title}: {error}")


def _enrich_lastfm(track_info: TrackInfo, lastfm_api_key: str | None) -> None:
    if not lastfm_api_key or track_info.genre:
        return
    try:
        tags = fetch_lastfm_tags(
            track_info.artist,
            track_info.title,
            api_key=lastfm_api_key,
            mbid=track_info.musicbrainz_trackid,
        )
        if tags:
            if not track_info.genre:
                raw_genre = tags[0]
                normalized_genre = normalize_genre(raw_genre)
                if normalized_genre:
                    track_info.genre = normalized_genre
                    LOG.info(f"   ∟ 🏷️ [Last.fm] Genre: [cyan]{normalized_genre}[/]")
            if len(tags) > 1 and not track_info.style:
                subgenres = [
                    t
                    for t in tags[1:4]
                    if t.lower() != (track_info.genre or "").lower()
                ]
                if subgenres:
                    track_info.style = ", ".join(subgenres)
            if not track_info.mood:
                mood_keywords = {
                    "chill",
                    "dark",
                    "sad",
                    "happy",
                    "energetic",
                    "melancholic",
                    "relax",
                    "relaxing",
                    "aggressive",
                    "ambient",
                    "party",
                    "romantic",
                    "hype",
                    "mellow",
                    "atmospheric",
                    "epic",
                    "somber",
                    "upbeat",
                }
                for t in tags:
                    if t.lower() in mood_keywords:
                        track_info.mood = t.title()
                        break
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"Last.fm enrichment failed for {track_info.title}: {error}")


def _enrich_discogs(
    track_info: TrackInfo,
    discogs_user_token: str | None,
    album_discogs_release: dict[str, object] | None,
) -> None:
    if not discogs_user_token:
        return
    try:
        release = album_discogs_release or search_discogs_release(
            track_info.artist, track_info.album, user_token=discogs_user_token
        )
        if not release:
            return

        styles_val = release.get("styles")
        if isinstance(styles_val, list) and styles_val and not track_info.style:
            track_info.style = ", ".join(str(s) for s in styles_val[:3])

        genres_val = release.get("genres")
        if isinstance(genres_val, list) and genres_val and not track_info.genre:
            track_info.genre = normalize_genre(str(genres_val[0])) or track_info.genre

        _apply_mapping(
            track_info,
            release,
            {
                "released": "date",
                "year": "date",
                "id": "discogs_release_id",
                "artist_id": "discogs_artist_id",
                "country": "release_country",
                "label": "label",
                "catalog_number": "catalog_number",
                "barcode": "barcode",
                "media": "media",
                "composer": "composer",
                "producers": "producers",
                "remixer": "remixer",
            },
        )

        track_credits_dict = release.get("track_credits")
        if isinstance(track_credits_dict, dict):
            track_key = (
                str(track_info.track_number) if track_info.track_number else None
            )
            specific = None
            if track_key and track_key in track_credits_dict:
                specific = track_credits_dict[track_key]
            elif track_info.title and track_info.title.lower() in track_credits_dict:
                specific = track_credits_dict[track_info.title.lower()]
            if isinstance(specific, dict):
                _apply_mapping(
                    track_info,
                    specific,
                    {
                        "producers": "producers",
                        "remixer": "remixer",
                        "composer": "composer",
                    },
                )
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"Discogs enrichment failed for {track_info.title}: {error}")


def _enrich_deezer(
    track_info: TrackInfo,
    album_deezer_details: dict[str, Any] | None,
    force: bool = False,
) -> None:
    try:
        album = album_deezer_details or fetch_deezer_album_details(
            track_info.artist, track_info.album
        )
        if album and isinstance(album, dict):
            _apply_mapping(
                track_info,
                album,
                {
                    "release_date": "date",
                    "genre": "genre",
                    "label": "label",
                    "barcode": "barcode",
                },
            )

        track: dict[str, Any] | None = None
        if album and isinstance(album, dict):
            raw_pos = album.get("tracks_by_position")
            t_by_pos: dict[Any, Any] = raw_pos if isinstance(raw_pos, dict) else {}
            raw_title = album.get("tracks_by_title")
            t_by_title: dict[Any, Any] = (
                raw_title if isinstance(raw_title, dict) else {}
            )
            if isinstance(t_by_pos, dict) and track_info.track_number in t_by_pos:
                track = t_by_pos[track_info.track_number]
            elif (
                isinstance(t_by_title, dict)
                and normalize_str(track_info.title) in t_by_title
            ):
                track = t_by_title[normalize_str(track_info.title)]

        if not track and (not track_info.isrc or not track_info.producers or force):
            track = fetch_deezer_track_details(track_info.artist, track_info.title)

        if not track or not isinstance(track, dict):
            return

        if track.get("featured_artists") and (not track_info.featured_artists or force):
            track_info.featured_artists = str(track["featured_artists"])
        if track.get("producers") and (not track_info.producers or force):
            track_info.producers = str(track["producers"])
        if track.get("track_position") is not None and track_info.track_number is None:
            track_info.track_number = int(str(track["track_position"]))
        if track.get("disk_number") is not None and track_info.disc_number is None:
            track_info.disc_number = int(str(track["disk_number"]))
        if track.get("explicit_lyrics") and not track_info.advisory:
            track_info.advisory = "Explicit"
        _apply_mapping(
            track_info,
            track,
            {
                "release_date": "date",
                "isrc": "isrc",
                "composer": "composer",
                "lyricist": "lyricist",
            },
        )
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"Deezer enrichment failed for {track_info.title}: {error}")


def _enrich_genius(
    track_info: TrackInfo,
    genius_api_token: str | None,
    force: bool = False,
) -> None:
    if not genius_api_token:
        return
    # Short-circuit if composer and producers are already populated (unless force)
    if track_info.composer and track_info.producers and not force:
        return
    try:
        genius_details = fetch_genius_song_details(
            track_info.artist, track_info.title, api_token=genius_api_token
        )
        if not genius_details:
            return
        if genius_details.get("description") and (not track_info.comment or force):
            track_info.comment = str(genius_details["description"])
        if genius_details.get("featured_artists") and (
            not track_info.featured_artists or force
        ):
            track_info.featured_artists = str(genius_details["featured_artists"])
        if genius_details.get("producers") and (not track_info.producers or force):
            track_info.producers = str(genius_details["producers"])
        _apply_mapping(
            track_info,
            genius_details,
            {
                "release_date": "date",
                "genius_song_id": "genius_song_id",
                "writers": "composer",
            },
        )
        if genius_details.get("writers") and not track_info.lyricist:
            track_info.lyricist = str(genius_details["writers"])
        LOG.debug("   ∟ 📝 [Genius] Fetched song details & credits")
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"Genius enrichment failed for {track_info.title}: {error}")


def _enrich_theaudiodb(track_info: TrackInfo, force: bool = False) -> None:
    if not (
        force
        or not track_info.music_video_url
        or not track_info.mood
        or not track_info.initial_key
        or track_info.rating is None
        or not track_info.comment
    ):
        return
    try:
        tadb_details = fetch_theaudiodb_track_details(
            track_info.artist, track_info.title
        )
        if not tadb_details:
            return
        if tadb_details.get("rating") is not None and track_info.rating is None:
            with contextlib.suppress(ValueError, TypeError):
                track_info.rating = float(str(tadb_details["rating"]))
        if tadb_details.get("description") and (not track_info.comment or force):
            desc_str = str(tadb_details["description"]).strip()
            if desc_str and desc_str.lower() not in ("none", "null"):
                track_info.comment = desc_str
        _apply_mapping(
            track_info,
            tadb_details,
            {
                "genre": "genre",
                "music_video_url": "music_video_url",
                "mood": "mood",
                "style": "style",
                "initial_key": "initial_key",
            },
        )
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"TheAudioDB enrichment failed for {track_info.title}: {error}")


def _enrich_cuesheet(
    track_info: TrackInfo,
    file_path: Path,
    cuesheet_content: str | None,
) -> None:
    if cuesheet_content:
        track_info.cuesheet = cuesheet_content
    elif not track_info.cuesheet:
        cue_files = list(file_path.parent.glob("*.cue"))
        if cue_files:
            track_info.cuesheet = read_cuesheet_content(cue_files[0])


def _enrich_bpm(
    track_info: TrackInfo,
    file_path: Path,
    fetch_bpm: bool,
    force: bool = False,
) -> None:
    if fetch_bpm and (track_info.bpm is None or force):
        try:
            bpm = calculate_bpm(file_path)
            if bpm is not None:
                track_info.bpm = bpm
                LOG.info(f"   ∟ 🎵 BPM Calculated: [green]{bpm}[/]")
        except (OSError, ValueError, RuntimeError) as error:
            LOG.debug(f"BPM calculation failed for {track_info.title}: {error}")


def _enrich_artwork(
    track_info: TrackInfo,
    file_path: Path,
    fetch_itunes_art: bool,
    force: bool = False,
    dry_run: bool = False,
) -> Path | None:
    if not fetch_itunes_art:
        return None
    try:
        return process_album_cover_art(
            file_path.parent,
            track_info.artist,
            track_info.album,
            musicbrainz_album_id=track_info.musicbrainz_albumid,
            force=force,
            dry_run=dry_run,
        )
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"Cover art downloading failed for {track_info.title}: {error}")
        return None


def _enrich_lyrics(
    track_info: TrackInfo,
    file_path: Path,
    fetch_lyrics: bool,
    force: bool = False,
    dry_run: bool = False,
) -> None:
    if not fetch_lyrics:
        return
    try:
        lrc_path = file_path.with_suffix(".lrc")
        had_lrc = lrc_path.exists() and lrc_path.stat().st_size > 0
        lyrics_text, tag_type = process_track_lyrics(
            file_path,
            track_info.artist,
            track_info.title,
            force=force,
            dry_run=dry_run,
            isrc=track_info.isrc,
        )
        if lyrics_text and tag_type:
            track_info.lyrics = lyrics_text
            if not had_lrc:
                LOG.info(
                    f"   ∟ [green]✅ Saved {tag_type} lyrics for {file_path.name}[/]"
                )
            elif force:
                LOG.info(
                    f"   ∟ [yellow]🔄 Updated {tag_type} lyrics for {file_path.name}[/]"
                )
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"Lyrics fetch failed for {track_info.title}: {error}")


def _render_tag_diffs(orig_info: TrackInfo, track_info: TrackInfo) -> list[str]:
    diff_lines: list[str] = []
    for field_info in dataclasses.fields(TrackInfo):
        if field_info.name in _SKIP_DIFF_FIELDS:
            continue
        old_val = getattr(orig_info, field_info.name)
        new_val = getattr(track_info, field_info.name)
        old_clean = None if old_val in (None, "", [], ()) else old_val
        new_clean = None if new_val in (None, "", [], ()) else new_val
        if old_clean != new_clean:
            if old_clean is None:
                color, sym = "green", "+"
            elif new_clean is None:
                color, sym = "red", "-"
            else:
                color, sym = "yellow", "*"
            diff_lines.append(
                f"\n       [{color}][{sym}] {field_info.name}: {old_clean} -> {new_clean}[/]"
            )
    return diff_lines


def process_single_track(
    file_path: Path,
    fetch_bpm: bool = True,
    fetch_lyrics: bool = True,
    fetch_itunes_art: bool = True,
    lastfm_api_key: str | None = None,
    acoustid_api_key: str | None = None,
    discogs_user_token: str | None = None,
    genius_api_token: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    album_mbid: str | None = None,
    album_track_mbids: dict[int, str] | None = None,
    cuesheet_content: str | None = None,
    album_mb_release_details: dict[str, object] | None = None,
    album_discogs_release: dict[str, object] | None = None,
    album_deezer_details: dict[str, Any] | None = None,
    album_itunes_details: dict[str, Any] | None = None,
    album_cover_path: Path | None = None,
) -> TrackInfo:
    wait_if_paused()
    LOG.start_buffering()
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        track_info = read_track_metadata(file_path)
        orig_info = dataclasses.replace(track_info)
        track_info.artist = resolve_artist_name(track_info.artist)

        LOG.info(f"🎧 Processing track: [white]{file_path.name}[/]")

        # 1. External metadata enrichment pipeline
        _enrich_acoustid(
            track_info,
            file_path,
            acoustid_api_key,
            album_track_mbids=album_track_mbids,
            force=force,
        )
        _enrich_musicbrainz(
            track_info,
            album_mbid,
            album_track_mbids,
            album_mb_release_details,
            force=force,
        )
        _enrich_itunes(track_info, album_itunes_details=album_itunes_details)
        _enrich_lastfm(track_info, lastfm_api_key)
        _enrich_discogs(track_info, discogs_user_token, album_discogs_release)
        _enrich_deezer(track_info, album_deezer_details, force=force)
        _enrich_genius(track_info, genius_api_token, force=force)
        _enrich_theaudiodb(track_info, force=force)

        # 2. Audio features, artwork, cuesheet & lyrics
        _enrich_cuesheet(track_info, file_path, cuesheet_content)
        _enrich_bpm(track_info, file_path, fetch_bpm, force=force)
        cover_image = (
            album_cover_path
            if album_cover_path and album_cover_path.exists()
            else _enrich_artwork(
                track_info,
                file_path,
                fetch_itunes_art,
                force=force,
                dry_run=dry_run,
            )
        )
        _enrich_lyrics(
            track_info,
            file_path,
            fetch_lyrics,
            force=force,
            dry_run=dry_run,
        )

        # 3. Compute tag diffs and persist metadata
        diff_lines = _render_tag_diffs(orig_info, track_info)
        if diff_lines or force:
            if not dry_run:
                write_track_metadata(track_info, cover_art_path=cover_image)
                get_library_state().record_track_state(file_path, status="TAGGED_OK")
                LOG.info(
                    f"   ∟ [green]✓[/] {file_path.name}: {len(diff_lines)} tag(s) updated.{''.join(diff_lines)}"
                )
            else:
                LOG.info(f"   ∟ [DRY-RUN] {file_path.name}{''.join(diff_lines)}")
        else:
            get_library_state().record_track_state(file_path, status="TAGGED_OK")
            LOG.info(
                f"   ∟ [bold dim]✨ SKIPPED:[/] [dim]{file_path.name}[/] [dim]is already perfect.[/]"
            )

        return track_info
    finally:
        LOG.stop_buffering()


def tag_album_folder(
    folder_path: Path,
    max_threads: int = 4,
    fetch_bpm: bool = True,
    fetch_replaygain: bool = True,
    fetch_lyrics: bool = True,
    fetch_itunes_art: bool = True,
    lastfm_api_key: str | None = None,
    acoustid_api_key: str | None = None,
    discogs_user_token: str | None = None,
    genius_api_token: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> list[TrackInfo]:
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Album folder not found: {folder_path}")

    global LAST_TAGGING_FAILURES, LAST_TAGGED_TRACKS
    LAST_TAGGING_FAILURES = []
    LAST_TAGGED_TRACKS = []

    all_audio_files = find_audio_files(folder_path, recursive=True)
    if not all_audio_files:
        return []

    # Incremental state index check
    state_mgr = get_library_state()
    if not force:
        outdated_files = set(state_mgr.filter_outdated_tracks(all_audio_files))
        if not outdated_files:
            LOG.info(
                f"✨ All {len(all_audio_files)} tracks are already up to date in library state index."
            )
            return [read_track_metadata(f) for f in all_audio_files]

    # Group tracks by album folder (parent directory)
    album_groups = group_files_by_parent(all_audio_files)

    results: list[TrackInfo] = []
    current_album_results: list[TrackInfo] = []

    with create_progress() as progress:
        task = progress.add_task("[cyan]Tagging tracks...", total=len(all_audio_files))

        with interactive_pause_listener(progress, task):
            executor = ThreadPoolExecutor(max_workers=max_threads)
            try:
                for album_dir, audio_files in album_groups.items():
                    wait_if_paused()
                    folder_name = album_dir.name
                    LOG.force_info(
                        f"📁 [bold cyan]Album:[/] [white]{folder_name}[/] [dim]({len(audio_files)} tracks)[/]"
                    )

                    # Pre-resolve Cuesheet content once for entire album
                    cue_files = list(album_dir.glob("*.cue"))
                    album_cue_content = (
                        read_cuesheet_content(cue_files[0]) if cue_files else None
                    )

                    # Batch Optimization: Fetch entire album track MBIDs, release details, Deezer, iTunes, and Discogs once per album
                    album_musicbrainz_id: str | None = None
                    album_track_mbids: dict[int, str] | None = None
                    album_mb_release_details: dict[str, object] | None = None
                    album_discogs_release: dict[str, object] | None = None
                    album_deezer_details: dict[str, Any] | None = None
                    album_itunes_details: dict[str, Any] | None = None
                    album_cover_path: Path | None = None

                    try:
                        sample_meta = read_track_metadata(audio_files[0])
                        sample_artist = sample_meta.album_artist or sample_meta.artist
                        sample_album = sample_meta.album
                        if sample_artist and sample_album:
                            release_info = search_musicbrainz_release(
                                sample_artist, sample_album
                            )
                            if release_info and release_info.get("id"):
                                album_musicbrainz_id = str(release_info["id"])
                                album_track_mbids = fetch_album_track_mbids(
                                    album_musicbrainz_id
                                )
                                album_mb_release_details = (
                                    fetch_musicbrainz_release_details(
                                        album_musicbrainz_id
                                    )
                                )
                            deezer_info = fetch_deezer_album_details(
                                sample_artist, sample_album
                            )
                            if deezer_info:
                                album_deezer_details = deezer_info
                            itunes_info = fetch_itunes_album_details(
                                sample_artist, sample_album
                            )
                            if itunes_info:
                                album_itunes_details = itunes_info
                            if discogs_user_token:
                                discogs_info = search_discogs_release(
                                    sample_artist,
                                    sample_album,
                                    user_token=discogs_user_token,
                                )
                                if discogs_info:
                                    album_discogs_release = discogs_info
                            if fetch_itunes_art:
                                cover_file = process_album_cover_art(
                                    album_dir,
                                    sample_artist,
                                    sample_album,
                                    musicbrainz_album_id=album_musicbrainz_id,
                                    force=force,
                                    dry_run=dry_run,
                                )
                                if cover_file:
                                    album_cover_path = cover_file
                    except _NETWORK_EXCEPTIONS as error:
                        LOG.debug(f"Pre-fetching album metadata failed: {error}")

                    current_album_results = []
                    future_to_file = {
                        executor.submit(
                            process_single_track,
                            file_path=audio_file,
                            fetch_bpm=fetch_bpm,
                            fetch_lyrics=fetch_lyrics,
                            fetch_itunes_art=fetch_itunes_art,
                            lastfm_api_key=lastfm_api_key,
                            acoustid_api_key=acoustid_api_key,
                            discogs_user_token=discogs_user_token,
                            genius_api_token=genius_api_token,
                            force=force,
                            dry_run=dry_run,
                            album_mbid=album_musicbrainz_id,
                            album_track_mbids=album_track_mbids,
                            cuesheet_content=album_cue_content,
                            album_mb_release_details=album_mb_release_details,
                            album_discogs_release=album_discogs_release,
                            album_deezer_details=album_deezer_details,
                            album_itunes_details=album_itunes_details,
                            album_cover_path=album_cover_path,
                        ): audio_file
                        for audio_file in audio_files
                    }

                    for future in as_completed(future_to_file):
                        wait_if_paused()
                        audio_file = future_to_file[future]
                        try:
                            track_info = future.result()
                            current_album_results.append(track_info)
                            LAST_TAGGED_TRACKS.append(track_info)
                        except _NETWORK_EXCEPTIONS as error:
                            LOG.warning(f"Failed to process {audio_file.name}: {error}")
                            LAST_TAGGING_FAILURES.append(
                                {
                                    "file": str(audio_file.resolve()),
                                    "filename": audio_file.name,
                                    "error": str(error),
                                    "error_type": type(error).__name__,
                                }
                            )
                        progress.advance(task)

                    if fetch_replaygain:
                        wait_if_paused()
                        calculate_album_replaygain(
                            audio_files,
                            force=force,
                            dry_run=dry_run,
                            max_threads=max_threads,
                        )

                    if current_album_results:
                        wait_if_paused()
                        primary_artist = (
                            current_album_results[0].album_artist
                            or current_album_results[0].artist
                        )
                        try:
                            process_artist_artwork(
                                album_dir, primary_artist, dry_run=dry_run
                            )
                        except _NETWORK_EXCEPTIONS as error:
                            LOG.debug(f"Artist art download failed: {error}")

                    results.extend(current_album_results)
                    current_album_results = []
                    future_to_file.clear()
                    gc.collect()
            except KeyboardInterrupt:
                executor.shutdown(wait=False, cancel_futures=True)
                results.extend(current_album_results)
                raise
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
                gc.collect()

    return results


LAST_NORMALIZED_COUNT: int = 0


def get_last_normalized_count() -> int:
    return LAST_NORMALIZED_COUNT


def normalize_single_track(
    file_path: Path,
    fetch_bpm: bool = False,
    force: bool = False,
    dry_run: bool = False,
) -> TrackInfo | None:
    """
    Locally cleans and normalizes metadata for a single audio file without any API requests.
    - Repairs mojibake/UTF-8 encoding via ftfy
    - Strips bracket junk: (Official Video), [FLAC], [320kbps], (2011 Remaster), [Explicit]
    - Cleans disambiguation suffixes: 'Armin (ROU)' -> 'Armin'
    - Canonicalizes genres via normalize_genre
    - Standardizes dates via normalize_date
    - Optionally calculates audio BPM locally
    """
    if not file_path.exists():
        return None

    try:
        current_info = read_track_metadata(file_path)
    except (OSError, ValueError, RuntimeError) as error:
        LOG.debug(f"Failed to read metadata for {file_path}: {error}")
        return None

    cleaned_artist = clean_disambiguation(ftfy.fix_text(current_info.artist or ""))
    cleaned_title_raw = clean_title(ftfy.fix_text(current_info.title or ""))
    cleaned_title = deduplicate_title_features(cleaned_title_raw)
    cleaned_album = ftfy.fix_text(current_info.album or "")
    if current_info.album_artist:
        cleaned_album_artist = clean_disambiguation(
            ftfy.fix_text(current_info.album_artist)
        )
    else:
        cleaned_album_artist = None

    cleaned_genre = normalize_genre(current_info.genre)
    cleaned_date = normalize_date(current_info.date)

    updated_bpm = current_info.bpm
    if fetch_bpm and (force or current_info.bpm is None):
        try:
            calculated = calculate_bpm(file_path)
            if calculated is not None:
                updated_bpm = calculated
        except (OSError, ValueError, RuntimeError) as error:
            LOG.debug(f"BPM calculation failed for {file_path}: {error}")

    updated_info = dataclasses.replace(
        current_info,
        artist=cleaned_artist or current_info.artist,
        title=cleaned_title or current_info.title,
        album=cleaned_album or current_info.album,
        album_artist=cleaned_album_artist,
        genre=cleaned_genre or current_info.genre,
        date=cleaned_date or current_info.date,
        bpm=updated_bpm,
    )

    if not dry_run:
        try:
            write_track_metadata(updated_info)
            get_library_state().record_track_state(file_path, "TAGGED_OK")
        except (OSError, ValueError, RuntimeError) as error:
            LOG.warning(f"Failed to save normalized tags for {file_path.name}: {error}")
            return None

    return updated_info


def normalize_library(
    directory: Path,
    fetch_bpm: bool = False,
    fetch_replaygain: bool = False,
    force: bool = False,
    dry_run: bool = False,
    max_threads: int = 4,
) -> list[TrackInfo]:
    global LAST_NORMALIZED_COUNT
    LAST_NORMALIZED_COUNT = 0

    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Directory not found: {directory}")

    LOG.info(f"Scanning for audio files in {directory}...")
    audio_files = find_audio_files(directory, recursive=True)
    if not audio_files:
        LOG.warning("No audio files found to normalize.")
        return []

    album_groups = group_files_by_parent(audio_files)
    results: list[TrackInfo] = []

    with create_progress() as progress:
        task = progress.add_task(
            "[cyan]Normalizing tracks (offline)...", total=len(audio_files)
        )
        with interactive_pause_listener(progress, task):
            executor = ThreadPoolExecutor(max_workers=max_threads)
            try:
                for files in album_groups.values():
                    wait_if_paused()
                    futures = {
                        executor.submit(
                            normalize_single_track,
                            file_path=f,
                            fetch_bpm=fetch_bpm,
                            force=force,
                            dry_run=dry_run,
                        ): f
                        for f in files
                    }
                    for future in as_completed(futures):
                        wait_if_paused()
                        res = future.result()
                        if res is not None:
                            results.append(res)
                            LAST_NORMALIZED_COUNT += 1
                        progress.advance(task)

                    if fetch_replaygain:
                        wait_if_paused()
                        calculate_album_replaygain(
                            files,
                            force=force,
                            dry_run=dry_run,
                            max_threads=max_threads,
                        )
                    futures.clear()
                    gc.collect()
            except KeyboardInterrupt:
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
                gc.collect()

    return results
