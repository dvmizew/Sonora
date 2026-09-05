import dataclasses
import gc
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import acoustid
import ftfy
import httpx
from musicbrainzngs import MusicBrainzError
from rapidfuzz import fuzz
from rich.markup import escape

from sonora.audio.art import process_album_cover_art, process_artist_artwork
from sonora.audio.bpm import calculate_bpm
from sonora.audio.cuesheet import read_cuesheet_content
from sonora.audio.key import detect_key_details, detect_musical_key
from sonora.audio.metadata import read_track_metadata, write_track_metadata
from sonora.audio.replaygain import calculate_album_replaygain
from sonora.core.config import get_config
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
    clean_unicode_punct,
    deduplicate_title_features,
    extract_balanced_features,
    find_audio_files,
    get_primary_artist,
    group_files_by_parent,
    is_valid_uuid,
    match_score,
    normalize_date,
    normalize_genre,
    normalize_str,
    resolve_artist_name,
    safe_float,
    safe_int,
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
        "is_alien",
    }
)


def get_last_tagging_failures() -> list[dict[str, str]]:
    """Return failures recorded during the latest tagging run."""
    return list(LAST_TAGGING_FAILURES)


def get_last_tagged_tracks() -> list[TrackInfo]:
    """Return successfully tagged tracks recorded during the latest tagging run."""
    return list(LAST_TAGGED_TRACKS)


def _has_diacritics(text: str | None) -> bool:
    if not text:
        return False
    return any(
        unicodedata.combining(c) for c in unicodedata.normalize("NFD", str(text))
    )


def _apply_mapping(
    track_info: TrackInfo,
    data: dict[str, Any],
    field_map: dict[str, str],
    force: bool = False,
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
        elif target_attr in ("date", "original_date"):
            normalized_d = normalize_date(val_str)
            if not normalized_d:
                continue
            val_str = normalized_d
        elif target_attr == "title":
            val_str = clean_unicode_punct(val_str)
            existing_title = getattr(track_info, "title", None)
            if (
                existing_title
                and _has_diacritics(str(existing_title))
                and not _has_diacritics(val_str)
            ):
                continue
            val_str = (
                deduplicate_title_features(val_str, primary_artist=track_info.artist)
                or val_str
            )
        elif target_attr in ("artist", "album", "album_artist"):
            val_str = clean_unicode_punct(val_str)
            existing_val = getattr(track_info, target_attr, None)
            if (
                existing_val
                and _has_diacritics(str(existing_val))
                and not _has_diacritics(val_str)
            ):
                continue

        if not getattr(track_info, target_attr) or force:
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


def _is_generic(text: str | None, placeholder: str) -> bool:
    if not text:
        return True
    val = text.strip().lower()
    return val in (
        "",
        placeholder.lower(),
        "unknown",
        f"unknown {placeholder.lower()}",
        "unknown track",
        "untitled",
    )


def _is_generic_title(title: str | None) -> bool:
    if _is_generic(title, "title"):
        return True
    val = str(title).strip().lower()
    return val.isdigit() or bool(
        re.match(r"^(track|audio\s*track|audiotrack|title)\s*\d*$", val)
    )


def _enrich_musicbrainz(
    track_info: TrackInfo,
    album_mbid: str | None,
    album_track_mbids: dict[int, str] | None,
    album_mb_release_details: dict[str, object] | None,
    force: bool = False,
) -> None:
    try:
        album_mbids = album_track_mbids or {}
        has_corrupt_identity = False
        if album_mb_release_details and track_info.track_number:
            tracks_by_pos = album_mb_release_details.get("tracks_by_position", {})
            if (
                isinstance(tracks_by_pos, dict)
                and track_info.track_number in tracks_by_pos
            ):
                candidate_rec = tracks_by_pos[track_info.track_number]
                if isinstance(candidate_rec, dict):
                    rec_isrc = candidate_rec.get("isrc")
                    rec_title = candidate_rec.get("title")
                    if (
                        track_info.isrc
                        and rec_isrc
                        and str(track_info.isrc).strip().upper()
                        == str(rec_isrc).strip().upper()
                        and rec_title
                        and clean_title(track_info.title).lower()
                        != clean_title(str(rec_title)).lower()
                    ):
                        has_corrupt_identity = True
                        LOG.info(
                            f"   ∟ 🩹 [Healer] Track {track_info.track_number} verified by ISRC match. "
                            f"Healing corrupted title '{track_info.title}' -> '{rec_title}'"
                        )

        candidate_mbid: str | None = None
        if (
            (
                not is_valid_uuid(track_info.musicbrainz_trackid)
                or force
                or has_corrupt_identity
            )
            and track_info.track_number
            and track_info.track_number in album_mbids
        ):
            potential_mbid = album_mbids[track_info.track_number]
            is_match = (
                has_corrupt_identity
                or not track_info.title
                or _is_generic_title(track_info.title)
            )
            if not is_match and album_mb_release_details:
                tracks_by_pos = album_mb_release_details.get("tracks_by_position", {})
                if (
                    isinstance(tracks_by_pos, dict)
                    and track_info.track_number in tracks_by_pos
                ):
                    cand_rec = tracks_by_pos[track_info.track_number]
                    if isinstance(cand_rec, dict):
                        cand_title = str(cand_rec.get("title") or "")
                        cand_artist = str(cand_rec.get("artist") or "")
                        if (
                            clean_title(track_info.title).lower()
                            == clean_title(cand_title).lower()
                            or match_score(
                                track_info.artist,
                                track_info.title,
                                cand_artist,
                                cand_title,
                            )
                            >= 70.0
                        ):
                            is_match = True
            elif not album_mb_release_details:
                is_match = True

            if is_match and is_valid_uuid(potential_mbid):
                candidate_mbid = potential_mbid
                track_info.musicbrainz_trackid = candidate_mbid
                LOG.info(
                    f"   ∟ 🏷️ [MusicBrainz Album Match] Found MBID: {candidate_mbid[:8]}..."
                )
        elif not is_valid_uuid(track_info.musicbrainz_trackid) or (
            force and not track_info.is_alien and not candidate_mbid
        ):
            mbid = fetch_track_mbid(track_info.artist, track_info.title)
            if is_valid_uuid(mbid):
                track_info.musicbrainz_trackid = mbid
                LOG.info(f"   ∟ 🏷️ [MusicBrainz] Found MBID: {mbid[:8]}...")

        if is_valid_uuid(album_mbid):
            if (
                force
                or not is_valid_uuid(track_info.musicbrainz_albumid)
                or album_mb_release_details is not None
            ):
                track_info.musicbrainz_albumid = str(album_mbid)
        elif not is_valid_uuid(track_info.musicbrainz_albumid):
            search_artist = (
                track_info.album_artist
                if track_info.album_artist
                else track_info.artist
            )
            release = search_musicbrainz_release(
                search_artist,
                track_info.album,
                expected_track_count=track_info.total_tracks,
            )
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
            fetched_rec = fetch_musicbrainz_recording_details(
                track_info.musicbrainz_trackid
            )
            if fetched_rec and isinstance(fetched_rec, dict):
                rec_isrc = fetched_rec.get("isrc")
                isrc_matches = bool(
                    track_info.isrc
                    and rec_isrc
                    and str(track_info.isrc).strip().upper()
                    == str(rec_isrc).strip().upper()
                )
                title_sim = match_score(
                    track_info.artist,
                    track_info.title,
                    str(fetched_rec.get("artist") or ""),
                    str(fetched_rec.get("title") or ""),
                )
                if isrc_matches or title_sim >= 70.0:
                    mb_rec = fetched_rec
                elif force:
                    LOG.debug(
                        f"Discarding mismatched pre-existing MBID {track_info.musicbrainz_trackid} "
                        f"('{fetched_rec.get('artist')} - {fetched_rec.get('title')}') "
                        f"for track '{track_info.artist} - {track_info.title}'"
                    )
                    track_info.musicbrainz_trackid = None

        if mb_rec and isinstance(mb_rec, dict):
            mb_map = {
                "isrc": "isrc",
                "disambiguation": "disambiguation",
                "composer": "composer",
                "lyricist": "lyricist",
                "producers": "producers",
                "remixer": "remixer",
                "musicbrainz_workid": "musicbrainz_workid",
            }
            rec_title = str(mb_rec.get("title") or "")
            title_matches = bool(
                rec_title
                and (
                    clean_title(track_info.title).lower()
                    == clean_title(rec_title).lower()
                    or match_score(
                        track_info.artist,
                        track_info.title,
                        str(mb_rec.get("artist") or ""),
                        rec_title,
                    )
                    >= 70.0
                )
            )
            if (
                has_corrupt_identity
                or not track_info.title
                or track_info.title == "Untitled"
                or (force and title_matches)
            ) and rec_title:
                mb_map["title"] = "title"
            if mb_rec.get("first-release-date") and (force or not track_info.date):
                mb_map["first-release-date"] = "date"
            _apply_mapping(
                track_info,
                mb_rec,
                mb_map,
                force=force or has_corrupt_identity,
            )
            rec_artist = mb_rec.get("artist")
            if rec_artist and isinstance(rec_artist, str):
                cleaned_artist = clean_unicode_punct(resolve_artist_name(rec_artist))
                if (
                    force
                    or has_corrupt_identity
                    or not track_info.artist
                    or track_info.artist.lower() in ("unknown", "unknown artist")
                    or (
                        album_mb_release_details is not None
                        and track_info.artist != cleaned_artist
                    )
                    or track_info.is_alien
                ):
                    track_info.artist = cleaned_artist

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
                mb_rel_map = {
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
                }
                if mb_rel.get("date") and (force or not track_info.date):
                    mb_rel_map["date"] = "date"
                if mb_rel.get("original_date"):
                    mb_rel_map["original_date"] = "original_date"
                _apply_mapping(
                    track_info,
                    mb_rel,
                    mb_rel_map,
                    force=force,
                )
                if mb_rel.get("title") and (
                    force
                    or not track_info.album
                    or track_info.album.lower() in ("unknown", "unknown album")
                    or (
                        album_mb_release_details is not None
                        and track_info.album != mb_rel.get("title")
                    )
                ):
                    track_info.album = clean_unicode_punct(str(mb_rel["title"]))
                if mb_rel.get("album_artist") and (
                    force
                    or not track_info.album_artist
                    or track_info.album_artist.lower() in ("unknown", "unknown artist")
                    or (
                        album_mb_release_details is not None
                        and track_info.album_artist != mb_rel.get("album_artist")
                    )
                ):
                    track_info.album_artist = clean_unicode_punct(
                        resolve_artist_name(str(mb_rel["album_artist"]))
                    )
                if force or not track_info.total_tracks:
                    total_tracks = safe_int(mb_rel.get("total_tracks"))
                    if total_tracks is not None:
                        track_info.total_tracks = total_tracks
                if force or not track_info.total_discs:
                    total_discs = safe_int(mb_rel.get("total_discs"))
                    if total_discs is not None:
                        track_info.total_discs = total_discs
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"MusicBrainz enrichment failed for {track_info.title}: {error}")


def _enrich_itunes(
    track_info: TrackInfo,
    album_itunes_details: dict[str, Any] | None = None,
    force: bool = False,
) -> None:
    try:
        data = None
        if album_itunes_details:
            t_by_num = album_itunes_details.get("tracks_by_number", {})
            t_by_title = album_itunes_details.get("tracks_by_title", {})
            clean_track_key = normalize_str(clean_title(track_info.title))
            if (
                isinstance(t_by_title, dict)
                and not _is_generic_title(track_info.title)
                and clean_track_key in t_by_title
            ):
                data = t_by_title[clean_track_key]
            elif (
                isinstance(t_by_title, dict)
                and not _is_generic_title(track_info.title)
                and normalize_str(track_info.title) in t_by_title
            ):
                data = t_by_title[normalize_str(track_info.title)]
            elif isinstance(t_by_num, dict) and track_info.track_number in t_by_num:
                data = t_by_num[track_info.track_number]

        if not data and (not track_info.genre or not track_info.advisory):
            data = fetch_itunes_track_metadata(track_info.artist, track_info.title)

        if data and isinstance(data, dict):
            itunes_map = {
                "genre": "genre",
                "advisory": "advisory",
                "copyright": "copyright",
                "itunes_trackid": "itunes_trackid",
                "itunes_collectionid": "itunes_collectionid",
                "itunes_artistid": "itunes_artistid",
                "release_country": "release_country",
            }
            if not track_info.date:
                itunes_map["date"] = "date"
            if (not track_info.title or track_info.title == "Untitled") and data.get(
                "trackName"
            ):
                itunes_map["trackName"] = "title"
            _apply_mapping(
                track_info,
                data,
                itunes_map,
                force=force,
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
    force: bool = False,
) -> None:
    if not discogs_user_token:
        return
    try:
        release = album_discogs_release or search_discogs_release(
            track_info.artist,
            track_info.album,
            user_token=discogs_user_token,
            expected_track_count=track_info.total_tracks,
        )
        if not release:
            return

        styles_val = release.get("styles")
        if isinstance(styles_val, list) and styles_val and not track_info.style:
            track_info.style = ", ".join(str(s) for s in styles_val[:3])

        genres_val = release.get("genres")
        if isinstance(genres_val, list) and genres_val and not track_info.genre:
            track_info.genre = normalize_genre(str(genres_val[0])) or track_info.genre

        discogs_rel_map = {
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
        }
        if not track_info.date:
            discogs_rel_map["released"] = "date"
            discogs_rel_map["year"] = "date"
        _apply_mapping(
            track_info,
            release,
            discogs_rel_map,
            force=force,
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
                discogs_track_map = {
                    "producers": "producers",
                    "remixer": "remixer",
                    "composer": "composer",
                }
                if (
                    not track_info.title or track_info.title == "Untitled"
                ) and specific.get("title"):
                    discogs_track_map["title"] = "title"
                _apply_mapping(
                    track_info,
                    specific,
                    discogs_track_map,
                    force=force,
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
            deezer_album_map = {
                "genre": "genre",
                "label": "label",
                "barcode": "barcode",
            }
            if not track_info.date:
                deezer_album_map["release_date"] = "date"
            _apply_mapping(
                track_info,
                album,
                deezer_album_map,
            )
            if (
                not is_valid_uuid(track_info.musicbrainz_albumid)
                and album.get("title")
                and (
                    force
                    or not track_info.album
                    or track_info.album.lower() in ("unknown", "unknown album")
                    or (
                        album_deezer_details is not None
                        and track_info.album != album.get("title")
                    )
                )
            ):
                track_info.album = clean_unicode_punct(str(album["title"]))
            if not is_valid_uuid(track_info.musicbrainz_albumid) and (
                force or not track_info.total_tracks
            ):
                nb_tracks = safe_int(album.get("nb_tracks"))
                if nb_tracks is not None:
                    track_info.total_tracks = nb_tracks

        track: dict[str, Any] | None = None
        if album and isinstance(album, dict):
            raw_pos = album.get("tracks_by_position")
            t_by_pos: dict[Any, Any] = raw_pos if isinstance(raw_pos, dict) else {}
            raw_title = album.get("tracks_by_title")
            t_by_title: dict[Any, Any] = (
                raw_title if isinstance(raw_title, dict) else {}
            )
            clean_track_key = normalize_str(clean_title(track_info.title))
            if isinstance(t_by_title, dict) and clean_track_key in t_by_title:
                track = t_by_title[clean_track_key]
            elif (
                isinstance(t_by_title, dict)
                and normalize_str(track_info.title) in t_by_title
            ):
                track = t_by_title[normalize_str(track_info.title)]
            elif isinstance(t_by_pos, dict) and track_info.track_number in t_by_pos:
                track = t_by_pos[track_info.track_number]

        if not track and (not track_info.isrc or not track_info.producers or force):
            track = fetch_deezer_track_details(track_info.artist, track_info.title)

        if not track or not isinstance(track, dict):
            return

        if track.get("featured_artists") and (not track_info.featured_artists or force):
            track_info.featured_artists = str(track["featured_artists"])
        if track.get("producers") and (not track_info.producers or force):
            track_info.producers = str(track["producers"])
        track_pos = safe_int(track.get("track_position"))
        if track_pos is not None and (
            track_info.track_number is None or force or album_deezer_details is not None
        ):
            track_info.track_number = track_pos
        if track_info.disc_number is None or force:
            disk_num = safe_int(track.get("disk_number"))
            if disk_num is not None:
                track_info.disc_number = disk_num
        if track.get("explicit_lyrics") and not track_info.advisory:
            track_info.advisory = "Explicit"
        deezer_track_map = {
            "isrc": "isrc",
            "composer": "composer",
            "lyricist": "lyricist",
        }
        if not track_info.date:
            deezer_track_map["release_date"] = "date"
        if (not track_info.title or track_info.title == "Untitled") and track.get(
            "title"
        ):
            deezer_track_map["title"] = "title"
        _apply_mapping(
            track_info,
            track,
            deezer_track_map,
            force=force,
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
        genius_map = {
            "genius_song_id": "genius_song_id",
            "writers": "composer",
        }
        if not track_info.date:
            genius_map["release_date"] = "date"
        _apply_mapping(
            track_info,
            genius_details,
            genius_map,
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
        if track_info.rating is None:
            rating_val = safe_float(tadb_details.get("rating"))
            if rating_val is not None:
                track_info.rating = rating_val
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


def _enrich_key(
    track_info: TrackInfo,
    file_path: Path,
    fetch_key: bool,
    force: bool = False,
) -> None:
    if fetch_key and (track_info.initial_key is None or force):
        try:
            key_details = detect_key_details(file_path)
            if key_details is not None:
                key_name, camelot, _ = key_details
                track_info.initial_key = key_name
                LOG.info(
                    f"   ∟ 🎵 Musical Key: [green]{escape(key_name)}[/] ({escape(camelot)})"
                )
        except (OSError, ValueError, RuntimeError) as error:
            LOG.debug(f"Key calculation failed for {track_info.title}: {error}")


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
                    f"   ∟ [green]✅ Saved {tag_type} lyrics for {escape(file_path.name)}[/]"
                )
            elif force:
                LOG.info(
                    f"   ∟ [yellow]🔄 Updated {tag_type} lyrics for {escape(file_path.name)}[/]"
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
                f"\n       [{color}][{sym}] {field_info.name}: {escape(str(old_clean))} -> {escape(str(new_clean))}[/]"
            )
    return diff_lines


def _resolve_album_track_position(
    track_info: TrackInfo,
    file_path: Path,
    album_mb_release_details: dict[str, object] | None = None,
    album_deezer_details: dict[str, Any] | None = None,
    album_itunes_details: dict[str, Any] | None = None,
    album_track_mbids: dict[int, str] | None = None,
) -> int | None:
    """
    Resolve the legitimate track position (1-indexed) of an audio file within an album release.
    Validates candidates against authoritative identifiers and title similarity to prevent
    corrupt positions from poisoning metadata.
    """
    # 1. Authoritative identifier match (ISRC / iTunes ID / MBID)
    clean_isrc = (
        str(track_info.isrc).strip().upper()
        if track_info.isrc
        and str(track_info.isrc).strip().upper() not in ("", "NONE", "NULL", "0")
        else None
    )
    if clean_isrc:
        if album_deezer_details:
            dz_tracks = album_deezer_details.get("tracks_by_position")
            if isinstance(dz_tracks, dict):
                for pos_key, trk in dz_tracks.items():
                    if (
                        isinstance(trk, dict)
                        and trk.get("isrc")
                        and str(trk["isrc"]).strip().upper() == clean_isrc
                    ):
                        pos_int = safe_int(pos_key)
                        if pos_int is not None:
                            return pos_int
        if album_mb_release_details:
            mb_tracks = album_mb_release_details.get("tracks_by_position")
            if isinstance(mb_tracks, dict):
                for pos_key, trk in mb_tracks.items():
                    if (
                        isinstance(trk, dict)
                        and trk.get("isrc")
                        and str(trk["isrc"]).strip().upper() == clean_isrc
                    ):
                        pos_int = safe_int(pos_key)
                        if pos_int is not None:
                            return pos_int

    clean_itunes_id = (
        str(track_info.itunes_trackid).strip()
        if track_info.itunes_trackid
        and str(track_info.itunes_trackid).strip() not in ("", "0", "None", "null")
        else None
    )
    if clean_itunes_id and album_itunes_details:
        it_tracks = album_itunes_details.get("tracks_by_position")
        if isinstance(it_tracks, dict):
            for pos_key, trk in it_tracks.items():
                if isinstance(trk, dict):
                    t_id = trk.get("itunes_trackid") or trk.get("trackId")
                    if t_id and str(t_id).strip() == clean_itunes_id:
                        pos_int = safe_int(pos_key)
                        if pos_int is not None:
                            return pos_int

    if is_valid_uuid(track_info.musicbrainz_trackid) and album_track_mbids:
        clean_mbid = str(track_info.musicbrainz_trackid).strip().lower()
        for pos_int, rec_mbid in album_track_mbids.items():
            if is_valid_uuid(rec_mbid) and str(rec_mbid).strip().lower() == clean_mbid:
                return pos_int

    def _get_album_track_details(pos: int) -> tuple[str | None, str | None]:
        cand_title: str | None = None
        cand_artist: str | None = None
        if album_mb_release_details:
            mb_tracks = album_mb_release_details.get("tracks_by_position")
            if isinstance(mb_tracks, dict) and pos in mb_tracks:
                rec = mb_tracks[pos]
                if isinstance(rec, dict):
                    cand_title = str(rec.get("title") or "") or None
                    cand_artist = str(rec.get("artist") or "") or None
        if not cand_title and album_deezer_details:
            dz_tracks = album_deezer_details.get("tracks_by_position")
            if isinstance(dz_tracks, dict) and pos in dz_tracks:
                rec = dz_tracks[pos]
                if isinstance(rec, dict):
                    cand_title = str(rec.get("title") or "") or None
                    cand_artist = str(rec.get("artist") or "") or None
        if not cand_title and album_itunes_details:
            it_tracks = album_itunes_details.get("tracks_by_position")
            if isinstance(it_tracks, dict) and pos in it_tracks:
                rec = it_tracks[pos]
                if isinstance(rec, dict):
                    cand_title = (
                        str(rec.get("trackName") or rec.get("title") or "") or None
                    )
                    cand_artist = (
                        str(rec.get("artistName") or rec.get("artist") or "") or None
                    )
        return cand_artist, cand_title

    def _pos_in_album(pos: int) -> bool:
        if album_track_mbids and pos in album_track_mbids:
            return True
        if album_deezer_details:
            dz = album_deezer_details.get("tracks_by_position")
            if isinstance(dz, dict) and pos in dz:
                return True
        if album_itunes_details:
            it = album_itunes_details.get("tracks_by_position")
            if isinstance(it, dict) and pos in it:
                return True
        if album_mb_release_details:
            mb = album_mb_release_details.get("tracks_by_position")
            if isinstance(mb, dict) and pos in mb:
                return True
        return False

    # 2. Candidate positions from filename prefix and existing track_number
    fn_pos: int | None = None
    fn_match = re.match(r"^(\d{1,3})\s*[-._\s]", file_path.name)
    if fn_match:
        fn_pos = safe_int(fn_match.group(1))

    candidates: list[int] = []
    if fn_pos is not None and _pos_in_album(fn_pos):
        candidates.append(fn_pos)
    if (
        track_info.track_number is not None
        and _pos_in_album(track_info.track_number)
        and track_info.track_number not in candidates
    ):
        candidates.append(track_info.track_number)

    # If title is generic or blank, candidate positions are our best information
    if not track_info.title or _is_generic_title(track_info.title):
        if candidates:
            return candidates[0]
        return None

    # Check candidate positions with title similarity
    has_any_cand_titles = False
    for cand_pos in candidates:
        cand_artist, cand_title = _get_album_track_details(cand_pos)
        if cand_title:
            has_any_cand_titles = True
            clean_cand = clean_title(cand_title).lower()
            clean_eff = clean_title(track_info.title).lower()
            if (
                clean_cand == clean_eff
                or match_score(
                    track_info.artist,
                    track_info.title,
                    str(cand_artist or ""),
                    cand_title,
                )
                >= 70.0
                or max(
                    fuzz.ratio(clean_eff, clean_cand),
                    fuzz.token_sort_ratio(clean_eff, clean_cand),
                )
                >= 75.0
            ):
                return cand_pos

    # If the album metadata has no track titles available (e.g. minimal MBID mapping),
    # fallback to the candidate position from filename/tag.
    if not has_any_cand_titles and candidates:
        return candidates[0]

    # 3. Search all positions in album release for matching title
    all_positions: set[int] = set()
    if album_track_mbids:
        all_positions.update(album_track_mbids.keys())
    for details in (
        album_mb_release_details,
        album_deezer_details,
        album_itunes_details,
    ):
        if details and isinstance(details, dict):
            t_by_p = details.get("tracks_by_position")
            if isinstance(t_by_p, dict):
                for p in t_by_p:
                    p_int = safe_int(p)
                    if p_int is not None:
                        all_positions.add(p_int)

    best_pos: int | None = None
    best_score: float = 0.0
    for pos in sorted(all_positions):
        cand_artist, cand_title = _get_album_track_details(pos)
        if cand_title:
            clean_cand = clean_title(cand_title).lower()
            clean_eff = clean_title(track_info.title).lower()
            if clean_cand == clean_eff:
                return pos
            score = max(
                match_score(
                    track_info.artist,
                    track_info.title,
                    str(cand_artist or ""),
                    cand_title,
                ),
                float(fuzz.ratio(clean_eff, clean_cand)),
                float(fuzz.token_sort_ratio(clean_eff, clean_cand)),
            )
            if score > best_score:
                best_score = score
                best_pos = pos

    if best_score >= 70.0 and best_pos is not None:
        return best_pos

    return None


def is_alien_album_track(
    track_info: TrackInfo,
    file_path: Path,
    target_album_artist: str | None,
    target_album_title: str | None,
    album_mb_release_details: dict[str, object] | None = None,
    album_deezer_details: dict[str, Any] | None = None,
    album_itunes_details: dict[str, Any] | None = None,
) -> bool:
    """
    Determine if a track located in an album directory is an outlier / alien track
    (e.g., an unrelated song by another artist accidentally mixed into an album folder).

    Returns True if the track has concrete metadata or filename evidence demonstrating
    it belongs to a completely different artist/song with no correlation to the album release.
    """
    # Without album context to compare against, the track cannot be deemed alien
    if (
        not target_album_artist
        and not target_album_title
        and not album_mb_release_details
        and not album_deezer_details
        and not album_itunes_details
    ):
        return False

    # Parse potential fallback artist/title from filename e.g. "03 - Lana Del Rey - Blue Jeans.flac"
    fn_stem = file_path.stem.strip()
    if fn_stem.isdigit():
        fn_clean = ""
    else:
        fn_clean = re.sub(r"^\d{1,3}\s*[-._\s]+", "", fn_stem).strip()
    fn_parts = [p.strip() for p in fn_clean.split(" - ") if p.strip()]

    fn_cand_artist: str | None = None
    fn_cand_title: str | None = None
    if len(fn_parts) >= 2:
        fn_cand_artist = fn_parts[0]
        fn_cand_title = " - ".join(fn_parts[1:])
    elif fn_clean and not fn_clean.isdigit():
        fn_cand_title = fn_clean

    eff_artist = (
        track_info.artist
        if not _is_generic(track_info.artist, "artist")
        else fn_cand_artist
    )
    eff_title = (
        track_info.title if not _is_generic_title(track_info.title) else fn_cand_title
    )
    eff_album = track_info.album if not _is_generic(track_info.album, "album") else None

    # Completely blank or generic identity cannot be definitively identified as alien
    if _is_generic(eff_artist, "artist") and _is_generic_title(eff_title):
        return False

    # Aggregate known album entities (artists, track titles, album titles, authoritative IDs)
    known_artists: set[str] = set()
    known_titles: set[str] = set()
    known_albums: set[str] = set()
    known_isrcs: set[str] = set()
    known_itunes_ids: set[str] = set()
    known_mbids: set[str] = set()

    if target_album_artist:
        known_artists.add(target_album_artist)
    if target_album_title:
        known_albums.add(target_album_title)

    for details in (
        album_mb_release_details,
        album_deezer_details,
        album_itunes_details,
    ):
        if not details or not isinstance(details, dict):
            continue
        art = details.get("album_artist") or details.get("artist")
        if art and isinstance(art, str):
            known_artists.add(art)
        alb = details.get("title")
        if alb and isinstance(alb, str):
            known_albums.add(alb)

        tracks_by_pos = details.get("tracks_by_position")
        if isinstance(tracks_by_pos, dict):
            for trk in tracks_by_pos.values():
                if isinstance(trk, dict):
                    if trk.get("title"):
                        known_titles.add(str(trk["title"]))
                    if trk.get("artist"):
                        known_artists.add(str(trk["artist"]))
                    isrc_val = trk.get("isrc")
                    if isrc_val:
                        known_isrcs.add(str(isrc_val).strip().upper())
                    rec_mbid = trk.get("recording_mbid")
                    if rec_mbid:
                        known_mbids.add(str(rec_mbid).strip().lower())
                    t_id = trk.get("itunes_trackid") or trk.get("trackId")
                    if t_id:
                        known_itunes_ids.add(str(t_id).strip())

        tracks_by_mbid = details.get("tracks_by_mbid")
        if isinstance(tracks_by_mbid, dict):
            for mbid_key in tracks_by_mbid:
                known_mbids.add(str(mbid_key).strip().lower())

        tracks_by_num = details.get("tracks_by_number")
        if isinstance(tracks_by_num, dict):
            for trk in tracks_by_num.values():
                if isinstance(trk, dict):
                    if trk.get("title"):
                        known_titles.add(str(trk["title"]))
                    t_name = trk.get("trackName")
                    if t_name:
                        known_titles.add(str(t_name))
                    t_id = trk.get("itunes_trackid") or trk.get("trackId")
                    if t_id:
                        known_itunes_ids.add(str(t_id).strip())

    # 0. Check authoritative metadata identifiers (ISRC, iTunes Track ID, Recording MBID).
    # If the track possesses an authoritative identifier that matches a track on the album,
    # it is conclusively a legitimate track of this release (even if its title tag was corrupted).
    clean_isrc = (
        str(track_info.isrc).strip().upper()
        if track_info.isrc
        and str(track_info.isrc).strip().upper() not in ("", "NONE", "NULL", "0")
        else None
    )
    if clean_isrc and clean_isrc in known_isrcs:
        return False

    clean_itunes_id = (
        str(track_info.itunes_trackid).strip()
        if track_info.itunes_trackid
        and str(track_info.itunes_trackid).strip() not in ("", "0", "None", "null")
        else None
    )
    if clean_itunes_id and clean_itunes_id in known_itunes_ids:
        return False

    if (
        is_valid_uuid(track_info.musicbrainz_trackid)
        and str(track_info.musicbrainz_trackid).strip().lower() in known_mbids
    ):
        return False

    # 1. Check title match across album tracklist
    title_matches = False
    if eff_title and not _is_generic_title(eff_title):
        clean_eff_t = clean_title(eff_title).lower()
        for kt in known_titles:
            clean_kt = clean_title(kt).lower()
            if clean_eff_t == clean_kt:
                title_matches = True
                break
            if len(clean_eff_t) > 3 and len(clean_kt) > 3:
                sim = max(
                    fuzz.ratio(clean_eff_t, clean_kt),
                    fuzz.token_sort_ratio(clean_eff_t, clean_kt),
                )
                if sim >= 70:
                    title_matches = True
                    break

    if title_matches:
        return False

    # If the album tracklist is known (3+ tracks) and this track has a concrete,
    # non-generic title that matches NONE of the album tracks:
    # It is an outlier / alien track (e.g. an unrelated artist's song or a track
    # from a different album) and must NOT be forced into the album's tracklist!
    if len(known_titles) >= 3 and not _is_generic_title(eff_title):
        return True

    # Check artist match against album artist or track artists
    artist_matches = False
    if eff_artist and not _is_generic(eff_artist, "artist"):
        norm_eff_a = normalize_str(eff_artist)
        pri_eff_a = normalize_str(get_primary_artist(eff_artist))
        resolved_eff_a = normalize_str(resolve_artist_name(eff_artist))

        for ka in known_artists:
            norm_ka = normalize_str(ka)
            pri_ka = normalize_str(get_primary_artist(ka))
            resolved_ka = normalize_str(resolve_artist_name(ka))

            if (
                norm_eff_a == norm_ka
                or pri_eff_a == pri_ka
                or resolved_eff_a == resolved_ka
            ):
                artist_matches = True
                break
            if (
                pri_eff_a
                and pri_ka
                and (pri_eff_a in pri_ka or pri_ka in pri_eff_a)
                and len(pri_eff_a) >= 4
                and len(pri_ka) >= 4
            ):
                artist_matches = True
                break
            if (
                max(
                    fuzz.ratio(norm_eff_a, norm_ka),
                    fuzz.token_set_ratio(norm_eff_a, norm_ka),
                )
                >= 65
            ):
                artist_matches = True
                break

    if artist_matches:
        return False

    # Check existing album tag match
    if eff_album and not _is_generic(eff_album, "album"):
        clean_eff_alb = clean_title(eff_album).lower()
        for kalb in known_albums:
            clean_kalb = clean_title(kalb).lower()
            if clean_eff_alb == clean_kalb:
                return False
            if fuzz.ratio(clean_eff_alb, clean_kalb) >= 70:
                return False

    # If artist fails to match, and artist is non-generic: Alien!
    return bool(
        eff_artist and not _is_generic(eff_artist, "artist") and not artist_matches
    )


def process_single_track(
    file_path: Path,
    fetch_bpm: bool = True,
    fetch_key: bool = True,
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
    target_album_artist: str | None = None,
    target_album_title: str | None = None,
) -> TrackInfo:
    wait_if_paused()
    LOG.start_buffering()
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        track_info = read_track_metadata(file_path)
        orig_info = dataclasses.replace(track_info)
        track_info.artist = resolve_artist_name(track_info.artist)

        # Check whether this track is an outlier / alien track in an album folder
        has_album_context = bool(
            album_track_mbids
            or album_mb_release_details
            or album_deezer_details
            or album_itunes_details
            or target_album_title
        )
        if has_album_context and is_alien_album_track(
            track_info=track_info,
            file_path=file_path,
            target_album_artist=target_album_artist,
            target_album_title=target_album_title,
            album_mb_release_details=album_mb_release_details,
            album_deezer_details=album_deezer_details,
            album_itunes_details=album_itunes_details,
        ):
            track_info.is_alien = True
            album_desc = (
                f"{target_album_artist} - {target_album_title}"
                if target_album_artist and target_album_title
                else (target_album_title or target_album_artist or "Album")
            )
            LOG.warning(
                f"⚠️  [bold yellow]Alien track detected in album folder:[/] [white]{escape(file_path.name)}[/]\n"
                f"   ∟ Found '[bold]{escape(track_info.artist)} - {escape(track_info.title)}'[/] inside '[bold]{escape(album_desc)}[/]'.\n"
                f"   ∟ [cyan]Shielding track from album metadata poisoning. Tagging independently as standalone track.[/]"
            )
            album_mbid = None
            album_track_mbids = None
            album_mb_release_details = None
            album_discogs_release = None
            album_deezer_details = None
            album_itunes_details = None
            album_cover_path = None
            cuesheet_content = None
            has_album_context = False

        # Align track position and verify identity against album release
        if has_album_context:
            resolved_pos = _resolve_album_track_position(
                track_info=track_info,
                file_path=file_path,
                album_mb_release_details=album_mb_release_details,
                album_deezer_details=album_deezer_details,
                album_itunes_details=album_itunes_details,
                album_track_mbids=album_track_mbids,
            )
            if resolved_pos is not None:
                if (
                    force
                    or track_info.track_number is None
                    or track_info.track_number != resolved_pos
                ):
                    track_info.track_number = resolved_pos
            elif not _is_generic_title(track_info.title):
                track_info.is_alien = True
                album_desc = (
                    f"{target_album_artist} - {target_album_title}"
                    if target_album_artist and target_album_title
                    else (target_album_title or target_album_artist or "Album")
                )
                LOG.warning(
                    f"⚠️  [bold yellow]Alien track detected in album folder:[/] [white]{escape(file_path.name)}[/]\n"
                    f"   ∟ Found '[bold]{escape(track_info.artist)} - {escape(track_info.title)}'[/] inside '[bold]{escape(album_desc)}[/]'.\n"
                    f"   ∟ [cyan]Shielding track from album metadata poisoning. Tagging independently as standalone track.[/]"
                )
                album_mbid = None
                album_track_mbids = None
                album_mb_release_details = None
                album_discogs_release = None
                album_deezer_details = None
                album_itunes_details = None
                album_cover_path = None
                cuesheet_content = None
                has_album_context = False

        LOG.info(f"🎧 Processing track: [white]{escape(file_path.name)}[/]")

        # 1. External metadata enrichment pipeline
        _enrich_musicbrainz(
            track_info,
            album_mbid,
            album_track_mbids,
            album_mb_release_details,
            force=force,
        )
        _enrich_acoustid(
            track_info,
            file_path,
            acoustid_api_key,
            album_track_mbids=album_track_mbids,
            force=force,
        )
        _enrich_itunes(
            track_info, album_itunes_details=album_itunes_details, force=force
        )
        _enrich_lastfm(track_info, lastfm_api_key)
        _enrich_discogs(
            track_info,
            discogs_user_token,
            album_discogs_release,
            force=force,
        )
        _enrich_deezer(track_info, album_deezer_details, force=force)
        _enrich_genius(track_info, genius_api_token, force=force)
        _enrich_theaudiodb(track_info, force=force)

        # 2. Audio features, artwork, cuesheet & lyrics
        _enrich_cuesheet(track_info, file_path, cuesheet_content)
        _enrich_bpm(track_info, file_path, fetch_bpm, force=force)
        _enrich_key(track_info, file_path, fetch_key, force=force)
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

        # 3. Deduplicate title features if present and clean unicode
        if track_info.title:
            track_info.title = clean_unicode_punct(track_info.title)
            track_info.title = deduplicate_title_features(
                track_info.title, primary_artist=track_info.artist
            )
        if track_info.artist:
            track_info.artist = clean_unicode_punct(track_info.artist)
        if track_info.album:
            track_info.album = clean_unicode_punct(track_info.album)
        if track_info.album_artist:
            track_info.album_artist = clean_unicode_punct(track_info.album_artist)

        # 4. Compute tag diffs and persist metadata
        diff_lines = _render_tag_diffs(orig_info, track_info)
        if diff_lines or force:
            if not dry_run:
                write_track_metadata(track_info, cover_art_path=cover_image)
                get_library_state().record_track_state(file_path, status="TAGGED_OK")
                LOG.info(
                    f"   ∟ [green]✓[/] {escape(file_path.name)}: {len(diff_lines)} tag(s) updated.{''.join(diff_lines)}"
                )
            else:
                LOG.info(
                    f"   ∟ [DRY-RUN] {escape(file_path.name)}{''.join(diff_lines)}"
                )
        else:
            get_library_state().record_track_state(file_path, status="TAGGED_OK")
            LOG.info(
                f"   ∟ [bold dim]✨ SKIPPED:[/] [dim]{escape(file_path.name)}[/] [dim]is already perfect.[/]"
            )

        return track_info
    finally:
        LOG.stop_buffering()


def _resolve_album_folder_identity(
    album_dir: Path, audio_files: list[Path]
) -> tuple[str | None, str | None]:
    """
    Resolve canonical artist and album name for a directory by combining
    directory structure ("Artist - Album" or parent artist directory) with
    metadata consensus across audio files.
    """
    effective_dir = album_dir
    if get_config().is_disc_folder(album_dir.name) and album_dir.parent != album_dir:
        effective_dir = album_dir.parent

    folder_name = effective_dir.name
    folder_artist: str | None = None
    folder_album: str | None = None

    if " - " in folder_name:
        parts = folder_name.split(" - ", 1)
        folder_artist, folder_album = parts[0].strip(), parts[1].strip()
    elif (
        effective_dir.parent
        and effective_dir.parent.name
        and not get_config().is_generic_container(effective_dir.parent.name)
    ):
        folder_artist = effective_dir.parent.name
        folder_album = folder_name

    if folder_artist and get_config().is_generic_container(folder_artist):
        folder_artist = None
    if folder_album and get_config().is_generic_container(folder_album):
        folder_album = None

    album_counts: dict[str, int] = {}
    artist_counts: dict[str, int] = {}

    for af in audio_files[:10]:
        try:
            m = read_track_metadata(af)
            if m.album and not get_config().is_generic_container(m.album):
                album_counts[m.album] = album_counts.get(m.album, 0) + 1
            art = m.album_artist or m.artist
            if art and not get_config().is_generic_container(art):
                artist_counts[art] = artist_counts.get(art, 0) + 1
        except (OSError, ValueError, RuntimeError):
            pass

    consensus_album = (
        max(album_counts, key=lambda k: album_counts[k]) if album_counts else None
    )
    consensus_artist = (
        max(artist_counts, key=lambda k: artist_counts[k]) if artist_counts else None
    )

    resolved_artist = folder_artist or consensus_artist
    resolved_album = folder_album or consensus_album

    return resolved_artist, resolved_album


def tag_album_folder(
    folder_path: Path,
    max_threads: int = 4,
    fetch_bpm: bool = True,
    fetch_key: bool = True,
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
                        f"📁 [bold cyan]Album:[/] [white]{escape(folder_name)}[/] [dim]({len(audio_files)} tracks)[/]"
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
                    sample_artist: str | None = None
                    sample_album: str | None = None

                    try:
                        (
                            sample_artist,
                            sample_album,
                        ) = _resolve_album_folder_identity(album_dir, audio_files)
                        if sample_artist and sample_album:
                            release_info = search_musicbrainz_release(
                                sample_artist,
                                sample_album,
                                expected_track_count=len(audio_files),
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
                                    expected_track_count=len(audio_files),
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
                            fetch_key=fetch_key,
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
                            target_album_artist=sample_artist,
                            target_album_title=sample_album,
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
                            LOG.warning(
                                f"Failed to process {escape(audio_file.name)}: {error}"
                            )
                            LAST_TAGGING_FAILURES.append(
                                {
                                    "file": str(audio_file.resolve()),
                                    "filename": audio_file.name,
                                    "error": str(error),
                                    "error_type": type(error).__name__,
                                }
                            )
                        progress.advance(task)

                    valid_album_files = [
                        t.file_path for t in current_album_results if not t.is_alien
                    ]
                    if fetch_replaygain and valid_album_files:
                        wait_if_paused()
                        calculate_album_replaygain(
                            valid_album_files,
                            force=force,
                            dry_run=dry_run,
                            max_threads=max_threads,
                        )

                    if current_album_results:
                        wait_if_paused()
                        valid_tracks = [
                            t for t in current_album_results if not t.is_alien
                        ]
                        rep_track = (
                            valid_tracks[0]
                            if valid_tracks
                            else current_album_results[0]
                        )
                        primary_artist = (
                            sample_artist or rep_track.album_artist or rep_track.artist
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
    fetch_key: bool = False,
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
    - Optionally detects musical key locally
    """
    if not file_path.exists():
        return None

    try:
        current_info = read_track_metadata(file_path)
    except (OSError, ValueError, RuntimeError) as error:
        LOG.debug(f"Failed to read metadata for {file_path}: {error}")
        return None

    cleaned_artist = clean_unicode_punct(
        clean_disambiguation(ftfy.fix_text(current_info.artist or ""))
    )
    cleaned_title = clean_unicode_punct(
        deduplicate_title_features(
            ftfy.fix_text(current_info.title or ""), primary_artist=cleaned_artist
        )
    )
    cleaned_album = clean_unicode_punct(ftfy.fix_text(current_info.album or ""))
    if current_info.album_artist:
        cleaned_album_artist = clean_unicode_punct(
            clean_disambiguation(ftfy.fix_text(current_info.album_artist))
        )
    else:
        cleaned_album_artist = None

    cleaned_genre = normalize_genre(current_info.genre)
    cleaned_date = normalize_date(current_info.date)

    cleaned_featured = current_info.featured_artists
    _, feat_list, _, _ = extract_balanced_features(cleaned_title)
    if feat_list:
        cleaned_featured = ", ".join(feat_list)

    updated_bpm = current_info.bpm
    if fetch_bpm and (force or current_info.bpm is None):
        try:
            calculated = calculate_bpm(file_path)
            if calculated is not None:
                updated_bpm = calculated
        except (OSError, ValueError, RuntimeError) as error:
            LOG.debug(f"BPM calculation failed for {file_path}: {error}")

    updated_key = current_info.initial_key
    if fetch_key and (force or current_info.initial_key is None):
        try:
            calculated_key = detect_musical_key(file_path)
            if calculated_key is not None:
                updated_key = calculated_key
        except (OSError, ValueError, RuntimeError) as error:
            LOG.debug(f"Key calculation failed for {file_path}: {error}")

    updated_info = dataclasses.replace(
        current_info,
        artist=cleaned_artist or current_info.artist,
        title=cleaned_title or current_info.title,
        album=cleaned_album or current_info.album,
        album_artist=cleaned_album_artist,
        featured_artists=cleaned_featured,
        genre=cleaned_genre or current_info.genre,
        date=cleaned_date or current_info.date,
        bpm=updated_bpm,
        initial_key=updated_key,
    )

    if not dry_run:
        try:
            write_track_metadata(updated_info)
            get_library_state().record_track_state(file_path, "TAGGED_OK")
        except (OSError, ValueError, RuntimeError) as error:
            LOG.warning(
                f"Failed to save normalized tags for {escape(file_path.name)}: {error}"
            )
            return None

    return updated_info


def normalize_library(
    directory: Path,
    fetch_bpm: bool = False,
    fetch_key: bool = False,
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
                            fetch_key=fetch_key,
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
