import dataclasses
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from sonora.audio.art import process_album_cover_art, process_artist_artwork
from sonora.audio.bpm import calculate_bpm
from sonora.audio.cuesheet import read_cuesheet_content
from sonora.audio.metadata import read_track_metadata, write_track_metadata
from sonora.audio.replaygain import calculate_album_replaygain
from sonora.core.logger import LOG, create_progress
from sonora.core.models import TrackInfo
from sonora.core.utils import (
    find_audio_files,
    is_valid_uuid,
    normalize_date,
    normalize_genre,
    resolve_artist_name,
)
from sonora.services.acoustid import lookup_acoustid
from sonora.services.deezer import (
    fetch_deezer_album_details,
    fetch_deezer_track_details,
)
from sonora.services.discogs import search_discogs_release
from sonora.services.genius import fetch_genius_song_details
from sonora.services.itunes import fetch_itunes_track_metadata
from sonora.services.lastfm import fetch_lastfm_tags, fetch_lastfm_track_stats
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
) -> TrackInfo:
    LOG.start_buffering()
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        track_info = read_track_metadata(file_path)
        orig_info = dataclasses.replace(track_info)
        track_info.artist = resolve_artist_name(track_info.artist)

        LOG.info(f"🎧 Processing track: [white]{file_path.name}[/]")

        # 1. AcoustID fingerprinting + WebService lookup
        if (
            not is_valid_uuid(track_info.musicbrainz_trackid) or force
        ) and acoustid_api_key:
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
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as error:
                LOG.debug(f"AcoustID lookup failed for {track_info.title}: {error}")

        # 2. Check pre-fetched album track MBIDs map first (1 single API call per album!)
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
            try:
                mbid = fetch_track_mbid(track_info.artist, track_info.title)
                if is_valid_uuid(mbid):
                    track_info.musicbrainz_trackid = mbid
                    LOG.info(f"   ∟ 🏷️ [MusicBrainz] Found MBID: {mbid[:8]}...")
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as error:
                LOG.debug(f"MusicBrainz lookup failed for {track_info.title}: {error}")

        # 3. Fetch MusicBrainz Album ID and Release Details
        if not is_valid_uuid(track_info.musicbrainz_albumid):
            if is_valid_uuid(album_mbid):
                track_info.musicbrainz_albumid = str(album_mbid)
            else:
                try:
                    search_artist = (
                        track_info.album_artist
                        if track_info.album_artist
                        else track_info.artist
                    )
                    release = search_musicbrainz_release(
                        search_artist, track_info.album
                    )
                    if release:
                        musicbrainz_id = release.get("id")
                        if is_valid_uuid(str(musicbrainz_id)):
                            track_info.musicbrainz_albumid = str(musicbrainz_id)
                        if not track_info.date:
                            date_str = release.get("date")
                            if isinstance(date_str, str) and len(date_str) >= 4:
                                track_info.date = normalize_date(date_str)
                except (httpx.HTTPError, OSError, ValueError, RuntimeError) as error:
                    LOG.debug(
                        f"MusicBrainz Album lookup failed for {track_info.title}: {error}"
                    )

        # 3b. Deep MusicBrainz Recording & Release Details
        if is_valid_uuid(track_info.musicbrainz_trackid):
            try:
                mb_rec = fetch_musicbrainz_recording_details(
                    track_info.musicbrainz_trackid
                )
                if mb_rec:
                    if mb_rec.get("isrc") and not track_info.isrc:
                        track_info.isrc = str(mb_rec["isrc"])
                    if mb_rec.get("disambiguation") and not track_info.disambiguation:
                        track_info.disambiguation = str(mb_rec["disambiguation"])
                    if mb_rec.get("composer") and not track_info.composer:
                        track_info.composer = str(mb_rec["composer"])
                    if mb_rec.get("lyricist") and not track_info.lyricist:
                        track_info.lyricist = str(mb_rec["lyricist"])
                    if mb_rec.get("producers") and not track_info.producers:
                        track_info.producers = str(mb_rec["producers"])
                    if mb_rec.get("remixer") and not track_info.remixer:
                        track_info.remixer = str(mb_rec["remixer"])
                    if (
                        mb_rec.get("musicbrainz_workid")
                        and not track_info.musicbrainz_workid
                    ):
                        track_info.musicbrainz_workid = str(
                            mb_rec["musicbrainz_workid"]
                        )
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as error:
                LOG.debug(
                    f"MusicBrainz recording details failed for {track_info.title}: {error}"
                )

        if is_valid_uuid(track_info.musicbrainz_albumid):
            try:
                mb_rel = fetch_musicbrainz_release_details(
                    track_info.musicbrainz_albumid
                )
                if mb_rel:
                    if mb_rel.get("barcode") and not track_info.barcode:
                        track_info.barcode = str(mb_rel["barcode"])
                    if mb_rel.get("release_country") and not track_info.release_country:
                        track_info.release_country = str(mb_rel["release_country"])
                    if mb_rel.get("release_status") and not track_info.release_status:
                        track_info.release_status = str(mb_rel["release_status"])
                    if mb_rel.get("release_type") and not track_info.release_type:
                        track_info.release_type = str(mb_rel["release_type"])
                    if (
                        mb_rel.get("musicbrainz_releasegroupid")
                        and not track_info.musicbrainz_releasegroupid
                    ):
                        track_info.musicbrainz_releasegroupid = str(
                            mb_rel["musicbrainz_releasegroupid"]
                        )
                    if mb_rel.get("label") and not track_info.label:
                        track_info.label = str(mb_rel["label"])
                    if mb_rel.get("catalog_number") and not track_info.catalog_number:
                        track_info.catalog_number = str(mb_rel["catalog_number"])
                    if mb_rel.get("media") and not track_info.media:
                        track_info.media = str(mb_rel["media"])
                    if mb_rel.get("total_tracks") and not track_info.total_tracks:
                        track_info.total_tracks = int(str(mb_rel["total_tracks"]))
                    if mb_rel.get("total_discs") and not track_info.total_discs:
                        track_info.total_discs = int(str(mb_rel["total_discs"]))
                    if mb_rel.get("language") and not track_info.language:
                        track_info.language = str(mb_rel["language"])
                    if mb_rel.get("script") and not track_info.script:
                        track_info.script = str(mb_rel["script"])
                    if mb_rel.get("artist_sort") and not track_info.artist_sort:
                        track_info.artist_sort = str(mb_rel["artist_sort"])
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as error:
                LOG.debug(
                    f"MusicBrainz release details failed for {track_info.title}: {error}"
                )

        # 4. iTunes metadata enrichment (Genre, Advisory, Copyright, Track IDs, Country)
        try:
            itunes_data = fetch_itunes_track_metadata(
                track_info.artist, track_info.title
            )
            if itunes_data:
                if itunes_data.get("genre") and not track_info.genre:
                    normalized_genre = normalize_genre(str(itunes_data["genre"]))
                    if normalized_genre:
                        track_info.genre = normalized_genre
                if itunes_data.get("advisory") and not track_info.advisory:
                    track_info.advisory = str(itunes_data["advisory"])
                if itunes_data.get("copyright") and not track_info.copyright:
                    track_info.copyright = str(itunes_data["copyright"])
                if itunes_data.get("itunes_trackid") and not track_info.itunes_trackid:
                    track_info.itunes_trackid = str(itunes_data["itunes_trackid"])
                if (
                    itunes_data.get("itunes_collectionid")
                    and not track_info.itunes_collectionid
                ):
                    track_info.itunes_collectionid = str(
                        itunes_data["itunes_collectionid"]
                    )
                if (
                    itunes_data.get("itunes_artistid")
                    and not track_info.itunes_artistid
                ):
                    track_info.itunes_artistid = str(itunes_data["itunes_artistid"])
                if (
                    itunes_data.get("release_country")
                    and not track_info.release_country
                ):
                    track_info.release_country = str(itunes_data["release_country"])
                if itunes_data.get("date") and not track_info.date:
                    track_info.date = normalize_date(str(itunes_data["date"]))
        except (httpx.HTTPError, OSError, ValueError, RuntimeError) as error:
            LOG.debug(f"iTunes track lookup failed for {track_info.title}: {error}")

        # 5. Fetch Last.fm genre/mood tags
        if lastfm_api_key:
            try:
                tags = fetch_lastfm_tags(
                    track_info.artist,
                    track_info.title,
                    api_key=lastfm_api_key,
                    mbid=track_info.musicbrainz_trackid,
                )
                if tags and not track_info.genre:
                    raw_genre = tags[0]
                    normalized_genre = normalize_genre(raw_genre)
                    if normalized_genre:
                        track_info.genre = normalized_genre
                        LOG.info(f"   ∟ 🏷️ [Last.fm] Genre: [cyan]{normalized_genre}[/]")
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as error:
                LOG.debug(f"Last.fm lookup failed for {track_info.title}: {error}")

        # 6. Discogs metadata enrichment
        if discogs_user_token:
            try:
                release = search_discogs_release(
                    track_info.artist,
                    track_info.album,
                    user_token=discogs_user_token,
                )
                if release:
                    if release.get("id") and not track_info.discogs_release_id:
                        track_info.discogs_release_id = str(release["id"])
                    if release.get("artist_id") and not track_info.discogs_artist_id:
                        track_info.discogs_artist_id = str(release["artist_id"])
                    if release.get("released") and not track_info.date:
                        track_info.date = normalize_date(str(release["released"]))
                    elif release.get("year") and not track_info.date:
                        track_info.date = normalize_date(str(release["year"]))

                    genres_val = release.get("genres")
                    if (
                        isinstance(genres_val, list)
                        and genres_val
                        and not track_info.genre
                    ):
                        raw_genre = str(genres_val[0])
                        normalized_genre = normalize_genre(raw_genre)
                        if normalized_genre:
                            track_info.genre = normalized_genre

                    styles_val = release.get("styles")
                    if (
                        isinstance(styles_val, list)
                        and styles_val
                        and not track_info.style
                    ):
                        track_info.style = ", ".join(
                            str(style_item) for style_item in styles_val[:3]
                        )

                    if release.get("country") and not track_info.release_country:
                        track_info.release_country = str(release["country"])
                    if release.get("label") and not track_info.label:
                        track_info.label = str(release["label"])
                    if release.get("catalog_number") and not track_info.catalog_number:
                        track_info.catalog_number = str(release["catalog_number"])
                    if release.get("barcode") and not track_info.barcode:
                        track_info.barcode = str(release["barcode"])
                    if release.get("media") and not track_info.media:
                        track_info.media = str(release["media"])
                    if release.get("composer") and not track_info.composer:
                        track_info.composer = str(release["composer"])

                    # Check track-specific credits or album-level credits for producers and remixer
                    track_credits_dict = release.get("track_credits")
                    track_specific_credits = None
                    if isinstance(track_credits_dict, dict):
                        track_key = (
                            str(track_info.track_number)
                            if track_info.track_number
                            else None
                        )
                        if track_key and track_key in track_credits_dict:
                            track_specific_credits = track_credits_dict[track_key]
                        elif (
                            track_info.title
                            and track_info.title.lower() in track_credits_dict
                        ):
                            track_specific_credits = track_credits_dict[
                                track_info.title.lower()
                            ]

                    if isinstance(track_specific_credits, dict):
                        if (
                            track_specific_credits.get("producers")
                            and not track_info.producers
                        ):
                            track_info.producers = str(
                                track_specific_credits["producers"]
                            )
                        if (
                            track_specific_credits.get("remixer")
                            and not track_info.remixer
                        ):
                            track_info.remixer = str(track_specific_credits["remixer"])

                    if release.get("producers") and not track_info.producers:
                        track_info.producers = str(release["producers"])
                    if release.get("remixer") and not track_info.remixer:
                        track_info.remixer = str(release["remixer"])
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as error:
                LOG.debug(f"Discogs lookup failed for {track_info.title}: {error}")

        # 7. Deezer metadata enrichment (BPM, Gain, Contributors, ISRC, Label, Barcode, Release Date, Genre)
        try:
            deezer_album = fetch_deezer_album_details(
                track_info.artist, track_info.album
            )
            if deezer_album:
                if deezer_album.get("label") and not track_info.label:
                    track_info.label = str(deezer_album["label"])
                if deezer_album.get("barcode") and not track_info.barcode:
                    track_info.barcode = str(deezer_album["barcode"])
                if deezer_album.get("release_date") and not track_info.date:
                    track_info.date = normalize_date(str(deezer_album["release_date"]))
                if deezer_album.get("genre") and not track_info.genre:
                    normalized_genre = normalize_genre(str(deezer_album["genre"]))
                    if normalized_genre:
                        track_info.genre = normalized_genre

            deezer_track = fetch_deezer_track_details(
                track_info.artist, track_info.title
            )
            if deezer_track:
                if deezer_track.get("isrc") and not track_info.isrc:
                    track_info.isrc = str(deezer_track["isrc"])
                if deezer_track.get("bpm") and (track_info.bpm is None or force):
                    track_info.bpm = float(str(deezer_track["bpm"]))
                if (
                    deezer_track.get("gain") is not None
                    and track_info.replaygain_track_gain is None
                ):
                    track_info.replaygain_track_gain = float(str(deezer_track["gain"]))
                if deezer_track.get("featured_artists") and (
                    not track_info.featured_artists or force
                ):
                    track_info.featured_artists = str(deezer_track["featured_artists"])
                if deezer_track.get("producers") and (
                    not track_info.producers or force
                ):
                    track_info.producers = str(deezer_track["producers"])
                if deezer_track.get("explicit_lyrics") and not track_info.advisory:
                    track_info.advisory = "Explicit"
        except (httpx.HTTPError, OSError, ValueError, RuntimeError) as error:
            LOG.debug(f"Deezer enrichment failed for {track_info.title}: {error}")

        # 8. Genius song details (description, genius_song_id, featured_artists, producers)
        if genius_api_token:
            try:
                genius_details = fetch_genius_song_details(
                    track_info.artist,
                    track_info.title,
                    api_token=genius_api_token,
                )
                if genius_details:
                    if genius_details.get("description") and (
                        not track_info.comment or force
                    ):
                        track_info.comment = str(genius_details["description"])
                    if genius_details.get("genius_song_id"):
                        track_info.genius_song_id = str(
                            genius_details["genius_song_id"]
                        )
                    if genius_details.get("featured_artists") and (
                        not track_info.featured_artists or force
                    ):
                        track_info.featured_artists = str(
                            genius_details["featured_artists"]
                        )
                    if genius_details.get("producers") and (
                        not track_info.producers or force
                    ):
                        track_info.producers = str(genius_details["producers"])
                    LOG.info("   ∟ 📝 [Genius] Fetched song details & credits")
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as error:
                LOG.debug(f"Genius lookup failed for {track_info.title}: {error}")

        # 9. Last.fm stats (listeners, playcount)
        if lastfm_api_key:
            try:
                stats = fetch_lastfm_track_stats(
                    track_info.artist, track_info.title, api_key=lastfm_api_key
                )
                if stats:
                    if stats.get("listeners"):
                        track_info.listeners = stats["listeners"]
                    if stats.get("playcount"):
                        track_info.playcount = stats["playcount"]
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as error:
                LOG.debug(
                    f"Last.fm stats lookup failed for {track_info.title}: {error}"
                )

        # 10. TheAudioDB Details (mood, style, initial_key, rating, music_video_url, description)
        try:
            tadb_details = fetch_theaudiodb_track_details(
                track_info.artist, track_info.title
            )
            if tadb_details:
                if (
                    tadb_details.get("music_video_url")
                    and not track_info.music_video_url
                ):
                    track_info.music_video_url = str(tadb_details["music_video_url"])
                if tadb_details.get("mood") and not track_info.mood:
                    track_info.mood = str(tadb_details["mood"])
                if tadb_details.get("style") and not track_info.style:
                    track_info.style = str(tadb_details["style"])
                if tadb_details.get("initial_key") and not track_info.initial_key:
                    track_info.initial_key = str(tadb_details["initial_key"])
                if tadb_details.get("rating") is not None and track_info.rating is None:
                    track_info.rating = float(str(tadb_details["rating"]))
                if tadb_details.get("description") and (
                    not track_info.comment or force
                ):
                    track_info.comment = str(tadb_details["description"])
        except (OSError, ValueError, RuntimeError) as error:
            LOG.debug(
                f"TheAudioDB details lookup failed for {track_info.title}: {error}"
            )

        # 11. Embed Cuesheet content
        if cuesheet_content:
            track_info.cuesheet = cuesheet_content
        elif not track_info.cuesheet:
            cue_files = list(file_path.parent.glob("*.cue"))
            if cue_files:
                track_info.cuesheet = read_cuesheet_content(cue_files[0])

        # 7. Calculate BPM
        if fetch_bpm and (track_info.bpm is None or force):
            try:
                bpm = calculate_bpm(file_path)
                if bpm is not None:
                    track_info.bpm = bpm
                    LOG.info(f"   ∟ 🎵 BPM Calculated: [green]{bpm}[/]")
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as error:
                LOG.debug(f"BPM calculation failed for {track_info.title}: {error}")

        # 8. Fetch iTunes Cover Art (Download only, don't embed yet)
        cover_image_file = None
        if fetch_itunes_art:
            try:
                cover_image_file = process_album_cover_art(
                    file_path.parent,
                    track_info.artist,
                    track_info.album,
                    musicbrainz_album_id=track_info.musicbrainz_albumid,
                    force=force,
                    dry_run=dry_run,
                )
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as error:
                LOG.debug(
                    f"Cover art downloading failed for {track_info.title}: {error}"
                )
                cover_image_file = None

        # 9. Fetch & write .lrc lyrics file (Quality Upgrade: enhanced (3) > line-synced (2) > plain (1))
        if fetch_lyrics:
            try:
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
                    LOG.info(f"   ∟ ✅ Saved {tag_type} lyrics for {file_path.name}")
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as error:
                LOG.debug(f"Lyrics fetch failed for {track_info.title}: {error}")

        # Compute exact tag diffs (compare dataclass fields)
        diff_lines = []
        _SKIP_FIELDS = {
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
        for field_info in dataclasses.fields(TrackInfo):
            if field_info.name in _SKIP_FIELDS:
                continue
            old_value = getattr(orig_info, field_info.name)
            new_value = getattr(track_info, field_info.name)
            if old_value != new_value:
                diff_lines.append(
                    f"\n       [*] {field_info.name}: {old_value} -> {new_value}"
                )

        # 10. Write metadata tags back to file only if changed
        if diff_lines or force:
            if not dry_run:
                write_track_metadata(track_info, cover_art_path=cover_image_file)
                LOG.info(
                    f"   ∟ [green]✓[/] {file_path.name}: {len(diff_lines)} tag(s) updated.{''.join(diff_lines)}"
                )
            else:
                LOG.info(f"   ∟ [DRY-RUN] {file_path.name}{''.join(diff_lines)}")
        else:
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

    all_audio_files = find_audio_files(folder_path, recursive=True)
    if not all_audio_files:
        return []

    # Group tracks by album folder (parent directory)
    album_groups: dict[Path, list[Path]] = {}
    for audio_file in all_audio_files:
        album_groups.setdefault(audio_file.parent, []).append(audio_file)

    results: list[TrackInfo] = []

    with create_progress() as progress:
        task = progress.add_task("[cyan]Tagging tracks...", total=len(all_audio_files))
        executor = ThreadPoolExecutor(max_workers=max_threads)
        try:
            for album_dir, audio_files in album_groups.items():
                folder_name = album_dir.name
                LOG.force_info(
                    f"📁 [bold cyan]Album:[/] [white]{folder_name}[/] [dim]({len(audio_files)} tracks)[/]"
                )

                # Pre-resolve Cuesheet content once for entire album
                cue_files = list(album_dir.glob("*.cue"))
                album_cue_content = (
                    read_cuesheet_content(cue_files[0]) if cue_files else None
                )

                # Batch Optimization: Fetch entire album track MBIDs and album MBID in 1 single API call
                album_musicbrainz_id: str | None = None
                album_track_mbids: dict[int, str] | None = None
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
                except (
                    httpx.HTTPError,
                    OSError,
                    ValueError,
                    RuntimeError,
                ) as error:
                    LOG.debug(f"Pre-fetching album track MBIDs failed: {error}")

                album_results: list[TrackInfo] = []
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
                    ): audio_file
                    for audio_file in audio_files
                }

                for future in as_completed(future_to_file):
                    audio_file = future_to_file[future]
                    try:
                        track_info = future.result()
                        album_results.append(track_info)
                    except (
                        httpx.HTTPError,
                        OSError,
                        ValueError,
                        RuntimeError,
                    ) as error:
                        LOG.warning(f"Failed to process {audio_file.name}: {error}")
                    progress.advance(task)

                if fetch_replaygain:
                    calculate_album_replaygain(
                        audio_files,
                        force=force,
                        dry_run=dry_run,
                        max_threads=max_threads,
                    )

                if album_results:
                    primary_artist = (
                        album_results[0].album_artist or album_results[0].artist
                    )
                    try:
                        process_artist_artwork(
                            album_dir, primary_artist, dry_run=dry_run
                        )
                    except (OSError, ValueError, RuntimeError) as error:
                        LOG.debug(f"Artist art download failed: {error}")

                results.extend(album_results)
        except KeyboardInterrupt:
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    return results
