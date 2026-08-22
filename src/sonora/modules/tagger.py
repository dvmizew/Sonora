import dataclasses
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from sonora.audio.art import process_album_cover_art, process_artist_artwork
from sonora.audio.bpm import calculate_bpm
from sonora.audio.cuesheet import read_cuesheet_content
from sonora.audio.metadata import read_track_metadata, write_track_metadata
from sonora.audio.replaygain import calculate_album_replaygain
from sonora.core.constants import ARTIST_ALIASES, SUPPORTED_EXTS
from sonora.core.logger import CONSOLE, LOG
from sonora.core.models import TrackInfo
from sonora.core.utils import normalize_str
from sonora.services.acoustid import lookup_acoustid
from sonora.services.discogs import search_discogs_release
from sonora.services.lastfm import fetch_lastfm_tags
from sonora.services.lyrics import process_track_lyrics
from sonora.services.musicbrainz import fetch_track_mbid


def normalize_artist_alias(artist: str) -> str:
    """Normalize artist name based on ARTIST_ALIASES table."""
    return ARTIST_ALIASES.get(normalize_str(artist), artist.strip())




def process_single_track(
    file_path: Path,
    fetch_bpm: bool = True,
    fetch_lyrics: bool = True,
    fetch_itunes_art: bool = True,
    lastfm_api_key: str | None = None,
    acoustid_api_key: str | None = None,
    discogs_user_token: str | None = None,
    genius_api_token: str | None = None,
    options: dict | None = None,
) -> TrackInfo:
    LOG.start_buffering()
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        orig_info = read_track_metadata(file_path)
        track_info = read_track_metadata(file_path)
        track_info.artist = normalize_artist_alias(track_info.artist)

        options = options or {}
        force = options.get("force", False)
        dry_run = options.get("dry_run", False)

        LOG.info(f"🎧 Processing track: [white]{file_path.name}[/]")

        # 1. Respect existing MBID or prioritize AcoustID (most exact) over text search
        if (not track_info.musicbrainz_trackid or force) and acoustid_api_key:
            try:
                acoustid_mbid = lookup_acoustid(file_path, api_key=acoustid_api_key)
                if acoustid_mbid:
                    track_info.musicbrainz_trackid = acoustid_mbid
                    LOG.info(f"   ∟ 🎯 [acoustid] Matched MBID: {acoustid_mbid[:8]}...")
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as e:
                LOG.debug(f"AcoustID lookup failed for {track_info.title}: {e}")

        # 2. Check pre-fetched album track MBIDs map first (1 single API call per album!)
        album_mbids = options.get("album_track_mbids", {}) if isinstance(options, dict) else {}
        if (not track_info.musicbrainz_trackid or force) and track_info.track_number and track_info.track_number in album_mbids:
            track_info.musicbrainz_trackid = album_mbids[track_info.track_number]
            mbid_str = str(track_info.musicbrainz_trackid)
            LOG.info(f"   ∟ 🏷️ [MusicBrainz Album Match] Found MBID: {mbid_str[:8]}...")
        elif not track_info.musicbrainz_trackid or force:
            try:
                mbid = fetch_track_mbid(track_info.artist, track_info.title)
                if mbid:
                    track_info.musicbrainz_trackid = mbid
                    LOG.info(f"   ∟ 🏷️ [MusicBrainz] Found MBID: {mbid[:8]}...")
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as e:
                LOG.debug(f"MusicBrainz lookup failed for {track_info.title}: {e}")

        # 3. Fetch MusicBrainz Album ID via Discography Optimization
        if not track_info.musicbrainz_albumid:
            try:
                from sonora.services.musicbrainz import search_musicbrainz_release
                search_artist = track_info.album_artist if track_info.album_artist else track_info.artist
                release = search_musicbrainz_release(search_artist, track_info.album)
                if release:
                    mb_id = release.get("id")
                    if mb_id is not None:
                        track_info.musicbrainz_albumid = str(mb_id)
                    # Also opportunistically set year/genre if missing
                    if not track_info.date:
                        from sonora.core.utils import normalize_date
                        date_str = release.get("date")
                        if isinstance(date_str, str) and len(date_str) >= 4:
                            track_info.date = normalize_date(date_str)
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as e:
                LOG.debug(f"MusicBrainz Album lookup failed for {track_info.title}: {e}")

        # 4. Fetch Last.fm genre/mood tags
        if lastfm_api_key:
            try:
                tags = fetch_lastfm_tags(
                    track_info.artist,
                    track_info.title,
                    api_key=lastfm_api_key,
                    mbid=track_info.musicbrainz_trackid,
                )
                if tags:
                    from sonora.core.utils import normalize_genre
                    raw_genre = tags[0]
                    norm_genre = normalize_genre(raw_genre)
                    if norm_genre:
                        track_info.genre = norm_genre
                        LOG.info(f"   ∟ 🏷️ [Last.fm] Genre: [cyan]{norm_genre}[/]")
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as e:
                LOG.debug(f"Last.fm lookup failed for {track_info.title}: {e}")

        # 5. Discogs fallback lookup for release metadata
        if discogs_user_token:
            try:
                release = search_discogs_release(track_info.artist, track_info.album, user_token=discogs_user_token)
                if release:
                    if release.get("id") and not track_info.discogs_release_id:
                        track_info.discogs_release_id = str(release["id"])
                    if release.get("year") and not track_info.date:
                        from sonora.core.utils import normalize_date
                        track_info.date = normalize_date(str(release["year"]))
                    genres_val = release.get("genres")
                    if isinstance(genres_val, list) and genres_val and not track_info.genre:
                        from sonora.core.utils import normalize_genre
                        raw_genre = str(genres_val[0])
                        norm_genre = normalize_genre(raw_genre)
                        if norm_genre:
                            track_info.genre = norm_genre
                    if release.get("country") and not track_info.release_country:
                        track_info.release_country = str(release["country"])
                    if release.get("label") and not track_info.label:
                        track_info.label = str(release["label"])
                    if release.get("catalog_number") and not track_info.catalog_number:
                        track_info.catalog_number = str(release["catalog_number"])
                    if release.get("barcode") and not track_info.barcode:
                        track_info.barcode = str(release["barcode"])
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as e:
                LOG.debug(f"Discogs lookup failed for {track_info.title}: {e}")

        # 6. Genius song details (description, genius_song_id, featured_artists, producers)
        if genius_api_token:
            try:
                from sonora.services.genius import fetch_genius_song_details
                g_details = fetch_genius_song_details(track_info.artist, track_info.title, api_token=genius_api_token)
                if g_details:
                    if g_details.get("description") and (not track_info.comment or force):
                        track_info.comment = str(g_details["description"])
                    if g_details.get("genius_song_id"):
                        track_info.genius_song_id = str(g_details["genius_song_id"])
                    if g_details.get("featured_artists") and (not track_info.featured_artists or force):
                        track_info.featured_artists = str(g_details["featured_artists"])
                    if g_details.get("producers") and (not track_info.producers or force):
                        track_info.producers = str(g_details["producers"])
                    LOG.info("   ∟ 📝 [Genius] Fetched song details & credits")
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as e:
                LOG.debug(f"Genius lookup failed for {track_info.title}: {e}")

        # 6a. Last.fm stats (listeners, playcount)
        if lastfm_api_key:
            try:
                from sonora.services.lastfm import fetch_lastfm_track_stats
                stats = fetch_lastfm_track_stats(track_info.artist, track_info.title, api_key=lastfm_api_key)
                if stats:
                    if stats.get("listeners"):
                        track_info.listeners = stats["listeners"]
                    if stats.get("playcount"):
                        track_info.playcount = stats["playcount"]
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as e:
                LOG.debug(f"Last.fm stats lookup failed for {track_info.title}: {e}")

        # 6b. TheAudioDB video URL
        try:
            from sonora.services.theaudiodb import fetch_track_video_url
            vid_url = fetch_track_video_url(track_info.artist, track_info.title)
            if vid_url:
                track_info.music_video_url = vid_url
        except (OSError, ValueError, RuntimeError) as e:
            LOG.debug(f"TheAudioDB video lookup failed for {track_info.title}: {e}")

        # 6b. Embed Cuesheet content if .cue file exists in directory
        cue_files = list(file_path.parent.glob("*.cue"))
        if cue_files:
            cuesheet_content = read_cuesheet_content(cue_files[0])
            if cuesheet_content:
                track_info.cuesheet = cuesheet_content

        # 7. Calculate BPM (Skip if already present unless force)
        if fetch_bpm and (track_info.bpm is None or force):
            try:
                bpm = calculate_bpm(file_path)
                if bpm is not None:
                    track_info.bpm = bpm
                    LOG.info(f"   ∟ 🎵 BPM Calculated: [green]{bpm}[/]")
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as e:
                LOG.debug(f"BPM calculation failed for {track_info.title}: {e}")

        # 8. Fetch iTunes Cover Art (Download only, don't embed yet)
        cover_jpg = None
        if fetch_itunes_art:
            try:
                cover_jpg = process_album_cover_art(
                    file_path.parent,
                    track_info.artist,
                    track_info.album,
                    mb_album_id=track_info.musicbrainz_albumid,
                    force=force,
                    dry_run=dry_run,
                )
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as e:
                LOG.debug(f"Cover art downloading failed for {track_info.title}: {e}")
                cover_jpg = None

        # 9. Fetch & write .lrc lyrics file (Quality Upgrade: enhanced (3) > line-synced (2) > plain (1))
        if fetch_lyrics:
            try:
                lrc_text, tag_type = process_track_lyrics(
                    file_path,
                    track_info.artist,
                    track_info.title,
                    force=force,
                    dry_run=dry_run,
                    isrc=track_info.isrc,
                )
                if lrc_text and tag_type:
                    track_info.lyrics = lrc_text
                    LOG.info(f"   ∟ ✅ Saved {tag_type} lyrics for {file_path.name}")
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as e:
                LOG.debug(f"Lyrics fetch failed for {track_info.title}: {e}")

        # Compute exact tag diffs (compare dataclass fields)
        diff_lines = []
        _SKIP_FIELDS = {"file_path", "lyrics", "synced_lyrics", "acoustid_fingerprint",
                        "sample_rate", "bitrate", "channels", "is_lossless",
                        "art_width", "art_height"}
        for f in dataclasses.fields(TrackInfo):
            if f.name in _SKIP_FIELDS:
                continue
            old_val = getattr(orig_info, f.name)
            new_val = getattr(track_info, f.name)
            if old_val != new_val:
                diff_lines.append(f"\n       [*] {f.name}: {old_val} -> {new_val}")

        # 10. Write metadata tags back to file only if changed
        if diff_lines or force:
            if not dry_run:
                write_track_metadata(track_info, cover_art_path=cover_jpg)
                LOG.info(f"   ∟ [green]✓[/] {file_path.name}: {len(diff_lines)} tag(s) updated.{''.join(diff_lines)}")
            else:
                LOG.info(f"   ∟ [DRY-RUN] {file_path.name}{''.join(diff_lines)}")
        else:
            LOG.info(f"   ∟ [bold dim]✨ SKIPPED:[/] [dim]{file_path.name}[/] [dim]is already perfect.[/]")

        return track_info
    finally:
        LOG.stop_buffering()


def tag_album_folder(
    folder_path: Path,
    max_workers: int = 4,
    **options: Any
) -> list[TrackInfo]:
    valid_options = {
        "fetch_bpm", "fetch_replaygain", "fetch_lyrics", "fetch_itunes_art",
        "lastfm_api_key", "acoustid_api_key", "discogs_user_token",
        "genius_api_token", "options"
    }
    invalid = set(options.keys()) - valid_options
    if invalid:
        raise ValueError(f"Invalid options passed to tag_album_folder: {invalid}")

    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Album folder not found: {folder_path}")

    # Check if folder_path contains child directories with audio files
    sub_dirs = [
        d for d in sorted(folder_path.iterdir())
        if d.is_dir() and any(p.is_file() and p.suffix.lower() in SUPPORTED_EXTS for p in d.rglob("*"))
    ]

    # If folder_path has child album directories and NO direct audio files in its root, tag each sub-album independently
    if sub_dirs and not any(p.is_file() and p.suffix.lower() in SUPPORTED_EXTS for p in folder_path.glob("*")):
        all_results: list[TrackInfo] = []
        for sub in sub_dirs:
            res = tag_album_folder(sub, max_workers=max_workers, **options)
            all_results.extend(res)
        return all_results

    audio_files = sorted([
        p for p in folder_path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ])

    folder_name = folder_path.name
    LOG.force_info(f"📁 [bold cyan]Album:[/] [white]{folder_name}[/] [dim]({len(audio_files)} tracks)[/]")

    if not audio_files:
        return []

    # Batch Optimization: Fetch entire album track MBIDs in 1 single API call
    try:
        from sonora.services.musicbrainz import (
            fetch_album_track_mbids,
            search_musicbrainz_release,
        )
        sample_meta = read_track_metadata(audio_files[0])
        s_artist = sample_meta.album_artist or sample_meta.artist
        s_album = sample_meta.album
        if s_artist and s_album:
            rel = search_musicbrainz_release(s_artist, s_album)
            if rel and rel.get("id"):
                album_mbids = fetch_album_track_mbids(str(rel["id"]))
                if options.get("options") is None:
                    options["options"] = {}
                options["options"]["album_track_mbids"] = album_mbids
    except (httpx.HTTPError, OSError, ValueError, RuntimeError) as e:
        LOG.debug(f"Pre-fetching album track MBIDs failed: {e}")

    results: list[TrackInfo] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {}
        for file_p in audio_files:
            future = executor.submit(
                process_single_track, 
                file_path=file_p,
                fetch_bpm=options.get("fetch_bpm", True),
                fetch_lyrics=options.get("fetch_lyrics", True),
                fetch_itunes_art=options.get("fetch_itunes_art", True),
                lastfm_api_key=options.get("lastfm_api_key"),
                acoustid_api_key=options.get("acoustid_api_key"),
                discogs_user_token=options.get("discogs_user_token"),
                genius_api_token=options.get("genius_api_token"),
                options=options.get("options")
            )
            future_to_file[future] = file_p
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=CONSOLE
        ) as progress:
            task = progress.add_task("[cyan]Tagging tracks...", total=len(audio_files))
            for future in as_completed(future_to_file):
                file_p = future_to_file[future]
                try:
                    info = future.result()
                    results.append(info)
                except (httpx.HTTPError, OSError, ValueError, RuntimeError) as e:
                    LOG.warning(f"Failed to process {file_p.name}: {e}")
                progress.advance(task)
    if options.get("fetch_replaygain", True):
        calculate_album_replaygain(audio_files, options=options)

    if results:
        primary_artist = results[0].album_artist or results[0].artist
        try:
            process_artist_artwork(folder_path, primary_artist, dry_run=options.get("dry_run", False))
        except (OSError, ValueError, RuntimeError) as e:
            LOG.debug(f"Artist art download failed: {e}")

    return results
