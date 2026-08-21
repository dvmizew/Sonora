import dataclasses
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from sonora.audio.art import check_image_similarity
from sonora.audio.bpm import calculate_bpm
from sonora.audio.cuesheet import read_cuesheet_content
from sonora.audio.metadata import read_track_metadata, write_track_metadata
from sonora.audio.replaygain import calculate_album_replaygain
from sonora.core.constants import ARTIST_ALIASES, SUPPORTED_EXTS
from sonora.core.exceptions import APIServiceError, AudioProcessingError, MetadataError
from sonora.core.logger import CONSOLE, LOG
from sonora.core.models import TrackInfo
from sonora.core.utils import normalize_str
from sonora.services.acoustid import lookup_acoustid
from sonora.services.discogs import search_discogs_release
from sonora.services.genius import fetch_genius_description
from sonora.services.itunes import fetch_itunes_cover_art_url
from sonora.services.lastfm import fetch_lastfm_tags
from sonora.services.lyrics import fetch_synced_lyrics
from sonora.services.musicbrainz import fetch_track_mbid
from sonora.services.theaudiodb import fetch_artist_images


def normalize_artist_alias(artist: str) -> str:
    """Normalize artist name based on ARTIST_ALIASES table."""
    lowered = normalize_str(artist)
    return ARTIST_ALIASES.get(lowered, artist.strip())

_cover_locks: dict[Path, threading.Lock] = {}
_cover_meta_lock = threading.Lock()

def _get_cover_lock(album_dir: Path) -> threading.Lock:
    with _cover_meta_lock:
        if len(_cover_locks) > 1000:
            _cover_locks.clear()
        if album_dir not in _cover_locks:
            _cover_locks[album_dir] = threading.Lock()
        return _cover_locks[album_dir]


def process_artist_art(artist_name: str, folder: Path) -> None:
    """Ensure artist.jpg (avatar) and banner.jpg (wide header) exist in artist's root folder."""
    if not artist_name or artist_name in ["Various Artists", "Unknown Artist"]:
        return

    parent = folder.parent
    artist_dir = parent if parent.name not in ["FLAC", "Music", ""] and folder.name != "Singles" else folder
    if parent.name == "Singles":
        artist_dir = parent.parent

    has_artist_img = any((artist_dir / n).exists() for n in ["artist.jpg", "artist.png", "folder.jpg"])
    has_banner_img = any((artist_dir / n).exists() for n in ["banner.jpg", "banner.png", "fanart.jpg"])

    if has_artist_img and has_banner_img:
        return

    thumb_bytes, banner_bytes = fetch_artist_images(artist_name)
    if thumb_bytes and not has_artist_img:
        try:
            (artist_dir / "artist.jpg").write_bytes(thumb_bytes)
            LOG.info(f"   ∟ 👤 Downloaded artist avatar: {artist_name} -> artist.jpg")
        except OSError as e:
            LOG.debug(f"Failed to write artist avatar: {e}")

    if banner_bytes and not has_banner_img:
        try:
            (artist_dir / "banner.jpg").write_bytes(banner_bytes)
            LOG.info(f"   ∟ 🎨 Downloaded artist banner: {artist_name} -> banner.jpg")
        except OSError as e:
            LOG.debug(f"Failed to write artist banner: {e}")


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
    """
    Process and tag a single audio file with metadata, artwork, lyrics, BPM, and ReplayGain.
    """
    LOG.start_buffering()
    try:
        if not file_path.exists():
            raise AudioProcessingError(f"File not found: {file_path}")

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
            except APIServiceError as e:
                LOG.debug(f"AcoustID lookup failed for {track_info.title}: {e}")

        # 2. Fallback to MusicBrainz text search by existing tags if AcoustID failed
        if not track_info.musicbrainz_trackid or force:
            try:
                mbid = fetch_track_mbid(track_info.artist, track_info.title)
                if mbid:
                    track_info.musicbrainz_trackid = mbid
                    LOG.info(f"   ∟ 🏷️ [MusicBrainz] Found MBID: {mbid[:8]}...")
            except APIServiceError as e:
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
            except APIServiceError as e:
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
            except APIServiceError as e:
                LOG.debug(f"Last.fm lookup failed for {track_info.title}: {e}")

        # 5. Discogs fallback lookup for release metadata
        if discogs_user_token and not track_info.genre:
            try:
                release = search_discogs_release(track_info.artist, track_info.album, user_token=discogs_user_token)
                if release:
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
            except APIServiceError as e:
                LOG.debug(f"Discogs lookup failed for {track_info.title}: {e}")

        # 6. Genius description as COMMENT tag
        if genius_api_token:
            try:
                desc = fetch_genius_description(track_info.artist, track_info.title, api_token=genius_api_token)
                if desc:
                    track_info.comment = desc
                    LOG.info("   ∟ 📝 [Genius] Fetched description")
            except APIServiceError as e:
                LOG.debug(f"Genius lookup failed for {track_info.title}: {e}")

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
            except AudioProcessingError as e:
                LOG.debug(f"BPM calculation failed for {track_info.title}: {e}")

        # 8. Fetch iTunes Cover Art (Download only, don't embed yet)
        cover_jpg = None
        if fetch_itunes_art and file_path.suffix.lower() == ".flac":
            try:
                cover_jpg = file_path.parent / "cover.jpg"
                # Do not block all threads on network I/O unless necessary
                with _get_cover_lock(file_path.parent):
                    art_downloaded = cover_jpg.exists() and not force
                    if not art_downloaded and not dry_run:
                        # Mark as downloaded (or downloading) so other threads skip
                        cover_jpg.touch()
                
                if not art_downloaded:
                    art_url = fetch_itunes_cover_art_url(track_info.artist, track_info.album)
                    if art_url:
                        from sonora.core.http import SESSION
                        resp = SESSION.get(art_url, timeout=15)
                        resp.raise_for_status()
                        new_art_bytes = resp.content

                        with _get_cover_lock(file_path.parent):
                            if not dry_run:
                                existing_bytes = cover_jpg.read_bytes() if (cover_jpg.exists() and cover_jpg.stat().st_size > 0) else None
                                if existing_bytes and not check_image_similarity(existing_bytes, new_art_bytes):
                                    LOG.info("   ∟ 🖼️  Skipped iTunes cover upgrade: visual mismatch")
                                else:
                                    cover_jpg.write_bytes(new_art_bytes)
                                    LOG.info("   ∟ 🖼️  Downloaded Cover Art")
                            else:
                                LOG.info(f"[DRY-RUN] Would download cover art to {cover_jpg.name}")
                    else:
                        # Remove placeholder if fetch failed
                        with _get_cover_lock(file_path.parent):
                            if not dry_run and cover_jpg.exists() and cover_jpg.stat().st_size == 0:
                                cover_jpg.unlink()
                
                # Ensure we only use valid cover jpgs
                with _get_cover_lock(file_path.parent):
                    if not dry_run and (not cover_jpg.exists() or cover_jpg.stat().st_size == 0):
                        cover_jpg = None
            except (APIServiceError, OSError) as e:
                LOG.debug(f"Cover art downloading failed for {track_info.title}: {e}")
                with _get_cover_lock(file_path.parent):
                    if not dry_run and cover_jpg and cover_jpg.exists() and cover_jpg.stat().st_size == 0:
                        cover_jpg.unlink()
                cover_jpg = None

        # 9. Fetch & write .lrc lyrics file (Skip if already present unless force)
        lrc_path = file_path.with_suffix(".lrc")
        enhanced_path = file_path.with_suffix(".enhanced.lrc")
        already_has_lyrics = lrc_path.exists() or enhanced_path.exists() or bool(track_info.synced_lyrics)
        if fetch_lyrics and (not already_has_lyrics or force):
            try:
                lrc = fetch_synced_lyrics(track_info.artist, track_info.title, isrc=track_info.isrc)
                if lrc:
                    if re.search(r"<\d{1,2}:\d{2}[\.:]\d{2,3}>", lrc):
                        plain_lrc = re.sub(r"<\d{1,2}:\d{2}[\.:]\d{2,3}>", "", lrc)
                        if not dry_run:
                            enhanced_path.write_text(lrc, encoding="utf-8")
                            lrc_path.write_text(plain_lrc, encoding="utf-8")
                            LOG.info(f"   ∟ [bold green]✅ Saved[/] [bold cyan]enhanced[/] lyrics for [white]{file_path.name}[/]")
                        else:
                            LOG.info(f"[DRY-RUN] Would write enhanced lyrics to {enhanced_path.name}")
                    else:
                        if not dry_run:
                            lrc_path.write_text(lrc, encoding="utf-8")
                            LOG.info(f"   ∟ [bold green]✅ Saved[/] [bold cyan]synced[/] lyrics for [white]{file_path.name}[/]")
                        else:
                            LOG.info(f"[DRY-RUN] Would write plain lyrics to {lrc_path.name}")
            except (APIServiceError, OSError) as e:
                LOG.debug(f"Lyrics fetch failed for {track_info.title}: {e}")

        # Compute exact tag diffs as in initial/script.py (compare dataclass fields)
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

        # 10. Write metadata tags back to file only if changed (matches initial/script.py)
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
        raise AudioProcessingError(f"Album folder not found: {folder_path}")

    audio_files = sorted([
        p for p in folder_path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ])

    folder_name = folder_path.name
    LOG.force_info(f"📁 [bold cyan]Album:[/] [white]{folder_name}[/] [dim]({len(audio_files)} tracks)[/]")

    if not audio_files:
        return []

    # Off-by-N track number normalization check (matches initial prototype)
    track_numbers = []
    for f_p in audio_files:
        try:
            meta = read_track_metadata(f_p)
            if meta.track_number is not None:
                track_numbers.append(meta.track_number)
        except (MetadataError, OSError) as e:
            LOG.debug(f"Could not read metadata for track number check on {f_p.name}: {e}")

    if track_numbers:
        min_t, max_t = min(track_numbers), max(track_numbers)
        num_t = len(track_numbers)
        if min_t > 1 and ((num_t == 1) or ((max_t - min_t + 1) == num_t)):
            shift = min_t - 1
            LOG.info(f"   ∟ ⚖️  Detecting off-by-{shift} numbering. Normalizing...")

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
            TimeRemainingColumn(),
            console=CONSOLE
        ) as progress:
            task = progress.add_task("[cyan]Tagging tracks...", total=len(audio_files))
            for future in as_completed(future_to_file):
                file_p = future_to_file[future]
                try:
                    info = future.result()
                    results.append(info)
                except (AudioProcessingError, MetadataError, APIServiceError) as e:
                    LOG.warning(f"Failed to process {file_p.name}: {e}")
                progress.advance(task)
    if options.get("fetch_replaygain", True):
        calculate_album_replaygain(audio_files, options=options)

    if results:
        primary_artist = results[0].album_artist or results[0].artist
        try:
            process_artist_art(primary_artist, folder_path)
        except (APIServiceError, OSError) as e:
            LOG.debug(f"Artist art download failed: {e}")

    return results
