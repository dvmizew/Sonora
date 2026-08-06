"""
Parallel autotagger engine using ThreadPoolExecutor for tagging tracks & albums.
"""

import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from sonora.audio.bpm import calculate_bpm
from sonora.audio.metadata import read_track_metadata, write_track_metadata
from sonora.audio.replaygain import calculate_album_replaygain
from sonora.core.constants import GENRE_MAP, SUPPORTED_EXTS
from sonora.core.exceptions import APIServiceError, AudioProcessingError, MetadataError
from sonora.core.logger import CONSOLE, LOG
from sonora.core.models import TrackInfo
from sonora.core.utils import normalize_str
from sonora.services.acoustid import lookup_acoustid
from sonora.services.discogs import search_discogs_release
from sonora.services.itunes import fetch_itunes_cover_art_url
from sonora.services.lastfm import fetch_lastfm_tags
from sonora.services.lyrics import fetch_synced_lyrics
from sonora.services.musicbrainz import fetch_track_mbid

# Artist alias mappings (e.g. Romanian & international artist variations)
ARTIST_ALIASES: dict[str, str] = {
    "killa fonic": "Killa Fonic",
    "nane": "Nane",
    "m.g.l": "M.G.L.",
    "mgl": "M.G.L.",
    "ian": "Ian",
    "deliric": "Deliric",
    "bug mafia": "B.U.G. Mafia",
    "b.u.g. mafia": "B.U.G. Mafia",
}


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


def process_single_track(
    file_path: Path,
    fetch_bpm: bool = True,
    fetch_replaygain: bool = True,
    fetch_lyrics: bool = True,
    fetch_itunes_art: bool = True,
    lastfm_api_key: str | None = None,
    acoustid_api_key: str | None = None,
    discogs_user_token: str | None = None,
) -> TrackInfo:
    """
    Process and tag a single audio file with metadata, artwork, lyrics, BPM, and ReplayGain.
    """
    if not file_path.exists():
        raise AudioProcessingError(f"File not found: {file_path}")

    track_info = read_track_metadata(file_path)
    track_info.artist = normalize_artist_alias(track_info.artist)

    # 1. Respect existing MBID or prioritize AcoustID (most exact) over text search
    if not track_info.musicbrainz_trackid and acoustid_api_key:
        try:
            acoustid_mbid = lookup_acoustid(file_path, api_key=acoustid_api_key)
            if acoustid_mbid:
                track_info.musicbrainz_trackid = acoustid_mbid
        except APIServiceError as e:
            LOG.debug(f"AcoustID lookup failed for {track_info.title}: {e}")

    # 2. Fallback to MusicBrainz text search by existing tags if AcoustID failed
    if not track_info.musicbrainz_trackid:
        try:
            mbid = fetch_track_mbid(track_info.artist, track_info.title)
            if mbid:
                track_info.musicbrainz_trackid = mbid
        except APIServiceError as e:
            LOG.debug(f"MusicBrainz lookup failed for {track_info.title}: {e}")

    # 3. Fetch MusicBrainz Album ID via Discography Optimization
    if not track_info.musicbrainz_albumid:
        try:
            from sonora.services.musicbrainz import search_musicbrainz_release
            release = search_musicbrainz_release(track_info.artist, track_info.album)
            if release:
                track_info.musicbrainz_albumid = release.get("id")
                # Also opportunistically set year/genre if missing
                if not track_info.date and release.get("date"):
                    track_info.date = release.get("date")[:4]
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
                raw_genre = tags[0]
                track_info.genre = GENRE_MAP.get(raw_genre, raw_genre)
        except APIServiceError as e:
            LOG.debug(f"Last.fm lookup failed for {track_info.title}: {e}")

    # 4. Discogs fallback lookup for release metadata
    if discogs_user_token and not track_info.genre:
        try:
            release = search_discogs_release(track_info.artist, track_info.album, user_token=discogs_user_token)
            if release:
                if release.get("year"):
                    track_info.date = str(release["year"])
                if release.get("genres") and not track_info.genre:
                    raw_genre = str(release["genres"][0])
                    track_info.genre = GENRE_MAP.get(raw_genre, raw_genre)
        except APIServiceError as e:
            LOG.debug(f"Discogs lookup failed for {track_info.title}: {e}")

    # 5. Calculate BPM
    if fetch_bpm:
        try:
            bpm = calculate_bpm(file_path)
            if bpm:
                track_info.bpm = bpm
        except AudioProcessingError as e:
            LOG.debug(f"BPM calculation failed for {track_info.title}: {e}")

    # 6. ReplayGain calculation is now deferred to the Album level in tag_album_folder.

    # 7. Fetch iTunes Cover Art (Download only, don't embed yet)
    cover_jpg = None
    if fetch_itunes_art and file_path.suffix.lower() == ".flac":
        try:
            cover_jpg = file_path.parent / "cover.jpg"
            # Optimization: Do not block all threads on network I/O
            with _get_cover_lock(file_path.parent):
                art_downloaded = cover_jpg.exists()
            
            if not art_downloaded:
                art_url = fetch_itunes_cover_art_url(track_info.artist, track_info.album)
                if art_url:
                    from sonora.core.http import SESSION
                    resp = SESSION.get(art_url, timeout=15)
                    resp.raise_for_status()
                    
                    with _get_cover_lock(file_path.parent):
                        if not cover_jpg.exists():
                            cover_jpg.write_bytes(resp.content)
            if not cover_jpg.exists():
                cover_jpg = None
        except (APIServiceError, OSError) as e:
            LOG.debug(f"Cover art downloading failed for {track_info.title}: {e}")
            cover_jpg = None

    # 8. Write metadata tags back to file AND embed cover art in one single Disk I/O operation!
    write_track_metadata(track_info, cover_art_path=cover_jpg)

    # 9. Fetch & write .lrc lyrics file (and .synced.lrc copy if enhanced)
    if fetch_lyrics:
        try:
            lrc = fetch_synced_lyrics(track_info.artist, track_info.title)
            if lrc:
                lrc_path = file_path.with_suffix(".lrc")
                if re.search(r"<\d{1,2}:\d{2}[\.:]\d{2,3}>", lrc):
                    enhanced_path = file_path.with_suffix(".enhanced.lrc")
                    enhanced_path.write_text(lrc, encoding="utf-8")
                    plain_lrc = re.sub(r"<\d{1,2}:\d{2}[\.:]\d{2,3}>", "", lrc)
                    lrc_path.write_text(plain_lrc, encoding="utf-8")
                else:
                    lrc_path.write_text(lrc, encoding="utf-8")
        except (APIServiceError, OSError) as e:
            LOG.debug(f"Lyrics fetch failed for {track_info.title}: {e}")

    return track_info


def tag_album_folder(
    folder_path: Path,
    max_workers: int = 4,
    **options: Any
) -> list[TrackInfo]:
    """
    Process and tag all audio files in an album folder (recursively) using a thread pool.
    """
    valid_options = {
        "fetch_bpm", "fetch_replaygain", "fetch_lyrics", "fetch_itunes_art",
        "lastfm_api_key", "acoustid_api_key", "discogs_user_token"
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

    if not audio_files:
        return []

    results: list[TrackInfo] = []
    
    # Check if rich is available for progress bar
    try:
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeRemainingColumn,
        )
        has_rich = True
    except ImportError:
        has_rich = False

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_single_track, file_p, **options): file_p
            for file_p in audio_files
        }
        
        if has_rich and CONSOLE:
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
        else:
            for future in as_completed(future_to_file):
                file_p = future_to_file[future]
                try:
                    info = future.result()
                    results.append(info)
                except (AudioProcessingError, MetadataError, APIServiceError) as e:
                    LOG.warning(f"Failed to process {file_p.name}: {e}")

    # After all tracks are tagged, compute Album ReplayGain (modifies files in-place via metaflac)
    if options.get("fetch_replaygain", True):
        calculate_album_replaygain(audio_files)

    return results
