"""
Parallel autotagger engine using ThreadPoolExecutor for tagging tracks & albums.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from sonora.audio.bpm import calculate_bpm
from sonora.audio.metadata import read_track_metadata, write_track_metadata
from sonora.audio.replaygain import calculate_replaygain
from sonora.core.constants import SUPPORTED_EXTS
from sonora.core.exceptions import APIServiceError, AudioProcessingError, MetadataError
from sonora.core.models import TrackInfo
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
    lowered = artist.strip().lower()
    return ARTIST_ALIASES.get(lowered, artist.strip())


def process_single_track(
    file_path: Path,
    fetch_bpm: bool = True,
    fetch_replaygain: bool = True,
    fetch_lyrics: bool = True,
    fetch_itunes_art: bool = True,
    lastfm_api_key: str | None = None,
) -> TrackInfo:
    """
    Process and tag a single audio file with metadata, artwork, lyrics, BPM, and ReplayGain.
    """
    if not file_path.exists():
        raise AudioProcessingError(f"File not found: {file_path}")

    track_info = read_track_metadata(file_path)
    track_info.artist = normalize_artist_alias(track_info.artist)

    # 1. Fetch MusicBrainz MBID
    try:
        mbid = fetch_track_mbid(track_info.artist, track_info.title)
        if mbid:
            track_info.musicbrainz_trackid = mbid
    except APIServiceError:
        pass

    # 2. Fetch Last.fm genre/mood tags
    if lastfm_api_key:
        try:
            tags = fetch_lastfm_tags(
                track_info.artist,
                track_info.title,
                api_key=lastfm_api_key,
                mbid=track_info.musicbrainz_trackid,
            )
            if tags:
                track_info.genre = tags[0]
        except APIServiceError:
            pass

    # 3. Calculate BPM
    if fetch_bpm:
        try:
            bpm = calculate_bpm(file_path)
            if bpm:
                track_info.bpm = bpm
        except AudioProcessingError:
            pass

    # 4. Write metadata tags back to file
    write_track_metadata(track_info)

    # 5. Calculate & write ReplayGain if FLAC
    if fetch_replaygain and file_path.suffix.lower() == ".flac":
        try:
            calculate_replaygain(file_path)
        except AudioProcessingError:
            pass

    # 6. Fetch & write .lrc lyrics file
    if fetch_lyrics:
        try:
            lrc = fetch_synced_lyrics(track_info.artist, track_info.title)
            if lrc:
                lrc_path = file_path.with_suffix(".lrc")
                with open(lrc_path, "w", encoding="utf-8") as f:
                    f.write(lrc)
        except (APIServiceError, OSError):
            pass

    # 7. Fetch iTunes Cover Art if enabled
    if fetch_itunes_art:
        try:
            fetch_itunes_cover_art_url(track_info.artist, track_info.album)
        except APIServiceError:
            pass

    return track_info


def tag_album_folder(
    folder_path: Path,
    max_workers: int = 4,
    **options: Any
) -> list[TrackInfo]:
    """
    Process and tag all audio files in an album folder using a thread pool.
    """
    if not folder_path.exists() or not folder_path.is_dir():
        raise AudioProcessingError(f"Album folder not found: {folder_path}")

    audio_files = sorted([
        p for p in folder_path.glob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ])

    if not audio_files:
        return []

    results: list[TrackInfo] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_single_track, file_p, **options): file_p
            for file_p in audio_files
        }
        for future in as_completed(future_to_file):
            try:
                info = future.result()
                results.append(info)
            except (AudioProcessingError, MetadataError, APIServiceError):
                pass

    return results
