"""
Folder organizer module for separating Singles from full Albums.
"""

import shutil
from pathlib import Path

from sonora.audio.metadata import read_track_metadata
from sonora.core.constants import SUPPORTED_EXTS
from sonora.core.exceptions import AudioProcessingError, MetadataError
from sonora.core.logger import LOG
from sonora.core.utils import normalize_str, sanitize_name


def is_single_folder(folder_path: Path) -> bool:
    """
    Determine if a folder contains standalone single tracks vs a full album.
    A folder is treated as a Single folder if it has <= 2 audio files or
    if audio files come from different albums.
    """
    if not folder_path.exists() or not folder_path.is_dir():
        return False

    audio_files = [
        p for p in folder_path.glob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]

    if not audio_files:
        return False

    if len(audio_files) <= 2:
        return True

    albums: set[str] = set()
    for p in audio_files:
        try:
            info = read_track_metadata(p)
            albums.add(normalize_str(info.album))
        except (MetadataError, OSError) as e:
            LOG.debug(f"Failed to read metadata for singles detection on {p}: {e}")

    # If tracks belong to multiple different album names, it's a Singles collection
    return len(albums) > 1


def organize_library_singles(source_dir: Path, target_singles_dir: Path) -> int:
    """
    Scan source_dir, detect single tracks, and move them to target_singles_dir
    organized as target_singles_dir / Artist / Artist - Title.ext.
    Returns the count of moved tracks.
    """
    if not source_dir.exists():
        raise AudioProcessingError(f"Source directory not found: {source_dir}")

    target_singles_dir.mkdir(parents=True, exist_ok=True)
    moved_count = 0
    single_folder_cache: dict[Path, bool] = {}

    for path in list(source_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            parent = path.parent
            if parent not in single_folder_cache:
                single_folder_cache[parent] = is_single_folder(parent)

            if not single_folder_cache[parent]:
                continue  # Skip tracks belonging to full album folders

            try:
                info = read_track_metadata(path)
                artist_dir = target_singles_dir / sanitize_name(info.artist)
                artist_dir.mkdir(parents=True, exist_ok=True)

                target_file = artist_dir / f"{sanitize_name(info.artist)} - {sanitize_name(info.title)}{path.suffix}"
                if path != target_file and not target_file.exists():
                    shutil.move(str(path), str(target_file))

                    # Move accompanying .lrc file if present
                    lrc_path = path.with_suffix(".lrc")
                    if lrc_path.exists():
                        target_lrc = target_file.with_suffix(".lrc")
                        if not target_lrc.exists():
                            shutil.move(str(lrc_path), str(target_lrc))

                    moved_count += 1
                    
                    # Cleanup empty parent directories
                    try:
                        if not any(parent.iterdir()):
                            parent.rmdir()
                    except OSError:
                        pass

            except (MetadataError, OSError) as e:
                LOG.warning(f"Failed to organize track {path}: {e}")

    return moved_count
