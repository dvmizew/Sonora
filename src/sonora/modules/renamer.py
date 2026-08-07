"""
File and directory renamer module with synchronized .lrc metadata header updater.
"""

import re
from pathlib import Path

from sonora.audio.metadata import read_track_metadata
from sonora.core.constants import SUPPORTED_EXTS
from sonora.core.exceptions import AudioProcessingError, MetadataError
from sonora.core.models import TrackInfo
from sonora.core.utils import sanitize_name


def sync_lrc_metadata(lrc_path: Path, track_info: TrackInfo) -> bool:
    """
    Update or insert [ar: Artist] and [ti: Title] metadata headers into an .lrc file.
    Preserves all synchronized timestamps [mm:ss.xx] intact.
    """
    if not lrc_path.exists():
        return False

    try:
        with open(lrc_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        new_lines: list[str] = []
        ar_found = False
        ti_found = False

        for line in lines:
            if line.startswith("[ar:"):
                new_lines.append(f"[ar:{track_info.artist}]\n")
                ar_found = True
            elif line.startswith("[ti:"):
                new_lines.append(f"[ti:{track_info.title}]\n")
                ti_found = True
            else:
                new_lines.append(line)

        # Prepend missing metadata headers at the top
        header_prefix: list[str] = []
        if not ar_found:
            header_prefix.append(f"[ar:{track_info.artist}]\n")
        if not ti_found:
            header_prefix.append(f"[ti:{track_info.title}]\n")

        final_content = "".join(header_prefix + new_lines)

        with open(lrc_path, "w", encoding="utf-8") as f:
            f.write(final_content)
        return True

    except Exception as e:
        raise AudioProcessingError(f"Failed to sync LRC metadata for {lrc_path}: {e}") from e


def rename_track_file(file_path: Path, format_pattern: str = "{track_number:02d} - {artist} - {title}", options: dict | None = None) -> Path:
    """
    Rename an audio file and its .lrc file based on metadata.
    """
    if not file_path.exists():
        raise AudioProcessingError(f"File not found: {file_path}")

    try:
        track_info = read_track_metadata(file_path)
    except MetadataError as e:
        raise AudioProcessingError(f"Cannot rename file without metadata: {e}") from e

    num = track_info.track_number or 1
    artist_clean = sanitize_name(track_info.artist)
    title_clean = sanitize_name(track_info.title)
    
    options = options or {}
    dry_run = options.get("dry_run", False)

    new_stem = format_pattern.format(
        track_number=num,
        artist=artist_clean,
        title=title_clean,
    )
    new_stem = re.sub(r"\s+", " ", new_stem).strip()
    new_path = file_path.with_name(f"{new_stem}{file_path.suffix}")

    if new_path != file_path:
        if new_path.exists():
            counter = 2
            while new_path.exists():
                new_path = file_path.with_name(f"{new_stem} ({counter}){file_path.suffix}")
                counter += 1
        if not dry_run:
            file_path.rename(new_path)
        else:
            from sonora.core.logger import LOG
            LOG.info(f"[DRY-RUN] Would rename {file_path.name} -> {new_path.name}")

    # Sync .lrc file if present
    old_lrc = file_path.with_suffix(".lrc")
    new_lrc = new_path.with_suffix(".lrc")
    if old_lrc.exists():
        if old_lrc != new_lrc and not new_lrc.exists():
            if not dry_run:
                old_lrc.rename(new_lrc)
            else:
                from sonora.core.logger import LOG
                LOG.info(f"[DRY-RUN] Would rename LRC {old_lrc.name} -> {new_lrc.name}")
        if not dry_run:
            sync_lrc_metadata(new_lrc if not dry_run else old_lrc, track_info)

    return new_path


def rename_directory_files(dir_path: Path, options: dict | None = None) -> list[Path]:
    """
    Scan a directory (recursively) and rename all supported audio files and their .lrc files.
    """
    if not dir_path.exists():
        raise AudioProcessingError(f"Directory not found: {dir_path}")

    from sonora.core.logger import LOG
    renamed: list[Path] = []
    for path in sorted(dir_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            try:
                new_p = rename_track_file(path, options=options)
                renamed.append(new_p)
            except AudioProcessingError as e:
                LOG.warning(f"Failed to rename file {path}: {e}")
    return renamed
