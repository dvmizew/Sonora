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


def rename_track_file(
    file_path: Path,
    format_pattern: str | None = None,
    track_info: TrackInfo | None = None,
    options: dict | None = None,
) -> Path:
    """
    Rename an audio file and its .lrc files based on metadata.
    """
    if not file_path.exists():
        raise AudioProcessingError(f"File not found: {file_path}")

    try:
        if track_info is None:
            track_info = read_track_metadata(file_path)
    except MetadataError as e:
        raise AudioProcessingError(f"Cannot rename file without metadata: {e}") from e

    num = track_info.track_number or 1
    artist_clean = sanitize_name(track_info.artist)
    title_clean = sanitize_name(track_info.title)

    options = options or {}
    dry_run = options.get("dry_run", False)

    # Multi-disc track number prefix (e.g. 2-01) if disc_number > 1
    disc_prefix = f"{track_info.disc_number}-" if (track_info.disc_number and track_info.disc_number > 1) else ""
    
    if format_pattern is None:
        new_stem = f"{disc_prefix}{num:02d} - {artist_clean} - {title_clean}"
    else:
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

    # Handle standard .lrc and .enhanced.lrc / .synced.lrc
    for lrc_ext in [".lrc", ".enhanced.lrc", ".synced.lrc"]:
        old_lrc = file_path.with_name(f"{file_path.stem}{lrc_ext}")
        new_lrc = new_path.with_name(f"{new_path.stem}{lrc_ext}")
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


def rename_album_folder(folder_path: Path, artist: str, album: str, options: dict | None = None) -> Path:
    """
    Rename an album directory to 'Artist - Album' if consensus metadata exists.
    Matches initial/rename.py logic.
    """
    if not album or album.lower() in ["singles", "unknown album", "unknown"]:
        return folder_path

    options = options or {}
    dry_run = options.get("dry_run", False)

    expected_name = sanitize_name(f"{artist} - {album}")
    if folder_path.name != expected_name:
        new_folder = folder_path.with_name(expected_name)
        if not new_folder.exists():
            if not dry_run:
                folder_path.rename(new_folder)
                from sonora.core.logger import LOG
                LOG.info(f"   ∟ 📂 Album folder renamed: [dim]{folder_path.name}[/] -> [cyan]{expected_name}[/]")
                return new_folder
            else:
                from sonora.core.logger import LOG
                LOG.info(f"[DRY-RUN] Would rename album folder {folder_path.name} -> {expected_name}")
    return folder_path


def rename_directory_files(dir_path: Path, options: dict | None = None) -> list[Path]:
    """
    Scan a directory (recursively) and rename all supported audio files, their .lrc files,
    and album folders based on consensus metadata.
    """
    if not dir_path.exists():
        raise AudioProcessingError(f"Directory not found: {dir_path}")

    from collections import Counter
    from sonora.core.logger import LOG

    renamed: list[Path] = []
    
    # Process files folder by folder to perform album folder rename consensus
    folder_files: dict[Path, list[Path]] = {}
    for path in sorted(dir_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            folder_files.setdefault(path.parent, []).append(path)

    for folder, files in folder_files.items():
        album_consensus: Counter[tuple[str, str]] = Counter()
        for path in files:
            try:
                info = read_track_metadata(path)
                search_artist = info.album_artist or info.artist
                if search_artist != "Unknown Artist" and info.album != "Unknown Album":
                    album_consensus[(search_artist, info.album)] += 1
                    
                new_p = rename_track_file(path, track_info=info, options=options)
                renamed.append(new_p)
            except AudioProcessingError as e:
                LOG.warning(f"Failed to rename file {path}: {e}")
        
        if album_consensus:
            top_artist, top_album = album_consensus.most_common(1)[0][0]
            count = album_consensus.most_common(1)[0][1]
            if count >= len(files) / 2:
                rename_album_folder(folder, top_artist, top_album, options=options)

    return renamed
