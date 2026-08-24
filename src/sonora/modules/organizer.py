import re
import shutil
from pathlib import Path

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from sonora.audio.metadata import read_track_metadata
from sonora.core.constants import PROTECTED_ARTISTS, SUPPORTED_EXTS
from sonora.core.logger import CONSOLE, LOG
from sonora.core.utils import normalize_str, sanitize_name

_ARTIST_SEPARATORS = [
    r"\s+fea?t\.?\s+",
    r"\s+featuring\s+",
    r"\s+and\s+",
    r"\s+și\s+",
    r"\s+si\s+",
    r"\s+cu\s+",
    r"\s+vs\.?\s+",
    r"\s+[xX×]\s+",
    r"\s*&\s*",
    r"\s*,\s*",
    r"\s*;\s*",
    r"\s*/\s*",
]
_ARTIST_SPLIT_PATTERN = re.compile("|".join(_ARTIST_SEPARATORS), re.IGNORECASE)


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
        except (OSError, ValueError, RuntimeError) as e:
            LOG.debug(f"Failed to read metadata for singles detection on {p}: {e}")
    return len(albums) > 1


def get_primary_artist(artist_name: str | None) -> str:
    """
    Extract primary artist from raw artist string by stripping featured artists/delimiters
    (feat., ft., &, comma, etc.), respecting PROTECTED_ARTISTS.
    """
    if not artist_name:
        return "Unknown"

    raw = str(artist_name).strip()
    is_protected = any(p.lower() == raw.lower() for p in PROTECTED_ARTISTS)
    if is_protected:
        return sanitize_name(raw)

    parts = _ARTIST_SPLIT_PATTERN.split(raw, maxsplit=1)
    primary = parts[0].strip() if parts else raw
    return sanitize_name(primary or "Unknown")


def organize_library_singles(source_dir: Path, target_singles_dir: Path, options: dict | None = None) -> int:
    """
    Scan source_dir, detect single tracks, and move them to target_singles_dir
    organized as target_singles_dir / Primary Artist / Artist - Title.ext.
    Returns the count of moved tracks.
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    options = options or {}
    dry_run = options.get("dry_run", False)
    
    if not dry_run:
        target_singles_dir.mkdir(parents=True, exist_ok=True)
    moved_count = 0
    single_folder_cache: dict[Path, bool] = {}

    all_audio_files = [
        path for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("[dim]/[/dim]"),
        TimeRemainingColumn(),
        console=CONSOLE,
    ) as progress:
        task = progress.add_task("[cyan]Organizing single tracks...", total=len(all_audio_files))
        for path in all_audio_files:
            parent = path.parent
            if parent not in single_folder_cache:
                single_folder_cache[parent] = is_single_folder(parent)

            if not single_folder_cache[parent]:
                progress.advance(task)
                continue  # Skip tracks belonging to full album folders

            try:
                info = read_track_metadata(path)
                primary_artist = get_primary_artist(info.artist)
                artist_dir = target_singles_dir / primary_artist
                if not dry_run:
                    artist_dir.mkdir(parents=True, exist_ok=True)

                target_file = artist_dir / f"{sanitize_name(info.artist)} - {sanitize_name(info.title)}{path.suffix}"
                if path != target_file and not target_file.exists():
                    if not dry_run:
                        shutil.move(str(path), str(target_file))
                    else:
                        LOG.info(f"[DRY-RUN] Would move {path.name} -> {target_file}")
                    lrc_path = path.with_suffix(".lrc")
                    if lrc_path.exists():
                        target_lrc = target_file.with_suffix(".lrc")
                        if not target_lrc.exists():
                            if not dry_run:
                                shutil.move(str(lrc_path), str(target_lrc))
                            else:
                                LOG.info(f"[DRY-RUN] Would move LRC {lrc_path.name} -> {target_lrc}")

                    moved_count += 1
                    try:
                        if not dry_run and not any(parent.iterdir()):
                            parent.rmdir()
                    except OSError as e:
                        LOG.debug(f"Could not remove parent dir {parent}: {e}")

            except (OSError, ValueError, RuntimeError) as e:
                LOG.warning(f"Failed to organize track {path}: {e}")
            progress.advance(task)

    # Phase 2 & 3: Deduplicate singles against albums
    if target_singles_dir.exists():
        album_fingerprints: set[str] = set()
        for p in source_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS and "Singles" not in p.parts:
                try:
                    meta = read_track_metadata(p)
                    p_art = get_primary_artist(meta.artist).lower()
                    key = f"{p_art} - {meta.title.lower()}"
                    album_fingerprints.add(key)
                except (OSError, ValueError, RuntimeError) as e:
                    LOG.debug(f"Could not read metadata for single deduplication: {e}")

        removed_dupes = 0
        for single_p in list(target_singles_dir.rglob("*")):
            if single_p.is_file() and single_p.suffix.lower() in SUPPORTED_EXTS:
                try:
                    meta = read_track_metadata(single_p)
                    p_art = get_primary_artist(meta.artist).lower()
                    key = f"{p_art} - {meta.title.lower()}"
                    if key in album_fingerprints:
                        if not dry_run:
                            single_p.unlink()
                            lrc_p = single_p.with_suffix(".lrc")
                            if lrc_p.exists():
                                lrc_p.unlink()
                            LOG.info(f"   ∟ 🗑️ Removed duplicate single: {key}")
                        else:
                            LOG.info(f"   ∟ [DRY-RUN] Would remove duplicate single: {key}")
                        removed_dupes += 1
                except (OSError, ValueError, RuntimeError) as e:
                    LOG.debug(f"Could not read metadata for duplicate check: {e}")

    # Cleanup empty directories
    if not dry_run:
        cleanup_empty_dirs(source_dir)

    return moved_count


def cleanup_empty_dirs(path: Path) -> int:
    removed = 0
    if not path.exists() or not path.is_dir():
        return 0
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_dir() and child.name not in (".git", ".idea", ".vscode", "__pycache__"):
            try:
                if not any(child.iterdir()):
                    child.rmdir()
                    removed += 1
            except OSError as e:
                LOG.debug(f"Could not remove empty dir {child}: {e}")
    return removed
