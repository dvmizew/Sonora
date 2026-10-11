import os
import re
import shutil
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

from sonora.core.artists import (
    _load_user_overrides,
    is_single_group_artist,
    resolve_artist_name,
)
from sonora.core.config import get_config
from sonora.core.constants import (
    ALBUM_COVER_NAMES,
    COMPANION_LYRICS_EXTS,
    SUPPORTED_EXTS,
)
from sonora.core.files import move_file_exclusively
from sonora.core.logger import LOG
from sonora.core.models import TrackInfo
from sonora.core.rate_limit import RateLimiter
from sonora.core.text import (
    clean_disambiguation,
    normalize_country_name,
    normalize_date,
    normalize_genre,
    normalize_language_name,
    normalize_legacy_diacritics,
    normalize_script_name,
    normalize_str,
    preserve_unicode_repertoire,
    safe_int,
)
from sonora.core.titles import clean_title


class InterruptedOperationError(KeyboardInterrupt):
    """Raised when an operation is cancelled via SIGINT / KeyboardInterrupt, carrying partial progress."""

    def __init__(self, partial_result: Any = None) -> None:
        super().__init__()
        self.partial_result = partial_result


def is_interruption(exc: BaseException) -> bool:
    """Recognize cancellation by exception type and chaining, without masking lock errors."""
    visited: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in visited:
        if isinstance(current, KeyboardInterrupt):
            return True
        visited.add(id(current))
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    return False


_T = TypeVar("_T")


def retry_network_call(
    func: Callable[[], _T],
    *,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    rate_limiter: RateLimiter | None = None,
    transient_exceptions: tuple[type[Exception], ...] = (
        OSError,
        TimeoutError,
    ),
    retry_condition: Callable[[Exception], bool] | None = None,
) -> _T:
    """
    Execute a network-dependent callable with exponential backoff retries.
    Optionally coordinates with a RateLimiter before each attempt to ensure rate
    limits are strictly respected.
    """
    if max_retries < 1:
        raise ValueError("max_retries must be positive")
    attempt = 0
    while True:
        if rate_limiter is not None:
            rate_limiter.wait()
        try:
            return func()
        except transient_exceptions as error:
            if retry_condition is not None and not retry_condition(error):
                raise
            if attempt >= max_retries - 1:
                raise
            delay = initial_delay * (backoff_factor**attempt)
            LOG.debug(
                f"Transient network error ({type(error).__name__}: {error}). "
                f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})..."
            )
            time.sleep(delay)
            attempt += 1


def find_audio_files(
    directory: Path, recursive: bool = True, include_hidden: bool = False
) -> list[Path]:
    """Find all supported audio files in a directory, ignoring hidden directories/files by default."""
    if not directory.exists() or not directory.is_dir():
        return []
    glob_iter = directory.rglob("*") if recursive else directory.glob("*")
    files: list[Path] = []
    for candidate in glob_iter:
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or candidate.suffix.lower() not in SUPPORTED_EXTS
        ):
            continue
        if not include_hidden:
            rel_parts = candidate.relative_to(directory).parts
            if any(part.startswith(".") for part in rel_parts):
                continue
        files.append(candidate)
    return sorted(files)


def find_companion_lyrics(
    audio_file: Path,
    track_number: int | str | None = None,
    in_singles: bool = False,
) -> list[Path]:
    """Find all existing companion lyric files (.lrc) for a given audio file."""
    parent = audio_file.parent
    stem = audio_file.stem
    results: list[Path] = []
    for ext in COMPANION_LYRICS_EXTS:
        candidate = parent / f"{stem}{ext}"
        if candidate.exists() and candidate.is_file():
            results.append(candidate)

    if not results and track_number is not None and parent.is_dir():
        parsed_track = safe_int(track_number)
        if parsed_track is not None:
            try:
                for candidate in parent.iterdir():
                    _, lyric_number, _ = parse_track_filename(candidate.name)
                    if (
                        candidate.suffix.lower() in COMPANION_LYRICS_EXTS
                        and lyric_number == parsed_track
                    ):
                        results.append(candidate)
                        break
            except OSError:
                pass

    if not results and in_singles and parent.is_dir():
        try:
            lrc_candidates = [
                c
                for c in parent.iterdir()
                if c.suffix.lower() in COMPANION_LYRICS_EXTS and c.is_file()
            ]
            if len(lrc_candidates) == 1:
                results.append(lrc_candidates[0])
        except OSError:
            pass

    return results


def group_files_by_parent(files: Sequence[Path]) -> dict[Path, list[Path]]:
    grouped: dict[Path, list[Path]] = {}
    for file_path in files:
        grouped.setdefault(file_path.parent, []).append(file_path)
    return grouped


_CD_PREFIXED_FILENAME_PATTERN = re.compile(
    r"^(?:cd|disc)\s*(\d{1,2})\s*[-_.]\s*(\d{1,3})\s*[-._\s]\s*(.*)$",
    re.IGNORECASE,
)

_MULTI_DISC_FILENAME_PATTERN = re.compile(r"^(\d{1,2})[-.](\d{1,3})\s*[-._\s]\s*(.*)$")

_SINGLE_TRACK_FILENAME_PATTERN = re.compile(r"^(\d{1,3})\s*[-._\s]\s*(.*)$")


def parse_track_filename(filename: str) -> tuple[int | None, int | None, str]:
    """
    Parse filename for disc number, track number, and clean title stem.
    Returns (disc_number, track_number, clean_title).
    """
    stem = Path(filename).stem
    cd_match = _CD_PREFIXED_FILENAME_PATTERN.match(stem)
    if cd_match:
        disc_num = safe_int(cd_match.group(1))
        track_num = safe_int(cd_match.group(2))
        title_part = cd_match.group(3).strip()
        return disc_num, track_num, title_part

    multi_match = _MULTI_DISC_FILENAME_PATTERN.match(stem)
    if multi_match:
        disc_num = safe_int(multi_match.group(1))
        track_num = safe_int(multi_match.group(2))
        title_part = multi_match.group(3).strip()
        return disc_num, track_num, title_part

    single_match = _SINGLE_TRACK_FILENAME_PATTERN.match(stem)
    if single_match:
        track_num = safe_int(single_match.group(1))
        title_part = single_match.group(2).strip()
        return None, track_num, title_part

    return None, None, stem.strip()


def extract_disc_number_from_folder(folder_name: str) -> int | None:
    """
    Extract disc number from disc folder name if it matches SonoraConfig.disc_folder_patterns.
    (e.g., 'CD 1' -> 1, 'Disc 02' -> 2, 'Side A' -> 1).
    """

    clean = folder_name.strip()
    if not get_config().is_disc_folder(clean):
        return None
    num_match = re.search(r"\d+", clean)
    if num_match:
        return safe_int(num_match.group(0))
    side_match = re.search(r"side\s*([a-z])", clean, re.IGNORECASE)
    if side_match:
        char_code = ord(side_match.group(1).lower()) - ord("a") + 1
        return char_code
    return None


def get_album_root_directory(folder: Path) -> Path:
    """Resolve disc subdirectory (e.g., 'CD 1', 'Disc 2') to its canonical album root directory."""

    if get_config().is_disc_folder(folder.name) and folder.parent != folder:
        return folder.parent
    return folder


def group_files_by_album_root(files: Sequence[Path]) -> dict[Path, list[Path]]:
    """
    Group audio files by their canonical album root directory.
    If files reside inside disc subdirectories (e.g., 'Album/CD 1/', 'Album/Disc 2/'),
    they are unified under 'Album/'.
    """
    grouped: dict[Path, list[Path]] = {}
    for file_path in files:
        album_root = get_album_root_directory(file_path.parent)
        grouped.setdefault(album_root, []).append(file_path)
    return grouped


def resolve_unique_path(target_path: Path, current_path: Path | None = None) -> Path:
    """
    Resolve a unique, non-colliding destination path by incrementing a counter
    'Stem (2).ext', 'Stem (3).ext' if target_path already exists on disk.
    """
    if not target_path.exists() or (
        current_path and target_path.resolve() == current_path.resolve()
    ):
        return target_path
    counter = 2
    parent_dir = target_path.parent
    base_stem = target_path.stem
    extension = target_path.suffix
    while True:
        candidate = parent_dir / f"{base_stem} ({counter}){extension}"
        if not candidate.exists() or (
            current_path and candidate.resolve() == current_path.resolve()
        ):
            return candidate
        counter += 1


def relocate_companion_artwork(
    source_dir: Path, target_dir: Path, dry_run: bool = False
) -> list[Path]:
    """
    Move companion album artwork ('cover.jpg', 'cover.png', etc.) from source_dir to target_dir.
    Returns list of relocated artwork files.
    """
    relocated: list[Path] = []
    if source_dir.resolve() == target_dir.resolve():
        return relocated

    for art_name in ALBUM_COVER_NAMES:
        source_art = source_dir / art_name
        target_art = target_dir / art_name
        if source_art.exists() and source_art.is_file() and not target_art.exists():
            if not dry_run:
                try:
                    shutil.move(str(source_art), str(target_art))
                    relocated.append(target_art)
                except OSError as error:
                    LOG.debug(
                        f"Failed to relocate companion artwork {source_art}: {error}"
                    )
            else:
                relocated.append(target_art)
    return relocated


def is_in_singles_hierarchy(path: Path, root_dir: Path | None = None) -> bool:
    """
    Check if a path is located inside a Singles container or directory.
    Universally matches 'singles' folder components and configured container names
    without misinterpreting root library folders (e.g. 'FLAC', 'Music').
    """
    if any(p.lower() in ("singles", "single") for p in path.parts):
        return True
    if root_dir is not None:
        try:
            rel_parts = path.relative_to(root_dir).parts
            dir_parts = rel_parts[:-1] if not path.is_dir() else rel_parts
            from sonora.core.config import get_config

            config = get_config()
            return any(config.is_generic_container(p) for p in dir_parts)
        except ValueError:
            pass
    return False


def get_single_release_title(track_info: TrackInfo) -> str:
    """
    Return the release title for a single folder, preserving version descriptors
    (e.g. 'Deluxe Edition', 'Monoir Remix', 'Extended Mix', 'Live')
    from the album tag if absent from the track title.
    """
    title = (track_info.title or "").strip() or "Untitled"
    album = (track_info.album or "").strip()

    if not album or get_config().is_generic_container(album):
        return title

    matches = re.findall(r"(\((?:[^()]+)\)|\[(?:[^\[\]]+)\])", album)
    descriptors_to_add: list[str] = []
    for version_descriptor in matches:
        if re.search(r"\b(?:feat|ft|featuring)\b", version_descriptor, re.IGNORECASE):
            continue
        if normalize_str(version_descriptor) not in normalize_str(title):
            descriptors_to_add.append(version_descriptor)

    if descriptors_to_add:
        return f"{title} " + " ".join(descriptors_to_add)
    return title


def safe_case_rename(src: Path, dst: Path) -> Path:
    """
    Safely rename a file or directory across all platforms, including case-only renames
    on case-insensitive filesystems (NTFS, FAT32, exFAT, APFS).
    """
    if src.resolve() == dst.resolve() and src.name == dst.name:
        return src

    if not os.access(src, os.W_OK) or not os.access(src.parent, os.W_OK):
        raise PermissionError(f"Cannot rename {src}: write permission denied")

    if dst.exists():
        if not src.samefile(dst):
            raise FileExistsError(f"Destination already exists: {dst}")
        if src.name != dst.name:
            src.rename(dst)
        return dst
    if src.is_file():
        move_file_exclusively(src, dst)
    else:
        src.rename(dst)
    return dst


def relocate_companion_lyrics(
    src_audio: Path,
    dst_audio: Path,
    dry_run: bool = False,
    companions: Sequence[Path] | None = None,
) -> list[Path]:
    """
    Move or rename all companion lyric files (.lrc) alongside an audio file to match the new audio location/stem.
    """
    moved_lyrics: list[Path] = []
    for companion in (
        companions if companions is not None else find_companion_lyrics(src_audio)
    ):
        if not companion.exists():
            continue
        suffix = companion.suffix
        target_companion = dst_audio.parent / f"{dst_audio.stem}{suffix}"
        if target_companion.exists() and target_companion != companion:
            continue

        if not dry_run:
            safe_case_rename(companion, target_companion)
        moved_lyrics.append(target_companion)
    return moved_lyrics


def format_filesize(size_bytes: float) -> str:
    """Format a byte count into a human-readable string (e.g. 4.25 MB, 1.20 GB)."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size) < 1024.0:
            return f"{int(size)} B" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def clear_utils_cache() -> None:
    """Clear cached artist lookups and metadata normalization."""
    clean_disambiguation.cache_clear()
    _load_user_overrides.cache_clear()
    resolve_artist_name.cache_clear()
    is_single_group_artist.cache_clear()
    clean_title.cache_clear()
    normalize_str.cache_clear()
    normalize_date.cache_clear()
    normalize_genre.cache_clear()
    normalize_country_name.cache_clear()
    normalize_language_name.cache_clear()
    normalize_script_name.cache_clear()
    preserve_unicode_repertoire.cache_clear()
    normalize_legacy_diacritics.cache_clear()
