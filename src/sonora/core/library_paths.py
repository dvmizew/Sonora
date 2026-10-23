import os
import re
from collections.abc import Sequence
from pathlib import Path

from sonora.core.config import get_config
from sonora.core.files import move_file_exclusively
from sonora.core.models import TrackInfo
from sonora.core.text import normalize_str, safe_int

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
