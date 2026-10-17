from collections.abc import Sequence
from pathlib import Path

from sonora.core.constants import SUPPORTED_EXTS


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


def group_files_by_parent(files: Sequence[Path]) -> dict[Path, list[Path]]:
    grouped: dict[Path, list[Path]] = {}
    for file_path in files:
        grouped.setdefault(file_path.parent, []).append(file_path)
    return grouped
