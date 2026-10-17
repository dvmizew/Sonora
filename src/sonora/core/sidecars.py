import shutil
from collections.abc import Sequence
from pathlib import Path

from sonora.core.constants import ALBUM_COVER_NAMES, COMPANION_LYRICS_EXTS
from sonora.core.library_paths import parse_track_filename, safe_case_rename
from sonora.core.logger import LOG
from sonora.core.text import safe_int


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


def relocate_companion_lyrics(
    src_audio: Path,
    dst_audio: Path,
    dry_run: bool = False,
    sidecars: Sequence[Path] | None = None,
) -> list[Path]:
    """
    Move or rename all companion lyric files (.lrc) alongside an audio file to match the new audio location/stem.
    """
    moved_lyrics: list[Path] = []
    for companion in (
        sidecars if sidecars is not None else find_companion_lyrics(src_audio)
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
