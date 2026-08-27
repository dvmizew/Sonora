import shutil
from pathlib import Path

from sonora.audio.metadata import read_track_metadata
from sonora.core.logger import LOG, create_progress
from sonora.core.models import TrackInfo
from sonora.core.utils import (
    find_audio_files,
    find_companion_lyrics,
    get_primary_artist,
    normalize_str,
    sanitize_name,
)


def is_single_folder(folder_path: Path) -> bool:
    """
    Determine if a folder contains standalone single tracks vs a full album.
    A folder is treated as a Single folder if it has <= 2 audio files or
    if audio files come from different albums.
    """
    if not folder_path.exists() or not folder_path.is_dir():
        return False

    audio_files = find_audio_files(folder_path, recursive=False)
    if not audio_files:
        return False

    if len(audio_files) <= 2:
        return True

    albums: set[str] = set()
    for audio_file in audio_files:
        try:
            track_info = read_track_metadata(audio_file)
            albums.add(normalize_str(track_info.album))
        except (OSError, ValueError, RuntimeError) as error:
            LOG.debug(
                f"Failed to read metadata for singles detection on {audio_file}: {error}"
            )
    return len(albums) > 1


def organize_library_singles(
    source_dir: Path, target_singles_dir: Path, dry_run: bool = False
) -> int:
    """
    Scan source_dir, detect single tracks, and move them to target_singles_dir
    organized as target_singles_dir / Primary Artist / Artist - Title.ext.
    Returns the count of moved tracks.
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    if not dry_run:
        target_singles_dir.mkdir(parents=True, exist_ok=True)

    moved_count = 0
    removed_dupes = 0
    all_audio_files = find_audio_files(source_dir, recursive=True)
    if not all_audio_files:
        return 0

    # Group tracks by directory to evaluate folder types in a single pass
    folder_files: dict[Path, list[Path]] = {}
    for path in all_audio_files:
        folder_files.setdefault(path.parent, []).append(path)

    album_fingerprints: set[str] = set()
    singles_to_process: list[tuple[Path, TrackInfo]] = []

    with create_progress() as progress:
        task = progress.add_task(
            "[cyan]Organizing single tracks...", total=len(all_audio_files)
        )

        for folder, files in folder_files.items():
            # Skip target singles directory itself during folder classification
            try:
                if folder == target_singles_dir or target_singles_dir in folder.parents:
                    for path in files:
                        progress.advance(task)
                    continue
            except (ValueError, OSError):
                pass

            # Determine if folder is a single folder (<= 2 tracks or multiple album tags)
            is_single = len(files) <= 2
            folder_track_infos: list[tuple[Path, TrackInfo]] = []
            albums_in_folder: set[str] = set()

            for path in files:
                try:
                    info = read_track_metadata(path)
                    folder_track_infos.append((path, info))
                    albums_in_folder.add(normalize_str(info.album))
                except (OSError, ValueError, RuntimeError) as error:
                    LOG.warning(f"Failed to read metadata for {path}: {error}")

            if not is_single and len(albums_in_folder) > 1:
                is_single = True

            if is_single:
                for path, info in folder_track_infos:
                    singles_to_process.append((path, info))
            else:
                for _path, info in folder_track_infos:
                    primary_artist_key = get_primary_artist(info.artist).lower()
                    track_identity_key = f"{primary_artist_key} - {info.title.lower()}"
                    album_fingerprints.add(track_identity_key)

            for _ in files:
                progress.advance(task)

    # Process and move collected singles (with deduplication against album tracks)
    for path, track_info in singles_to_process:
        primary_artist = get_primary_artist(track_info.artist)
        primary_artist_key = primary_artist.lower()
        track_identity_key = f"{primary_artist_key} - {track_info.title.lower()}"

        # Deduplicate: if an identical track exists inside a full album, remove the single
        if track_identity_key in album_fingerprints:
            if not dry_run:
                try:
                    path.unlink(missing_ok=True)
                    for companion in find_companion_lyrics(path):
                        companion.unlink(missing_ok=True)
                    LOG.info(f"   ∟ 🗑️ Removed duplicate single: {track_identity_key}")
                except OSError as error:
                    LOG.debug(f"Failed to remove duplicate single {path}: {error}")
            else:
                LOG.info(
                    f"[DRY-RUN] Would remove duplicate single: {track_identity_key}"
                )
            removed_dupes += 1
            continue

        artist_dir = target_singles_dir / primary_artist
        if not dry_run:
            artist_dir.mkdir(parents=True, exist_ok=True)

        target_file = (
            artist_dir
            / f"{sanitize_name(track_info.artist)} - {sanitize_name(track_info.title)}{path.suffix}"
        )

        if path != target_file and not target_file.exists():
            if not dry_run:
                shutil.move(str(path), str(target_file))
            else:
                LOG.info(f"[DRY-RUN] Would move {path.name} -> {target_file}")

            for companion in find_companion_lyrics(path):
                suffix = companion.name[len(path.stem) :]
                target_companion = target_file.parent / f"{target_file.stem}{suffix}"
                if not target_companion.exists():
                    if not dry_run:
                        shutil.move(str(companion), str(target_companion))
                    else:
                        LOG.info(
                            f"[DRY-RUN] Would move lyrics {companion.name} -> {target_companion}"
                        )

            moved_count += 1

    if removed_dupes > 0:
        LOG.info(f"🗑️ Removed {removed_dupes} duplicate single(s).")

    # Cleanup empty directories
    if not dry_run:
        cleanup_empty_dirs(source_dir)

    return moved_count


def cleanup_empty_dirs(path: Path) -> int:
    removed_count = 0
    if not path.exists() or not path.is_dir():
        return 0
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_dir() and child.name not in (
            ".git",
            ".idea",
            ".vscode",
            "__pycache__",
        ):
            try:
                if not any(child.iterdir()):
                    child.rmdir()
                    removed_count += 1
            except OSError as error:
                LOG.debug(f"Could not remove empty dir {child}: {error}")
    return removed_count
