import shutil
from pathlib import Path

from sonora.audio.metadata import read_track_metadata
from sonora.core.logger import (
    LOG,
    create_progress,
    interactive_pause_listener,
    wait_if_paused,
)
from sonora.core.models import TrackInfo
from sonora.core.utils import (
    find_audio_files,
    find_companion_lyrics,
    get_primary_artist,
    group_files_by_parent,
    normalize_str,
    relocate_companion_lyrics,
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
            if track_info.album and track_info.album != "Unknown Album":
                albums.add(normalize_str(track_info.album))
        except (OSError, ValueError, RuntimeError) as error:
            LOG.debug(
                f"Failed to read metadata for singles detection on {audio_file}: {error}"
            )
    return len(albums) > 1


LAST_ORGANIZED_COUNT: int = 0


def get_last_organized_count() -> int:
    return LAST_ORGANIZED_COUNT


def organize_library_singles(
    source_dir: Path, target_singles_dir: Path, dry_run: bool = False
) -> int:
    """
    Scan source_dir, detect single tracks, and move them to target_singles_dir
    organized as target_singles_dir / Primary Artist / Artist - Title.ext.
    Returns the count of moved tracks.
    """
    global LAST_ORGANIZED_COUNT
    LAST_ORGANIZED_COUNT = 0

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    if not dry_run:
        target_singles_dir.mkdir(parents=True, exist_ok=True)

    moved_count = 0
    removed_dupes = 0
    all_audio_files = find_audio_files(source_dir, recursive=True)
    if not all_audio_files:
        return 0

    folder_files = group_files_by_parent(all_audio_files)
    album_fingerprints: set[str] = set()
    singles_to_process: list[tuple[Path, TrackInfo]] = []

    with create_progress() as progress:
        task = progress.add_task(
            "[cyan]Organizing single tracks...", total=len(all_audio_files)
        )
        with interactive_pause_listener(progress, task):
            for folder, files in folder_files.items():
                wait_if_paused()
                is_single = len(files) <= 2 or "singles" in (
                    p.lower() for p in folder.parts
                )
                folder_track_infos: list[tuple[Path, TrackInfo]] = []
                albums_in_folder: set[str] = set()

                for path in files:
                    wait_if_paused()
                    try:
                        info = read_track_metadata(path)
                        folder_track_infos.append((path, info))
                        if info.album and info.album.lower() not in [
                            "singles",
                            "unknown album",
                            "unknown",
                        ]:
                            albums_in_folder.add(normalize_str(info.album))
                    except (OSError, ValueError, RuntimeError) as error:
                        LOG.warning(f"Failed to read metadata for {path}: {error}")
                    progress.advance(task)

                if not is_single and len(albums_in_folder) > 1:
                    is_single = True

                if is_single:
                    for path, info in folder_track_infos:
                        singles_to_process.append((path, info))
                else:
                    for _path, info in folder_track_infos:
                        primary_artist_key = get_primary_artist(info.artist).lower()
                        track_identity_key = (
                            f"{primary_artist_key} - {info.title.lower()}"
                        )
                        album_fingerprints.add(track_identity_key)

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

        single_folder_name = sanitize_name(f"{primary_artist} - {track_info.title}")
        primary_artist_clean = sanitize_name(primary_artist)

        if target_singles_dir and target_singles_dir != source_dir / "Singles":
            base_parent = target_singles_dir
        elif source_dir.name.lower() == primary_artist_clean.lower():
            base_parent = source_dir
        else:
            try:
                subdirs = [
                    p
                    for p in source_dir.iterdir()
                    if p.is_dir() and not p.name.startswith(".")
                ]
                if len(subdirs) > 5 and any(" - " not in p.name for p in subdirs):
                    base_parent = source_dir / primary_artist_clean
                else:
                    base_parent = source_dir
            except OSError:
                base_parent = source_dir

        single_folder = base_parent / single_folder_name
        if not dry_run:
            single_folder.mkdir(parents=True, exist_ok=True)

        target_file = (
            single_folder / f"01 - {sanitize_name(track_info.title)}{path.suffix}"
        )

        # Handle destination collisions cleanly
        if path.resolve() != target_file.resolve():
            if target_file.exists():
                try:
                    if target_file.stat().st_size == path.stat().st_size:
                        if not dry_run:
                            path.unlink(missing_ok=True)
                            for companion in find_companion_lyrics(path):
                                companion.unlink(missing_ok=True)
                        removed_dupes += 1
                        continue
                    else:
                        counter = 2
                        base_stem = f"01 - {sanitize_name(track_info.title)}"
                        while target_file.exists():
                            target_file = (
                                single_folder / f"{base_stem} ({counter}){path.suffix}"
                            )
                            counter += 1
                except OSError:
                    pass

            if not dry_run:
                # Also move companion artwork from old single folder if changing folders
                if path.parent != single_folder:
                    for art_name in [
                        "cover.jpg",
                        "cover.png",
                        "folder.jpg",
                        "front.jpg",
                    ]:
                        old_art = path.parent / art_name
                        new_art = single_folder / art_name
                        if old_art.exists() and not new_art.exists():
                            try:
                                shutil.move(str(old_art), str(new_art))
                            except OSError:
                                pass
                shutil.move(str(path), str(target_file))
            else:
                LOG.info(f"[DRY-RUN] Would move {path.name} -> {target_file}")

            relocate_companion_lyrics(path, target_file, dry_run=dry_run)

            moved_count += 1
            LAST_ORGANIZED_COUNT = moved_count

    if removed_dupes > 0:
        LOG.info(f"🗑️ Removed {removed_dupes} duplicate single(s).")

    # Cleanup empty/orphaned directories
    if not dry_run:
        cleanup_empty_dirs(source_dir)

    return moved_count


def cleanup_empty_dirs(path: Path, target_singles_dir: Path | None = None) -> int:
    removed_count = 0
    if not path.exists() or not path.is_dir():
        return 0
    for child in sorted(path.rglob("*"), reverse=True):
        if not child.is_dir():
            continue
        if child.name in (
            ".git",
            ".idea",
            ".vscode",
            "__pycache__",
        ):
            continue
        if target_singles_dir and (
            child == target_singles_dir
            or target_singles_dir in child.parents
            or child in target_singles_dir.parents
        ):
            continue
        try:
            audio_files = find_audio_files(child, recursive=True)
            if not audio_files:
                shutil.rmtree(str(child), ignore_errors=True)
                removed_count += 1
        except OSError as error:
            LOG.debug(f"Could not remove empty/orphaned dir {child}: {error}")
    return removed_count
