import contextlib
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.markup import escape

from sonora.audio.metadata import read_track_metadata
from sonora.core.config import get_config
from sonora.core.logger import (
    LOG,
    create_progress,
    interactive_pause_listener,
    wait_if_paused,
)
from sonora.core.models import TrackInfo
from sonora.core.utils import (
    InterruptedOperationError,
    deduplicate_title_features,
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
    A folder is treated as a Single folder if it is in Singles, has <= 2 audio files,
    or if audio files come from different albums.
    """
    if not folder_path.exists() or not folder_path.is_dir():
        return False

    if any(get_config().is_generic_container(p) for p in folder_path.parts):
        return True

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
                if len(albums) > 1:
                    return True
        except (OSError, ValueError, RuntimeError) as error:
            LOG.debug(
                f"Failed to read metadata for singles detection on {audio_file}: {error}"
            )
    return len(albums) > 1


def _quarantine_file(file_path: Path, quarantine_dir: Path) -> Path:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    target = quarantine_dir / file_path.name
    counter = 1
    while target.exists() and target.resolve() != file_path.resolve():
        target = quarantine_dir / f"{file_path.stem} ({counter}){file_path.suffix}"
        counter += 1
    if target.resolve() != file_path.resolve():
        shutil.move(str(file_path), str(target))
    for companion in find_companion_lyrics(file_path):
        comp_target = quarantine_dir / companion.name
        c_counter = 1
        while comp_target.exists() and comp_target.resolve() != companion.resolve():
            comp_target = (
                quarantine_dir / f"{companion.stem} ({c_counter}){companion.suffix}"
            )
            c_counter += 1
        if comp_target.resolve() != companion.resolve():
            shutil.move(str(companion), str(comp_target))
    return target


def _read_file_info(file_path: Path) -> tuple[Path, TrackInfo | None]:
    try:
        return file_path, read_track_metadata(file_path)
    except (OSError, ValueError, RuntimeError) as err:
        LOG.warning(f"Failed to read metadata for {escape(str(file_path))}: {err}")
        return file_path, None


def _track_fingerprints(info: TrackInfo) -> list[str]:
    primary_key = normalize_str(get_primary_artist(info.artist))
    title_key = normalize_str(deduplicate_title_features(info.title))
    prints = [f"{primary_key} - {title_key}"]
    if info.isrc:
        prints.append(f"isrc:{info.isrc.strip().upper()}")
    if info.musicbrainz_trackid:
        prints.append(f"mbid:{info.musicbrainz_trackid.strip().lower()}")
    return prints


def _quarantine_duplicate_single(
    path: Path,
    key: str,
    quarantine_dir: Path,
    dry_run: bool = False,
    reason: str = "",
) -> None:
    if not dry_run:
        try:
            _quarantine_file(path, quarantine_dir)
            LOG.info(
                f"   ∟ 📦 Quarantined duplicate single{reason}: {escape(key)} -> .duplicates/"
            )
        except OSError as error:
            LOG.debug(f"Failed to quarantine duplicate single {path}: {error}")
    else:
        LOG.info(f"[DRY-RUN] Would quarantine duplicate single{reason}: {escape(key)}")


def organize_library_singles(
    source_dir: Path,
    target_singles_dir: Path,
    dry_run: bool = False,
    max_threads: int = 4,
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

    folder_files = group_files_by_parent(all_audio_files)
    album_fingerprints: set[str] = set()
    singles_to_process: list[tuple[Path, TrackInfo]] = []

    with create_progress() as progress:
        task = progress.add_task(
            "[cyan]Organizing single tracks...", total=len(all_audio_files)
        )
        with (
            interactive_pause_listener(progress, task),
            ThreadPoolExecutor(max_workers=max_threads) as executor,
        ):
            try:
                for folder, files in folder_files.items():
                    wait_if_paused()
                    is_single = len(files) <= 2 or any(
                        get_config().is_generic_container(p) for p in folder.parts
                    )
                    folder_track_infos: list[tuple[Path, TrackInfo]] = []
                    albums_in_folder: set[str] = set()

                    file_results = (
                        (
                            fut.result()
                            for fut in as_completed(
                                [executor.submit(_read_file_info, p) for p in files]
                            )
                        )
                        if max_threads > 1 and len(files) > 1
                        else (_read_file_info(p) for p in files)
                    )
                    for p, info in file_results:
                        wait_if_paused()
                        if info is not None:
                            folder_track_infos.append((p, info))
                            if info.album and not get_config().is_generic_container(
                                info.album
                            ):
                                albums_in_folder.add(normalize_str(info.album))
                        progress.advance(task)

                    if not is_single and len(albums_in_folder) > 1:
                        is_single = True

                    if is_single:
                        for path, info in folder_track_infos:
                            singles_to_process.append((path, info))
                    else:
                        for _path, info in folder_track_infos:
                            album_fingerprints.update(_track_fingerprints(info))
            except KeyboardInterrupt:
                executor.shutdown(wait=True, cancel_futures=True)
                raise InterruptedOperationError(moved_count) from None

    seen_single_fingerprints: set[str] = set()
    quarantine_dir = (target_singles_dir or (source_dir / "Singles")) / ".duplicates"

    # Process and move collected singles (with safe quarantining against album tracks and duplicate singles)
    try:
        for path, track_info in singles_to_process:
            primary_artist = get_primary_artist(track_info.artist)
            clean_title = deduplicate_title_features(track_info.title)
            primary_artist_key = normalize_str(primary_artist)
            track_identity_key = f"{primary_artist_key} - {normalize_str(clean_title)}"

            track_fps = _track_fingerprints(track_info)
            is_duplicate = any(
                fp in album_fingerprints or fp in seen_single_fingerprints
                for fp in track_fps
            )

            # Deduplicate: if an identical track exists inside a full album or another single, quarantine the duplicate
            if is_duplicate:
                _quarantine_duplicate_single(
                    path, track_identity_key, quarantine_dir, dry_run
                )
                removed_dupes += 1
                continue

            seen_single_fingerprints.update(track_fps)

            single_folder_name = sanitize_name(f"{primary_artist} - {track_info.title}")
            primary_artist_clean = sanitize_name(primary_artist)

            if target_singles_dir and target_singles_dir != source_dir / "Singles":
                base_parent = target_singles_dir
            elif source_dir.name.lower() == primary_artist_clean.lower():
                base_parent = source_dir / "Singles"
            else:
                try:
                    subdirs = [
                        p
                        for p in source_dir.iterdir()
                        if p.is_dir() and not p.name.startswith(".")
                    ]
                    if len(subdirs) > 5 and any(" - " not in p.name for p in subdirs):
                        base_parent = source_dir / primary_artist_clean / "Singles"
                    else:
                        base_parent = source_dir / "Singles"
                except OSError:
                    base_parent = source_dir / "Singles"

            single_folder = base_parent / single_folder_name
            if not dry_run:
                single_folder.mkdir(parents=True, exist_ok=True)

            target_file = (
                single_folder / f"01 - {sanitize_name(track_info.title)}{path.suffix}"
            )

            # Handle destination collisions cleanly (quarantine redundant duplicate single if target already exists)
            if path.resolve() != target_file.resolve() and target_file.exists():
                _quarantine_duplicate_single(
                    path,
                    track_identity_key,
                    quarantine_dir,
                    dry_run,
                    reason=" (target exists)",
                )
                removed_dupes += 1
                continue

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
                            with contextlib.suppress(OSError):
                                shutil.move(str(old_art), str(new_art))
                shutil.move(str(path), str(target_file))
            else:
                LOG.info(
                    f"[DRY-RUN] Would move {escape(path.name)} -> {escape(str(target_file))}"
                )

            relocate_companion_lyrics(path, target_file, dry_run=dry_run)

            moved_count += 1
    except KeyboardInterrupt:
        raise InterruptedOperationError(moved_count) from None

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
            for junk in child.iterdir():
                if junk.is_file() and junk.name in (
                    ".DS_Store",
                    "Thumbs.db",
                    "desktop.ini",
                ):
                    with contextlib.suppress(OSError):
                        junk.unlink()
            child.rmdir()
            removed_count += 1
        except OSError as error:
            LOG.debug(f"Could not remove non-empty dir {child}: {error}")
    return removed_count
