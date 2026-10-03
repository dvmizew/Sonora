import contextlib
import os
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rapidfuzz import fuzz
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
    clean_title,
    find_audio_files,
    find_companion_lyrics,
    get_primary_artist,
    get_single_release_title,
    group_files_by_parent,
    is_in_singles_hierarchy,
    is_interruption,
    normalize_str,
    relocate_companion_artwork,
    relocate_companion_lyrics,
    resolve_unique_path,
    sanitize_name,
)


def _quarantine_file(file_path: Path, quarantine_dir: Path) -> Path:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    target = resolve_unique_path(
        quarantine_dir / file_path.name, current_path=file_path
    )
    if target.resolve() != file_path.resolve():
        shutil.move(str(file_path), str(target))
    for companion in find_companion_lyrics(file_path):
        comp_target = resolve_unique_path(
            quarantine_dir / companion.name, current_path=companion
        )
        if comp_target.resolve() != companion.resolve():
            shutil.move(str(companion), str(comp_target))
    return target


def _read_file_info(file_path: Path) -> tuple[Path, TrackInfo | None]:
    try:
        return file_path, read_track_metadata(file_path)
    except OSError as err:
        LOG.warning(f"Failed to read metadata for {escape(str(file_path))}: {err}")
        return file_path, None


class SingleDeduplicator:
    """
    Registry for detecting duplicate singles without cross-song collisions
    or false-positive elimination of distinct mixes, versions, or audio qualities.
    """

    def __init__(self) -> None:
        self._tracks: dict[
            str,
            list[
                tuple[
                    str,
                    str | None,
                    str | None,
                    float | None,
                    int | None,
                    int | None,
                    Path,
                ]
            ],
        ] = {}

    def register(self, track_info: TrackInfo, target_path: Path | None = None) -> None:
        primary_artist_key = normalize_str(get_primary_artist(track_info.artist))
        clean_title_key = normalize_str(clean_title(track_info.title))
        isrc_key = track_info.isrc.strip().upper() if track_info.isrc else None
        mbid_key = (
            track_info.musicbrainz_trackid.strip().lower()
            if track_info.musicbrainz_trackid
            else None
        )
        resolved_path = target_path or track_info.file_path
        entry = (
            clean_title_key,
            isrc_key,
            mbid_key,
            track_info.duration,
            track_info.bits_per_sample,
            track_info.sample_rate,
            resolved_path,
        )
        self._tracks.setdefault(primary_artist_key, []).append(entry)

    def find_duplicate(
        self, track_info: TrackInfo
    ) -> tuple[bool, Path | None, int | None, int | None]:
        """
        Check if track_info is a duplicate of a previously registered single.
        Returns: (is_duplicate, existing_target_path, existing_bits, existing_sample_rate)
        """
        primary_artist_key = normalize_str(get_primary_artist(track_info.artist))
        existing_list = self._tracks.get(primary_artist_key)
        if not existing_list:
            return False, None, None, None

        clean_title_key = normalize_str(clean_title(track_info.title))
        isrc_key = track_info.isrc.strip().upper() if track_info.isrc else None
        mbid_key = (
            track_info.musicbrainz_trackid.strip().lower()
            if track_info.musicbrainz_trackid
            else None
        )
        curr_duration = track_info.duration

        for (
            ex_title,
            ex_isrc,
            ex_mbid,
            ex_duration,
            ex_bits,
            ex_sr,
            ex_path,
        ) in existing_list:
            if (
                curr_duration is not None
                and ex_duration is not None
                and abs(curr_duration - ex_duration) > 15.0
            ):
                continue

            if clean_title_key == ex_title:
                return True, ex_path, ex_bits, ex_sr

            if (
                isrc_key
                and ex_isrc
                and isrc_key == ex_isrc
                and fuzz.ratio(clean_title_key, ex_title) >= 70.0
            ):
                return True, ex_path, ex_bits, ex_sr

            if (
                mbid_key
                and ex_mbid
                and mbid_key == ex_mbid
                and fuzz.ratio(clean_title_key, ex_title) >= 70.0
            ):
                return True, ex_path, ex_bits, ex_sr

        return False, None, None, None

    def is_duplicate(self, track_info: TrackInfo) -> bool:
        dup, _, _, _ = self.find_duplicate(track_info)
        return dup


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


def _is_single_folder(
    folder: Path,
    files: list[Path],
    source_dir: Path,
    folder_track_infos: list[tuple[Path, TrackInfo]],
) -> bool:
    # Never dismantle disc subfolders (e.g. 'CD 1', 'Disc 2')
    if get_config().is_disc_folder(folder.name):
        return False

    # Loose files placed directly in library root are loose singles
    if folder.resolve() == source_dir.resolve():
        return True

    # Folders already located within a Singles hierarchy
    if is_in_singles_hierarchy(folder, source_dir):
        return True

    # Immediate generic dump bucket relative to source_dir (e.g. source_dir / 'Downloads')
    if get_config().is_generic_container(folder.name):
        return True

    if not folder_track_infos:
        return False

    cleaned_albums: Counter[str] = Counter()
    raw_albums: Counter[str] = Counter()
    track_numbers: set[int] = set()
    total_tracks_counts: Counter[int] = Counter()

    for _, track_metadata in folder_track_infos:
        if track_metadata.album and not get_config().is_generic_container(
            track_metadata.album
        ):
            raw_album_norm = normalize_str(track_metadata.album)
            raw_albums[raw_album_norm] += 1
            cleaned_album = normalize_str(clean_title(track_metadata.album))
            if cleaned_album:
                cleaned_albums[cleaned_album] += 1
        if track_metadata.track_number and track_metadata.track_number > 0:
            track_numbers.add(track_metadata.track_number)
        if track_metadata.total_tracks and track_metadata.total_tracks > 0:
            total_tracks_counts[track_metadata.total_tracks] += 1

    file_count = len(files)

    # Multi-track album evaluation (>= 3 tracks)
    if file_count >= 3:
        # Album consensus: majority of tracks share the same raw or cleaned album title
        if cleaned_albums and cleaned_albums.most_common(1)[0][1] >= (file_count * 0.5):
            return False
        if raw_albums and raw_albums.most_common(1)[0][1] >= (file_count * 0.5):
            return False

        # Sequence continuity: >= 3 unique positive track numbers forming an album sequence
        if (
            len(track_numbers) >= 3
            and max(track_numbers) >= 3
            and (len(track_numbers) / file_count) >= 0.5
        ):
            return False

        # Declared album release: majority of tracks declare total_tracks >= 3
        multi_track_declared = sum(
            cnt for tot, cnt in total_tracks_counts.items() if tot >= 3
        )
        if multi_track_declared >= (file_count * 0.5):
            return False

        # Structural coherence: folder name reflects an album title present in track metadata
        norm_folder = normalize_str(folder.name)
        for alb in cleaned_albums:
            if alb in norm_folder or norm_folder in alb:
                return False

        return True

    # 1-2 tracks: verify whether track metadata declares a multi-track album
    for _, track_meta in folder_track_infos:
        if (track_meta.total_tracks and track_meta.total_tracks > 2) or (
            track_meta.track_number and track_meta.track_number > 2
        ):
            return False
        if track_meta.album and not get_config().is_generic_container(track_meta.album):
            norm_alb = normalize_str(track_meta.album)
            norm_folder = normalize_str(folder.name)
            if (
                (norm_alb in norm_folder or norm_folder in norm_alb)
                and track_meta.total_tracks
                and track_meta.total_tracks > 1
            ):
                return False

    return True


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
    deduplicator = SingleDeduplicator()
    singles_to_process: list[tuple[Path, TrackInfo]] = []

    with create_progress() as progress:
        task_scan = progress.add_task(
            "[cyan]Scanning library & analyzing albums...", total=len(all_audio_files)
        )
        with (
            interactive_pause_listener(progress, task_scan),
            ThreadPoolExecutor(max_workers=max_threads) as executor,
        ):
            try:
                for folder, files in folder_files.items():
                    wait_if_paused()
                    folder_track_infos: list[tuple[Path, TrackInfo]] = []

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
                    for file_path, track_info in file_results:
                        wait_if_paused()
                        if track_info is not None:
                            folder_track_infos.append((file_path, track_info))
                        progress.advance(task_scan)

                    is_single = _is_single_folder(
                        folder=folder,
                        files=files,
                        source_dir=source_dir,
                        folder_track_infos=folder_track_infos,
                    )

                    if is_single:
                        for path, track_info in folder_track_infos:
                            singles_to_process.append((path, track_info))
            except (KeyboardInterrupt, RuntimeError) as exc:
                if not is_interruption(exc):
                    raise
                executor.shutdown(wait=True, cancel_futures=True)
                raise InterruptedOperationError(moved_count) from None

        progress.update(task_scan, visible=False)

        task_organize = progress.add_task(
            "[cyan]Organizing single tracks...", total=len(singles_to_process)
        )

        quarantine_dir = (
            target_singles_dir or (source_dir / "Singles")
        ) / ".duplicates"

        # Process and move collected singles (with safe deduplication and quality downgrade protection)
        try:
            for path, track_info in singles_to_process:
                wait_if_paused()
                primary_artist = get_primary_artist(track_info.artist)
                clean_title_str = clean_title(track_info.title)
                primary_artist_key = normalize_str(primary_artist)
                track_identity_key = (
                    f"{primary_artist_key} - {normalize_str(clean_title_str)}"
                )

                single_release_title = get_single_release_title(track_info)
                single_folder_name = sanitize_name(
                    f"{primary_artist} - {single_release_title}"
                )
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
                        if len(subdirs) > 5 and any(
                            " - " not in p.name for p in subdirs
                        ):
                            base_parent = source_dir / primary_artist_clean / "Singles"
                        else:
                            base_parent = source_dir / "Singles"
                    except OSError:
                        base_parent = source_dir / "Singles"

                single_folder = base_parent / single_folder_name
                target_file = (
                    single_folder
                    / f"01 - {sanitize_name(track_info.title)}{path.suffix}"
                )

                # 1. Canonical location check: if already in its canonical target location, register and skip cleanly
                if path.resolve() == target_file.resolve():
                    deduplicator.register(track_info, target_file)
                    progress.advance(task_organize)
                    continue

                # 2. Check for duplicate against previously organized singles
                is_dup, ex_path, ex_bits, ex_sr = deduplicator.find_duplicate(
                    track_info
                )
                if is_dup:
                    curr_bits = track_info.bits_per_sample or 16
                    prior_bits = ex_bits or 16
                    curr_sr = track_info.sample_rate or 44100
                    prior_sr = ex_sr or 44100

                    if (
                        (
                            curr_bits > prior_bits
                            or (curr_bits == prior_bits and curr_sr > prior_sr)
                        )
                        and ex_path
                        and ex_path.exists()
                    ):
                        # Existing is lower quality -> quarantine existing, allow incoming higher-quality file to take its place
                        _quarantine_duplicate_single(
                            ex_path,
                            track_identity_key,
                            quarantine_dir,
                            dry_run,
                            reason=" (lower quality replaced)",
                        )
                        removed_dupes += 1
                    else:
                        # Current track is lower or equal quality duplicate -> quarantine incoming file
                        _quarantine_duplicate_single(
                            path, track_identity_key, quarantine_dir, dry_run
                        )
                        removed_dupes += 1
                        progress.advance(task_organize)
                        continue

                if not dry_run:
                    if not os.access(path, os.R_OK):
                        LOG.warning(
                            f"Permission denied reading file: {escape(str(path))}"
                        )
                        progress.advance(task_organize)
                        continue
                    if not os.access(path.parent, os.W_OK):
                        LOG.warning(
                            f"Permission denied modifying directory: {escape(str(path.parent))}"
                        )
                        progress.advance(task_organize)
                        continue
                    single_folder.mkdir(parents=True, exist_ok=True)

                # 3. Handle destination collisions (target already exists on disk from an unmanaged or prior file)
                if target_file.exists():
                    target_meta = _read_file_info(target_file)[1]
                    target_bits = (
                        target_meta.bits_per_sample if target_meta else None
                    ) or 16
                    target_sr = (
                        target_meta.sample_rate if target_meta else None
                    ) or 44100
                    curr_bits = track_info.bits_per_sample or 16
                    curr_sr = track_info.sample_rate or 44100

                    if curr_bits > target_bits or (
                        curr_bits == target_bits and curr_sr > target_sr
                    ):
                        _quarantine_duplicate_single(
                            target_file,
                            track_identity_key,
                            quarantine_dir,
                            dry_run,
                            reason=" (lower quality replaced)",
                        )
                        removed_dupes += 1
                    else:
                        _quarantine_duplicate_single(
                            path,
                            track_identity_key,
                            quarantine_dir,
                            dry_run,
                            reason=" (target exists)",
                        )
                        removed_dupes += 1
                        progress.advance(task_organize)
                        continue

                if not dry_run:
                    if path.parent != single_folder:
                        relocate_companion_artwork(
                            path.parent, single_folder, dry_run=False
                        )
                    shutil.move(str(path), str(target_file))
                    LOG.info(
                        f"   ∟ 📁 Moved single: {escape(path.name)} -> {escape(single_folder.name)}/"
                    )
                else:
                    LOG.info(
                        f"[DRY-RUN] Would move {escape(path.name)} -> {escape(str(target_file))}"
                    )

                relocate_companion_lyrics(path, target_file, dry_run=dry_run)
                deduplicator.register(track_info, target_file)

                moved_count += 1
                progress.advance(task_organize)
        except (KeyboardInterrupt, RuntimeError) as exc:
            if not is_interruption(exc):
                raise
            raise InterruptedOperationError(moved_count) from None

    if removed_dupes > 0:
        LOG.info(f"🗑️ Removed {removed_dupes} duplicate single(s).")

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
