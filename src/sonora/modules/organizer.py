import contextlib
import os
import re
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
        feat_key = normalize_str(track_info.featured_artists or "")
        isrc_key = track_info.isrc.strip().upper() if track_info.isrc else None
        mbid_key = (
            track_info.musicbrainz_trackid.strip().lower()
            if track_info.musicbrainz_trackid
            else None
        )
        resolved_path = target_path or track_info.file_path
        entry = (
            clean_title_key,
            feat_key,
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
        feat_key = normalize_str(track_info.featured_artists or "")
        isrc_key = track_info.isrc.strip().upper() if track_info.isrc else None
        mbid_key = (
            track_info.musicbrainz_trackid.strip().lower()
            if track_info.musicbrainz_trackid
            else None
        )
        curr_duration = track_info.duration

        for (
            ex_title,
            ex_feat,
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

            # Authoritative truth anchor: distinct non-empty ISRCs signify different recordings/releases
            if isrc_key and ex_isrc and isrc_key != ex_isrc:
                continue

            # Distinct non-matching featured artists signify different collaborations
            if feat_key != ex_feat:
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


def _resolve_single_target_file(
    path: Path,
    track_info: TrackInfo,
    source_dir: Path,
    target_singles_dir: Path,
) -> Path:
    """Calculates canonical target path for a single audio track."""
    primary_artist = get_primary_artist(track_info.artist)
    single_release_title = get_single_release_title(track_info)
    has_feat_in_title = bool(
        re.search(r"\b(?:feat|ft|featuring)\b", single_release_title, re.IGNORECASE)
    )
    canonical_with_feat = (
        sanitize_name(
            f"{primary_artist} - {single_release_title} (feat. {track_info.featured_artists})"
        )
        if track_info.featured_artists and not has_feat_in_title
        else None
    )
    canonical_without_feat = sanitize_name(f"{primary_artist} - {single_release_title}")

    if path.parent.name in (canonical_with_feat, canonical_without_feat):
        single_folder_name = path.parent.name
    else:
        single_folder_name = canonical_with_feat or canonical_without_feat

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
                resolved_dir = next(
                    (
                        p.name
                        for p in subdirs
                        if p.name.lower() == primary_artist_clean.lower()
                        or normalize_str(p.name) == normalize_str(primary_artist_clean)
                    ),
                    primary_artist_clean,
                )
                base_parent = source_dir / resolved_dir / "Singles"
            else:
                base_parent = source_dir / "Singles"
        except OSError:
            base_parent = source_dir / "Singles"

    single_folder = base_parent / single_folder_name
    return single_folder / f"01 - {sanitize_name(track_info.title)}{path.suffix}"


def _handle_single_deduplication(
    path: Path,
    track_info: TrackInfo,
    target_file: Path,
    track_identity_key: str,
    deduplicator: SingleDeduplicator,
    quarantine_dir: Path,
    dry_run: bool,
) -> tuple[bool, int]:
    """
    Checks for duplicates in memory and on disk with quality downgrade protection.
    Returns (should_skip, removed_count).
    """
    removed_count = 0
    is_dup, ex_path, ex_bits, ex_sr = deduplicator.find_duplicate(track_info)
    if is_dup:
        curr_bits = track_info.bits_per_sample or 16
        prior_bits = ex_bits or 16
        curr_sr = track_info.sample_rate or 44100
        prior_sr = ex_sr or 44100

        if (
            (curr_bits > prior_bits or (curr_bits == prior_bits and curr_sr > prior_sr))
            and ex_path
            and ex_path.exists()
        ):
            _quarantine_duplicate_single(
                ex_path,
                track_identity_key,
                quarantine_dir,
                dry_run,
                reason=" (lower quality replaced)",
            )
            removed_count += 1
        else:
            _quarantine_duplicate_single(
                path, track_identity_key, quarantine_dir, dry_run
            )
            return True, 1

    if target_file.exists():
        target_meta = _read_file_info(target_file)[1]
        target_bits = (target_meta.bits_per_sample if target_meta else None) or 16
        target_sr = (target_meta.sample_rate if target_meta else None) or 44100
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
            removed_count += 1
        else:
            _quarantine_duplicate_single(
                path,
                track_identity_key,
                quarantine_dir,
                dry_run,
                reason=" (target exists)",
            )
            return True, removed_count + 1

    return False, removed_count


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
                target_file = _resolve_single_target_file(
                    path, track_info, source_dir, target_singles_dir
                )
                if path.resolve() == target_file.resolve():
                    deduplicator.register(track_info, target_file)
                    progress.advance(task_organize)
                    continue

                primary_artist = get_primary_artist(track_info.artist)
                track_identity_key = f"{normalize_str(primary_artist)} - {normalize_str(clean_title(track_info.title))}"
                should_skip, dup_count = _handle_single_deduplication(
                    path,
                    track_info,
                    target_file,
                    track_identity_key,
                    deduplicator,
                    quarantine_dir,
                    dry_run,
                )
                removed_dupes += dup_count
                if should_skip:
                    progress.advance(task_organize)
                    continue

                if not dry_run:
                    if not os.access(path, os.R_OK) or not os.access(
                        path.parent, os.W_OK
                    ):
                        LOG.warning(f"Permission denied modifying: {escape(str(path))}")
                        progress.advance(task_organize)
                        continue
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    if path.parent != target_file.parent:
                        relocate_companion_artwork(
                            path.parent, target_file.parent, dry_run=False
                        )
                    shutil.move(str(path), str(target_file))
                    LOG.info(
                        f"   ∟ 📁 Moved single: {escape(path.name)} -> {escape(target_file.parent.name)}/"
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
        consolidate_duplicate_artist_dirs(source_dir, dry_run=False)
        cleanup_empty_dirs(source_dir)
    else:
        consolidate_duplicate_artist_dirs(source_dir, dry_run=True)

    return moved_count


def _consolidate_directory_entries(parent_dir: Path, dry_run: bool = False) -> int:
    """Consolidates case-variant and duplicate subdirectories within parent_dir."""
    if not parent_dir.is_dir():
        return 0

    consolidated_count = 0
    dir_groups: dict[str, list[Path]] = {}
    for entry in parent_dir.iterdir():
        if entry.is_dir() and not entry.name.startswith("."):
            norm_key = normalize_str(entry.name)
            if norm_key:
                dir_groups.setdefault(norm_key, []).append(entry)

    for candidate_dirs in dir_groups.values():
        if len(candidate_dirs) <= 1:
            continue

        candidate_stats: list[tuple[Path, int, list[str]]] = []
        for cand_dir in candidate_dirs:
            audio_files = find_audio_files(cand_dir)
            tag_artists: list[str] = []
            for af in audio_files[:10]:
                _, meta = _read_file_info(af)
                if meta and meta.artist:
                    tag_artists.append(meta.artist)
            candidate_stats.append((cand_dir, len(audio_files), tag_artists))

        canonical_dir: Path | None = None
        all_tags = [art for _, _, tags in candidate_stats for art in tags]
        if all_tags:
            most_common_tag = Counter(all_tags).most_common(1)[0][0]
            clean_tag = sanitize_name(most_common_tag)
            for cand_dir, _, _ in candidate_stats:
                if cand_dir.name == clean_tag:
                    canonical_dir = cand_dir
                    break

        if not canonical_dir:
            non_screaming = [
                d
                for d, _, _ in candidate_stats
                if not (d.name.isupper() and len(d.name) > 3)
            ]
            if non_screaming:
                canonical_dir = max(
                    [s for s in candidate_stats if s[0] in non_screaming],
                    key=lambda s: s[1],
                )[0]
            else:
                canonical_dir = max(candidate_stats, key=lambda s: s[1])[0]

        for cand_dir, _, _ in candidate_stats:
            if cand_dir == canonical_dir:
                continue

            LOG.info(
                f"Consolidating duplicate directory: {cand_dir.name} -> {canonical_dir.name}"
            )
            if not dry_run:
                try:
                    shutil.copytree(
                        str(cand_dir), str(canonical_dir), dirs_exist_ok=True
                    )
                    shutil.rmtree(str(cand_dir))
                except OSError as error:
                    LOG.debug(
                        f"Failed to consolidate {cand_dir} into {canonical_dir}: {error}"
                    )
            consolidated_count += 1

    return consolidated_count


def consolidate_duplicate_artist_dirs(source_dir: Path, dry_run: bool = False) -> int:
    """
    Detects case-variant and normalized duplicate artist directories and subdirectories
    in source_dir (e.g. 'Ian' vs 'IAN', 'Nosfe' vs 'NOSFE', 'Crush' vs 'CRUSH').
    Consolidates subdirectories, audio files, and companion assets into the canonical directory.
    """
    if not source_dir.is_dir():
        return 0

    consolidated_count = _consolidate_directory_entries(source_dir, dry_run=dry_run)

    for entry in list(source_dir.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            singles_dir = entry / "Singles"
            if singles_dir.is_dir():
                consolidated_count += _consolidate_directory_entries(
                    singles_dir, dry_run=dry_run
                )
            consolidated_count += _consolidate_directory_entries(entry, dry_run=dry_run)

    return consolidated_count


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
