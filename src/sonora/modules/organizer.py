import contextlib
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

from rapidfuzz import fuzz
from rich.markup import escape

from sonora.audio.metadata import read_track_metadata
from sonora.core.artists import extract_featured_artist_tokens, get_primary_artist
from sonora.core.audio_files import find_audio_files, group_files_by_parent
from sonora.core.cache import api_cache_session
from sonora.core.cancellation import InterruptedOperationError, is_interruption
from sonora.core.config import get_config
from sonora.core.library_paths import (
    get_single_release_title,
    is_in_singles_hierarchy,
    resolve_unique_path,
    safe_case_rename,
)
from sonora.core.logger import (
    LOG,
    create_progress,
    interactive_pause_listener,
    resume_workers,
    wait_if_paused,
)
from sonora.core.models import TrackInfo
from sonora.core.playlists import update_local_playlists
from sonora.core.sidecars import (
    find_companion_lyrics,
    relocate_companion_artwork,
    relocate_companion_lyrics,
)
from sonora.core.text import normalize_str, sanitize_name
from sonora.core.titles import clean_title
from sonora.modules.normalization import normalize_track_metadata


def _quarantine_file(file_path: Path, quarantine_dir: Path) -> Path:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    target = resolve_unique_path(
        quarantine_dir / file_path.name, current_path=file_path
    )
    if target.resolve() != file_path.resolve():
        safe_case_rename(file_path, target)
    for companion in find_companion_lyrics(file_path):
        comp_target = resolve_unique_path(
            quarantine_dir / companion.name, current_path=companion
        )
        if comp_target.resolve() != companion.resolve():
            safe_case_rename(companion, comp_target)
    return target


@api_cache_session()
def _read_file_info(file_path: Path) -> tuple[Path, TrackInfo | None]:
    try:
        return file_path, read_track_metadata(file_path)
    except OSError as err:
        LOG.warning(
            f"Failed to read metadata for {escape(str(file_path))}: {escape(str(err))}"
        )
        return file_path, None


def _audio_quality_rank(track_info: TrackInfo) -> tuple[int, int, int, int]:
    return (
        int(track_info.is_lossless),
        (track_info.bits_per_sample or 16) if track_info.is_lossless else 0,
        (track_info.sample_rate or 44100) if track_info.is_lossless else 0,
        0 if track_info.is_lossless else (track_info.bitrate or 0),
    )


@dataclass(frozen=True)
class _RegisteredSingle:
    primary_artist_key: str
    artist_key: frozenset[str]
    title_key: str
    featured_key: frozenset[str]
    isrc: str | None
    recording_mbid: str | None
    duration: float | None
    quality: tuple[int, int, int, int]
    file_path: Path


def _single_recording(
    track_info: TrackInfo, target_path: Path | None = None
) -> _RegisteredSingle:
    normalized_track = normalize_track_metadata(replace(track_info))
    return _RegisteredSingle(
        primary_artist_key=normalize_str(get_primary_artist(normalized_track.artist)),
        artist_key=frozenset(
            normalize_str(artist)
            for artist in extract_featured_artist_tokens(normalized_track.artist)
        ),
        title_key=normalize_str(clean_title(normalized_track.title)),
        featured_key=frozenset(
            normalize_str(artist)
            for artist in extract_featured_artist_tokens(
                normalized_track.featured_artists
            )
        ),
        isrc=normalized_track.isrc.strip().upper() if normalized_track.isrc else None,
        recording_mbid=normalized_track.musicbrainz_trackid.strip().lower()
        if normalized_track.musicbrainz_trackid
        else None,
        duration=normalized_track.duration,
        quality=_audio_quality_rank(track_info),
        file_path=target_path or track_info.file_path,
    )


class SingleDeduplicator:
    """Keep distinct recording editions while ranking duplicate audio quality."""

    def __init__(self) -> None:
        self._tracks: dict[str, list[_RegisteredSingle]] = {}

    def register(self, track_info: TrackInfo, target_path: Path | None = None) -> None:
        recording = _single_recording(track_info, target_path)
        self.forget(recording.file_path)
        self._tracks.setdefault(recording.primary_artist_key, []).append(recording)

    def find_duplicate(self, track_info: TrackInfo) -> _RegisteredSingle | None:
        candidate = _single_recording(track_info)
        for recording in self._tracks.get(candidate.primary_artist_key, []):
            if (
                track_info.duration is not None
                and recording.duration is not None
                and abs(track_info.duration - recording.duration)
                > get_config().single_duration_tolerance_seconds
            ):
                continue
            if candidate.isrc and recording.isrc and candidate.isrc != recording.isrc:
                continue
            if (
                candidate.recording_mbid
                and recording.recording_mbid
                and candidate.recording_mbid != recording.recording_mbid
            ):
                continue
            if (
                candidate.artist_key != recording.artist_key
                or candidate.featured_key != recording.featured_key
            ):
                continue
            if candidate.title_key == recording.title_key:
                return recording
            if (
                (candidate.isrc and candidate.isrc == recording.isrc)
                or (
                    candidate.recording_mbid
                    and candidate.recording_mbid == recording.recording_mbid
                )
            ) and fuzz.ratio(
                candidate.title_key, recording.title_key
            ) >= get_config().single_identifier_title_threshold:
                return recording
        return None

    def is_duplicate(self, track_info: TrackInfo) -> bool:
        return self.find_duplicate(track_info) is not None

    def forget(self, audio_path: Path) -> None:
        for primary_artist, recordings in self._tracks.items():
            self._tracks[primary_artist] = [
                recording
                for recording in recordings
                if recording.file_path != audio_path
            ]


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
            LOG.detail(
                f"   ∟ 📦 Quarantined duplicate single{reason}: {escape(key)} -> .duplicates/"
            )
        except OSError as error:
            LOG.debug(f"Failed to quarantine duplicate single {path}: {error}")
            raise
    else:
        LOG.detail(
            f"[DRY-RUN] Would quarantine duplicate single{reason}: {escape(key)}"
        )


def _is_single_folder(
    folder: Path,
    files: list[Path],
    source_dir: Path,
    folder_track_infos: list[tuple[Path, TrackInfo]],
) -> bool:
    # Never dismantle disc subfolders (e.g. 'CD 1', 'Disc 2')
    if get_config().is_disc_folder(folder.name):
        return False

    if folder.resolve() == source_dir.resolve():
        return True

    if is_in_singles_hierarchy(folder, source_dir) or get_config().is_generic_container(
        folder.name
    ):
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

    if file_count >= 3:
        if cleaned_albums and cleaned_albums.most_common(1)[0][1] >= (file_count * 0.5):
            return False
        if raw_albums and raw_albums.most_common(1)[0][1] >= (file_count * 0.5):
            return False

        if (
            len(track_numbers) >= 3
            and max(track_numbers) >= 3
            and (len(track_numbers) / file_count) >= 0.5
        ):
            return False

        multi_track_declared = sum(
            count
            for total_tracks, count in total_tracks_counts.items()
            if total_tracks >= 3
        )
        if multi_track_declared >= (file_count * 0.5):
            return False

        norm_folder = normalize_str(folder.name)
        for album in cleaned_albums:
            if album in norm_folder or norm_folder in album:
                return False

        return True

    for _, track_meta in folder_track_infos:
        if (track_meta.total_tracks and track_meta.total_tracks > 2) or (
            track_meta.track_number and track_meta.track_number > 2
        ):
            return False
        if track_meta.album and not get_config().is_generic_container(track_meta.album):
            normalized_album = normalize_str(track_meta.album)
            norm_folder = normalize_str(folder.name)
            if (
                (normalized_album in norm_folder or norm_folder in normalized_album)
                and track_meta.total_tracks
                and track_meta.total_tracks > 1
            ):
                return False

    return True


def _resolve_single_target_file(
    path: Path,
    track_info: TrackInfo,
    target_singles_dir: Path,
) -> Path:
    """Calculates canonical target path for a single audio track."""
    normalized_track = normalize_track_metadata(replace(track_info))
    artist_credit = normalized_track.artist or get_primary_artist(track_info.artist)
    single_release_title = get_single_release_title(normalized_track)
    has_feat_in_title = bool(
        re.search(r"\b(?:feat|ft|featuring)\b", single_release_title, re.IGNORECASE)
    )
    canonical_with_feat = (
        sanitize_name(
            f"{artist_credit} - {single_release_title} (feat. {normalized_track.featured_artists})"
        )
        if normalized_track.featured_artists and not has_feat_in_title
        else None
    )
    canonical_without_feat = sanitize_name(f"{artist_credit} - {single_release_title}")

    if path.parent.name in (canonical_with_feat, canonical_without_feat):
        single_folder_name = path.parent.name
    else:
        single_folder_name = canonical_with_feat or canonical_without_feat

    single_folder = target_singles_dir / single_folder_name
    return single_folder / f"01 - {sanitize_name(normalized_track.title)}{path.suffix}"


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
    registered = deduplicator.find_duplicate(track_info)
    if registered and not dry_run and not registered.file_path.exists():
        deduplicator.forget(registered.file_path)
        registered = None
    if registered:
        if _audio_quality_rank(track_info) > registered.quality:
            _quarantine_duplicate_single(
                registered.file_path,
                track_identity_key,
                quarantine_dir,
                dry_run,
                reason=" (lower quality replaced)",
            )
            deduplicator.forget(registered.file_path)
            removed_count += 1
        else:
            _quarantine_duplicate_single(
                path, track_identity_key, quarantine_dir, dry_run
            )
            return True, 1

    if target_file.exists() and not (
        dry_run and registered and registered.file_path == target_file and removed_count
    ):
        target_meta = _read_file_info(target_file)[1]
        destination_registry = SingleDeduplicator()
        if target_meta is None:
            return False, removed_count
        destination_registry.register(target_meta)
        if not destination_registry.is_duplicate(track_info):
            return False, removed_count
        if _audio_quality_rank(track_info) > _audio_quality_rank(target_meta):
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
    failures: list[str] | None = None,
    updated_playlists: list[Path] | None = None,
) -> int:
    """Move standalone tracks into release folders at the selected destination."""
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
    playlist_renames: dict[Path, Path] = {}

    with create_progress() as progress:
        task_scan = progress.add_task(
            "[cyan]Scanning library & analyzing albums...", total=len(all_audio_files)
        )
        with (
            interactive_pause_listener(progress, task_scan) as pause_stop_event,
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
                        elif failures is not None:
                            failures.append(str(file_path))
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
                resume_workers(pause_stop_event)
                executor.shutdown(wait=True, cancel_futures=True)
                raise InterruptedOperationError(moved_count) from None

        progress.update(task_scan, visible=False)

        task_organize = progress.add_task(
            "[cyan]Organizing single tracks...", total=len(singles_to_process)
        )

        quarantine_dir = target_singles_dir / ".duplicates"

        try:
            for path, track_info in singles_to_process:
                try:
                    wait_if_paused()
                    target_file = _resolve_single_target_file(
                        path, track_info, target_singles_dir
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

                    if target_file.exists() and target_file.resolve() != path.resolve():
                        edition_id = track_info.isrc or track_info.musicbrainz_trackid
                        edition_label = (
                            edition_id or track_info.album or path.parent.name
                        )
                        target_file = (
                            target_file.parent.with_name(
                                f"{target_file.parent.name} [{sanitize_name(edition_label)}]"
                            )
                            / target_file.name
                        )
                        target_file = resolve_unique_path(
                            target_file, current_path=path
                        )

                    if not dry_run:
                        if not os.access(path, os.R_OK) or not os.access(
                            path.parent, os.W_OK
                        ):
                            LOG.warning(
                                f"Permission denied modifying: {escape(str(path))}"
                            )
                            if failures is not None:
                                failures.append(str(path))
                            progress.advance(task_organize)
                            continue
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        if path.parent != target_file.parent:
                            relocate_companion_artwork(
                                path.parent, target_file.parent, dry_run=False
                            )
                        safe_case_rename(path, target_file)
                        LOG.detail(
                            f"   ∟ 📁 Moved single: {escape(path.name)} -> {escape(target_file.parent.name)}/"
                        )
                    else:
                        LOG.detail(
                            f"[DRY-RUN] Would move {escape(path.name)} -> {escape(str(target_file))}"
                        )

                    playlist_renames[path] = target_file
                    relocate_companion_lyrics(path, target_file, dry_run=dry_run)
                    deduplicator.register(track_info, target_file)
                    moved_count += 1
                    progress.advance(task_organize)
                except OSError as error:
                    if failures is not None:
                        failures.append(str(path))
                    LOG.warning(
                        f"Failed to organize {escape(str(path))}: {escape(str(error))}"
                    )
                    progress.advance(task_organize)
        except (KeyboardInterrupt, RuntimeError) as exc:
            if not is_interruption(exc):
                raise
            raise InterruptedOperationError(moved_count) from None

    for folder in {path.parent for path in playlist_renames}:
        changed_playlists = update_local_playlists(
            folder, playlist_renames, dry_run=dry_run
        )
        if updated_playlists is not None:
            updated_playlists.extend(changed_playlists)

    if removed_dupes > 0:
        LOG.detail(f"🗑️ Removed {removed_dupes} duplicate single(s).")

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

            LOG.detail(
                f"Consolidating duplicate directory: {cand_dir.name} -> {canonical_dir.name}"
            )
            if not dry_run:
                try:
                    for source_file in sorted(cand_dir.rglob("*")):
                        if not source_file.is_file():
                            continue
                        destination_file = canonical_dir / source_file.relative_to(
                            cand_dir
                        )
                        destination_file.parent.mkdir(parents=True, exist_ok=True)
                        destination_file = resolve_unique_path(destination_file)
                        safe_case_rename(source_file, destination_file)
                    cleanup_empty_dirs(cand_dir)
                    cand_dir.rmdir()
                except OSError as error:
                    LOG.debug(
                        f"Failed to consolidate {cand_dir} into {canonical_dir}: {error}"
                    )
                    continue
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
