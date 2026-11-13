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
    companions = find_companion_lyrics(file_path)
    reserved_paths: set[Path] = set()
    while True:
        target = resolve_unique_path(
            quarantine_dir / file_path.name,
            current_path=file_path,
            reserved_paths=reserved_paths,
        )
        if all(
            not target.with_suffix(companion.suffix).exists()
            or target.with_suffix(companion.suffix) == companion
            for companion in companions
        ):
            break
        reserved_paths.add(target)
    if target.resolve() != file_path.resolve():
        safe_case_rename(file_path, target)
    relocate_companion_lyrics(file_path, target, sidecars=companions)
    return target


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
        self._artist_by_path: dict[Path, str] = {}

    def register(self, track_info: TrackInfo, target_path: Path | None = None) -> None:
        recording = _single_recording(track_info, target_path)
        self.forget(recording.file_path)
        self._tracks.setdefault(recording.primary_artist_key, []).append(recording)
        self._artist_by_path[recording.file_path] = recording.primary_artist_key

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
        primary_artist = self._artist_by_path.pop(audio_path, None)
        if primary_artist is None:
            return
        recordings = self._tracks[primary_artist]
        self._tracks[primary_artist] = [
            recording for recording in recordings if recording.file_path != audio_path
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


def _is_album_folder(
    folder: Path,
    files: list[Path],
    folder_track_infos: list[tuple[Path, TrackInfo]],
) -> bool:
    """Determine if a folder is a dedicated album container that should not be dismantled."""
    if get_config().is_disc_folder(folder.name):
        return True

    if not folder_track_infos:
        return False

    cleaned_albums: Counter[str] = Counter()
    for _, track_metadata in folder_track_infos:
        if track_metadata.album and not get_config().is_generic_container(
            track_metadata.album
        ):
            cleaned_album = normalize_str(clean_title(track_metadata.album))
            if cleaned_album:
                cleaned_albums[cleaned_album] += 1

    norm_folder = normalize_str(folder.name)
    for album in cleaned_albums:
        if (album in norm_folder or norm_folder in album) and len(album) > 2:
            return True

    file_count = len(files)
    if file_count >= 3 and cleaned_albums:
        _, top_count = cleaned_albums.most_common(1)[0]
        if top_count >= (file_count * 0.8):
            return True

    return False


def _is_standalone_single_track(
    track_info: TrackInfo,
    multi_track_albums: set[str],
) -> bool:
    """Check if an individual track in a non-album folder represents a standalone single."""
    track_num = track_info.track_number
    if track_num is not None and track_num > 1:
        return False

    total_tracks = track_info.total_tracks
    if total_tracks is not None and total_tracks > 1:
        return False

    album = track_info.album.strip() if track_info.album else ""
    cleaned_album = normalize_str(clean_title(album)) if album else ""
    if cleaned_album in multi_track_albums:
        return False

    if total_tracks == 1:
        return True

    if not album or get_config().is_generic_container(album):
        return True

    title = track_info.title.strip() if track_info.title else ""
    cleaned_title = normalize_str(clean_title(title)) if title else ""
    return bool(cleaned_album and cleaned_title and cleaned_album == cleaned_title)


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
                track_info_by_path: dict[Path, TrackInfo] = {}
                file_results = (
                    (
                        future.result()
                        for future in as_completed(
                            [
                                executor.submit(_read_file_info, path)
                                for path in all_audio_files
                            ]
                        )
                    )
                    if max_threads > 1
                    else map(_read_file_info, all_audio_files)
                )
                for file_path, track_info in file_results:
                    wait_if_paused()
                    if track_info is not None:
                        track_info_by_path[file_path] = track_info
                    elif failures is not None:
                        failures.append(str(file_path))
                    progress.advance(task_scan)

                for folder, files in folder_files.items():
                    wait_if_paused()
                    folder_track_infos = [
                        (path, track_info_by_path[path])
                        for path in files
                        if path in track_info_by_path
                    ]
                    is_single = _is_single_folder(
                        folder=folder,
                        files=files,
                        source_dir=source_dir,
                        folder_track_infos=folder_track_infos,
                    )

                    if is_single:
                        singles_to_process.extend(folder_track_infos)
                    elif not _is_album_folder(
                        folder=folder,
                        files=files,
                        folder_track_infos=folder_track_infos,
                    ):
                        multi_track_albums: set[str] = set()
                        for _, track_meta in folder_track_infos:
                            if (
                                track_meta.album
                                and not get_config().is_generic_container(
                                    track_meta.album
                                )
                                and (
                                    (
                                        track_meta.track_number
                                        and track_meta.track_number > 1
                                    )
                                    or (
                                        track_meta.total_tracks
                                        and track_meta.total_tracks > 1
                                    )
                                )
                            ):
                                cleaned = normalize_str(clean_title(track_meta.album))
                                if cleaned:
                                    multi_track_albums.add(cleaned)

                        for item in folder_track_infos:
                            _, track_info = item
                            if _is_standalone_single_track(
                                track_info, multi_track_albums
                            ):
                                singles_to_process.append(item)
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

                    if (
                        target_file.exists() or target_file in playlist_renames.values()
                    ) and target_file.resolve() != path.resolve():
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
                            target_file,
                            current_path=path,
                            reserved_paths=set(playlist_renames.values()),
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
                        companion_lyrics = find_companion_lyrics(path)
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        if (
                            path.parent != target_file.parent
                            and not get_config().is_generic_container(path.parent.name)
                        ):
                            relocate_companion_artwork(
                                path.parent, target_file.parent, dry_run=False
                            )
                        safe_case_rename(path, target_file)
                        LOG.detail(
                            f"   ∟ Moved single: {escape(path.name)} -> {escape(target_file.parent.name)}/"
                        )
                    else:
                        companion_lyrics = find_companion_lyrics(path)
                        LOG.detail(
                            f"[DRY-RUN] Would move {escape(str(path))} -> {escape(str(target_file))}"
                        )

                    playlist_renames[path] = target_file
                    relocate_companion_lyrics(
                        path, target_file, dry_run=dry_run, sidecars=companion_lyrics
                    )
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

    consolidate_duplicate_artist_dirs(
        source_dir,
        dry_run=dry_run,
        renames=playlist_renames,
        failures=failures,
        track_info_by_path={
            playlist_renames.get(path, path): track_info
            for path, track_info in track_info_by_path.items()
        },
    )
    playlist_folders = (
        {
            path.parent
            for path in source_dir.rglob("*")
            if path.suffix.casefold() in {".m3u", ".m3u8"} and path.is_file()
        }
        if playlist_renames
        else set()
    )
    for folder in playlist_folders:
        changed_playlists = update_local_playlists(
            folder, playlist_renames, dry_run=dry_run
        )
        if updated_playlists is not None:
            updated_playlists.extend(changed_playlists)

    if removed_dupes > 0:
        action = "Would quarantine" if dry_run else "Removed"
        LOG.detail(f"{action} {removed_dupes} duplicate single(s).")

    if not dry_run:
        cleanup_empty_dirs(source_dir, target_singles_dir=target_singles_dir)

    return moved_count


def _consolidate_directory_entries(
    parent_dir: Path,
    dry_run: bool = False,
    renames: dict[Path, Path] | None = None,
    failures: list[str] | None = None,
    track_info_by_path: dict[Path, TrackInfo] | None = None,
) -> int:
    """Consolidates case-variant and duplicate subdirectories within parent_dir."""
    if not parent_dir.is_dir():
        return 0
    if renames is None:
        renames = {}

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

        candidate_stats: list[tuple[Path, int, list[str], dict[str, set[str]]]] = []
        for cand_dir in candidate_dirs:
            audio_files = find_audio_files(cand_dir)
            if dry_run and renames:
                audio_files = [path for path in audio_files if path not in renames]
            tag_artists: list[str] = []
            release_ids: dict[str, set[str]] = {}
            for af in audio_files:
                meta = track_info_by_path.get(af) if track_info_by_path else None
                if meta is None:
                    _, meta = _read_file_info(af)
                if meta:
                    if meta.artist:
                        tag_artists.append(meta.artist)
                    if meta.musicbrainz_albumid:
                        relative_parent = af.relative_to(cand_dir).parent
                        parts = relative_parent.parts
                        album_part = ""
                        if parts and normalize_str(parts[0]) == "singles":
                            album_part = parts[1] if len(parts) > 1 else ""
                        elif not parts:
                            album_part = cand_dir.name
                        elif parts:
                            album_part = parts[0]
                        album_key = normalize_str(album_part)
                        if album_key:
                            release_ids.setdefault(album_key, set()).add(
                                meta.musicbrainz_albumid
                            )
            candidate_stats.append(
                (cand_dir, len(audio_files), tag_artists, release_ids)
            )

        canonical_dir: Path | None = None
        release_ids_by_album: dict[str, set[str]] = {}
        for _, _, _, albums in candidate_stats:
            for album, ids in albums.items():
                release_ids_by_album.setdefault(album, set()).update(ids)
        if any(len(ids) > 1 for ids in release_ids_by_album.values()):
            continue
        all_tags = [art for _, _, tags, _ in candidate_stats for art in tags]
        if all_tags:
            most_common_tag = Counter(all_tags).most_common(1)[0][0]
            clean_tag = sanitize_name(most_common_tag)
            for cand_dir, _, _, _ in candidate_stats:
                if cand_dir.name == clean_tag:
                    canonical_dir = cand_dir
                    break

        if not canonical_dir:
            non_screaming = [
                candidate
                for candidate in candidate_stats
                if not (candidate[0].name.isupper() and len(candidate[0].name) > 3)
            ]
            canonical_dir = max(non_screaming or candidate_stats, key=lambda s: s[1])[0]

        for cand_dir, _, _, _ in candidate_stats:
            if cand_dir == canonical_dir:
                continue

            action = "[DRY-RUN] Would consolidate" if dry_run else "Consolidating"
            LOG.detail(
                f"{action} duplicate directory: {cand_dir.name} -> {canonical_dir.name}"
            )
            try:
                audio_files = find_audio_files(cand_dir)
                companions = {
                    audio: find_companion_lyrics(audio) for audio in audio_files
                }
                lyric_files = {path for paths in companions.values() for path in paths}
                other_files = sorted(
                    path
                    for path in cand_dir.rglob("*")
                    if path.is_file()
                    and path not in audio_files
                    and path not in lyric_files
                )
                for source_file in [*audio_files, *other_files]:
                    if dry_run and renames and source_file in renames:
                        continue
                    target = canonical_dir / source_file.relative_to(cand_dir)
                    reserved_paths = set(renames.values())
                    sidecars = companions.get(source_file, [])
                    while True:
                        destination_file = resolve_unique_path(
                            target, reserved_paths=reserved_paths
                        )
                        if all(
                            not destination_file.with_suffix(sidecar.suffix).exists()
                            and destination_file.with_suffix(sidecar.suffix)
                            not in reserved_paths
                            for sidecar in sidecars
                        ):
                            break
                        reserved_paths.add(destination_file)
                    if not dry_run:
                        destination_file.parent.mkdir(parents=True, exist_ok=True)
                        safe_case_rename(source_file, destination_file)
                    moved_sidecars = relocate_companion_lyrics(
                        source_file,
                        destination_file,
                        dry_run=dry_run,
                        sidecars=sidecars,
                    )
                    for original, previous_target in tuple(renames.items()):
                        if previous_target == source_file:
                            renames[original] = destination_file
                    renames[source_file] = destination_file
                    renames.update(zip(sidecars, moved_sidecars))
                if not dry_run:
                    cleanup_empty_dirs(cand_dir)
                    cand_dir.rmdir()
            except OSError as error:
                if failures is not None:
                    failures.append(str(cand_dir))
                LOG.warning(
                    f"Failed to consolidate {escape(str(cand_dir))} into {escape(str(canonical_dir))}: {escape(str(error))}"
                )
                continue
            consolidated_count += 1

    return consolidated_count


def consolidate_duplicate_artist_dirs(
    source_dir: Path,
    dry_run: bool = False,
    renames: dict[Path, Path] | None = None,
    failures: list[str] | None = None,
    track_info_by_path: dict[Path, TrackInfo] | None = None,
) -> int:
    """
    Detects case-variant and normalized duplicate artist directories and subdirectories
    in source_dir (e.g. 'Ian' vs 'IAN', 'Nosfe' vs 'NOSFE', 'Crush' vs 'CRUSH').
    Consolidates subdirectories, audio files, and companion assets into the canonical directory.
    """
    if not source_dir.is_dir():
        return 0
    if renames is None:
        renames = {}

    consolidated_count = _consolidate_directory_entries(
        source_dir,
        dry_run=dry_run,
        renames=renames,
        failures=failures,
        track_info_by_path=track_info_by_path,
    )

    for entry in list(source_dir.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            singles_dir = entry / "Singles"
            if singles_dir.is_dir():
                consolidated_count += _consolidate_directory_entries(
                    singles_dir,
                    dry_run=dry_run,
                    renames=renames,
                    failures=failures,
                    track_info_by_path=track_info_by_path,
                )
            consolidated_count += _consolidate_directory_entries(
                entry,
                dry_run=dry_run,
                renames=renames,
                failures=failures,
                track_info_by_path=track_info_by_path,
            )

    return consolidated_count


def cleanup_empty_dirs(path: Path, target_singles_dir: Path | None = None) -> int:
    removed_count = 0
    if not path.is_dir():
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
            if (
                is_in_singles_hierarchy(child)
                and child.name.lower() not in ("singles", "single")
                and not find_audio_files(child)
            ):
                for orphan in list(child.iterdir()):
                    if orphan.is_file() and orphan.suffix.lower() in (
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".webp",
                        ".bmp",
                        ".gif",
                        ".txt",
                        ".nfo",
                        ".ini",
                        ".db",
                    ):
                        with contextlib.suppress(OSError):
                            orphan.unlink()

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
