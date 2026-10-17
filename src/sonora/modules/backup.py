import base64
import datetime
import gzip
import os
import zlib
from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, BinaryIO

import ijson
import orjson
import taglib
from rich.markup import escape

from sonora.audio.metadata import clear_metadata_cache
from sonora.core.audio_files import find_audio_files
from sonora.core.cancellation import InterruptedOperationError, is_interruption
from sonora.core.files import atomic_audio_edit, atomic_write_file
from sonora.core.logger import (
    LOG,
    create_progress,
    interactive_pause_listener,
    resume_workers,
    wait_if_paused,
)

_GZIP_MAGIC_HEADER = b"\x1f\x8b"


def _read_track_for_backup(audio_path: Path) -> tuple[str, dict[str, Any] | None]:
    wait_if_paused()
    try:
        with taglib.File(str(audio_path)) as audio_tags:
            complex_tags = {
                key: [
                    {
                        field: {"base64": base64.b64encode(value).decode("ascii")}
                        if isinstance(value, bytes)
                        else value
                        for field, value in properties.items()
                    }
                    for properties in audio_tags.complex_properties(key)
                ]
                for key in audio_tags.complex_property_keys
            }
            return str(audio_path), {
                "tags": audio_tags.tags,
                "complex_tags": complex_tags,
            }
    except OSError as error:
        LOG.debug(
            f"Error reading {escape(str(audio_path))} for backup: {escape(str(error))}"
        )
        return str(audio_path), None


def _restore_single_track(
    relative_path: str,
    backup_tags: dict[str, Any],
    restore_root: Path,
) -> tuple[bool, bool]:
    wait_if_paused()
    target_path = restore_root / relative_path
    if not target_path.is_file():
        return False, True
    try:
        if not os.access(target_path, os.W_OK):
            raise PermissionError(f"Read-only audio file: {target_path}")
        with (
            atomic_audio_edit(target_path) as staging_path,
            taglib.File(str(staging_path)) as audio_tags,
        ):
            audio_tags.tags.clear()
            audio_tags.tags.update(backup_tags["tags"])
            for key in set(audio_tags.complex_property_keys) | set(
                backup_tags["complex_tags"]
            ):
                properties = [
                    {
                        field: base64.b64decode(value["base64"], validate=True)
                        if isinstance(value, dict)
                        else value
                        for field, value in property_fields.items()
                    }
                    for property_fields in backup_tags["complex_tags"].get(key, [])
                ]
                if not audio_tags.set_complex_properties(key, properties):
                    raise OSError(
                        f"Cannot restore binary property {key} on {target_path}"
                    )
            if audio_tags.save():
                raise OSError(f"Some backup tags could not be saved to {target_path}")
        clear_metadata_cache()
        return True, False
    except OSError as error:
        LOG.warning(
            f"Failed to restore {escape(str(target_path))}: {escape(str(error))}"
        )
        return False, False


def backup_library_tags(
    directory: Path, output_file: Path | None = None, max_threads: int = 4
) -> Path:
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Directory not found: {directory}")

    LOG.info(f"🔄 Scanning for files in {escape(str(directory))}...")
    audio_files = find_audio_files(directory, recursive=True)

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    output_path = output_file or Path(f"backup_{timestamp}.json")

    LOG.info(
        f"🔄 Creating full backup for {len(audio_files)} files (threads={max_threads})..."
    )
    backed_up = 0
    failed = 0

    try:
        with (
            atomic_write_file(output_path) as backup_output,
            ExitStack() as output_streams,
            create_progress() as progress,
            ThreadPoolExecutor(max_workers=max_threads) as executor,
        ):
            archive_stream = (
                output_streams.enter_context(
                    gzip.GzipFile(
                        filename="",
                        mode="wb",
                        compresslevel=6,
                        fileobj=backup_output,
                        mtime=0,
                    )
                )
                if output_path.name.endswith(".gz")
                else backup_output
            )
            archive_stream.write(
                b'{"schema":"sonora_backup_v2","root":'
                + orjson.dumps(str(directory.resolve()))
                + b',"tracks":{'
            )
            task = progress.add_task(
                "[cyan]Backing up audio tags...", total=len(audio_files)
            )
            remaining_files = iter(audio_files)
            # Older Python versions lack Executor.map(buffersize); keep artwork bounded here.
            pending_reads = deque(
                executor.submit(_read_track_for_backup, file_path)
                for file_path in islice(remaining_files, max_threads)
            )
            with interactive_pause_listener(progress, task) as pause_stop_event:
                try:
                    while pending_reads:
                        audio_path_string, backup_tags = (
                            pending_reads.popleft().result()
                        )
                        if backup_tags is not None:
                            if backed_up:
                                archive_stream.write(b",")
                            archive_stream.write(
                                orjson.dumps(
                                    str(Path(audio_path_string).relative_to(directory))
                                )
                            )
                            archive_stream.write(b":")
                            archive_stream.write(orjson.dumps(backup_tags))
                            backed_up += 1
                        else:
                            failed += 1
                        next_file = next(remaining_files, None)
                        if next_file is not None:
                            pending_reads.append(
                                executor.submit(_read_track_for_backup, next_file)
                            )
                        progress.advance(task)
                    archive_stream.write(b"}}")
                except (KeyboardInterrupt, RuntimeError) as exc:
                    if not is_interruption(exc):
                        raise
                    resume_workers(pause_stop_event)
                    executor.shutdown(wait=True, cancel_futures=True)
                    raise

        LOG.info(
            f"✅ Successfully backed up {backed_up}/{len(audio_files)} files to {escape(str(output_path))}"
        )
        if failed > 0:
            LOG.warning(f"   ⚠️  {failed} files could not be read")
        if failed:
            raise OSError(
                f"Partial backup saved to {output_path}: {failed} unreadable files"
            )
        return output_path
    except OSError as error:
        LOG.error(f"Failed to save backup: {escape(str(error))}")
        raise


@dataclass
class _BackupManifest:
    schema: str | None = None
    root: str | None = None
    has_tracks: bool = False


@contextmanager
def _open_backup_stream(backup_file: Path) -> Iterator[BinaryIO | gzip.GzipFile]:
    with backup_file.open("rb") as archive_file:
        compressed = archive_file.read(2) == _GZIP_MAGIC_HEADER
        archive_file.seek(0)
        if compressed or backup_file.name.endswith(".gz"):
            with gzip.GzipFile(fileobj=archive_file, mode="rb") as archive_stream:
                yield archive_stream
        else:
            yield archive_file


def _backup_manifest_events(
    archive_stream: BinaryIO | gzip.GzipFile, manifest: _BackupManifest
) -> Iterator[tuple[str, str, Any]]:
    top_level_keys: set[str] = set()
    for prefix, event, value in ijson.parse(archive_stream, use_float=True):
        if prefix == "" and event == "start_array":
            raise ValueError("Expected a canonical sonora_backup_v2 archive")
        if prefix == "" and event == "map_key":
            if value in top_level_keys:
                raise ValueError(f"Duplicate backup manifest field: {value}")
            top_level_keys.add(value)
        if prefix == "schema":
            manifest.schema = value if event == "string" else None
        elif prefix == "root":
            manifest.root = value if event == "string" else None
        elif prefix == "tracks" and event not in {"end_map", "map_key"}:
            if event != "start_map":
                raise TypeError("Invalid backup manifest")
            manifest.has_tracks = True
        yield prefix, event, value


def _validate_backup_tags(relative_path: str, raw_tags: Any) -> dict[str, Any]:
    if not isinstance(raw_tags, dict):
        raise TypeError("Invalid backup track entry")
    if not isinstance(raw_tags.get("tags"), dict) or not isinstance(
        raw_tags.get("complex_tags"), dict
    ):
        raise TypeError(f"Invalid raw metadata for {relative_path}")
    if any(
        not isinstance(tag_name, str)
        or not isinstance(values, list)
        or any(not isinstance(value, str) for value in values)
        for tag_name, values in raw_tags["tags"].items()
    ):
        raise ValueError(f"Invalid tag values for {relative_path}")
    for property_key, property_maps in raw_tags["complex_tags"].items():
        if not isinstance(property_key, str) or not isinstance(property_maps, list):
            raise TypeError(f"Invalid complex property for {relative_path}")
        for property_fields in property_maps:
            if not isinstance(property_fields, dict):
                raise TypeError(f"Invalid complex property fields for {relative_path}")
            for property_field, property_value in property_fields.items():
                if not isinstance(property_field, str):
                    raise TypeError(
                        f"Invalid complex property field for {relative_path}"
                    )
                if isinstance(property_value, dict):
                    if set(property_value) != {"base64"} or not isinstance(
                        property_value["base64"], str
                    ):
                        raise ValueError(f"Invalid binary property for {relative_path}")
                    base64.b64decode(property_value["base64"], validate=True)
                elif not isinstance(property_value, (str, int, float, bool)):
                    raise TypeError(
                        f"Invalid complex property value for {relative_path}"
                    )
    return raw_tags


def _validate_backup_manifest(
    archive_stream: BinaryIO | gzip.GzipFile, target_directory: Path | None
) -> tuple[Path, int]:
    manifest = _BackupManifest()
    relative_paths: set[str] = set()
    events = _backup_manifest_events(archive_stream, manifest)
    try:
        for relative_path, raw_tags in ijson.kvitems(events, "tracks", use_float=True):
            if relative_path in relative_paths:
                raise ValueError(f"Duplicate backup track entry: {relative_path}")
            _validate_backup_tags(relative_path, raw_tags)
            relative_paths.add(relative_path)
    except (ijson.JSONError, EOFError, zlib.error) as error:
        raise ValueError(f"Invalid backup archive: {error}") from error
    if manifest.schema != "sonora_backup_v2":
        raise ValueError("Expected a canonical sonora_backup_v2 archive")
    if manifest.root is None or not manifest.has_tracks:
        raise TypeError("Invalid backup manifest")
    restore_root = (target_directory or Path(manifest.root)).resolve()
    resolved_targets: set[Path] = set()
    for relative_path in relative_paths:
        target_path = (restore_root / relative_path).resolve()
        if (
            Path(relative_path).is_absolute()
            or not target_path.is_relative_to(restore_root)
            or target_path in resolved_targets
        ):
            raise ValueError(
                f"Ambiguous or unsafe restore destination: {relative_path}"
            )
        resolved_targets.add(target_path)
    return restore_root, len(relative_paths)


def restore_library_tags(
    backup_file: Path,
    target_directory: Path | None = None,
    max_threads: int = 4,
    failures: list[str] | None = None,
) -> int:
    """Validate an entire archive, then restore native tags with bounded memory."""
    if not backup_file.exists():
        raise FileNotFoundError(f"Backup file not found: {backup_file}")
    if max_threads < 1:
        raise ValueError("max_threads must be positive")

    LOG.info(
        f"🔄 Starting tag restoration from {escape(str(backup_file))} (threads={max_threads})..."
    )
    restored_count = 0
    failed_count = 0
    missing_count = 0
    with _open_backup_stream(backup_file) as archive_stream:
        # Complete preflight precedes every write; a malformed final entry must not cause a partial restore.
        restore_root, track_count = _validate_backup_manifest(
            archive_stream, target_directory
        )
        archive_stream.seek(0)
        track_entries = ijson.kvitems(archive_stream, "tracks", use_float=True)
        with (
            create_progress() as progress,
            ThreadPoolExecutor(max_workers=max_threads) as executor,
        ):
            task = progress.add_task("[cyan]Restoring audio tags...", total=track_count)
            with interactive_pause_listener(progress, task) as pause_stop_event:
                try:
                    pending_restores = deque(
                        (
                            relative_path,
                            executor.submit(
                                _restore_single_track,
                                relative_path,
                                raw_tags,
                                restore_root,
                            ),
                        )
                        for relative_path, raw_tags in islice(
                            track_entries, max_threads
                        )
                    )
                    while pending_restores:
                        relative_path, restore_future = pending_restores.popleft()
                        restored, is_missing = restore_future.result()
                        if restored:
                            restored_count += 1
                        elif is_missing:
                            missing_count += 1
                        else:
                            failed_count += 1
                        if not restored and failures is not None:
                            failures.append(relative_path)
                        progress.advance(task)
                        next_entry = next(track_entries, None)
                        if next_entry is not None:
                            next_path, next_tags = next_entry
                            pending_restores.append(
                                (
                                    next_path,
                                    executor.submit(
                                        _restore_single_track,
                                        next_path,
                                        next_tags,
                                        restore_root,
                                    ),
                                )
                            )
                except (KeyboardInterrupt, RuntimeError) as exc:
                    if not is_interruption(exc):
                        raise
                    resume_workers(pause_stop_event)
                    executor.shutdown(wait=True, cancel_futures=True)
                    raise InterruptedOperationError(restored_count) from None

    LOG.info(f"✅ Successfully restored {restored_count} files")
    if missing_count > 0:
        LOG.warning(f"   ⚠️  {missing_count} files in backup are missing from disk")
    if failed_count > 0:
        LOG.warning(f"   ⚠️  {failed_count} files failed to restore")
    return restored_count
