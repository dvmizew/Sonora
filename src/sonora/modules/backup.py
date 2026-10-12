import base64
import datetime
import gzip
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from itertools import islice
from pathlib import Path
from typing import Any

import orjson
import taglib
from rich.markup import escape

from sonora.audio.metadata import clear_metadata_cache
from sonora.core.files import atomic_audio_edit, atomic_write_file
from sonora.core.logger import (
    LOG,
    create_progress,
    interactive_pause_listener,
    resume_workers,
    wait_if_paused,
)
from sonora.core.utils import (
    InterruptedOperationError,
    find_audio_files,
    is_interruption,
)

_GZIP_MAGIC_HEADER = b"\x1f\x8b"


def _read_track_for_backup(audio_file: Path) -> tuple[str, dict[str, Any] | None]:
    wait_if_paused()
    try:
        with taglib.File(str(audio_file)) as audio_tags:
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
            return str(audio_file), {
                "tags": audio_tags.tags,
                "complex_tags": complex_tags,
            }
    except OSError as error:
        LOG.debug(f"Error reading {audio_file} for backup: {error}")
        return str(audio_file), None


def _restore_single_track(
    file_path_str: str,
    tags_dict: dict[str, Any],
    base_dir: Path,
) -> tuple[bool, bool]:
    wait_if_paused()
    target_path = base_dir / file_path_str
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
            audio_tags.tags.update(tags_dict["tags"])
            for key in set(audio_tags.complex_property_keys) | set(
                tags_dict["complex_tags"]
            ):
                properties = [
                    {
                        field: base64.b64decode(value["base64"], validate=True)
                        if isinstance(value, dict)
                        else value
                        for field, value in property_fields.items()
                    }
                    for property_fields in tags_dict["complex_tags"].get(key, [])
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
                        path_str, tag_dict = pending_reads.popleft().result()
                        if tag_dict is not None:
                            if backed_up:
                                archive_stream.write(b",")
                            archive_stream.write(
                                orjson.dumps(str(Path(path_str).relative_to(directory)))
                            )
                            archive_stream.write(b":")
                            archive_stream.write(orjson.dumps(tag_dict))
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


def restore_library_tags(
    backup_file: Path,
    target_directory: Path | None = None,
    max_threads: int = 4,
    failures: list[str] | None = None,
) -> int:
    """
    Restore audio metadata tags from a JSON or GZipped JSON backup file.
    Automatically resolves relative paths if tracks were moved to target_directory or backup folder.
    """
    if not backup_file.exists():
        raise FileNotFoundError(f"Backup file not found: {backup_file}")

    LOG.info(
        f"🔄 Starting tag restoration from {escape(str(backup_file))} (threads={max_threads})..."
    )
    try:
        content = backup_file.read_bytes()
        if content.startswith(_GZIP_MAGIC_HEADER) or backup_file.name.endswith(".gz"):
            content = gzip.decompress(content)

        backup_payload: Any = orjson.loads(content)
    except (orjson.JSONDecodeError, OSError) as error:
        LOG.error(f"Failed to read backup file: {escape(str(error))}")
        raise

    if (
        not isinstance(backup_payload, dict)
        or backup_payload.get("schema") != "sonora_backup_v2"
    ):
        raise ValueError("Expected a canonical sonora_backup_v2 archive")
    backup_dict = backup_payload.get("tracks")
    if not isinstance(backup_dict, dict) or not isinstance(
        backup_payload.get("root"), str
    ):
        raise TypeError("Invalid backup manifest")
    search_base_dir = (target_directory or Path(backup_payload["root"])).resolve()
    resolved_targets: set[Path] = set()
    for relative_path, raw_tags in backup_dict.items():
        if not isinstance(relative_path, str) or not isinstance(raw_tags, dict):
            raise TypeError("Invalid backup track entry")
        target_path = (search_base_dir / relative_path).resolve()
        if (
            Path(relative_path).is_absolute()
            or not target_path.is_relative_to(search_base_dir)
            or target_path in resolved_targets
        ):
            raise ValueError(
                f"Ambiguous or unsafe restore destination: {relative_path}"
            )
        resolved_targets.add(target_path)
        if not isinstance(raw_tags.get("tags"), dict) or not isinstance(
            raw_tags.get("complex_tags"), dict
        ):
            raise TypeError(f"Invalid raw metadata for {relative_path}")
        if any(
            not isinstance(values, list)
            or any(not isinstance(value, str) for value in values)
            for values in raw_tags["tags"].values()
        ):
            raise ValueError(f"Invalid tag values for {relative_path}")
        for property_key, property_maps in raw_tags["complex_tags"].items():
            if not isinstance(property_key, str) or not isinstance(property_maps, list):
                raise TypeError(f"Invalid complex property for {relative_path}")
            for property_fields in property_maps:
                if not isinstance(property_fields, dict):
                    raise TypeError(
                        f"Invalid complex property fields for {relative_path}"
                    )
                for property_value in property_fields.values():
                    if isinstance(property_value, dict):
                        if set(property_value) != {"base64"} or not isinstance(
                            property_value["base64"], str
                        ):
                            raise ValueError(
                                f"Invalid binary property for {relative_path}"
                            )
                        base64.b64decode(property_value["base64"], validate=True)
                    elif not isinstance(property_value, (str, int, float, bool)):
                        raise TypeError(
                            f"Invalid complex property value for {relative_path}"
                        )
    count = 0
    failed = 0
    missing = 0

    with (
        create_progress() as progress,
        ThreadPoolExecutor(max_workers=max_threads) as executor,
    ):
        task = progress.add_task(
            "[cyan]Restoring audio tags...", total=len(backup_dict)
        )
        futures = {
            executor.submit(
                _restore_single_track,
                file_str,
                tags,
                search_base_dir,
            ): file_str
            for file_str, tags in backup_dict.items()
        }
        with interactive_pause_listener(progress, task) as pause_stop_event:
            try:
                for future in as_completed(futures):
                    success, is_missing = future.result()
                    if success:
                        count += 1
                    elif is_missing:
                        missing += 1
                    else:
                        failed += 1
                    if not success and failures is not None:
                        failures.append(futures[future])
                    progress.advance(task)
            except (KeyboardInterrupt, RuntimeError) as exc:
                if not is_interruption(exc):
                    raise
                resume_workers(pause_stop_event)
                executor.shutdown(wait=True, cancel_futures=True)
                raise InterruptedOperationError(count) from None

    LOG.info(f"✅ Successfully restored {count} files")
    if missing > 0:
        LOG.warning(f"   ⚠️  {missing} files in backup are missing from disk")
    if failed > 0:
        LOG.warning(f"   ⚠️  {failed} files failed to restore")
    return count
