import datetime
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import orjson
from rich.markup import escape

from sonora.audio.metadata import read_track_metadata, write_track_metadata
from sonora.core.logger import (
    LOG,
    create_progress,
    interactive_pause_listener,
    wait_if_paused,
)
from sonora.core.models import TrackInfo
from sonora.core.utils import (
    InterruptedOperationError,
    find_audio_files,
    is_interruption,
)

_GZIP_MAGIC_HEADER = b"\x1f\x8b"


def _read_track_for_backup(audio_file: Path) -> tuple[str, dict[str, Any] | None]:
    wait_if_paused()
    try:
        track_info = read_track_metadata(audio_file)
        return str(audio_file), track_info.to_dict()
    except OSError as error:
        LOG.debug(f"Error reading {audio_file} for backup: {error}")
        return str(audio_file), None


def _restore_single_track(
    file_path_str: str,
    tags_dict: Any,
    base_dir: Path | None = None,
    candidate_lookup: dict[str, Path] | None = None,
) -> tuple[bool, bool]:
    wait_if_paused()
    target_path = Path(file_path_str)

    # Portable path resolution fallback if original absolute path was moved or mounted elsewhere
    if not target_path.exists() and base_dir is not None:
        direct_candidate = base_dir / target_path.name
        if direct_candidate.exists():
            target_path = direct_candidate
        elif candidate_lookup and target_path.name in candidate_lookup:
            target_path = candidate_lookup[target_path.name]
        elif candidate_lookup is None and base_dir.exists():
            target_path = next(base_dir.rglob(target_path.name), target_path)

    if not target_path.exists():
        return False, True

    try:
        if isinstance(tags_dict, dict):
            clean_tags = {
                k: v
                for k, v in tags_dict.items()
                if k not in ("file_path", "file_name") and hasattr(TrackInfo, k)
            }
            write_track_metadata(TrackInfo(file_path=target_path, **clean_tags))
            return True, False
    except OSError as error:
        LOG.debug(f"Failed to restore {target_path}: {error}")
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

    if not audio_files:
        LOG.warning("No audio files found to back up.")
        output_path.write_bytes(b"{}\n")
        return output_path

    LOG.info(
        f"🔄 Creating full backup for {len(audio_files)} files (threads={max_threads})..."
    )
    backup_manifest: dict[str, Any] = {}
    failed = 0

    with (
        create_progress() as progress,
        ThreadPoolExecutor(max_workers=max_threads) as executor,
    ):
        task = progress.add_task(
            "[cyan]Backing up audio tags...", total=len(audio_files)
        )
        futures = [
            executor.submit(_read_track_for_backup, file_path)
            for file_path in audio_files
        ]
        with interactive_pause_listener(progress, task):
            try:
                for future in as_completed(futures):
                    path_str, tag_dict = future.result()
                    if tag_dict is not None:
                        backup_manifest[path_str] = tag_dict
                    else:
                        failed += 1
                    progress.advance(task)
            except (KeyboardInterrupt, RuntimeError) as exc:
                if not is_interruption(exc):
                    raise
                executor.shutdown(wait=True, cancel_futures=True)
                raise

    try:
        raw_json_bytes = orjson.dumps(backup_manifest, option=orjson.OPT_INDENT_2)
        payload = (
            gzip.compress(raw_json_bytes, compresslevel=6)
            if output_path.name.endswith(".gz")
            else raw_json_bytes
        )

        temp_output = output_path.with_suffix(f"{output_path.suffix}.tmp")
        temp_output.write_bytes(payload)
        temp_output.replace(output_path)

        LOG.info(
            f"✅ Successfully backed up {len(backup_manifest)}/{len(audio_files)} files to {escape(str(output_path))}"
        )
        if failed > 0:
            LOG.warning(f"   ⚠️  {failed} files could not be read")
        return output_path
    except OSError as error:
        LOG.error(f"Failed to save backup: {error}")
        raise


def restore_library_tags(
    backup_file: Path,
    target_directory: Path | None = None,
    max_threads: int = 4,
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
        LOG.error(f"Failed to read backup file: {error}")
        raise

    if not isinstance(backup_payload, dict):
        error_msg = "Backup file is not a valid JSON object"
        LOG.error(f"Failed to read backup file: {error_msg}")
        raise TypeError(error_msg)
    backup_dict: dict[str, Any] = backup_payload

    count = 0
    failed = 0
    missing = 0
    search_base_dir = target_directory or backup_file.parent
    candidate_lookup: dict[str, Path] = {}
    if search_base_dir.exists():
        for audio_file_path in find_audio_files(search_base_dir, recursive=True):
            candidate_lookup.setdefault(audio_file_path.name, audio_file_path)

    with (
        create_progress() as progress,
        ThreadPoolExecutor(max_workers=max_threads) as executor,
    ):
        task = progress.add_task(
            "[cyan]Restoring audio tags...", total=len(backup_dict)
        )
        futures = [
            executor.submit(
                _restore_single_track,
                file_str,
                tags,
                search_base_dir,
                candidate_lookup,
            )
            for file_str, tags in backup_dict.items()
        ]
        with interactive_pause_listener(progress, task):
            try:
                for future in as_completed(futures):
                    success, is_missing = future.result()
                    if success:
                        count += 1
                    elif is_missing:
                        missing += 1
                    else:
                        failed += 1
                    progress.advance(task)
            except (KeyboardInterrupt, RuntimeError) as exc:
                if not is_interruption(exc):
                    raise
                executor.shutdown(wait=True, cancel_futures=True)
                raise InterruptedOperationError(count) from None

    LOG.info(f"✅ Successfully restored {count} files")
    if missing > 0:
        LOG.warning(f"   ⚠️  {missing} files in backup are missing from disk")
    if failed > 0:
        LOG.warning(f"   ⚠️  {failed} files failed to restore")
    return count
