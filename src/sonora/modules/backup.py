import datetime
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import orjson

from sonora.audio.metadata import read_track_metadata, write_track_metadata
from sonora.core.logger import LOG, create_progress
from sonora.core.models import TrackInfo
from sonora.core.utils import find_audio_files

_GZIP_MAGIC_HEADER = b"\x1f\x8b"


def _read_track_for_backup(audio_file: Path) -> tuple[str, dict[str, Any] | None]:
    try:
        track_info = read_track_metadata(audio_file)
        return str(audio_file), track_info.to_dict()
    except (OSError, ValueError, RuntimeError) as error:
        LOG.debug(f"Error reading {audio_file} for backup: {error}")
        return str(audio_file), None


def _restore_single_track(
    file_path_str: str, tags_dict: Any, base_dir: Path | None = None
) -> tuple[bool, bool]:
    target_path = Path(file_path_str)

    # Portable path resolution fallback if original absolute path was moved or mounted elsewhere
    if not target_path.exists() and base_dir is not None:
        direct_candidate = base_dir / target_path.name
        if direct_candidate.exists():
            target_path = direct_candidate
        else:
            matches = list(base_dir.rglob(target_path.name))
            if matches:
                target_path = matches[0]

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
    except (OSError, ValueError, RuntimeError) as error:
        LOG.debug(f"Failed to restore {target_path}: {error}")
    return False, False


def backup_library_tags(
    directory: Path, output_file: Path | None = None, max_threads: int = 4
) -> Path:
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Directory not found: {directory}")

    LOG.info(f"🔄 Scanning for files in {directory}...")
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
    backup_data: dict[str, Any] = {}
    failed = 0

    with create_progress() as progress:
        task = progress.add_task(
            "[cyan]Backing up audio tags...", total=len(audio_files)
        )
        executor = ThreadPoolExecutor(max_workers=max_threads)
        try:
            futures = [
                executor.submit(_read_track_for_backup, file_path)
                for file_path in audio_files
            ]
            for future in as_completed(futures):
                path_str, data = future.result()
                if data is not None:
                    backup_data[path_str] = data
                else:
                    failed += 1
                progress.advance(task)
        except KeyboardInterrupt:
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    try:
        raw_json_bytes = orjson.dumps(backup_data, option=orjson.OPT_INDENT_2)
        payload = (
            gzip.compress(raw_json_bytes, compresslevel=6)
            if output_path.name.endswith(".gz")
            else raw_json_bytes
        )

        temp_output = output_path.with_suffix(f"{output_path.suffix}.tmp")
        temp_output.write_bytes(payload)
        temp_output.replace(output_path)

        LOG.info(
            f"✅ Successfully backed up {len(backup_data)}/{len(audio_files)} files to {output_path}"
        )
        if failed > 0:
            LOG.warning(f"   ⚠️  {failed} files could not be read")
        return output_path
    except (OSError, TypeError) as error:
        LOG.error(f"Failed to save backup: {error}")
        raise


LAST_RESTORED_COUNT: int = 0


def get_last_restored_count() -> int:
    """Return the number of files restored during the most recent or interrupted restore run."""
    return LAST_RESTORED_COUNT


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

    global LAST_RESTORED_COUNT
    LAST_RESTORED_COUNT = 0

    LOG.info(
        f"🔄 Starting tag restoration from {backup_file} (threads={max_threads})..."
    )
    try:
        content = backup_file.read_bytes()
        if content.startswith(_GZIP_MAGIC_HEADER) or backup_file.name.endswith(".gz"):
            content = gzip.decompress(content)

        backup_dict: dict[str, Any] = orjson.loads(content)
        if not isinstance(backup_dict, dict):
            raise TypeError("Backup file is not a valid JSON object")
    except (orjson.JSONDecodeError, OSError, ValueError) as error:
        LOG.error(f"Failed to read backup file: {error}")
        raise

    count = 0
    failed = 0
    missing = 0
    search_base_dir = target_directory or backup_file.parent

    with create_progress() as progress:
        task = progress.add_task(
            "[cyan]Restoring audio tags...", total=len(backup_dict)
        )
        executor = ThreadPoolExecutor(max_workers=max_threads)
        try:
            futures = [
                executor.submit(_restore_single_track, file_str, tags, search_base_dir)
                for file_str, tags in backup_dict.items()
            ]
            for future in as_completed(futures):
                success, is_missing = future.result()
                if success:
                    count += 1
                    LAST_RESTORED_COUNT = count
                elif is_missing:
                    missing += 1
                else:
                    failed += 1
                progress.advance(task)
        except KeyboardInterrupt:
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    LOG.info(f"✅ Successfully restored {count} files")
    if missing > 0:
        LOG.warning(f"   ⚠️  {missing} files in backup are missing from disk")
    if failed > 0:
        LOG.warning(f"   ⚠️  {failed} files failed to restore")
    return count
