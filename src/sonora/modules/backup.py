import datetime
import gc
from pathlib import Path
from typing import Any

import orjson
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from sonora.audio.metadata import read_track_metadata, write_track_metadata
from sonora.core.constants import SUPPORTED_EXTS
from sonora.core.logger import CONSOLE, LOG

_ORJSON_OPTIONS = (
    orjson.OPT_SERIALIZE_DATACLASS
    | orjson.OPT_SERIALIZE_NUMPY
    | orjson.OPT_NON_STR_KEYS
)


def backup_library_tags(directory: Path, output_file: Path | None = None) -> Path:
    """
    Stream-based backup.
    Excludes cover art data to keep backup size lightweight.
    """
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Directory not found: {directory}")

    LOG.info(f"🔄 Scanning for files in {directory}...")
    audio_files = sorted(
        [
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS
        ]
    )

    timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    output_path = output_file or Path(f"backup_{timestamp_str}.json")

    if not audio_files:
        LOG.warning("No audio files found to back up.")
        output_path.write_bytes(b"{}\n")
        return output_path

    LOG.info(
        f"🔄 Creating full backup for {len(audio_files)} files (streaming mode)..."
    )
    count = 0
    failed = 0

    try:
        with open(output_path, "wb") as file_handle:
            file_handle.write(b"{\n")
            first = True
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TextColumn("[dim]/[/dim]"),
                TimeRemainingColumn(),
                console=CONSOLE,
            ) as progress:
                task = progress.add_task(
                    "[cyan]Backing up audio tags...", total=len(audio_files)
                )
                for index, audio_file in enumerate(audio_files):
                    try:
                        track_info = read_track_metadata(audio_file)
                        track_data = track_info.to_dict()

                        if not first:
                            file_handle.write(b",\n")
                        key_bytes = orjson.dumps(str(audio_file), option=_ORJSON_OPTIONS)
                        value_bytes = orjson.dumps(track_data, option=_ORJSON_OPTIONS)
                        file_handle.write(b"  " + key_bytes + b": " + value_bytes)
                        first = False
                        count += 1
                    except (OSError, ValueError, RuntimeError) as error:
                        LOG.debug(f"Error reading {audio_file} for backup: {error}")
                        failed += 1

                    progress.advance(task)
                    if (index + 1) % 500 == 0:
                        gc.collect()

            file_handle.write(b"\n}\n")

        LOG.info(
            f"✅ Successfully backed up {count}/{len(audio_files)} files to {output_path}"
        )
        if failed > 0:
            LOG.warning(f"   ⚠️  {failed} files could not be read")
        return output_path
    except (OSError, ValueError, TypeError) as error:
        LOG.error(f"Failed to save backup: {error}")
        raise


def restore_library_tags(backup_file: Path) -> int:
    """
    High-performance restore: parses JSON via orjson at native C/Rust speed.
    Returns the number of restored files.
    """
    if not backup_file.exists():
        raise FileNotFoundError(f"Backup file not found: {backup_file}")

    LOG.info(f"🔄 Starting tag restoration from {backup_file} (streaming mode)...")
    count = 0
    failed = 0
    missing = 0

    try:
        content = backup_file.read_bytes()
        backup_dict: dict[str, Any] = orjson.loads(content)
        if not isinstance(backup_dict, dict):
            raise TypeError("Backup file is not a valid JSON object")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=CONSOLE,
        ) as progress:
            task = progress.add_task(
                "[cyan]Restoring audio tags...", total=len(backup_dict)
            )
            processed = 0
            for file_path_string, tags_dictionary in backup_dict.items():
                target_file_path = Path(file_path_string)
                if not target_file_path.exists():
                    missing += 1
                else:
                    try:
                        track_info = read_track_metadata(target_file_path)
                        if isinstance(tags_dictionary, dict):
                            for tag_key, tag_value in tags_dictionary.items():
                                if tag_key in ("file_path", "file_name"):
                                    continue
                                if hasattr(track_info, tag_key) and tag_value is not None:
                                    setattr(track_info, tag_key, tag_value)
                            write_track_metadata(track_info)
                            count += 1
                    except (OSError, ValueError, KeyError, RuntimeError) as error:
                        LOG.debug(f"Failed to restore {target_file_path}: {error}")
                        failed += 1

                processed += 1
                progress.advance(task)
                if processed % 500 == 0:
                    gc.collect()

        gc.collect()
        LOG.info(f"✅ Successfully restored {count} files")
        if missing > 0:
            LOG.warning(f"   ⚠️  {missing} files in backup are missing from disk")
        if failed > 0:
            LOG.warning(f"   ⚠️  {failed} files failed to restore")
        return count
    except (orjson.JSONDecodeError, OSError, ValueError, KeyError) as error:
        LOG.error(f"Failed to read backup file: {error}")
        raise
