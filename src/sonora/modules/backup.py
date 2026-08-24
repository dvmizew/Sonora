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

_ORJSON_OPTS = (
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
            p
            for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        ]
    )

    timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    out_path = output_file or Path(f"backup_{timestamp_str}.json")

    if not audio_files:
        LOG.warning("No audio files found to back up.")
        out_path.write_bytes(b"{}\n")
        return out_path

    LOG.info(
        f"🔄 Creating full backup for {len(audio_files)} files (streaming mode)..."
    )
    count = 0
    failed = 0

    try:
        with open(out_path, "wb") as f:
            f.write(b"{\n")
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
                for idx, file_p in enumerate(audio_files):
                    try:
                        info = read_track_metadata(file_p)
                        data = info.to_dict()

                        if not first:
                            f.write(b",\n")
                        key_bytes = orjson.dumps(str(file_p), option=_ORJSON_OPTS)
                        val_bytes = orjson.dumps(data, option=_ORJSON_OPTS)
                        f.write(b"  " + key_bytes + b": " + val_bytes)
                        first = False
                        count += 1
                    except (OSError, ValueError, RuntimeError) as e:
                        LOG.debug(f"Error reading {file_p} for backup: {e}")
                        failed += 1

                    progress.advance(task)
                    if (idx + 1) % 500 == 0:
                        gc.collect()

            f.write(b"\n}\n")

        LOG.info(
            f"✅ Successfully backed up {count}/{len(audio_files)} files to {out_path}"
        )
        if failed > 0:
            LOG.warning(f"   ⚠️  {failed} files could not be read")
        return out_path
    except (OSError, ValueError, TypeError) as e:
        LOG.error(f"Failed to save backup: {e}")
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
            for f_path_str, tags_dict in backup_dict.items():
                f_path = Path(f_path_str)
                if not f_path.exists():
                    missing += 1
                else:
                    try:
                        info = read_track_metadata(f_path)
                        if isinstance(tags_dict, dict):
                            for k, v in tags_dict.items():
                                if k in ("file_path", "file_name"):
                                    continue
                                if hasattr(info, k) and v is not None:
                                    setattr(info, k, v)
                            write_track_metadata(info)
                            count += 1
                    except (OSError, ValueError, KeyError, RuntimeError) as e:
                        LOG.debug(f"Failed to restore {f_path}: {e}")
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
    except (orjson.JSONDecodeError, OSError, ValueError, KeyError) as e:
        LOG.error(f"Failed to read backup file: {e}")
        raise
