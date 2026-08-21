import datetime
import gc
import json
from pathlib import Path
from typing import Any

from sonora.audio.metadata import read_track_metadata, write_track_metadata
from sonora.core.constants import SUPPORTED_EXTS
from sonora.core.logger import LOG


def backup_library_tags(directory: Path, output_file: Path | None = None) -> Path:
    """
    Stream-based backup: writes JSON line-by-line to avoid RAM overload.
    Excludes cover art data to keep backup size lightweight.
    """
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Directory not found: {directory}")

    LOG.info(f"🔄 Scanning for files in {directory}...")
    audio_files = sorted([
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ])

    if not audio_files:
        LOG.warning("No audio files found to back up.")
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = output_file or Path(f"backup_{timestamp_str}.json")
        out_path.write_text("{}\n", encoding="utf-8")
        return out_path

    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = output_file or Path(f"backup_{timestamp_str}.json")

    LOG.info(f"🔄 Creating full backup for {len(audio_files)} files (streaming mode)...")
    count = 0
    failed = 0

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("{\n")
            first = True
            for idx, file_p in enumerate(audio_files):
                try:
                    info = read_track_metadata(file_p)
                    data = info.to_dict()

                    if not first:
                        f.write(",\n")
                    f.write(f"  {json.dumps(str(file_p))}: ")
                    f.write(json.dumps(data, ensure_ascii=False))
                    first = False
                    count += 1
                except Exception as e:
                    LOG.debug(f"Error reading {file_p} for backup: {e}")
                    failed += 1

                if (idx + 1) % 500 == 0:
                    gc.collect()
                    LOG.info(f"   ∟ Progress: {idx + 1}/{len(audio_files)} files backed up...")

            f.write("\n}\n")

        LOG.info(f"✅ Successfully backed up {count}/{len(audio_files)} files to {out_path}")
        if failed > 0:
            LOG.warning(f"   ⚠️  {failed} files could not be read")
        return out_path
    except Exception as e:
        LOG.error(f"Failed to save backup: {e}")
        raise


def restore_library_tags(backup_file: Path) -> int:
    """
    Stream-based restore: parses JSON incrementally to avoid RAM overload.
    Returns the number of restored files.
    """
    if not backup_file.exists():
        raise FileNotFoundError(f"Backup file not found: {backup_file}")

    LOG.info(f"🔄 Starting tag restoration from {backup_file} (streaming mode)...")
    count = 0
    failed = 0
    missing = 0

    try:
        with open(backup_file, "r", encoding="utf-8") as f:
            dec = json.JSONDecoder()
            buf = ""
            idx = 0
            eof = False
            processed = 0

            def _fill(min_avail: int = 1) -> None:
                nonlocal buf, eof
                while (len(buf) - idx) < min_avail and not eof:
                    chunk = f.read(65536)
                    if chunk:
                        buf += chunk
                    else:
                        eof = True

            def _skip_ws() -> None:
                nonlocal idx
                while True:
                    _fill(1)
                    if idx >= len(buf) or buf[idx] not in " \t\r\n":
                        return
                    idx += 1

            def _decode_one() -> Any:
                nonlocal idx
                while True:
                    _fill(1)
                    try:
                        val, end = dec.raw_decode(buf, idx)
                        idx = end
                        return val
                    except json.JSONDecodeError:
                        if eof:
                            raise
                        _fill(65536)

            def _trim_buffer() -> None:
                nonlocal buf, idx
                if idx > 512 * 1024:
                    buf = buf[idx:]
                    idx = 0

            _skip_ws()
            _fill(1)
            if idx >= len(buf) or buf[idx] != "{":
                raise ValueError("Backup file is not a valid JSON object")
            idx += 1

            while True:
                _skip_ws()
                _fill(1)
                if idx >= len(buf) or buf[idx] == "}":
                    if idx < len(buf):
                        idx += 1
                    break

                try:
                    f_path_str = _decode_one()
                    if not isinstance(f_path_str, str):
                        raise ValueError("Backup key is not a string path")

                    _skip_ws()
                    _fill(1)
                    if idx >= len(buf) or buf[idx] != ":":
                        raise ValueError("Missing ':' between path and payload")
                    idx += 1

                    _skip_ws()
                    tags_dict = _decode_one()

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
                        except Exception as e:
                            LOG.debug(f"Failed to restore {f_path}: {e}")
                            failed += 1

                    processed += 1
                    if processed % 200 == 0:
                        gc.collect()
                        LOG.info(f"   Restored progress: {processed} entries...")

                    _skip_ws()
                    _fill(1)
                    if idx < len(buf) and buf[idx] == ",":
                        idx += 1
                    elif idx < len(buf) and buf[idx] == "}":
                        idx += 1
                        break

                    _trim_buffer()

                except Exception as e:
                    LOG.debug(f"Skipping malformed restore entry: {e}")
                    failed += 1
                    _fill(1)
                    while idx < len(buf) and buf[idx] not in ",}":
                        idx += 1
                    if idx < len(buf) and buf[idx] == ",":
                        idx += 1
                    elif idx < len(buf) and buf[idx] == "}":
                        idx += 1
                        break
                    _trim_buffer()

        gc.collect()
        LOG.info(f"✅ Successfully restored {count} files")
        if missing > 0:
            LOG.warning(f"   ⚠️  {missing} files in backup are missing from disk")
        if failed > 0:
            LOG.warning(f"   ⚠️  {failed} files failed to restore")
        return count
    except Exception as e:
        LOG.error(f"Failed to read backup file: {e}")
        raise
