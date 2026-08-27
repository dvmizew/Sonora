import subprocess
from pathlib import Path

from sonora.core.constants import FLAC_CMD


def verify_flac_checksum(file_path: Path) -> bool:
    """
    Verify the audio stream MD5 checksum of a FLAC file using `flac -t`.
    Returns True if valid, False if corrupted.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.suffix.lower() != ".flac":
        return True

    try:
        result = subprocess.run(
            [FLAC_CMD, "-t", "--silent", str(file_path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        return result.returncode == 0
    except FileNotFoundError as error:
        raise RuntimeError(
            f"STRICT check failed: '{FLAC_CMD}' binary not found on system path."
        ) from error
    except (subprocess.SubprocessError, OSError) as error:
        raise RuntimeError(
            f"Checksum verification failed for {file_path}: {error}"
        ) from error
