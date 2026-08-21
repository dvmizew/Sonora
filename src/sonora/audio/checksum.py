import subprocess
from pathlib import Path

from sonora.core.constants import FLAC_CMD
from sonora.core.exceptions import AudioProcessingError


def verify_flac_checksum(file_path: Path) -> bool:
    """
    Verify the audio stream MD5 checksum of a FLAC file using `flac -t`.
    Returns True if valid, False if corrupted.
    Raises AudioProcessingError if flac executable is missing.
    """
    if not file_path.exists():
        raise AudioProcessingError(f"File not found: {file_path}")
    if file_path.suffix.lower() != ".flac":
        return True

    try:
        result = subprocess.run(
            [FLAC_CMD, "-t", "--silent", str(file_path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=60

        )
        return result.returncode == 0
    except FileNotFoundError as e:
        raise AudioProcessingError(
            f"STRICT check failed: '{FLAC_CMD}' binary not found on system path."
        ) from e
    except (subprocess.SubprocessError, OSError) as e:
        raise AudioProcessingError(f"Checksum verification failed for {file_path}: {e}") from e
