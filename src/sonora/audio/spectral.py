"""
Spectral analysis for detecting fake lossless (upsampled) audio files.
"""

import re
import subprocess
from pathlib import Path

from sonora.core.constants import SOX_CMD
from sonora.core.exceptions import AudioProcessingError


def analyze_spectral_cutoff(file_path: Path, cutoff_hz: int = 16000) -> float:
    """
    Measure the RMS amplitude above the specified cutoff frequency (default 16kHz) using SoX.
    Returns RMS amplitude value. Lower values (< 0.001) indicate likely fake/upsampled audio.
    """
    if not file_path.exists():
        raise AudioProcessingError(f"File not found: {file_path}")

    try:
        result = subprocess.run(
            [SOX_CMD, str(file_path), "-n", "highpass", str(cutoff_hz), "stat"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode != 0:
            raise AudioProcessingError(f"SoX execution failed: {result.stderr}")

        match = re.search(r"RMS\s+amplitude:\s+([\d.]+)", result.stderr)
        if match:
            return float(match.group(1))
        return 0.0
    except FileNotFoundError as e:
        raise AudioProcessingError(
            f"Spectral check failed: '{SOX_CMD}' binary not found on system path."
        ) from e
    except Exception as e:
        if isinstance(e, AudioProcessingError):
            raise
        raise AudioProcessingError(f"Spectral analysis error for {file_path}: {e}") from e


def is_fake_lossless(file_path: Path, threshold_rms: float = 0.001) -> bool:
    """
    Returns True if the file has low energy above 16kHz, indicating upsampled audio.
    """
    rms = analyze_spectral_cutoff(file_path)
    return rms < threshold_rms
