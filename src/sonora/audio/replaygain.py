"""
ReplayGain & EBU R128 loudness analysis using FFmpeg.
"""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from sonora.core.constants import FFMPEG_CMD
from sonora.core.exceptions import AudioProcessingError


@dataclass
class ReplayGainResult:
    track_gain_db: float
    track_peak: float


def calculate_replaygain(file_path: Path, target_loudness_lufs: float = -18.0) -> ReplayGainResult:
    """
    Analyze audio track loudness using FFmpeg ebur128 filter and compute ReplayGain values.
    Standard target loudness is -18.0 LUFS which is the ReplayGain 2.0 specification.
    """
    if not file_path.exists():
        raise AudioProcessingError(f"File not found: {file_path}")

    try:
        cmd = [
            FFMPEG_CMD,
            "-hide_banner",
            "-i", str(file_path),
            "-af", "ebur128=peak=true",
            "-f", "null",
            "-"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=120)
        if result.returncode != 0:
            raise AudioProcessingError(f"FFmpeg loudness analysis failed: {result.stderr}")

        # Parse Integrated Loudness (LUFS)
        integrated_match = re.search(r"Integrated loudness:\s+I:\s+([-\d.]+)\s+LUFS", result.stderr)
        peak_match = re.search(r"Peak:\s+([-\d.]+)\s+dBFS", result.stderr)

        if not integrated_match:
            raise AudioProcessingError("Could not parse integrated loudness from FFmpeg output.")

        integrated_lufs = float(integrated_match.group(1))
        track_gain_db = round(target_loudness_lufs - integrated_lufs, 2)

        track_peak = 1.0
        if peak_match:
            peak_db = float(peak_match.group(1))
            track_peak = round(10 ** (peak_db / 20.0), 6)

        return ReplayGainResult(track_gain_db=track_gain_db, track_peak=track_peak)

    except FileNotFoundError as e:
        raise AudioProcessingError(f"ReplayGain failed: '{FFMPEG_CMD}' binary not found on system path.") from e
    except Exception as e:
        if isinstance(e, AudioProcessingError):
            raise
        raise AudioProcessingError(f"ReplayGain calculation error for {file_path}: {e}") from e
