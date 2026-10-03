import subprocess
from pathlib import Path

import numpy as np
import soundfile

from sonora.core.logger import LOG


def load_audio(
    file_path: Path,
    mono: bool = False,
    max_seconds: float | None = None,
    offset_seconds: float = 0.0,
) -> tuple[np.ndarray, int] | None:
    """
    Load raw audio samples into a numpy float32 array and return (audio_data, sample_rate).
    Tries fast soundfile direct decoding first, falling back to ffmpeg for exotic formats.
    """
    try:
        with soundfile.SoundFile(str(file_path)) as sf:
            sample_rate = int(sf.samplerate)
            if offset_seconds > 0 and sf.frames > int(
                sample_rate * (offset_seconds + 2.0)
            ):
                sf.seek(int(sample_rate * offset_seconds))
            frames = (
                int(sample_rate * max_seconds)
                if (max_seconds and max_seconds > 0)
                else -1
            )
            audio_data = sf.read(frames=frames, dtype="float32", always_2d=True)

        if mono:
            audio_data = (
                np.mean(audio_data, axis=1)
                if audio_data.shape[1] > 1
                else audio_data[:, 0]
            )
        return audio_data, int(sample_rate)
    except (soundfile.LibsndfileError, OSError) as error:
        LOG.debug(f"soundfile decode failed for {file_path}: {error}")

    try:
        command = [
            "ffmpeg",
            *(["-ss", str(offset_seconds)] if offset_seconds > 0 else []),
            "-i",
            str(file_path),
            *(["-t", str(max_seconds)] if max_seconds and max_seconds > 0 else []),
            "-f",
            "f32le",
            "-ar",
            "44100",
            "-ac",
            "1" if mono else "2",
            "-v",
            "quiet",
            "-",
        ]
        result = subprocess.run(command, capture_output=True, check=True, timeout=15.0)
        if result.stdout:
            buffer_data = np.frombuffer(result.stdout, dtype=np.float32)
            audio_array: np.ndarray = (
                buffer_data.reshape(-1, 2) if not mono else buffer_data
            )
            if len(audio_array) > 0:
                return audio_array, 44100
    except (subprocess.SubprocessError, OSError) as error:
        LOG.debug(f"ffmpeg decode failed for {file_path}: {error}")

    return None
