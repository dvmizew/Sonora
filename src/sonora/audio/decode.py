import shutil
import subprocess
from pathlib import Path
from typing import cast

import mutagen
import numpy as np
import soundfile
from mutagen import MutagenError
from numpy.typing import NDArray

from sonora.core.logger import LOG


def _probe_channels(file_path: Path) -> int:
    """Return native audio channel count for FFmpeg decoding, defaulting to stereo."""
    try:
        audio = mutagen.File(file_path)
        channels = getattr(getattr(audio, "info", None), "channels", None)
        if isinstance(channels, int) and channels > 0:
            return channels
    except (MutagenError, OSError):
        pass

    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            probe_command = [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=channels",
                "-of",
                "default=nw=1:nk=1",
                str(file_path),
            ]
            probe = subprocess.run(
                probe_command,
                capture_output=True,
                check=True,
                timeout=5.0,
            )
            raw = probe.stdout.decode("ascii", errors="ignore").strip()
            if raw.isdigit() and int(raw) > 0:
                return int(raw)
        except (subprocess.SubprocessError, OSError):
            pass

    return 2


def load_audio(
    file_path: Path,
    mono: bool = False,
    max_seconds: float | None = None,
    offset_seconds: float = 0.0,
) -> tuple[NDArray[np.float32], int] | None:
    """
    Load raw audio samples into a numpy float32 array and return (audio_data, sample_rate).
    Tries fast soundfile direct decoding first, falling back to ffmpeg for exotic formats.
    """
    try:
        with soundfile.SoundFile(str(file_path)) as sf:
            sample_rate = int(sf.samplerate)
            if offset_seconds > 0:
                sf.seek(min(int(sample_rate * offset_seconds), sf.frames))
            frames = (
                int(sample_rate * max_seconds)
                if (max_seconds and max_seconds > 0)
                else -1
            )
            audio_data = cast(
                NDArray[np.float32],
                sf.read(frames=frames, dtype="float32", always_2d=True),
            )

        if mono:
            audio_data = (
                np.mean(audio_data, axis=1)
                if audio_data.shape[1] > 1
                else audio_data[:, 0]
            )
        return audio_data, sample_rate
    except (soundfile.LibsndfileError, OSError) as error:
        LOG.debug(f"soundfile decode failed for {file_path}: {error}")

    try:
        channels = 1 if mono else _probe_channels(file_path)
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
            str(channels),
            "-v",
            "quiet",
            "-",
        ]
        result = subprocess.run(command, capture_output=True, check=True, timeout=15.0)
        if result.stdout:
            buffer_data = np.frombuffer(result.stdout, dtype=np.float32)
            if mono:
                audio_array: NDArray[np.float32] = buffer_data
            else:
                usable_samples = (len(buffer_data) // channels) * channels
                audio_array = buffer_data[:usable_samples].reshape(-1, channels)
            return audio_array, 44100
    except (subprocess.SubprocessError, OSError) as error:
        LOG.debug(f"ffmpeg decode failed for {file_path}: {error}")

    return None
