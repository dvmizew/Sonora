import subprocess
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile

from sonora.core.logger import LOG


def load_audio(
    file_path: Path, mono: bool = False, max_seconds: float | None = None
) -> tuple[np.ndarray, int] | None:
    try:
        if max_seconds and max_seconds > 0:
            with soundfile.SoundFile(str(file_path)) as sf:
                sample_rate = int(sf.samplerate)
                frames = int(sample_rate * max_seconds)
                audio_data = sf.read(frames=frames, dtype="float32", always_2d=True)
        else:
            audio_data, sample_rate_val = soundfile.read(
                str(file_path), dtype="float32", always_2d=True
            )
            sample_rate = int(sample_rate_val)

        if mono:
            audio_data = (
                np.mean(audio_data, axis=1)
                if audio_data.shape[1] > 1
                else audio_data[:, 0]
            )
        return audio_data, int(sample_rate)
    except (soundfile.LibsndfileError, OSError, ValueError, RuntimeError) as error:
        LOG.debug(f"soundfile decode failed for {file_path}: {error}")

    try:
        command = [
            "ffmpeg",
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
        result = subprocess.run(command, capture_output=True, check=True)
        if result.stdout:
            buffer_data = np.frombuffer(result.stdout, dtype=np.float32)
            audio_array: np.ndarray = (
                buffer_data.reshape(-1, 2) if not mono else buffer_data
            )
            if len(audio_array) > 0:
                return audio_array, 44100
    except (subprocess.SubprocessError, OSError, ValueError, RuntimeError) as error:
        LOG.debug(f"ffmpeg decode failed for {file_path}: {error}")

    return None


def calculate_bpm(file_path: Path) -> float | None:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        loaded = load_audio(file_path, mono=True, max_seconds=90.0)
        if loaded is None:
            return None

        audio_mono, sample_rate = loaded
        if len(audio_mono) == 0:
            return None

        # STFT Spectrogram onset envelope autocorrelation via SciPy
        _, _, spectrogram = scipy.signal.spectrogram(
            audio_mono, fs=sample_rate, nperseg=1024, noverlap=512
        )
        onset_env = np.diff(np.mean(spectrogram, axis=0))
        onset_env = np.maximum(0, onset_env)

        if len(onset_env) == 0 or np.all(onset_env == 0):
            return None

        autocorr = scipy.signal.correlate(onset_env, onset_env, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]

        frame_rate = sample_rate / 512.0
        min_lag = int(frame_rate * 60 / 200)  # 200 BPM
        max_lag = int(frame_rate * 60 / 60)  # 60 BPM

        if max_lag <= min_lag or len(autocorr) <= max_lag:
            return None

        peak_idx = min_lag + np.argmax(autocorr[min_lag:max_lag])
        if peak_idx == 0:
            return None

        bpm_value = (frame_rate * 60.0) / peak_idx

        # Octave normalization to standard 75-190 BPM music tempo range
        while bpm_value < 75.0:
            bpm_value *= 2.0
        while bpm_value > 190.0:
            bpm_value /= 2.0

        return round(float(bpm_value), 1)

    except (OSError, ValueError, RuntimeError) as error:
        LOG.debug(f"BPM calculation failed for {file_path}: {error}")
        return None
