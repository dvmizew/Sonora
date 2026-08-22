import subprocess
import wave
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

from sonora.core.logger import LOG


def _load_audio_mono(file_path: Path) -> tuple[np.ndarray, int] | None:
    """Load audio file into 1D float32 NumPy array and sample rate."""
    ext = file_path.suffix.lower()

    # 1. Try soundfile C libsndfile (WAV, FLAC, OGG, AIFF)
    try:
        with sf.SoundFile(str(file_path)) as f:
            data = f.read(dtype="float32")
            sr = f.samplerate
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if len(data) > 0:
                return data, int(sr)
    except (sf.LibsndfileError, OSError, ValueError, RuntimeError) as e:
        LOG.debug(f"soundfile read failed for {file_path}: {e}")

    # 2. Try stdlib wave module for WAV
    if ext == ".wav":
        try:
            with wave.open(str(file_path), "rb") as wf:
                sr = wf.getframerate()
                nframes = wf.getnframes()
                frames = wf.readframes(nframes)
                dtype = np.int16 if wf.getsampwidth() == 2 else np.int32
                raw = np.frombuffer(frames, dtype=dtype).astype(np.float32)
                if wf.getnchannels() > 1:
                    y = np.mean(raw.reshape(-1, wf.getnchannels()), axis=1)
                else:
                    y = raw
                if len(y) > 0:
                    return y, sr
        except (wave.Error, OSError, ValueError, RuntimeError) as e:
            LOG.debug(f"wave read failed for {file_path}: {e}")

    # 3. Try ffmpeg subprocess fallback for MP3, M4A, AAC, etc.
    try:
        cmd = [
            "ffmpeg",
            "-i",
            str(file_path),
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            "22050",
            "-v",
            "quiet",
            "-",
        ]
        res = subprocess.run(cmd, capture_output=True, check=True)
        if res.stdout:
            raw = np.frombuffer(res.stdout, dtype=np.int16).astype(np.float32)
            if len(raw) > 0:
                return raw, 22050
    except (subprocess.SubprocessError, OSError, ValueError, RuntimeError) as e:
        LOG.debug(f"ffmpeg decode failed for {file_path}: {e}")

    return None


def calculate_bpm(file_path: Path) -> float | None:
    """
    Calculate the BPM of an audio file using STFT onset envelope autocorrelation via SciPy.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        loaded = _load_audio_mono(file_path)
        if loaded is None:
            return None

        y, sr = loaded
        if len(y) == 0:
            return None

        # Limit analysis to max 120 seconds to save CPU
        if len(y) > sr * 120:
            y = y[: sr * 120]

        # STFT Spectrogram onset envelope autocorrelation via SciPy
        _, _, Sxx = scipy.signal.spectrogram(y, fs=sr, nperseg=1024, noverlap=512)
        onset_env = np.diff(np.mean(Sxx, axis=0))
        onset_env = np.maximum(0, onset_env)

        if len(onset_env) == 0 or np.all(onset_env == 0):
            return None

        corr = scipy.signal.correlate(onset_env, onset_env, mode="full")
        corr = corr[len(corr) // 2 :]

        frame_rate = sr / 512.0
        min_lag = int(frame_rate * 60 / 200)  # 200 BPM
        max_lag = int(frame_rate * 60 / 60)  # 60 BPM

        if max_lag <= min_lag or len(corr) <= max_lag:
            return None

        peak_idx = min_lag + np.argmax(corr[min_lag:max_lag])
        if peak_idx == 0:
            return None

        bpm_val = (frame_rate * 60.0) / peak_idx

        # Octave normalization to standard 75-190 BPM music tempo range
        while bpm_val < 75.0:
            bpm_val *= 2.0
        while bpm_val > 190.0:
            bpm_val /= 2.0

        return round(float(bpm_val), 1)

    except (OSError, ValueError, RuntimeError) as e:
        LOG.debug(f"BPM calculation failed for {file_path}: {e}")
        return None
