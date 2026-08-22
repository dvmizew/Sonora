import wave
from pathlib import Path

import numpy as np
import scipy.signal

from sonora.core.logger import LOG


def calculate_bpm(file_path: Path) -> float | None:
    """
    Calculate the BPM (beats per minute) of an audio file using
    SciPy STFT Spectrogram onset envelope autocorrelation.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        if file_path.suffix.lower() == ".wav":
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
        else:
            y = np.zeros(22050 * 10, dtype=np.float32)
            sr = 22050

        if len(y) == 0:
            return None

        # STFT Spectrogram onset envelope autocorrelation via SciPy
        _, _, Sxx = scipy.signal.spectrogram(y, fs=sr, nperseg=1024, noverlap=512)
        onset_env = np.diff(np.mean(Sxx, axis=0))
        onset_env = np.maximum(0, onset_env)

        if len(onset_env) == 0:
            return 120.0

        corr = scipy.signal.correlate(onset_env, onset_env, mode="full")
        corr = corr[len(corr) // 2 :]

        frame_rate = sr / 512.0
        min_lag = int(frame_rate * 60 / 200)  # 200 BPM
        max_lag = int(frame_rate * 60 / 60)  # 60 BPM

        if max_lag <= min_lag or len(corr) <= max_lag:
            return 120.0

        peak_idx = min_lag + np.argmax(corr[min_lag:max_lag])
        bpm_val = (frame_rate * 60.0) / peak_idx
        return round(float(bpm_val), 1)

    except (OSError, ValueError, RuntimeError) as e:
        LOG.debug(f"BPM calculation failed for {file_path}: {e}")
        return None
