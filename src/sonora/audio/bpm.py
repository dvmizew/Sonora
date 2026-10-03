from pathlib import Path

import numpy as np
import scipy.signal

from sonora.audio.decode import load_audio
from sonora.core.logger import LOG


def calculate_bpm(file_path: Path) -> float | None:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        loaded = load_audio(file_path, mono=True, max_seconds=60.0)
        if loaded is None:
            return None

        audio_mono, sample_rate = loaded
        if len(audio_mono) == 0:
            return None

        # Downsample to ~22,050 Hz
        decimation = sample_rate // 22050
        if decimation > 1:
            audio_mono = audio_mono[::decimation]
            sample_rate = sample_rate // decimation

        nperseg = 512 if sample_rate <= 24000 else 1024
        noverlap = nperseg // 2
        _, _, spectrogram = scipy.signal.spectrogram(
            audio_mono, fs=sample_rate, nperseg=nperseg, noverlap=noverlap
        )
        onset_env = np.diff(np.mean(spectrogram, axis=0))
        onset_env = np.maximum(0, onset_env)

        if len(onset_env) == 0 or np.all(onset_env == 0):
            return None

        autocorr = scipy.signal.correlate(
            onset_env, onset_env, mode="full", method="fft"
        )
        autocorr = autocorr[len(autocorr) // 2 :]

        frame_rate = sample_rate / float(nperseg - noverlap)
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

    except OSError as error:
        LOG.debug(f"BPM calculation failed for {file_path}: {error}")
        return None
