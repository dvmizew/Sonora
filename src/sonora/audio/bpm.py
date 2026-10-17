from math import gcd
from pathlib import Path

import numpy as np
import scipy.signal

from sonora.audio.decode import load_audio
from sonora.core.config import get_config
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

        if sample_rate > 22050:
            common_rate = gcd(sample_rate, 22050)
            audio_mono = scipy.signal.resample_poly(
                audio_mono, 22050 // common_rate, sample_rate // common_rate
            )
            sample_rate = 22050

        nperseg = 512
        if len(audio_mono) < nperseg:
            return None
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
        tempo_policy = get_config()
        min_lag = max(1, int(frame_rate * 60 / tempo_policy.bpm_search_max))
        max_lag = int(frame_rate * 60 / tempo_policy.bpm_search_min)

        if max_lag <= min_lag or len(autocorr) <= max_lag:
            return None

        peak_idx = min_lag + np.argmax(autocorr[min_lag:max_lag])
        bpm_value = (frame_rate * 60.0) / peak_idx

        if (
            tempo_policy.bpm_octave_min is not None
            and tempo_policy.bpm_octave_max is not None
        ):
            while bpm_value < tempo_policy.bpm_octave_min:
                bpm_value *= 2.0
            while bpm_value > tempo_policy.bpm_octave_max:
                bpm_value /= 2.0

        return round(float(bpm_value), 1)

    except OSError as error:
        LOG.debug(f"BPM calculation failed for {file_path}: {error}")
        return None
