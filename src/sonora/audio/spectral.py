from pathlib import Path

import numpy as np
import scipy.signal

from sonora.audio.bpm import _load_audio_mono
from sonora.core.logger import LOG


def detect_fake_lossless(file_path: Path) -> tuple[bool, float, str | None]:
    """
    Returns: (is_fake_lossless, ratio, cutoff_description)
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        loaded = _load_audio_mono(file_path)
        if loaded is None:
            return False, 1.0, None

        mono, samplerate = loaded
        if samplerate < 32000 or len(mono) == 0:
            return False, 1.0, None

        # Limit analysis to first 30 seconds
        if len(mono) > samplerate * 30:
            mono = mono[: samplerate * 30]

        # Compute Fast Fourier Transform spectrogram via SciPy
        f_axis, _, Sxx = scipy.signal.spectrogram(mono, fs=samplerate, nperseg=2048)
        power_spectrum = np.mean(Sxx, axis=1)

        # Nyquist & cutoff frequency bins
        cutoff_16k_idx = np.argmin(np.abs(f_axis - 16000))
        cutoff_20k_idx = np.argmin(np.abs(f_axis - 20000))

        low_band = power_spectrum[:cutoff_16k_idx]
        high_band = power_spectrum[cutoff_16k_idx:cutoff_20k_idx]

        low_power = float(np.mean(low_band)) if len(low_band) > 0 else 1e-9
        high_power = float(np.mean(high_band)) if len(high_band) > 0 else 0.0

        ratio = (high_power / low_power) if low_power > 0 else 1.0

        # Hard brickwall cutoff detection: if high band energy is less than 0.1% of low band energy
        if ratio < 0.001:
            desc = "Brickwall spectral cutoff detected at ~16-18kHz (likely upscaled 128-192kbps MP3 fake lossless)"
            return True, ratio, desc

        return False, ratio, None

    except (OSError, ValueError, RuntimeError) as e:
        LOG.debug(f"Spectral analysis skipped for {file_path}: {e}")
        return False, 1.0, None
