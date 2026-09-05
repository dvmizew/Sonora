from pathlib import Path

import numpy as np
import scipy.signal

from sonora.audio.bpm import load_audio
from sonora.core.logger import LOG


def detect_fake_lossless(file_path: Path) -> tuple[bool, float, str | None]:
    """
    Returns: (is_fake_lossless, ratio, cutoff_description)
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        loaded = load_audio(file_path, mono=True, max_seconds=30.0)
        if loaded is None:
            return False, 1.0, None

        audio_mono, sample_rate = loaded
        if sample_rate < 32000 or len(audio_mono) == 0:
            return False, 1.0, None

        # Limit analysis to first 30 seconds
        if len(audio_mono) > sample_rate * 30:
            audio_mono = audio_mono[: sample_rate * 30]

        # Compute Fast Fourier Transform spectrogram via SciPy
        frequency_axis, _, spectrogram = scipy.signal.spectrogram(
            audio_mono, fs=sample_rate, nperseg=2048
        )
        power_spectrum = np.mean(spectrogram, axis=1)

        # Nyquist & cutoff frequency bins
        cutoff_16k_idx = np.argmin(np.abs(frequency_axis - 16000))
        cutoff_20k_idx = np.argmin(np.abs(frequency_axis - 20000))

        low_band = power_spectrum[:cutoff_16k_idx]
        high_band = power_spectrum[cutoff_16k_idx:cutoff_20k_idx]

        low_power = float(np.mean(low_band)) if len(low_band) > 0 else 1e-9
        high_power = float(np.mean(high_band)) if len(high_band) > 0 else 0.0

        ratio = (high_power / low_power) if low_power > 0 else 1.0

        # Hard brickwall cutoff detection: if high band energy is less than 0.1% of low band energy
        if ratio < 0.001:
            description = "Brickwall spectral cutoff detected at ~16-18kHz (likely upscaled 128-192kbps MP3 fake lossless)"
            return True, ratio, description

        return False, ratio, None

    except (OSError, ValueError, RuntimeError) as error:
        LOG.debug(f"Spectral analysis skipped for {file_path}: {error}")
        return False, 1.0, None
