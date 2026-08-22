import wave
from pathlib import Path

import numpy as np
import scipy.signal

from sonora.core.logger import LOG


def detect_fake_lossless(file_path: Path) -> tuple[bool, float, str | None]:
    """
    Returns: (is_fake_lossless, ratio, cutoff_description)
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        if file_path.suffix.lower() == ".wav":
            with wave.open(str(file_path), "rb") as wf:
                samplerate = wf.getframerate()
                if samplerate < 32000:
                    return False, 1.0, None
                nframes = wf.getnframes()
                frames = wf.readframes(min(nframes, samplerate * 30))
                dtype = np.int16 if wf.getsampwidth() == 2 else np.int32
                raw = np.frombuffer(frames, dtype=dtype).astype(np.float32)
                if wf.getnchannels() > 1:
                    mono = np.mean(raw.reshape(-1, wf.getnchannels()), axis=1)
                else:
                    mono = raw
        else:
            return False, 1.0, None

        if len(mono) == 0:
            return False, 1.0, None

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
