from pathlib import Path

import numpy as np
import scipy.signal
import soundfile

from sonora.audio.bpm import load_audio
from sonora.core.logger import LOG


def detect_fake_lossless(file_path: Path) -> tuple[bool, float, str | None]:
    """
    Returns: tuple of (is_fake_lossless, detected_cutoff_khz, description)
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        sample_rate: int = 44100
        audio_mono: np.ndarray | None = None

        # Partial read: seek to active middle section (30% in) to avoid silent/spoken intros
        try:
            with soundfile.SoundFile(str(file_path)) as audio_file:
                sample_rate = int(audio_file.samplerate)
                total_frames = audio_file.frames
                duration_seconds = (
                    total_frames / sample_rate if sample_rate > 0 else 0.0
                )

                if duration_seconds > 60.0:
                    offset_seconds = min(45.0, duration_seconds * 0.30)
                else:
                    offset_seconds = max(0.0, (duration_seconds - 20.0) / 2.0)

                audio_file.seek(int(sample_rate * offset_seconds))
                read_frames = int(sample_rate * 20.0)
                raw_samples = audio_file.read(
                    frames=read_frames, dtype="float32", always_2d=True
                )
                if raw_samples.shape[1] > 1:
                    audio_mono = np.mean(raw_samples, axis=1)
                else:
                    audio_mono = raw_samples[:, 0]
        except (soundfile.LibsndfileError, OSError):
            audio_mono = None

        if audio_mono is None:
            loaded_audio = load_audio(
                file_path, mono=True, max_seconds=20.0, offset_seconds=30.0
            )
            if loaded_audio is None:
                return False, 0.0, None
            audio_mono, sample_rate = loaded_audio

        if sample_rate < 32000 or len(audio_mono) < sample_rate * 5:
            return False, 0.0, None

        # Compute Fast Fourier Transform spectrogram
        frequency_axis, _, spectrogram_matrix = scipy.signal.spectrogram(
            audio_mono, fs=sample_rate, nperseg=2048, noverlap=1024
        )

        # 95th percentile power across time slices captures high-frequency transient peaks (hi-hats, vocal air)
        peak_spectrum_power = np.percentile(spectrogram_matrix, 95, axis=1)
        max_power = float(np.max(peak_spectrum_power))
        if max_power <= 1e-12:
            return False, 0.0, None

        peak_spectrum_dbfs = 10.0 * np.log10(
            np.maximum(peak_spectrum_power, 1e-12) / max_power
        )

        # Evaluate transition bands between 14.5 kHz and 19.0 kHz (step 250 Hz)
        # 19 kHz is the maximum boundary because 44.1/48 kHz anti-aliasing reconstruction filters roll off at 20-21 kHz
        for target_frequency in np.arange(14500, 19250, 250):
            frequency_index = int(np.argmin(np.abs(frequency_axis - target_frequency)))
            transition_index = int(
                np.argmin(np.abs(frequency_axis - (target_frequency + 1000)))
            )
            stopband_end_index = int(
                np.argmin(np.abs(frequency_axis - (target_frequency + 2500)))
            )

            passband_level = float(peak_spectrum_dbfs[frequency_index])
            transition_level = float(peak_spectrum_dbfs[transition_index])
            stopband_slice = peak_spectrum_dbfs[
                transition_index : stopband_end_index + 1
            ]
            stopband_peak = (
                float(np.max(stopband_slice)) if len(stopband_slice) > 0 else -100.0
            )

            cliff_drop = passband_level - transition_level

            # Brickwall criterion: audible signal before cutoff (> -73 dBFS), steep cliff drop (>= 18 dB/kHz),
            # and complete stopband attenuation (< -75 dBFS) into quantization noise floor
            if passband_level > -73.0 and cliff_drop >= 18.0 and stopband_peak < -75.0:
                detected_cutoff_khz = float(target_frequency / 1000.0)
                if detected_cutoff_khz <= 16.5:
                    encoder_profile = "128kbps MP3/AAC"
                elif detected_cutoff_khz <= 17.8:
                    encoder_profile = "160kbps MP3"
                else:
                    encoder_profile = "192kbps MP3"

                description = (
                    f"Brickwall spectral cutoff detected at ~{detected_cutoff_khz:.1f}kHz "
                    f"(likely upscaled {encoder_profile} fake lossless)"
                )
                return True, detected_cutoff_khz, description

        return False, 0.0, None

    except OSError as error:
        LOG.debug(f"Spectral analysis skipped for {file_path}: {error}")
        return False, 0.0, None
