from pathlib import Path
from typing import cast

import numpy as np
import soundfile
from numpy.typing import NDArray

from sonora.audio.decode import load_audio
from sonora.core.logger import LOG


def _analyze_audio_spectrum(
    audio_data: NDArray[np.float32], sample_rate: int
) -> tuple[bool, float, str | None]:
    import scipy.signal

    channels = audio_data[:, None] if audio_data.ndim == 1 else audio_data
    spectra = [
        scipy.signal.spectrogram(
            channels[:, channel], fs=sample_rate, nperseg=2048, noverlap=1024
        )
        for channel in range(channels.shape[1])
    ]
    frequency_axis = spectra[0][0]
    spectrogram_matrix = np.mean([spectrum[2] for spectrum in spectra], axis=0)

    # 95th percentile power across time slices captures high-frequency transient peaks (hi-hats, vocal air)
    peak_spectrum_power = np.percentile(spectrogram_matrix, 95, axis=1)
    max_power = float(np.max(peak_spectrum_power))
    if max_power <= 1e-12:
        return False, 0.0, None

    peak_spectrum_dbfs = 10.0 * np.log10(
        np.maximum(peak_spectrum_power, 1e-12) / max_power
    )
    max_transient_power = np.max(spectrogram_matrix, axis=1)
    max_transient_dbfs = 10.0 * np.log10(
        np.maximum(max_transient_power, 1e-12) / max_power
    )

    # Maximum cutoff search frequency:
    # At 44.1 kHz, standard CD anti-aliasing reconstruction filters roll off between 20.0 and 22.05 kHz.
    # Evaluating target frequencies >= 18.75 kHz with a 2.5 kHz lookahead measures into the reconstruction
    # roll-off, falsely flagging authentic CD masters. Cap search at 18.5 kHz for <= 44.1 kHz.
    max_target = 18750 if sample_rate <= 44100 else 19250

    for target_frequency in np.arange(14500, max_target, 250):
        frequency_index = int(np.argmin(np.abs(frequency_axis - target_frequency)))
        transition_index = int(
            np.argmin(np.abs(frequency_axis - (target_frequency + 1000)))
        )
        stopband_end_index = int(
            np.argmin(np.abs(frequency_axis - (target_frequency + 2500)))
        )

        passband_level = float(peak_spectrum_dbfs[frequency_index])
        transition_level = float(peak_spectrum_dbfs[transition_index])
        stopband_slice = peak_spectrum_dbfs[transition_index : stopband_end_index + 1]
        stopband_peak = (
            float(np.max(stopband_slice)) if len(stopband_slice) > 0 else -100.0
        )

        cliff_drop = passband_level - transition_level

        # In a genuine lossy transcode (MP3/AAC lowpass), all frequencies above cutoff
        # remain suppressed down to the noise floor until Nyquist.
        # If significant energy (>= -65 dBFS) exists higher up, the drop is a musical notch
        # or instrumentation profile, not a brickwall encoder cutoff.
        high_band_slice = peak_spectrum_dbfs[transition_index:]
        high_band_max = (
            float(np.max(high_band_slice)) if len(high_band_slice) > 0 else -100.0
        )
        if high_band_max >= -65.0:
            continue

        stopband_transient_slice = max_transient_dbfs[stopband_end_index:]
        high_transient_max = (
            float(np.max(stopband_transient_slice))
            if len(stopband_transient_slice) > 0
            else -100.0
        )
        if high_transient_max >= -60.0:
            continue

        # Brickwall criterion: audible signal before cutoff (> -73 dBFS), steep cliff drop (>= 18 dB/kHz),
        # complete stopband attenuation (< -75 dBFS), and persistent high-band suppression.
        if (
            passband_level > -73.0
            and cliff_drop >= 18.0
            and stopband_peak < -75.0
            and high_band_max < -68.0
        ):
            detected_cutoff_khz = float(target_frequency / 1000.0)
            description = (
                f"Sharp spectral cutoff near {detected_cutoff_khz:.1f} kHz; "
                "possible lossy source or mastering filter. Source quality requires review."
            )
            return True, detected_cutoff_khz, description

    return False, 0.0, None


def detect_fake_lossless(file_path: Path) -> tuple[bool, float, str | None]:
    """
    Returns (has_suspicious_cutoff, detected_cutoff_khz, description).
    A cutoff alone cannot establish the source codec or prove audio corruption.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        sample_rate: int = 44100
        audio_data: NDArray[np.float32] | None = None
        duration_seconds: float = 0.0

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
                raw_samples = cast(
                    NDArray[np.float32],
                    audio_file.read(
                        frames=read_frames, dtype="float32", always_2d=True
                    ),
                )
                audio_data = raw_samples
        except (soundfile.LibsndfileError, OSError):
            audio_data = None

        if audio_data is None:
            loaded_audio = load_audio(file_path, max_seconds=20.0, offset_seconds=30.0)
            if loaded_audio is not None and len(loaded_audio[0]) < loaded_audio[1] * 5:
                loaded_audio = load_audio(
                    file_path, max_seconds=20.0, offset_seconds=0.0
                )
            if loaded_audio is None:
                return False, 0.0, None
            audio_data, sample_rate = loaded_audio

        if sample_rate < 32000 or len(audio_data) < sample_rate * 5:
            return False, 0.0, None

        is_fake, detected_cutoff, description = _analyze_audio_spectrum(
            audio_data, sample_rate
        )

        # If a cutoff was detected in the first window and the track is long enough,
        # sample a second window (65% in) to avoid false alarms on acoustic intros or breakdowns.
        if is_fake and duration_seconds > 60.0:
            second_offset = duration_seconds * 0.65
            try:
                with soundfile.SoundFile(str(file_path)) as audio_file:
                    audio_file.seek(int(sample_rate * second_offset))
                    second_samples = cast(
                        NDArray[np.float32],
                        audio_file.read(
                            frames=int(sample_rate * 20.0),
                            dtype="float32",
                            always_2d=True,
                        ),
                    )
                    if len(second_samples) >= sample_rate * 5:
                        is_second_fake, second_cutoff, _ = _analyze_audio_spectrum(
                            second_samples, sample_rate
                        )
                        # If the second window has authentic full bandwidth, suppress the false alarm
                        if not is_second_fake:
                            return False, 0.0, None
                        return True, max(detected_cutoff, second_cutoff), description
            except (soundfile.LibsndfileError, OSError):
                pass

        return is_fake, detected_cutoff, description

    except OSError as error:
        LOG.debug(f"Spectral analysis skipped for {file_path}: {error}")
        return False, 0.0, None
