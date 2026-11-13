"""
Unit and integration tests for Sonora DSP audio engine modules:
BPM detection, musical key / Camelot wheel tonality, ReplayGain loudness, and spectral analysis.
"""

import dataclasses
import wave
from pathlib import Path
from unittest.mock import Mock, call, patch

import numpy as np
import soundfile

from sonora.audio import bpm, key, replaygain, spectral
from sonora.audio.key import (
    detect_key_details,
    detect_musical_key,
    key_to_camelot,
)
from sonora.audio.metadata import read_track_metadata
from sonora.audio.replaygain import calculate_album_replaygain
from sonora.audio.spectral import detect_fake_lossless
from sonora.core.config import SonoraConfig
from tests.support import IsolatedTestCase


def create_dummy_wav_file(dest_path: Path) -> Path:
    sample_rate = 44100
    time_axis = np.linspace(0, 1, sample_rate, endpoint=False)
    samples = (np.sin(2 * np.pi * 440 * time_axis) * 16000).astype(np.int16)
    stereo = np.column_stack([samples, samples]).flatten()
    with wave.open(str(dest_path), "wb") as wave_file:
        wave_file.setnchannels(2)
        wave_file.setsampwidth(2)
        wave_file.setframerate(sample_rate)
        wave_file.writeframes(stereo.tobytes())
    return dest_path


def create_chord_wav(
    dest_path: Path,
    freqs: list[float],
    duration: float = 2.0,
    sample_rate: int = 22050,
) -> Path:
    time_axis = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.zeros_like(time_axis)
    for f in freqs:
        signal += np.sin(2 * np.pi * f * time_axis)
    signal = signal / (len(freqs) * 1.2)
    samples = (signal * 32767).astype(np.int16)
    stereo = np.column_stack([samples, samples]).flatten()

    with wave.open(str(dest_path), "wb") as wave_file:
        wave_file.setnchannels(2)
        wave_file.setsampwidth(2)
        wave_file.setframerate(sample_rate)
        wave_file.writeframes(stereo.tobytes())
    return dest_path


class TestTempoDetection(IsolatedTestCase):
    def setUp(self) -> None:
        self.audio_path = self.test_directory / "recording.wav"
        self.audio_path.touch()

    def test_missing_or_unusable_audio_has_no_tempo(self) -> None:
        with self.assertRaises(FileNotFoundError):
            bpm.calculate_bpm(self.audio_path.with_name("absent.wav"))
        for decoded in (
            None,
            (np.zeros(0), 22050),
            (np.ones(100), 22050),
            (np.zeros(22050), 22050),
        ):
            with (
                self.subTest(decoded=decoded),
                patch.object(bpm, "load_audio", return_value=decoded),
            ):
                self.assertIsNone(bpm.calculate_bpm(self.audio_path))
        with patch.object(bpm, "load_audio", side_effect=OSError("decoder failed")):
            self.assertIsNone(bpm.calculate_bpm(self.audio_path))

    def test_insufficient_onset_window_has_no_tempo(self) -> None:
        for onset_frames, sample_rate in ((1, 22050), (3, 22050), (100, 1)):
            with (
                self.subTest(onset_frames=onset_frames, sample_rate=sample_rate),
                patch.object(
                    bpm, "load_audio", return_value=(np.ones(22050), sample_rate)
                ),
                patch(
                    "scipy.signal.spectrogram",
                    return_value=(
                        None,
                        None,
                        np.tile(np.arange(onset_frames), (10, 1)),
                    ),
                ),
            ):
                self.assertIsNone(bpm.calculate_bpm(self.audio_path))

    def test_shared_audio_keeps_the_existing_tempo_window(self) -> None:
        samples = np.ones(22050 * 70, dtype=np.float32)
        correlation = np.zeros(500)
        correlation[100] = 1
        with (
            patch.object(bpm, "load_audio") as load_audio,
            patch(
                "scipy.signal.spectrogram",
                return_value=(None, None, np.tile(np.arange(251), (10, 1))),
            ) as spectrogram,
            patch("scipy.signal.correlate", return_value=correlation),
        ):
            bpm.calculate_bpm(self.audio_path, audio_data=(samples, 22050))
        load_audio.assert_not_called()
        self.assertEqual(len(spectrogram.call_args.args[0]), 22050 * 60)

    def test_octave_policy_can_fold_or_preserve_detected_tempo(self) -> None:
        for lag, minimum, maximum, expected in (
            (100, 80, 160, 103.4),
            (30, 80, 160, 86.1),
            (100, None, None, 51.7),
        ):
            correlation = np.zeros(501)
            correlation[250 + lag] = 1
            policy = dataclasses.replace(
                SonoraConfig(),
                bpm_search_min=30,
                bpm_octave_min=minimum,
                bpm_octave_max=maximum,
            )
            with (
                self.subTest(lag=lag, minimum=minimum),
                patch.object(bpm, "get_config", return_value=policy),
                patch.object(bpm, "load_audio", return_value=(np.ones(22050), 22050)),
                patch(
                    "scipy.signal.spectrogram",
                    return_value=(None, None, np.tile(np.arange(251), (10, 1))),
                ),
                patch("scipy.signal.correlate", return_value=correlation),
            ):
                self.assertEqual(bpm.calculate_bpm(self.audio_path), expected)


class TestMusicalKeyDetection(IsolatedTestCase):
    def test_key_to_camelot_major_keys(self) -> None:
        mappings = {
            "C": "8B",
            "C#": "3B",
            "Db": "3B",
            "D": "10B",
            "D#": "5B",
            "Eb": "5B",
            "E": "12B",
            "F": "7B",
            "F#": "2B",
            "Gb": "2B",
            "G": "9B",
            "G#": "4B",
            "Ab": "4B",
            "A": "11B",
            "A#": "6B",
            "Bb": "6B",
            "B": "1B",
        }
        for k, c in mappings.items():
            self.assertEqual(key_to_camelot(k), c)

    def test_key_to_camelot_minor_keys(self) -> None:
        mappings = {
            "Am": "8A",
            "A#m": "3A",
            "Bbm": "3A",
            "Bm": "10A",
            "Cm": "5A",
            "C#m": "12A",
            "Dbm": "12A",
            "Dm": "7A",
            "D#m": "2A",
            "Ebm": "2A",
            "Em": "9A",
            "Fm": "4A",
            "F#m": "11A",
            "Gbm": "11A",
            "Gm": "6A",
            "G#m": "1A",
            "Abm": "1A",
        }
        for k, c in mappings.items():
            self.assertEqual(key_to_camelot(k), c)

    def test_key_to_camelot_extended_formats(self) -> None:
        self.assertEqual(key_to_camelot("C# minor"), "12A")
        self.assertEqual(key_to_camelot("A minor"), "8A")
        self.assertEqual(key_to_camelot("F# min"), "11A")
        self.assertEqual(key_to_camelot("Bb Major"), "6B")
        self.assertEqual(key_to_camelot("Am (8A)"), "8A")
        self.assertEqual(key_to_camelot("8A - Am"), "8A")
        self.assertEqual(key_to_camelot("D♯"), "5B")
        self.assertEqual(key_to_camelot("B♭m"), "3A")
        self.assertEqual(key_to_camelot("8A"), "8A")
        self.assertEqual(key_to_camelot("12b"), "12B")
        self.assertIsNone(key_to_camelot("B#"))
        self.assertEqual(key_to_camelot("B# (8A)"), "8A")
        self.assertIsNone(key_to_camelot(None))

    def test_chroma_resampling_preserves_reference_pitch_and_unit_norm(self) -> None:
        sample_rate = 44100
        time_axis = np.arange(sample_rate * 2) / sample_rate
        reference_tone = np.sin(2 * np.pi * 440 * time_axis).astype(np.float32)
        chroma = key.calculate_chroma(reference_tone, sample_rate)
        assert chroma is not None
        self.assertEqual(chroma.shape, (12,))
        self.assertEqual(int(np.argmax(chroma)), 9)
        self.assertAlmostEqual(float(np.linalg.norm(chroma)), 1, places=6)

    def test_silent_short_and_constant_audio_has_no_chroma(self) -> None:
        for samples in (np.zeros(0), np.zeros(5000), np.ones(100), np.ones(5000)):
            with self.subTest(sample_count=len(samples)):
                self.assertIsNone(key.calculate_chroma(samples, 22050))
        self.assertEqual(
            key._pearson_r(np.ones(12), np.arange(12, dtype=np.float64)), 0
        )
        self.assertIsNone(key.detect_key_from_chroma(np.ones(11)))
        self.assertIsNone(key.detect_key_from_chroma(np.ones(12)))

    def test_detect_c_major_chord(self) -> None:
        c_major_wav = self.test_directory / "c_major.wav"
        create_chord_wav(c_major_wav, [261.63, 329.63, 392.00])
        details = detect_key_details(c_major_wav)
        assert details is not None
        key_name, camelot, conf = details
        self.assertEqual(key_name, "C")
        self.assertEqual(camelot, "8B")
        self.assertGreater(conf, 0.5)

        detected_key = detect_musical_key(c_major_wav)
        self.assertEqual(detected_key, "C")

    def test_detect_a_minor_chord(self) -> None:
        a_minor_wav = self.test_directory / "a_minor.wav"
        create_chord_wav(a_minor_wav, [220.00, 261.63, 329.63])
        details = detect_key_details(a_minor_wav)
        assert details is not None
        key_name, camelot, conf = details
        self.assertEqual(key_name, "Am")
        self.assertEqual(camelot, "8A")
        self.assertGreater(conf, 0.5)

        detected_key = detect_musical_key(a_minor_wav)
        self.assertEqual(detected_key, "Am")


class TestReplayGain(IsolatedTestCase):
    def setUp(self) -> None:
        self.path = self.test_directory / "recording.flac"
        self.path.touch()
        self.measurement = replaygain.LoudnessMeasurement(self.path, 2, 0.5, (-20,))

    def test_calculate_album_replaygain_success(self) -> None:
        track1_path = self.test_directory / "1.wav"
        track2_path = self.test_directory / "2.wav"
        create_dummy_wav_file(track1_path)
        create_dummy_wav_file(track2_path)

        result = calculate_album_replaygain([track1_path, track2_path], force=True)
        self.assertTrue(result)

        m1 = read_track_metadata(track1_path)
        m2 = read_track_metadata(track2_path)
        self.assertIsInstance(m1.replaygain_track_gain, float)
        self.assertIsInstance(m1.replaygain_album_gain, float)
        self.assertEqual(m1.replaygain_album_gain, m2.replaygain_album_gain)
        self.assertEqual(m1.replaygain_album_peak, m2.replaygain_album_peak)

    def test_calculate_album_replaygain_corrupted_or_empty(self) -> None:
        self.assertFalse(calculate_album_replaygain([]))
        fail_wav = self.test_directory / "corrupt.wav"
        fail_wav.write_bytes(b"not an audio file")
        self.assertFalse(calculate_album_replaygain([fail_wav]))

    def test_measurement_programming_errors_are_not_silenced(self) -> None:
        with (
            patch.object(
                replaygain,
                "_measure_track_loudness",
                side_effect=RuntimeError("measurement bug"),
            ),
            self.assertRaisesRegex(RuntimeError, "measurement bug"),
        ):
            replaygain.calculate_album_replaygain([self.path], force=True)

    def test_measurement_and_write_pools_share_the_worker_limit(self) -> None:
        executor = Mock()
        executor.__enter__ = Mock(return_value=executor)
        executor.__exit__ = Mock(return_value=False)
        completed = Mock()
        completed.result.return_value = self.measurement
        executor.submit.return_value = completed
        executor.map.return_value = [True]
        with (
            patch.object(
                replaygain, "ThreadPoolExecutor", return_value=executor
            ) as executor_factory,
            patch.object(
                replaygain, "_write_album_replaygain_track", return_value=True
            ),
        ):
            replaygain.calculate_album_replaygain(
                [self.path], force=True, max_threads=100
            )

        self.assertEqual(
            executor_factory.call_args_list,
            [call(max_workers=4), call(max_workers=4)],
        )


class TestSpectralAnalysis(IsolatedTestCase):
    def test_detect_fake_lossless_authentic_audio(self) -> None:
        authentic_path = self.test_directory / "authentic.wav"
        rng = np.random.default_rng(42)
        samples = rng.normal(0.0, 0.15, int(44100 * 25.0)).astype(np.float32)
        soundfile.write(str(authentic_path), samples, 44100)

        is_fake, cutoff_khz, description = detect_fake_lossless(authentic_path)
        self.assertFalse(is_fake)
        self.assertEqual(cutoff_khz, 0.0)
        self.assertIsNone(description)

    def test_detect_fake_lossless_brickwall_transcode(self) -> None:
        fake_path = self.test_directory / "transcode_128k.wav"
        rng = np.random.default_rng(42)
        time_axis = np.linspace(0, 25.0, int(44100 * 25.0), endpoint=False)
        audio_signal = (
            np.sin(2 * np.pi * 100 * time_axis) * 0.5
            + np.sin(2 * np.pi * 1000 * time_axis) * 0.2
            + rng.normal(0.0, 0.05, len(time_axis))
        )
        fft_coefficients = np.fft.rfft(audio_signal)
        fft_frequencies = np.fft.rfftfreq(len(audio_signal), d=1 / 44100)
        fft_coefficients[fft_frequencies > 16000] = 1e-6
        brickwall_audio = np.fft.irfft(fft_coefficients, n=len(audio_signal)).astype(
            np.float32
        )
        soundfile.write(str(fake_path), brickwall_audio, 44100)

        is_fake, cutoff_khz, description = detect_fake_lossless(fake_path)
        self.assertTrue(is_fake)
        self.assertGreaterEqual(cutoff_khz, 14.5)
        self.assertLessEqual(cutoff_khz, 16.5)
        assert description is not None
        self.assertIn("Source quality requires review", description)
        self.assertNotIn("kbps", description)

    def test_antiphase_stereo_does_not_hide_a_brickwall_cutoff(self) -> None:
        path = self.test_directory / "antiphase.wav"
        sample_rate = 44100
        time_axis = np.arange(sample_rate * 25) / sample_rate
        rng = np.random.default_rng(42)
        samples = (
            0.5 * np.sin(2 * np.pi * 100 * time_axis)
            + 0.2 * np.sin(2 * np.pi * 1000 * time_axis)
            + rng.normal(0.0, 0.05, len(time_axis))
        )
        coefficients = np.fft.rfft(samples)
        frequencies = np.fft.rfftfreq(len(samples), 1 / sample_rate)
        coefficients[frequencies > 16000] = 1e-6
        mono = np.fft.irfft(coefficients, n=len(samples)).astype(np.float32)
        soundfile.write(path, np.column_stack((mono, -mono)), sample_rate)

        fake, cutoff, _ = spectral.detect_fake_lossless(path)
        self.assertTrue(fake)
        self.assertLessEqual(cutoff, 16.5)

    def test_long_stereo_silence_does_not_resemble_a_transcode(self) -> None:
        path = self.test_directory / "silence.wav"
        soundfile.write(path, np.zeros((32000 * 61, 2), dtype=np.float32), 32000)
        self.assertEqual(spectral.detect_fake_lossless(path), (False, 0, None))

    def test_detect_fake_lossless_short_audio(self) -> None:
        short_path = self.test_directory / "short.wav"
        rng = np.random.default_rng(42)
        samples = rng.normal(0.0, 0.15, int(44100 * 1.0)).astype(np.float32)
        soundfile.write(str(short_path), samples, 44100)

        is_fake, cutoff_khz, description = detect_fake_lossless(short_path)
        self.assertFalse(is_fake)
        self.assertEqual(cutoff_khz, 0.0)
        self.assertIsNone(description)

    def test_detect_fake_lossless_nonexistent_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            detect_fake_lossless(self.test_directory / "nonexistent.flac")
