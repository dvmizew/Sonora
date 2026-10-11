import dataclasses
import subprocess
from unittest.mock import Mock, patch

import numpy as np
import soundfile

from sonora.audio import bpm, checksum, key, replaygain, spectral
from sonora.core.config import SonoraConfig
from sonora.core.models import TrackInfo
from tests.support import IsolatedTestCase


class TestTempoContracts(IsolatedTestCase):
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


class TestMusicalKeyContracts(IsolatedTestCase):
    def test_unsupported_enharmonic_spelling_does_not_invent_a_camelot_code(
        self,
    ) -> None:
        self.assertIsNone(key.key_to_camelot("B#"))
        self.assertEqual(key.key_to_camelot("B# (8A)"), "8A")

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

    def test_decoder_failure_and_short_track_fallback(self) -> None:
        path = self.test_directory / "recording.wav"
        path.touch()
        with self.assertRaises(FileNotFoundError):
            key.detect_key_details(path.with_name("missing.wav"))
        for first_read in (None, (np.zeros(100), 22050)):
            with (
                self.subTest(first_read=first_read),
                patch.object(
                    key, "load_audio", side_effect=[first_read, None]
                ) as decode,
            ):
                self.assertIsNone(key.detect_key_details(path))
            self.assertEqual(decode.call_count, 2)
            self.assertEqual(decode.call_args.kwargs["offset_seconds"], 0)
        with patch.object(key, "load_audio", return_value=(np.zeros(5000), 22050)):
            self.assertIsNone(key.detect_key_details(path))
            self.assertIsNone(key.detect_musical_key(path))
        with patch.object(key, "load_audio", side_effect=OSError("decode failed")):
            self.assertIsNone(key.detect_key_details(path))


class TestSpectralContracts(IsolatedTestCase):
    def test_long_stereo_silence_does_not_resemble_a_transcode(self) -> None:
        path = self.test_directory / "silence.wav"
        soundfile.write(path, np.zeros((32000 * 61, 2), dtype=np.float32), 32000)
        self.assertEqual(spectral.detect_fake_lossless(path), (False, 0, None))

    def test_decoding_fallback_and_low_sample_rate_skip_analysis(self) -> None:
        path = self.test_directory / "unsupported.m4a"
        path.touch()
        for decoded in (None, (np.zeros(22050 * 5), 22050), (np.zeros(44100), 44100)):
            with (
                self.subTest(decoded=decoded),
                patch("soundfile.SoundFile", side_effect=OSError("unsupported format")),
                patch.object(spectral, "load_audio", return_value=decoded) as decode,
            ):
                self.assertEqual(spectral.detect_fake_lossless(path), (False, 0, None))
            self.assertEqual(decode.call_args.kwargs["offset_seconds"], 30)
        with (
            patch("soundfile.SoundFile", side_effect=OSError("unsupported format")),
            patch.object(spectral, "load_audio", side_effect=OSError("decoder failed")),
        ):
            self.assertEqual(spectral.detect_fake_lossless(path), (False, 0, None))

    def test_encoder_profile_boundaries_use_spectral_cutoff(self) -> None:
        path = self.test_directory / "spectral.wav"
        path.touch()
        frequencies = np.arange(0, 22051, 250)
        for cutoff, profile in (
            (16500, "128kbps MP3/AAC"),
            (17500, "160kbps MP3"),
            (18500, "192kbps MP3"),
        ):
            power = np.full((len(frequencies), 10), 1e-9)
            power[np.flatnonzero(frequencies == cutoff)[0], :] = 1
            with (
                self.subTest(cutoff=cutoff),
                patch("soundfile.SoundFile", side_effect=OSError("unsupported format")),
                patch.object(
                    spectral, "load_audio", return_value=(np.ones(44100 * 5), 44100)
                ),
                patch(
                    "scipy.signal.spectrogram", return_value=(frequencies, None, power)
                ),
            ):
                fake, detected, description = spectral.detect_fake_lossless(path)
            self.assertTrue(fake)
            self.assertEqual(detected, cutoff / 1000)
            assert description is not None
            self.assertIn(profile, description)


class TestReplayGainFailureContracts(IsolatedTestCase):
    def test_measurement_programming_errors_are_not_silenced(self) -> None:
        path = self.test_directory / "recording.flac"
        path.touch()
        with (
            patch.object(
                replaygain,
                "_measure_track_loudness",
                side_effect=RuntimeError("measurement bug"),
            ),
            self.assertRaisesRegex(RuntimeError, "measurement bug"),
        ):
            replaygain.calculate_album_replaygain([path], force=True)

    def setUp(self) -> None:
        self.path = self.test_directory / "recording.flac"
        self.path.touch()
        self.measurement = replaygain.LoudnessMeasurement(
            self.path, 2, 0.5, -20, 10, (-20,)
        )

    def test_album_gate_uses_block_energy_and_remains_stable_at_extreme_levels(
        self,
    ) -> None:
        for offset in (0.0, 4000.0):
            measurement = dataclasses.replace(
                self.measurement,
                block_loudness=tuple(offset + level for level in (-20, -20, -50)),
            )
            with self.subTest(offset=offset):
                album_loudness = replaygain._album_loudness([measurement])
                assert album_loudness is not None
                self.assertAlmostEqual(album_loudness, offset - 20)

    def test_replaygain_writes_rounded_measurements_and_supports_previews(self) -> None:
        recording = TrackInfo(self.path)
        with (
            patch.object(replaygain, "read_track_metadata", return_value=recording),
            patch.object(replaygain, "write_track_metadata") as write,
        ):
            self.assertTrue(
                replaygain._write_album_replaygain_track(
                    self.measurement, 1.234, 0.12345678
                )
            )
            self.assertTrue(
                replaygain._write_album_replaygain_track(
                    self.measurement, 1.234, 0.12345678, dry_run=True
                )
            )
        write.assert_called_once_with(recording)
        self.assertEqual(
            (
                recording.replaygain_track_gain,
                recording.replaygain_track_peak,
                recording.replaygain_album_gain,
                recording.replaygain_album_peak,
            ),
            (2, 0.5, 1.23, 0.123457),
        )

    def test_measurement_failures_and_ungated_silence(self) -> None:
        self.assertIsNone(replaygain._album_loudness([]))
        self.assertIsNone(
            replaygain._album_loudness(
                [
                    dataclasses.replace(
                        self.measurement, block_loudness=(-80, float("nan"))
                    )
                ]
            )
        )
        for decoded in (None, (np.zeros(0), 44100), (np.ones(100), 0)):
            with (
                self.subTest(decoded=decoded),
                patch.object(replaygain, "load_audio", return_value=decoded),
            ):
                self.assertIsNone(replaygain._measure_track_loudness(self.path))

    def test_write_failure_is_reported_as_unsuccessful(self) -> None:
        with patch.object(
            replaygain, "read_track_metadata", side_effect=OSError("locked file")
        ):
            self.assertFalse(
                replaygain._write_album_replaygain_track(self.measurement, 2, 0.5)
            )

    def test_all_tagged_files_skip_measurement(self) -> None:
        tagged = TrackInfo(self.path, replaygain_track_gain=1, replaygain_album_gain=1)
        with (
            patch.object(replaygain, "read_track_metadata", return_value=tagged),
            patch.object(replaygain, "_measure_track_loudness") as measure,
        ):
            self.assertFalse(replaygain.calculate_album_replaygain([self.path]))
        measure.assert_not_called()

    def test_unreadable_tags_and_failed_measurements_produce_no_writes(self) -> None:
        for failure in (None, OSError("decoder failed")):
            with (
                self.subTest(failure=failure),
                patch.object(
                    replaygain,
                    "read_track_metadata",
                    side_effect=OSError("unreadable tags"),
                ),
                patch.object(
                    replaygain,
                    "_measure_track_loudness",
                    return_value=None,
                    side_effect=failure,
                ),
                patch.object(replaygain, "write_track_metadata") as write,
            ):
                self.assertFalse(replaygain.calculate_album_replaygain([self.path]))
            write.assert_not_called()

    def test_write_interruption_cancels_pending_work(self) -> None:
        executor = Mock()
        executor.__enter__ = Mock(return_value=executor)
        executor.__exit__ = Mock(return_value=False)
        completed = Mock()
        completed.result.return_value = self.measurement
        executor.submit.return_value = completed
        interrupted = RuntimeError("writer interrupted")
        interrupted.__cause__ = KeyboardInterrupt()
        for failure, cancelled in (
            (KeyboardInterrupt(), True),
            (interrupted, True),
            (RuntimeError("writer bug"), False),
        ):
            executor.reset_mock()
            executor.map.side_effect = failure
            with (
                self.subTest(failure=failure),
                patch.object(replaygain, "ThreadPoolExecutor", return_value=executor),
                self.assertRaises(type(failure)),
            ):
                replaygain.calculate_album_replaygain([self.path], force=True)
            if cancelled:
                executor.shutdown.assert_called_once_with(
                    wait=True, cancel_futures=True
                )
            else:
                executor.shutdown.assert_not_called()


class TestChecksumFailureContracts(IsolatedTestCase):
    def test_verification_uses_process_exit_status(self) -> None:
        path = self.test_directory / "recording.flac"
        path.touch()
        for status in (0, 1):
            with (
                self.subTest(status=status),
                patch(
                    "subprocess.run",
                    return_value=subprocess.CompletedProcess([], status),
                ) as execute,
            ):
                self.assertEqual(checksum.verify_flac_checksum(path), status == 0)
            self.assertEqual(execute.call_args.kwargs["timeout"], 60)
            self.assertFalse(execute.call_args.kwargs["check"])

    def test_missing_executable_and_process_failures_preserve_io_error(self) -> None:
        path = self.test_directory / "recording.flac"
        path.touch()
        failures = (
            (FileNotFoundError("missing flac"), "binary not found"),
            (subprocess.TimeoutExpired("flac", 60), "Checksum verification failed"),
        )
        for failure, message in failures:
            with (
                self.subTest(failure=failure),
                patch("subprocess.run", side_effect=failure),
                self.assertRaisesRegex(OSError, message) as raised,
            ):
                checksum.verify_flac_checksum(path)
            self.assertIs(raised.exception.__cause__, failure)
