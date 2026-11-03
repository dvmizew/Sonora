import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile

from sonora.audio.cuesheet import find_companion_cuesheet, read_cuesheet_content
from sonora.audio.decode import load_audio
from tests.support import IsolatedTestCase


class TestCueSheetContracts(IsolatedTestCase):
    def test_missing_unreadable_and_absent_cuesheets(self) -> None:
        cue_path = self.test_directory / "album.cue"
        self.assertIsNone(read_cuesheet_content(cue_path))
        self.assertIsNone(find_companion_cuesheet(cue_path))
        self.assertIsNone(find_companion_cuesheet(self.test_directory))
        cue_path.write_text('TITLE "Album"')
        self.assertEqual(find_companion_cuesheet(self.test_directory), cue_path)
        with patch.object(
            Path, "read_bytes", side_effect=PermissionError("locked cue")
        ):
            self.assertIsNone(read_cuesheet_content(cue_path))

    def test_cuesheet_encodings_preserve_metadata(self) -> None:
        for encoded, expected in (
            (b'\xef\xbb\xbfTITLE "Caf\xc3\xa9"', 'TITLE "Café"'),
            (b'TITLE "Prix \x80"', 'TITLE "Prix €"'),
            (b'TITLE "\x81"', 'TITLE "\x81"'),
        ):
            cue_path = self.test_directory / "album.cue"
            cue_path.write_bytes(encoded)
            with self.subTest(encoded=encoded):
                self.assertEqual(read_cuesheet_content(cue_path), expected)


class TestAudioDecodingContracts(IsolatedTestCase):
    def test_seek_near_or_beyond_end_never_restarts_the_track(self) -> None:
        path = self.test_directory / "short.wav"
        signal = np.arange(2000, dtype=np.float32) / 2000
        soundfile.write(path, signal, 1000, subtype="FLOAT")
        for offset, expected in (
            (1.5, signal[1500:]),
            (2, signal[:0]),
            (3, signal[:0]),
        ):
            with self.subTest(offset=offset):
                loaded = load_audio(path, mono=True, offset_seconds=offset)
                assert loaded is not None
                samples, _ = loaded
                np.testing.assert_array_equal(samples, expected)

    def test_real_seek_and_duration_limit(self) -> None:
        audio_path = self.test_directory / "album.wav"
        signal = np.zeros(5000, dtype=np.float32)
        signal[1000:2000] = 0.5
        soundfile.write(audio_path, signal, 1000, subtype="FLOAT")
        loaded = load_audio(audio_path, mono=True, offset_seconds=1, max_seconds=1)
        assert loaded is not None
        audio, rate = loaded
        self.assertEqual(rate, 1000)
        self.assertEqual(audio.shape, (1000,))
        np.testing.assert_array_equal(audio, np.full(1000, 0.5))

    def test_ffmpeg_fallback_respects_channels_seek_and_duration(self) -> None:
        audio_path = self.test_directory / "track.m4a"
        raw_audio = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        for mono, expected_shape in ((True, (4,)), (False, (2, 2))):
            completed = subprocess.CompletedProcess([], 0, stdout=raw_audio.tobytes())
            with (
                self.subTest(mono=mono),
                patch("soundfile.SoundFile", side_effect=OSError("unsupported format")),
                patch("subprocess.run", return_value=completed) as run,
            ):
                loaded = load_audio(
                    audio_path, mono=mono, offset_seconds=2, max_seconds=5
                )
            assert loaded is not None
            audio, rate = loaded
            self.assertEqual(rate, 44100)
            self.assertEqual(audio.shape, expected_shape)
            np.testing.assert_array_equal(audio.ravel(), raw_audio)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("-ss") + 1], "2")
            self.assertEqual(command[command.index("-t") + 1], "5")
            self.assertEqual(command[command.index("-ac") + 1], "1" if mono else "2")

    def test_ffmpeg_failure_and_empty_output_return_no_audio(self) -> None:
        for outcome in (
            subprocess.CompletedProcess([], 0, stdout=b""),
            subprocess.TimeoutExpired("ffmpeg", 15),
        ):
            with (
                self.subTest(outcome=outcome),
                patch("soundfile.SoundFile", side_effect=OSError("decode failed")),
                patch(
                    "subprocess.run",
                    side_effect=outcome if isinstance(outcome, Exception) else None,
                    return_value=outcome,
                ),
            ):
                self.assertIsNone(load_audio(self.test_directory / "damaged.wav"))
