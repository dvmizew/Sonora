"""
Unit and integration tests for Sonora audio engine modules.
"""

import sys
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Guarantee src/ is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import unittest

from sonora.audio.bpm import calculate_bpm
from sonora.audio.checksum import verify_flac_checksum
from sonora.audio.metadata import (
    read_track_metadata,
    write_track_metadata,
)
from sonora.audio.replaygain import (
    calculate_album_replaygain,
    calculate_track_replaygain,
)
from sonora.core.models import TrackInfo


def create_dummy_wav_file(dest_path: Path) -> Path:
    """Create a temporary 1-second WAV audio file with standard audio properties."""
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


class TestAudioEngine(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        self.dummy_audio_path = create_dummy_wav_file(self.tmp_path / "dummy_audio.wav")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_read_real_audio_metadata(self):
        track_info = read_track_metadata(self.dummy_audio_path)
        self.assertIsInstance(track_info, TrackInfo)
        self.assertEqual(track_info.sample_rate, 44100)
        self.assertEqual(track_info.channels, 2)

    @patch("taglib.File")
    def test_write_flac_audio_metadata(self, mock_taglib_cls):
        mock_file_instance = MagicMock()
        mock_file_instance.tags = {}
        mock_taglib_cls.return_value.__enter__.return_value = mock_file_instance

        flac_path = self.tmp_path / "test_track.flac"
        flac_path.write_bytes(b"dummy flac data")

        track_info = TrackInfo(
            file_path=flac_path,
            artist="Test Artist",
            title="Test Track",
            album="Test Album",
            replaygain_track_gain=-4.25,
            replaygain_track_peak=0.951234,
        )
        write_track_metadata(track_info)

        mock_file_instance.save.assert_called_once()
        self.assertEqual(mock_file_instance.tags["ARTIST"], ["Test Artist"])
        self.assertEqual(mock_file_instance.tags["TITLE"], ["Test Track"])
        self.assertEqual(mock_file_instance.tags["REPLAYGAIN_TRACK_GAIN"], ["-4.25 dB"])
        self.assertEqual(mock_file_instance.tags["REPLAYGAIN_TRACK_PEAK"], ["0.951234"])

    def test_read_nonexistent_file_raises_metadata_error(self):
        bogus_path = self.tmp_path / "nonexistent_audio_track_9999.flac"
        with self.assertRaises(FileNotFoundError):
            read_track_metadata(bogus_path)

    def test_verify_nonexistent_file_raises_audio_error(self):
        bogus_path = self.tmp_path / "nonexistent_audio_track_9999.flac"
        with self.assertRaises(FileNotFoundError):
            verify_flac_checksum(bogus_path)

    def test_verify_non_flac_returns_true(self):
        self.assertTrue(verify_flac_checksum(self.dummy_audio_path))

    def test_calculate_track_replaygain_success(self):
        replaygain_result = calculate_track_replaygain(self.dummy_audio_path)
        self.assertIsNotNone(replaygain_result)
        if replaygain_result:
            gain, peak = replaygain_result
            self.assertIsInstance(gain, float)
            self.assertIsInstance(peak, float)
            self.assertTrue(0.0 <= peak <= 1.0)

    def test_calculate_album_replaygain_success(self):
        track1_path = self.tmp_path / "1.wav"
        track2_path = self.tmp_path / "2.wav"
        create_dummy_wav_file(track1_path)
        create_dummy_wav_file(track2_path)

        result = calculate_album_replaygain([track1_path, track2_path], force=True)
        self.assertTrue(result)

        info1 = read_track_metadata(track1_path)
        self.assertIsNotNone(info1.replaygain_track_gain)
        self.assertIsNotNone(info1.replaygain_album_gain)
        self.assertIsNotNone(info1.replaygain_track_peak)
        self.assertIsNotNone(info1.replaygain_album_peak)

    @patch(
        "sonora.audio.bpm.load_audio",
        return_value=(np.random.rand(44100 * 10).astype(np.float32), 44100),
    )
    @patch(
        "scipy.signal.spectrogram",
        return_value=(None, None, np.tile(np.linspace(1, 10, 100), (10, 1))),
    )
    def test_calculate_bpm_with_scipy(self, mock_spec, mock_load):
        bpm = calculate_bpm(self.dummy_audio_path)
        self.assertIsNotNone(bpm)
        self.assertIsInstance(bpm, float)
        if bpm:
            self.assertTrue(40.0 <= bpm <= 220.0)

    @patch("taglib.File")
    def test_read_metadata_unsupported_format(self, mock_file):
        mock_file.return_value = None
        dummy_path = self.tmp_path / "song.xyz"
        dummy_path.write_bytes(b"dummy")

        with self.assertRaises(ValueError):
            read_track_metadata(dummy_path)

    @patch("subprocess.run")
    def test_checksum_binary_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        flac_path = self.tmp_path / "dummy_flac_check_99.flac"
        flac_path.write_bytes(b"dummy")
        with self.assertRaises(RuntimeError):
            verify_flac_checksum(flac_path)

    def test_calculate_album_replaygain_failure_empty(self):
        result = calculate_album_replaygain([])
        self.assertFalse(result)

    def test_calculate_album_replaygain_corrupted_file(self):
        fail_wav = self.tmp_path / "corrupt.wav"
        fail_wav.write_bytes(b"not an audio file")
        result = calculate_album_replaygain([fail_wav])
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
