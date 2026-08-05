"""
Unit and integration tests for Sonora audio engine modules.
"""

import struct
import sys
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

# Guarantee src/ is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import unittest

from sonora.audio.bpm import calculate_bpm
from sonora.audio.checksum import verify_flac_checksum
from sonora.audio.metadata import read_track_metadata, write_track_metadata
from sonora.audio.replaygain import ReplayGainResult, calculate_replaygain
from sonora.core.exceptions import AudioProcessingError, MetadataError
from sonora.core.models import TrackInfo


def create_dummy_wav_file() -> Path:
    """Create a temporary 1-second WAV audio file with standard audio properties."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    with wave.open(str(temp_path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(struct.pack("<h", 0) * 88200)

    return temp_path


class TestAudioEngine(unittest.TestCase):
    def setUp(self):
        self.dummy_audio_path = create_dummy_wav_file()

    def tearDown(self):
        if self.dummy_audio_path.exists():
            self.dummy_audio_path.unlink()

    def test_read_real_audio_metadata(self):
        track_info = read_track_metadata(self.dummy_audio_path)
        self.assertIsInstance(track_info, TrackInfo)
        self.assertEqual(track_info.sample_rate, 44100)
        self.assertEqual(track_info.channels, 2)

    @patch("sonora.audio.metadata.FLAC")
    def test_write_flac_audio_metadata(self, mock_flac_cls):
        mock_flac_instance = MagicMock()
        mock_flac_cls.return_value = mock_flac_instance

        flac_path = Path("/tmp/test_track.flac")
        with patch.object(Path, "exists", return_value=True):
            track_info = TrackInfo(
                file_path=flac_path,
                artist="Test Artist",
                title="Test Track",
                album="Test Album"
            )
            write_track_metadata(track_info)

        mock_flac_instance.save.assert_called_once()
        mock_flac_instance.__setitem__.assert_any_call("ARTIST", ["Test Artist"])
        mock_flac_instance.__setitem__.assert_any_call("TITLE", ["Test Track"])

    def test_read_nonexistent_file_raises_metadata_error(self):
        bogus_path = Path("/tmp/nonexistent_audio_track_9999.flac")
        with self.assertRaises(MetadataError):
            read_track_metadata(bogus_path)

    def test_verify_nonexistent_file_raises_audio_error(self):
        bogus_path = Path("/tmp/nonexistent_audio_track_9999.flac")
        with self.assertRaises(AudioProcessingError):
            verify_flac_checksum(bogus_path)

    def test_verify_non_flac_returns_true(self):
        self.assertTrue(verify_flac_checksum(self.dummy_audio_path))

    def test_replaygain_result_dataclass(self):
        res = ReplayGainResult(track_gain_db=-3.5, track_peak=0.987)
        self.assertEqual(res.track_gain_db, -3.5)
        self.assertEqual(res.track_peak, 0.987)

    @patch("subprocess.run")
    def test_calculate_replaygain_with_mocked_ffmpeg(self, mock_run):
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stderr = "Integrated loudness: I: -14.5 LUFS\nPeak: -1.2 dBFS"
        mock_run.return_value = mock_process

        res = calculate_replaygain(self.dummy_audio_path)
        self.assertEqual(res.track_gain_db, -3.5)
        self.assertGreater(res.track_peak, 0.0)

    @patch("librosa.beat.beat_track")
    @patch("librosa.load")
    def test_calculate_bpm_with_mocked_librosa(self, mock_load, mock_beat_track):
        mock_load.return_value = (None, 44100)
        mock_beat_track.return_value = (124.5, None)

        bpm = calculate_bpm(self.dummy_audio_path)
        self.assertEqual(bpm, 124.5)


if __name__ == "__main__":
    unittest.main()
