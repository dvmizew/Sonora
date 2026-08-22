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
from sonora.audio.metadata import (
    read_track_metadata,
    write_track_metadata,
)
from sonora.audio.replaygain import calculate_album_replaygain
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

    @patch("taglib.File")
    def test_write_flac_audio_metadata(self, mock_taglib_cls):
        mock_file_instance = MagicMock()
        mock_file_instance.tags = {}
        mock_taglib_cls.return_value.__enter__.return_value = mock_file_instance

        flac_path = Path("/tmp/test_track.flac")
        with patch.object(Path, "exists", return_value=True):
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
        bogus_path = Path("/tmp/nonexistent_audio_track_9999.flac")
        with self.assertRaises(FileNotFoundError):
            read_track_metadata(bogus_path)

    def test_verify_nonexistent_file_raises_audio_error(self):
        bogus_path = Path("/tmp/nonexistent_audio_track_9999.flac")
        with self.assertRaises(FileNotFoundError):
            verify_flac_checksum(bogus_path)

    def test_verify_non_flac_returns_true(self):
        self.assertTrue(verify_flac_checksum(self.dummy_audio_path))

    @patch("subprocess.run")
    @patch("sonora.audio.replaygain.FLAC")
    def test_calculate_album_replaygain_success(self, mock_flac, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        mock_flac_instance = MagicMock()
        mock_flac_instance.info.sample_rate = 44100
        mock_flac_instance.info.channels = 2
        mock_flac_instance.info.bits_per_sample = 16
        # Simulate REPLAYGAIN_ALBUM_GAIN not being present
        mock_flac_instance.__contains__.return_value = False
        mock_flac.return_value = mock_flac_instance
        
        with patch.object(Path, "exists", return_value=True), patch("sonora.audio.replaygain.shutil.which", return_value="metaflac"):
            p1 = Path("/tmp/1.flac")
            p2 = Path("/tmp/2.flac")
            result = calculate_album_replaygain([p1, p2])
            
        self.assertTrue(result)
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        self.assertEqual(args[0], "metaflac")
        self.assertEqual(args[1], "--add-replay-gain")
        self.assertIn("/tmp/1.flac", args)
        self.assertIn("/tmp/2.flac", args)

    @patch.dict("sys.modules", {"librosa": MagicMock()})
    @patch("shutil.which", return_value=None)
    def test_calculate_bpm_with_mocked_librosa(self, mock_which):
        import sys
        mock_librosa = sys.modules["librosa"]
        mock_librosa.load.return_value = (None, 44100)
        mock_librosa.beat.beat_track.return_value = (124.5, None)

        bpm = calculate_bpm(self.dummy_audio_path)
        self.assertEqual(bpm, 124.5)

    @patch("taglib.File")
    def test_read_metadata_unsupported_format(self, mock_file):
        mock_file.return_value = None
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            dummy_path = Path(f.name)

        try:
            with self.assertRaises(ValueError):
                read_track_metadata(dummy_path)
        finally:
            if dummy_path.exists():
                dummy_path.unlink()

    @patch("subprocess.run")
    def test_checksum_binary_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        flac_p = Path("/tmp/dummy_flac_check_99.flac")
        with patch.object(Path, "exists", return_value=True), self.assertRaises(RuntimeError):
            verify_flac_checksum(flac_p)

    @patch("subprocess.run")
    @patch("sonora.audio.replaygain.FLAC")
    def test_calculate_album_replaygain_failure(self, mock_flac, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="error")
        mock_flac_instance = MagicMock()
        mock_flac_instance.info.sample_rate = 44100
        mock_flac_instance.info.channels = 2
        mock_flac_instance.info.bits_per_sample = 16
        mock_flac_instance.__contains__.return_value = False
        mock_flac.return_value = mock_flac_instance
        
        with patch.object(Path, "exists", return_value=True), patch("sonora.audio.replaygain.shutil.which", return_value="metaflac"):
            result = calculate_album_replaygain([Path("/tmp/fail.flac")])
        self.assertFalse(result)

if __name__ == "__main__":
    unittest.main()
