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

from sonora.audio.art import _find_artist_directory
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
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        self.dummy_audio_path = create_dummy_wav_file(self.tmp_path / "dummy_audio.wav")

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_read_real_audio_metadata(self) -> None:
        track_info = read_track_metadata(self.dummy_audio_path)
        self.assertIsInstance(track_info, TrackInfo)
        self.assertEqual(track_info.sample_rate, 44100)
        self.assertEqual(track_info.channels, 2)

    @patch("taglib.File")
    def test_write_flac_audio_metadata(self, mock_taglib_cls: MagicMock) -> None:
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

    def test_read_nonexistent_file_raises_metadata_error(self) -> None:
        bogus_path = self.tmp_path / "nonexistent_audio_track_9999.flac"
        with self.assertRaises(FileNotFoundError):
            read_track_metadata(bogus_path)

    def test_verify_nonexistent_file_raises_audio_error(self) -> None:
        bogus_path = self.tmp_path / "nonexistent_audio_track_9999.flac"
        with self.assertRaises(FileNotFoundError):
            verify_flac_checksum(bogus_path)

    def test_verify_non_flac_returns_true(self) -> None:
        self.assertTrue(verify_flac_checksum(self.dummy_audio_path))

    def test_calculate_track_replaygain_success(self) -> None:
        replaygain_result = calculate_track_replaygain(self.dummy_audio_path)
        self.assertIsNotNone(replaygain_result)
        if replaygain_result:
            gain, peak = replaygain_result
            self.assertIsInstance(gain, float)
            self.assertIsInstance(peak, float)
            self.assertTrue(0.0 <= peak <= 1.0)

    def test_calculate_album_replaygain_success(self) -> None:
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

    def test_calculate_bpm_with_scipy(self) -> None:
        with (
            patch(
                "sonora.audio.bpm.load_audio",
                return_value=(np.random.rand(44100 * 10).astype(np.float32), 44100),
            ),
            patch(
                "scipy.signal.spectrogram",
                return_value=(None, None, np.tile(np.linspace(1, 10, 100), (10, 1))),
            ),
        ):
            bpm = calculate_bpm(self.dummy_audio_path)
            self.assertIsNotNone(bpm)
            self.assertIsInstance(bpm, float)
            if bpm:
                self.assertTrue(40.0 <= bpm <= 220.0)

    @patch("taglib.File")
    def test_read_metadata_unsupported_format(self, mock_file: MagicMock) -> None:
        mock_file.return_value = None
        dummy_path = self.tmp_path / "song.xyz"
        dummy_path.write_bytes(b"dummy")

        with self.assertRaises(ValueError):
            read_track_metadata(dummy_path)

    @patch("subprocess.run")
    def test_checksum_binary_not_found(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = FileNotFoundError()
        flac_path = self.tmp_path / "dummy_flac_check_99.flac"
        flac_path.write_bytes(b"dummy")
        with self.assertRaises(RuntimeError):
            verify_flac_checksum(flac_path)

    def test_calculate_album_replaygain_failure_empty(self) -> None:
        result = calculate_album_replaygain([])
        self.assertFalse(result)

    def test_calculate_album_replaygain_corrupted_file(self) -> None:
        fail_wav = self.tmp_path / "corrupt.wav"
        fail_wav.write_bytes(b"not an audio file")
        result = calculate_album_replaygain([fail_wav])
        self.assertFalse(result)

    def test_metadata_in_memory_cache(self) -> None:
        wav_path = self.tmp_path / "cached_track.wav"
        create_dummy_wav_file(wav_path)

        with patch("taglib.File") as mock_taglib_file:
            mock_song = MagicMock()
            mock_song.tags = {"ARTIST": ["Cached Artist"], "TITLE": ["Cached Title"]}
            mock_song.sampleRate = 44100
            mock_song.bitrate = 1411
            mock_song.channels = 2
            mock_song.pictures = []
            mock_taglib_file.return_value.__enter__.return_value = mock_song

            info1 = read_track_metadata(wav_path)
            self.assertEqual(info1.artist, "Cached Artist")
            self.assertEqual(mock_taglib_file.call_count, 1)

            # Second read from same unmodified file returns from in-memory cache without taglib.File call
            info2 = read_track_metadata(wav_path)
            self.assertEqual(info2.artist, "Cached Artist")
            self.assertEqual(mock_taglib_file.call_count, 1)

    def test_find_artist_directory_singles_hierarchy(self) -> None:
        artist_dir = self.tmp_path / "3 Doors Down"
        singles_dir = artist_dir / "Singles"
        track_folder = singles_dir / "3 Doors Down - Here Without You"
        track_folder.mkdir(parents=True)

        found = _find_artist_directory(track_folder, "3 Doors Down")
        self.assertEqual(found.resolve(), artist_dir.resolve())

    def test_find_artist_directory_fallback(self) -> None:
        folder = self.tmp_path / "Other Artist" / "Album"
        folder.mkdir(parents=True)
        found = _find_artist_directory(folder, "Unknown Artist")
        self.assertEqual(found.resolve(), folder.resolve())

    def test_metadata_safe_numeric_parsing(self) -> None:
        wav_path = self.tmp_path / "safe_numeric.wav"
        create_dummy_wav_file(wav_path)

        with patch("taglib.File") as mock_taglib_file:
            mock_song = MagicMock()
            mock_song.tags = {
                "ARTIST": ["Artist"],
                "TITLE": ["Title"],
                "TRACKTOTAL": ["12/12"],
                "DISCTOTAL": ["2/2"],
                "RATING": ["not_a_float"],
            }
            mock_song.sampleRate = 44100
            mock_song.bitrate = 1411
            mock_song.channels = 2
            mock_song.pictures = []
            mock_taglib_file.return_value.__enter__.return_value = mock_song

            info = read_track_metadata(wav_path)
            self.assertEqual(info.total_tracks, 12)
            self.assertEqual(info.total_discs, 2)
            self.assertIsNone(info.rating)


if __name__ == "__main__":
    unittest.main()
