"""
Unit tests for Sonora core business logic modules (Auditor, Renamer, Tagger, Organizer).
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Guarantee src/ is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import unittest

from sonora.core.models import TrackInfo
from sonora.modules.auditor import audit_library, check_brackets_corruption
from sonora.modules.organizer import is_single_folder, organize_library_singles
from sonora.modules.renamer import (
    rename_directory_files,
    rename_track_file,
    sync_lrc_metadata,
)
from sonora.modules.tagger import (
    normalize_artist_alias,
    process_single_track,
    tag_album_folder,
)


def create_dummy_wav(path: Path) -> None:
    """Create a minimal valid 44.1kHz mono WAV file."""
    sample_rate = 44100
    num_samples = 44100 // 10  # 0.1s
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)
    data_size = num_samples * block_align
    chunk_size = 36 + data_size

    header = bytearray()
    header.extend(b"RIFF")
    header.extend(chunk_size.to_bytes(4, "little"))
    header.extend(b"WAVEfmt ")
    header.extend((16).to_bytes(4, "little"))  # Subchunk1Size
    header.extend((1).to_bytes(2, "little"))   # AudioFormat (PCM)
    header.extend(num_channels.to_bytes(2, "little"))
    header.extend(sample_rate.to_bytes(4, "little"))
    header.extend(byte_rate.to_bytes(4, "little"))
    header.extend(block_align.to_bytes(2, "little"))
    header.extend(bits_per_sample.to_bytes(2, "little"))
    header.extend(b"data")
    header.extend(data_size.to_bytes(4, "little"))
    header.extend(b"\x00" * data_size)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)


class TestCoreModules(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_normalize_artist_alias(self):
        self.assertEqual(normalize_artist_alias("mgl"), "M.G.L.")
        self.assertEqual(normalize_artist_alias("killa fonic"), "Killa Fonic")
        self.assertEqual(normalize_artist_alias("nane"), "Nane")

    def test_check_brackets_corruption(self):
        issues = check_brackets_corruption("Song Title [Official Video]")
        self.assertTrue(any("Corrupt bracket" in issue for issue in issues))

        no_issues = check_brackets_corruption("Song Title (Remix)")
        self.assertEqual(no_issues, [])

    def test_sync_lrc_metadata(self):
        lrc_file = self.tmp_path / "test.lrc"
        lrc_file.write_text("[00:10.00] Line 1\n", encoding="utf-8")

        info = TrackInfo(file_path=Path("dummy.flac"), artist="Artist Name", title="Track Title")
        success = sync_lrc_metadata(lrc_file, info)
        self.assertTrue(success)

        content = lrc_file.read_text(encoding="utf-8")
        self.assertIn("[ar:Artist Name]", content)
        self.assertIn("[ti:Track Title]", content)
        self.assertIn("[00:10.00] Line 1", content)

    @patch("sonora.modules.renamer.read_track_metadata")
    def test_rename_track_file(self, mock_read):
        wav_file = self.tmp_path / "old_name.wav"
        create_dummy_wav(wav_file)

        mock_read.return_value = TrackInfo(
            file_path=wav_file,
            artist="Beyoncé",
            title="Halo",
            track_number=1
        )

        new_path = rename_track_file(wav_file)
        self.assertTrue(new_path.exists())
        self.assertEqual(new_path.name, "01 - Beyoncé - Halo.wav")

    @patch("sonora.modules.renamer.read_track_metadata")
    def test_rename_directory_files(self, mock_read):
        f1 = self.tmp_path / "song1.wav"
        f2 = self.tmp_path / "song2.wav"
        create_dummy_wav(f1)
        create_dummy_wav(f2)

        mock_read.side_effect = [
            TrackInfo(file_path=f1, artist="Artist", title="Title 1", track_number=1),
            TrackInfo(file_path=f2, artist="Artist", title="Title 2", track_number=2),
        ]

        renamed = rename_directory_files(self.tmp_path)
        self.assertEqual(len(renamed), 2)

    def test_is_single_folder(self):
        album_dir = self.tmp_path / "album"
        album_dir.mkdir()

        f1 = album_dir / "01.wav"
        f2 = album_dir / "02.wav"
        f3 = album_dir / "03.wav"
        create_dummy_wav(f1)
        create_dummy_wav(f2)
        create_dummy_wav(f3)

        with patch("sonora.modules.organizer.read_track_metadata") as mock_read:
            mock_read.side_effect = [
                TrackInfo(file_path=f1, album="Same Album"),
                TrackInfo(file_path=f2, album="Same Album"),
                TrackInfo(file_path=f3, album="Same Album"),
            ]
            self.assertFalse(is_single_folder(album_dir))

    @patch("sonora.modules.organizer.read_track_metadata")
    def test_organize_library_singles(self, mock_read):
        src_dir = self.tmp_path / "source"
        target_dir = self.tmp_path / "Singles"
        f1 = src_dir / "single.wav"
        create_dummy_wav(f1)

        mock_read.return_value = TrackInfo(file_path=f1, artist="Single Artist", title="Single Song")

        moved = organize_library_singles(src_dir, target_dir)
        self.assertEqual(moved, 1)
        self.assertTrue((target_dir / "Single Artist" / "single.wav").exists())

    @patch("sonora.modules.auditor.read_track_metadata")
    def test_audit_library(self, mock_read):
        f1 = self.tmp_path / "song.wav"
        create_dummy_wav(f1)

        mock_read.return_value = TrackInfo(file_path=f1, artist="Artist [Official]", title="Title")

        report = audit_library(self.tmp_path, output_json=self.tmp_path / "report.json")
        self.assertEqual(report.total_files, 1)
        self.assertTrue((self.tmp_path / "report.json").exists())

    @patch("sonora.modules.tagger.write_track_metadata")
    @patch("sonora.modules.tagger.fetch_synced_lyrics")
    @patch("sonora.modules.tagger.fetch_track_mbid")
    @patch("sonora.modules.tagger.read_track_metadata")
    def test_process_single_track(self, mock_read, mock_mbid, mock_lyrics, mock_write):
        wav_file = self.tmp_path / "song.wav"
        create_dummy_wav(wav_file)

        mock_read.return_value = TrackInfo(file_path=wav_file, artist="nane", title="Piesa")
        mock_mbid.return_value = "mbid-12345"
        mock_lyrics.return_value = "[00:01.00] Vers"

        info = process_single_track(wav_file, fetch_bpm=False, fetch_replaygain=False)
        self.assertEqual(info.artist, "Nane")
        self.assertEqual(info.musicbrainz_trackid, "mbid-12345")
        mock_write.assert_called_once()

    @patch("sonora.modules.tagger.process_single_track")
    def test_tag_album_folder(self, mock_process):
        album_dir = self.tmp_path / "album"
        f1 = album_dir / "01.wav"
        f2 = album_dir / "02.wav"
        create_dummy_wav(f1)
        create_dummy_wav(f2)

        mock_process.side_effect = [
            TrackInfo(file_path=f1, artist="Artist", title="T1"),
            TrackInfo(file_path=f2, artist="Artist", title="T2"),
        ]

        results = tag_album_folder(album_dir, max_workers=2)
        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
