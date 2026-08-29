"""
Unit tests for Sonora CLI main entrypoint and subcommands.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Guarantee src/ is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import unittest

from sonora.cli.main import main
from sonora.core.models import CheckReport, TrackInfo


class TestCLIInterface(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.temporary_path = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_main_no_args_returns_zero(self):
        exit_code = main([])
        self.assertEqual(exit_code, 0)

    def test_handle_tag_subcommand(self):
        with patch("sonora.cli.main.tag_album_folder") as mock_tag_album_folder:
            mock_tag_album_folder.return_value = [
                TrackInfo(file_path=Path("dummy.flac"))
            ]
            exit_code = main(["tag", str(self.temporary_path), "-t", "2"])
            self.assertEqual(exit_code, 0)
            mock_tag_album_folder.assert_called_once()

    def test_handle_tag_subcommand_with_json_report(self):
        with patch("sonora.cli.main.tag_album_folder") as mock_tag_album_folder:
            mock_tag_album_folder.return_value = [
                TrackInfo(
                    file_path=Path("dummy.flac"),
                    title="Song",
                    artist="Artist",
                    bpm=120.0,
                    genre="Pop",
                )
            ]
            json_output_path = self.temporary_path / "tag_report.json"
            exit_code = main(
                ["tag", str(self.temporary_path), "--json", str(json_output_path)]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(json_output_path.exists())
            content = json_output_path.read_text(encoding="utf-8")
            self.assertIn("summary_text", content)
            self.assertIn('"bpm_calculated_count": 1', content)

    def test_handle_check_subcommand(self):
        with patch("sonora.cli.main.check_library") as mock_check:
            mock_check.return_value = CheckReport(total_files=1, corrupt_files=0)
            exit_code = main(["check", str(self.temporary_path)])
            self.assertEqual(exit_code, 0)
            mock_check.assert_called_once()

    def test_handle_rename_subcommand(self):
        with patch("sonora.cli.main.rename_directory_files") as mock_rename:
            mock_rename.return_value = [Path("01 - Artist - Title.flac")]
            exit_code = main(["rename", str(self.temporary_path)])
            self.assertEqual(exit_code, 0)
            mock_rename.assert_called_once()

    def test_handle_organize_subcommand(self):
        with patch("sonora.cli.main.organize_library_singles") as mock_organize:
            mock_organize.return_value = 5
            target_directory = self.temporary_path / "Singles"
            exit_code = main(
                [
                    "organize",
                    str(self.temporary_path),
                    "--target-singles",
                    str(target_directory),
                ]
            )
            self.assertEqual(exit_code, 0)
            mock_organize.assert_called_once()

    def test_handle_backup_subcommand(self):
        with patch("sonora.cli.main.backup_library_tags") as mock_backup:
            mock_backup.return_value = self.temporary_path / "backup.json"
            exit_code = main(["backup", str(self.temporary_path)])
            self.assertEqual(exit_code, 0)
            mock_backup.assert_called_once()

    def test_handle_rename_subcommand_with_json(self):
        with patch("sonora.cli.main.rename_directory_files") as mock_rename:
            mock_rename.return_value = [Path("01 - Artist - Title.flac")]
            json_output = self.temporary_path / "rename_report.json"
            exit_code = main(
                ["rename", str(self.temporary_path), "--json", str(json_output)]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(json_output.exists())

    def test_handle_organize_subcommand_with_json(self):
        with patch("sonora.cli.main.organize_library_singles") as mock_organize:
            mock_organize.return_value = 3
            json_output = self.temporary_path / "organize_report.json"
            exit_code = main(
                ["organize", str(self.temporary_path), "--json", str(json_output)]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(json_output.exists())

    def test_handle_restore_subcommand_with_json(self):
        with patch("sonora.cli.main.restore_library_tags") as mock_restore:
            mock_restore.return_value = 5
            dummy_backup = self.temporary_path / "backup.json"
            dummy_backup.write_text("{}", encoding="utf-8")
            json_output = self.temporary_path / "restore_report.json"
            exit_code = main(["restore", str(dummy_backup), "--json", str(json_output)])
            self.assertEqual(exit_code, 0)
            self.assertTrue(json_output.exists())

    def test_handle_normalize_subcommand(self):
        with patch("sonora.cli.main.normalize_library") as mock_normalize:
            mock_normalize.return_value = [TrackInfo(file_path=Path("song.flac"))]
            exit_code = main(
                ["normalize", str(self.temporary_path), "--bpm", "--replaygain"]
            )
            self.assertEqual(exit_code, 0)
            mock_normalize.assert_called_once()

    def test_handle_bpm_subcommand(self):
        song = self.temporary_path / "song.flac"
        song.write_bytes(b"dummy")
        with (
            patch("sonora.cli.main.calculate_bpm") as mock_calc_bpm,
            patch("sonora.cli.main.read_track_metadata") as mock_read,
            patch("sonora.cli.main.write_track_metadata") as mock_write,
        ):
            mock_read.return_value = TrackInfo(file_path=song, bpm=None)
            mock_calc_bpm.return_value = 128.0
            exit_code = main(["bpm", str(self.temporary_path)])
            self.assertEqual(exit_code, 0)
            mock_write.assert_called_once()
            mock_read.assert_called_once()
            mock_calc_bpm.assert_called_once()

    def test_handle_replaygain_subcommand(self):
        song = self.temporary_path / "song.flac"
        song.write_bytes(b"dummy")
        with patch("sonora.cli.main.calculate_album_replaygain") as mock_rg:
            mock_rg.return_value = True
            exit_code = main(["replaygain", str(self.temporary_path)])
            self.assertEqual(exit_code, 0)
            mock_rg.assert_called_once()

    def test_handle_lyrics_subcommand(self):
        song = self.temporary_path / "song.flac"
        song.write_bytes(b"dummy")
        with (
            patch("sonora.cli.main.process_track_lyrics") as mock_lyrics,
            patch("sonora.cli.main.read_track_metadata") as mock_read,
            patch("sonora.cli.main.write_track_metadata") as mock_write,
        ):
            mock_read.return_value = TrackInfo(
                file_path=song, artist="Artist", title="Title"
            )
            mock_lyrics.return_value = ("[00:01.00] Line", "synced")
            exit_code = main(["lyrics", str(self.temporary_path)])
            self.assertEqual(exit_code, 0)
            mock_read.assert_called_once()
            mock_lyrics.assert_called_once()
            mock_write.assert_called_once()

    def test_handle_keyboard_interrupts(self):
        dummy_backup = self.temporary_path / "backup.json"
        dummy_backup.write_text("{}", encoding="utf-8")

        with (
            patch("sonora.cli.main.tag_album_folder", side_effect=KeyboardInterrupt),
            patch("sonora.cli.main.check_library", side_effect=KeyboardInterrupt),
            patch(
                "sonora.cli.main.rename_directory_files",
                side_effect=KeyboardInterrupt,
            ),
            patch(
                "sonora.cli.main.organize_library_singles",
                side_effect=KeyboardInterrupt,
            ),
            patch(
                "sonora.cli.main.restore_library_tags",
                side_effect=KeyboardInterrupt,
            ),
            patch(
                "sonora.cli.main.normalize_library",
                side_effect=KeyboardInterrupt,
            ),
        ):
            for args in [
                ["tag", str(self.temporary_path)],
                ["check", str(self.temporary_path)],
                ["rename", str(self.temporary_path)],
                ["organize", str(self.temporary_path)],
                ["restore", str(dummy_backup)],
                ["normalize", str(self.temporary_path)],
            ]:
                with self.subTest(args=args):
                    exit_code = main(args)
                    self.assertEqual(exit_code, 130)


if __name__ == "__main__":
    unittest.main()
