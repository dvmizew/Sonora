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

from sonora.cli.main import build_parser, main
from sonora.core.models import CheckReport, TrackInfo


class TestCLIInterface(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.temporary_path = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_build_parser_version(self):
        parser = build_parser()
        self.assertEqual(parser.prog, "sonora")

    def test_main_no_args_returns_zero(self):
        exit_code = main([])
        self.assertEqual(exit_code, 0)

    @patch("sonora.cli.main.tag_album_folder")
    def test_handle_tag_subcommand(self, mock_tag_album_folder):
        mock_tag_album_folder.return_value = [TrackInfo(file_path=Path("dummy.flac"))]
        exit_code = main(["tag", str(self.temporary_path), "-w", "2"])
        self.assertEqual(exit_code, 0)
        mock_tag_album_folder.assert_called_once()

    @patch("sonora.cli.main.tag_album_folder")
    def test_handle_tag_subcommand_with_json_report(self, mock_tag_album_folder):
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

    @patch("sonora.cli.main.check_library")
    def test_handle_check_subcommand(self, mock_check):
        mock_check.return_value = CheckReport(total_files=1, corrupt_files=0)
        exit_code = main(["check", str(self.temporary_path)])
        self.assertEqual(exit_code, 0)
        mock_check.assert_called_once()

    @patch("sonora.cli.main.rename_directory_files")
    def test_handle_rename_subcommand(self, mock_rename):
        mock_rename.return_value = [Path("01 - Artist - Title.flac")]
        exit_code = main(["rename", str(self.temporary_path)])
        self.assertEqual(exit_code, 0)
        mock_rename.assert_called_once()

    @patch("sonora.cli.main.organize_library_singles")
    def test_handle_organize_subcommand(self, mock_organize):
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

    @patch("sonora.cli.main.backup_library_tags")
    def test_handle_backup_subcommand(self, mock_backup):
        mock_backup.return_value = self.temporary_path / "backup.json"
        exit_code = main(
            [
                "backup",
                str(self.temporary_path),
                "--out",
                str(self.temporary_path / "backup.json"),
            ]
        )
        self.assertEqual(exit_code, 0)
        mock_backup.assert_called_once()

    @patch("sonora.cli.main.restore_library_tags")
    def test_handle_restore_subcommand(self, mock_restore):
        mock_restore.return_value = 10
        dummy_backup = self.temporary_path / "backup.json"
        dummy_backup.write_text("{}", encoding="utf-8")
        exit_code = main(["restore", str(dummy_backup)])
        self.assertEqual(exit_code, 0)
        mock_restore.assert_called_once()


if __name__ == "__main__":
    unittest.main()
