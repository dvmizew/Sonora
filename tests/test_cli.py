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
from sonora.core.models import AuditReport, TrackInfo


class TestCLIInterface(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_build_parser_version(self):
        parser = build_parser()
        self.assertEqual(parser.prog, "sonora")

    def test_main_no_args_returns_zero(self):
        code = main([])
        self.assertEqual(code, 0)

    @patch("sonora.cli.main.tag_album_folder")
    def test_handle_tag_subcommand(self, mock_tag_folder):
        mock_tag_folder.return_value = [TrackInfo(file_path=Path("dummy.flac"))]
        code = main(["tag", str(self.tmp_path), "-w", "2"])
        self.assertEqual(code, 0)
        mock_tag_folder.assert_called_once()

    @patch("sonora.cli.main.audit_library")
    def test_handle_audit_subcommand(self, mock_audit):
        mock_audit.return_value = AuditReport(total_files=1, corrupt_files=0)
        code = main(["audit", str(self.tmp_path)])
        self.assertEqual(code, 0)
        mock_audit.assert_called_once()

    @patch("sonora.cli.main.rename_directory_files")
    def test_handle_rename_subcommand(self, mock_rename):
        mock_rename.return_value = [Path("01 - Artist - Title.flac")]
        code = main(["rename", str(self.tmp_path)])
        self.assertEqual(code, 0)
        mock_rename.assert_called_once()

    @patch("sonora.cli.main.organize_library_singles")
    def test_handle_organize_subcommand(self, mock_organize):
        mock_organize.return_value = 5
        target = self.tmp_path / "Singles"
        code = main(["organize", str(self.tmp_path), "--target-singles", str(target)])
        self.assertEqual(code, 0)
        mock_organize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
