import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import unittest

from sonora.core.models import CheckReport, TrackInfo
from sonora.core.utils import normalize_str, sanitize_name


class TestCoreUtils(unittest.TestCase):
    def test_normalize_str_basic(self):
        self.assertEqual(normalize_str("Hello World"), "hello world")
        self.assertEqual(normalize_str("$tring!"), "string")

    def test_normalize_str_diacritics_and_symbols(self):
        self.assertEqual(normalize_str("Beyoncé"), "beyonce")
        self.assertEqual(normalize_str("Mötley Crüe"), "motley crue")

    def test_sanitize_name_filesystem_chars(self):
        self.assertEqual(sanitize_name("AC/DC"), "AC_DC")
        self.assertEqual(sanitize_name("Artist: Album?"), "Artist Album")
        self.assertEqual(sanitize_name("Track 01.flac."), "Track 01.flac")

    def test_sanitize_name_empty(self):
        self.assertEqual(sanitize_name(""), "Unknown")
        self.assertEqual(sanitize_name(None), "Unknown")

    def test_normalize_str_edge_cases(self):
        self.assertEqual(normalize_str(None), "")
        self.assertEqual(normalize_str(""), "")
        self.assertEqual(normalize_str("   "), "")
        self.assertEqual(normalize_str("A$AP Rocky!"), "asap rocky")
        self.assertEqual(normalize_str("Beyoncé - Nöél"), "beyonce noel")

    def test_sanitize_name_complex_edge_cases(self):
        self.assertEqual(sanitize_name("Artist / Title <HQ>:"), "Artist _ Title HQ")


class TestCoreModels(unittest.TestCase):
    def test_track_info_to_dict(self):
        track = TrackInfo(
            file_path=Path("/music/song.flac"),
            artist="Beyoncé",
            title="HALO",
            album="I Am... Sasha Fierce",
            track_number=1,
            bpm=120.0
        )
        data = track.to_dict()
        self.assertEqual(data["artist"], "Beyoncé")
        self.assertEqual(data["title"], "HALO")
        self.assertEqual(data["track_number"], 1)
        self.assertEqual(data["bpm"], 120.0)


    def test_check_report_initialization(self):
        report = CheckReport(file_path=Path("/music/1.flac"), is_valid=True)
        self.assertTrue(report.is_valid)
        self.assertEqual(report.missing_tags, [])

    def test_track_info_default_values(self):
        track = TrackInfo(file_path=Path("/music/song.flac"))
        self.assertEqual(track.artist, "Unknown Artist")
        self.assertEqual(track.title, "Unknown Title")
        self.assertEqual(track.album, "Unknown Album")
        self.assertTrue(track.is_lossless)

    def test_track_info_to_dict_empty_fields(self):
        track = TrackInfo(file_path=Path("/music/song.flac"))
        data = track.to_dict()
        self.assertIsNone(data["track_number"])
        self.assertIsNone(data["bpm"])



if __name__ == "__main__":
    unittest.main()
