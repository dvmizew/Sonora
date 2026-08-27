import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import unittest

from sonora.core.models import CheckReport, TrackInfo
from sonora.core.utils import (
    clean_title,
    is_valid_uuid,
    normalize_genre,
    normalize_str,
    sanitize_name,
)


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

    def test_is_valid_uuid(self):
        self.assertTrue(is_valid_uuid("c8b03190-306c-4125-9b32-3f9d86d60a12"))
        self.assertTrue(is_valid_uuid("C8B03190-306C-4125-9B32-3F9D86D60A12"))
        self.assertFalse(is_valid_uuid("not-a-uuid"))
        self.assertFalse(
            is_valid_uuid("c8b03190306c41259b323f9d86d60a12")
        )  # 32 chars without hyphens
        self.assertFalse(is_valid_uuid("urn:uuid:c8b03190-306c-4125-9b32-3f9d86d60a12"))
        self.assertFalse(is_valid_uuid("{c8b03190-306c-4125-9b32-3f9d86d60a12}"))
        self.assertFalse(is_valid_uuid(None))
        self.assertFalse(is_valid_uuid(""))

    def test_extract_series_number_all_formats(self):
        from sonora.core.utils import extract_series_number

        # Arabic numbers
        self.assertEqual(extract_series_number("Savage Mode 2"), 2)
        self.assertEqual(extract_series_number("Part 3"), 3)
        self.assertEqual(extract_series_number("Vol. 4"), 4)

        # Roman numerals
        self.assertEqual(extract_series_number("Savage Mode II"), 2)
        self.assertEqual(extract_series_number("Act IV"), 4)
        self.assertEqual(extract_series_number("Chapter IX"), 9)
        self.assertEqual(extract_series_number("Volume XIV"), 14)
        self.assertEqual(extract_series_number("Part XX"), 20)

        # Number words (1..20)
        self.assertEqual(extract_series_number("Volume One"), 1)
        self.assertEqual(extract_series_number("Part Three"), 3)
        self.assertEqual(extract_series_number("Act Seven"), 7)
        self.assertEqual(extract_series_number("Chapter Twelve"), 12)
        self.assertEqual(extract_series_number("Book Twenty"), 20)

        # None / Edge cases
        self.assertIsNone(extract_series_number("Savage Mode"))
        self.assertIsNone(extract_series_number("Live in Paris"))
        self.assertIsNone(extract_series_number(""))
        self.assertIsNone(extract_series_number(None))

    def test_clean_title_remaster_and_features(self):
        self.assertEqual(clean_title("In the End (2020 Remaster)"), "In the End")
        self.assertEqual(clean_title("Rockstar (feat. 21 Savage)"), "Rockstar")
        self.assertEqual(clean_title("Song [Explicit]"), "Song")
        self.assertEqual(clean_title("Track (Deluxe Edition)"), "Track")
        self.assertEqual(clean_title("Video [Official Music Video]"), "Video")
        self.assertEqual(clean_title("Audio (Official Audio)"), "Audio")
        self.assertEqual(clean_title("Clean [Clean Version]"), "Clean")
        self.assertEqual(clean_title("Parody [Parody]"), "Parody")
        self.assertEqual(clean_title("Spaces\u200b\u00a0Track"), "Spaces Track")
        self.assertEqual(clean_title(""), "")

    def test_load_user_overrides_corrupt_json(self):
        from unittest.mock import patch

        from sonora.core.utils import _load_user_overrides

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_text", return_value="INVALID JSON {{"),
        ):
            overrides = _load_user_overrides()
            self.assertEqual(overrides, {})

    def test_normalize_genre_mapping_and_filtering(self):
        self.assertEqual(normalize_genre("Hip Hop"), "Hip-Hop/Rap")
        self.assertEqual(normalize_genre("Rap"), "Hip-Hop/Rap")
        self.assertEqual(normalize_genre("Electronic"), "Electronic")
        self.assertEqual(normalize_genre("Synthpop"), "Synth-pop")
        self.assertIsNone(normalize_genre("Billboard Top 40"))  # Blacklisted
        self.assertIsNone(normalize_genre("Unknown"))  # Blacklisted
        self.assertIsNone(normalize_genre("12345"))  # Digits
        self.assertIsNone(normalize_genre(""))
        self.assertIsNone(normalize_genre(None))


class TestCoreModels(unittest.TestCase):
    def test_track_info_to_dict(self):
        track = TrackInfo(
            file_path=Path("/music/song.flac"),
            artist="Beyoncé",
            title="HALO",
            album="I Am... Sasha Fierce",
            track_number=1,
            bpm=120.0,
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
