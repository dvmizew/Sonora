import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import unittest

from sonora.core.exceptions import AudioProcessingError, SonoraError
from sonora.core.models import AlbumInfo, AuditReport, TrackInfo
from sonora.core.utils import normalize_str, sanitize_name


class TestCoreUtils(unittest.TestCase):
    def test_normalize_str_basic(self):
        self.assertEqual(normalize_str("Hello World"), "hello world")
        self.assertEqual(normalize_str("$tring!"), "stringi")

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
        self.assertEqual(data["track_number"], "1")
        self.assertEqual(data["bpm"], "120.0")

    def test_album_info_track_count(self):
        t1 = TrackInfo(file_path=Path("/music/1.flac"), title="Track 1")
        t2 = TrackInfo(file_path=Path("/music/2.flac"), title="Track 2")
        album = AlbumInfo(title="Test Album", artist="Test Artist", tracks=[t1, t2])
        self.assertEqual(album.track_count, 2)

    def test_audit_report_initialization(self):
        report = AuditReport(file_path=Path("/music/1.flac"), is_valid=True)
        self.assertTrue(report.is_valid)
        self.assertEqual(report.missing_tags, [])


class TestCoreExceptions(unittest.TestCase):
    def test_exception_inheritance(self):
        err = AudioProcessingError("FFmpeg failed")
        self.assertIsInstance(err, SonoraError)
        self.assertEqual(str(err), "FFmpeg failed")


if __name__ == "__main__":
    unittest.main()
