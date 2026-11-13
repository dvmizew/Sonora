"""
Unit and integration tests for Sonora lyrics subsystem:
LRC parsing, timestamp validation, junk/watermark cleanup, local .lrc companion sidecars,
and provider synchronization contracts.
"""

from unittest.mock import MagicMock, patch

from sonora.core.config import SonoraConfig
from sonora.core.sidecars import (
    clean_lyrics_text,
    extract_max_timestamp,
    find_companion_lyrics,
    get_lyrics_quality,
    is_lyrics_duration_valid,
    lyrics_identity_matches,
    lyrics_word_body,
    relocate_companion_lyrics,
)
from sonora.modules.renamer import sync_lrc_metadata
from sonora.modules.track_processing import process_track_lyrics
from sonora.services.lyrics import fetch_synced_lyrics
from tests.support import IsolatedTestCase

SYNCED = "[00:01.00]First line\n[00:03.50]Second line\n[00:06.00]End"
ENHANCED = "<00:01.00>First <00:02.00>line\n<00:03.50>Second <00:04.50>line"
PLAIN = "First line\nSecond line\nEnd"


class TestLyricsParsingAndValidation(IsolatedTestCase):
    def test_lyrics_word_body_normalization(self) -> None:
        self.assertEqual(
            lyrics_word_body("[Verse 1: Artist]\nDon't stop, 心!\n[Chorus]\nNow."),
            lyrics_word_body("[00:01]<00:01>DON’T <00:02>STOP 心\n[00:03]now"),
        )
        self.assertNotEqual(lyrics_word_body("don't go"), lyrics_word_body("do go"))
        self.assertNotEqual(lyrics_word_body("心"), lyrics_word_body("xin"))
        self.assertEqual(lyrics_word_body("[Stay with me]"), "stay with me")
        self.assertEqual(lyrics_word_body("[Verse 99: Sung by Artist]"), "")
        self.assertEqual(
            lyrics_word_body(
                "[Versuri pentru „Recording” cu Artist]\n[Strofă 1: Artist]\n"
                "Nu pleca\n[Refren x2: Artist]\nRămâi"
            ),
            lyrics_word_body("[ti:Recording]\nNu pleca\nRămâi"),
        )
        self.assertEqual(lyrics_word_body("Recording Lyrics\nWords"), "words")

    def test_timestamp_extraction_and_quality(self) -> None:
        self.assertIsNone(extract_max_timestamp("[length:99:00]\n[ti:01:02]\n[00:10>"))
        self.assertEqual(
            extract_max_timestamp("[00:01.5][00:03.125]Words\n<00:05:1>More"),
            5.1,
        )
        self.assertEqual(get_lyrics_quality("<00:01>One <00:02>two"), 3)
        self.assertEqual(get_lyrics_quality("[00:01.5]Words"), 2)
        self.assertEqual(get_lyrics_quality("Plain text lyrics without timestamps"), 1)
        self.assertEqual(get_lyrics_quality(""), 0)
        self.assertEqual(get_lyrics_quality(None), 0)

    def test_duration_validation_rules(self) -> None:
        self.assertTrue(is_lyrics_duration_valid("[offset:20000]\n[01:50]End", 100, 0))
        self.assertFalse(
            is_lyrics_duration_valid("[offset:-20000]\n[01:30]End", 100, 0)
        )
        self.assertFalse(is_lyrics_duration_valid("[00:10]<00:12>One <00:11>two", 100))
        self.assertTrue(is_lyrics_duration_valid("[00:10][00:05]Repeated", 100))
        self.assertFalse(is_lyrics_duration_valid("[00:01]Words", float("nan")))
        self.assertFalse(is_lyrics_duration_valid("[00:99]Words", 100))
        self.assertFalse(is_lyrics_duration_valid("[00:10>Words", 100))
        self.assertFalse(is_lyrics_duration_valid("<00:12>One <00:11>two", None))

    def test_cleanup_strips_watermarks_without_removing_lyrics(self) -> None:
        text = (
            "[offset:200]\n"
            "[00:01][00:03]You and me\n"
            "[00:05][00:07]12 Contributors\n"
            "[00:08]Sing along\n"
            "Embed"
        )
        expected = "[offset:200]\n[00:01][00:03]You and me\n[00:08]Sing along"
        self.assertEqual(clean_lyrics_text(text), expected)
        self.assertEqual(clean_lyrics_text(expected), expected)


class TestLocalLyricsSidecars(IsolatedTestCase):
    def test_find_and_relocate_companion_lyrics(self) -> None:
        audio = self.test_directory / "01 - Track.flac"
        audio.touch()
        lrc = self.test_directory / "01 - Track.lrc"
        lrc.write_text("[00:01.00]Song lyrics", encoding="utf-8")

        found = find_companion_lyrics(audio)
        self.assertEqual(found, [lrc])

        new_audio = self.test_directory / "01 - Renamed.flac"
        relocated = relocate_companion_lyrics(audio, new_audio)
        self.assertEqual(relocated, [self.test_directory / "01 - Renamed.lrc"])
        self.assertTrue((self.test_directory / "01 - Renamed.lrc").exists())
        self.assertFalse(lrc.exists())

    def test_lyrics_identity_matches(self) -> None:
        lrc_text = "[ar:Adele]\n[ti:Easy On Me]\n[00:01.00]There ain't no gold"
        self.assertTrue(lyrics_identity_matches(lrc_text, "Adele", "Easy On Me"))
        self.assertFalse(lyrics_identity_matches(lrc_text, "Drake", "God's Plan"))

    def test_sync_lrc_metadata_updates_headers_atomically(self) -> None:
        lrc_file = self.test_directory / "lyrics.lrc"
        lrc_file.write_text(
            "[ar:adele]\n[ti:easy on me]\n[00:01.00]There ain't no gold\n",
            encoding="utf-8",
        )

        updated = sync_lrc_metadata(lrc_file, "Adele", "Easy On Me")
        self.assertTrue(updated)

        content = lrc_file.read_text(encoding="utf-8")
        self.assertIn("[ar:Adele]", content)
        self.assertIn("[ti:Easy On Me]", content)
        self.assertIn("[00:01.00]There ain't no gold", content)


class TestLyricsServiceRequests(IsolatedTestCase):
    def setUp(self) -> None:
        super().setUp()
        config_patcher = patch(
            "sonora.services.lyrics.get_config",
            return_value=SonoraConfig(allow_unverified_lyrics=True),
        )
        config_patcher.start()
        self.addCleanup(config_patcher.stop)

    @patch("sonora.services.lyrics._query_lrclib", return_value=None)
    @patch("syncedlyrics.search")
    def test_fetch_synced_lyrics_basic_and_enhanced(
        self, mock_search: MagicMock, mock_lrclib: MagicMock
    ) -> None:
        mock_search.return_value = SYNCED
        lyrics = fetch_synced_lyrics("Artist", "Title")
        self.assertEqual(lyrics, SYNCED)
        mock_search.assert_called_with(
            "artist - title",
            plain_only=False,
            synced_only=False,
            enhanced=False,
            providers=["Musixmatch", "NetEase", "Megalobiz", "Genius"],
        )

        mock_search.return_value = ENHANCED
        lyrics_enhanced = fetch_synced_lyrics(
            "Artist", "Title", synced_only=True, enhanced=True
        )
        self.assertEqual(lyrics_enhanced, ENHANCED)

    @patch("sonora.services.lyrics._query_lrclib", return_value=None)
    @patch("syncedlyrics.search", return_value=None)
    def test_fetch_synced_lyrics_not_found(
        self, mock_search: MagicMock, mock_lrclib: MagicMock
    ) -> None:
        self.assertIsNone(fetch_synced_lyrics("Nonexistent", "Track"))


class TestLyricsProcessingWorkflow(IsolatedTestCase):
    def test_process_track_lyrics_quality_upgrade_and_protection(self) -> None:
        audio_file = self.test_directory / "track.flac"
        audio_file.write_bytes(b"dummy audio")
        lrc_file = self.test_directory / "track.lrc"

        # 1. Enhanced (Quality 3) skips remote call when not force
        lrc_file.write_text("<00:01.00> Enhanced word synced", encoding="utf-8")
        with patch("sonora.services.lyrics.fetch_synced_lyrics") as mock_fetch:
            lyrics, tag_type = process_track_lyrics(
                audio_file, "Artist", "Title", force=False, duration=180.0
            )
            self.assertEqual(lyrics, "<00:01.00> Enhanced word synced")
            self.assertEqual(tag_type, "existing")
            mock_fetch.assert_not_called()

        # 2. Plain text upgrades to Synced
        lrc_file.write_text("Just plain lyrics", encoding="utf-8")
        with patch(
            "sonora.services.lyrics.fetch_synced_lyrics",
            return_value="[00:01.00] Just plain lyrics",
        ) as mock_fetch:
            lyrics, tag_type = process_track_lyrics(
                audio_file, "Artist", "Title", force=False, duration=180.0
            )
            self.assertEqual(lyrics, "[00:01.00] Just plain lyrics")
            self.assertEqual(tag_type, "synced")

        # 3. Force=True with plain fallback does NOT downgrade existing synced lyrics
        lrc_file.write_text("[00:01.00] Synced line", encoding="utf-8")
        with patch(
            "sonora.services.lyrics.fetch_synced_lyrics",
            return_value="Plain text fallback",
        ) as mock_fetch:
            lyrics, tag_type = process_track_lyrics(
                audio_file, "Artist", "Title", force=True, duration=180.0
            )
            self.assertEqual(lyrics, "[00:01.00] Synced line")
            self.assertEqual(tag_type, "existing")
