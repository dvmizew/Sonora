import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from sonora.core.config import SonoraConfig
from sonora.core.sidecars import (
    clean_lyrics_text,
    extract_max_timestamp,
    get_lyrics_quality,
    is_lyrics_duration_valid,
    lyrics_word_body,
)
from sonora.modules.track_processing import process_track_lyrics
from sonora.services.genius import fetch_genius_song_details
from sonora.services.lyrics import fetch_synced_lyrics
from tests.service_support import ServiceResponseTestCase


class TestLyricsResponses(ServiceResponseTestCase):
    def test_lyric_comparison_ignores_labels_and_typography_but_preserves_words(
        self,
    ) -> None:
        self.assertEqual(
            lyrics_word_body("[Verse 1: Artist]\nDon't stop, 心!\n[Chorus]\nNow."),
            lyrics_word_body("[00:01]<00:01>DON’T <00:02>STOP 心\n[00:03]now"),
        )
        self.assertNotEqual(lyrics_word_body("don't go"), lyrics_word_body("do go"))
        self.assertNotEqual(lyrics_word_body("心"), lyrics_word_body("xin"))
        self.assertEqual(lyrics_word_body("[Stay with me]"), "stay with me")
        self.assertEqual(lyrics_word_body("[Verse 99: Sung by Artist]"), "")

    def test_timestamp_extraction_ignores_metadata_and_mismatched_brackets(
        self,
    ) -> None:
        self.assertIsNone(extract_max_timestamp("[length:99:00]\n[ti:01:02]\n[00:10>"))
        self.assertEqual(
            extract_max_timestamp("[00:01.5][00:03.125]Words\n<00:05:1>More"),
            5.1,
        )
        self.assertEqual(get_lyrics_quality("<00:01>One <00:02>two"), 3)
        self.assertEqual(get_lyrics_quality("[00:01.5]Words"), 2)

    def test_duration_validation_accounts_for_offsets_and_word_order(self) -> None:
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

    def test_cleanup_handles_repeated_timestamps_without_removing_sung_text(
        self,
    ) -> None:
        text = "[offset:200]\n[00:01][00:03]You and me\n[00:05][00:07]12 Contributors\n[00:08]"
        expected = "[offset:200]\n[00:01][00:03]You and me\n[00:08]"
        self.assertEqual(clean_lyrics_text(text), expected)
        self.assertEqual(clean_lyrics_text(expected), expected)

    def setUp(self) -> None:
        super().setUp()
        configuration = patch(
            "sonora.services.lyrics.get_config",
            return_value=SonoraConfig(allow_unverified_lyrics=True),
        )
        configuration.start()
        self.addCleanup(configuration.stop)

    @patch("sonora.services.lyrics.syncedlyrics")
    def test_fetch_synced_lyrics(self, mock_syncedlyrics: MagicMock) -> None:
        # 1. Basic search
        mock_syncedlyrics.search.return_value = "[00:12.34] Test lyric line"
        lyrics = fetch_synced_lyrics("Artist", "Title")
        self.assertEqual(lyrics, "[00:12.34] Test lyric line")
        mock_syncedlyrics.search.assert_called_with(
            "artist - title",
            plain_only=False,
            synced_only=False,
            enhanced=False,
            providers=["Musixmatch", "NetEase", "Megalobiz", "Genius"],
        )

        # 2. Options search
        mock_syncedlyrics.search.return_value = "<00:12.34> Enhanced lyric line"
        lyrics_enhanced = fetch_synced_lyrics(
            "Artist",
            "Title",
            synced_only=True,
            enhanced=True,
            providers=["Lrclib"],
            lang="en",
        )
        self.assertEqual(lyrics_enhanced, "<00:12.34> Enhanced lyric line")
        mock_syncedlyrics.search.assert_called_with(
            "artist - title",
            plain_only=False,
            synced_only=True,
            enhanced=True,
            providers=["Lrclib"],
            lang="en",
        )

    @patch("sonora.services.lyrics.syncedlyrics")
    def test_fetch_synced_lyrics_raises_api_service_error_on_failure(
        self, mock_syncedlyrics: MagicMock
    ) -> None:
        mock_syncedlyrics.search.side_effect = RuntimeError("Network timeout")
        with self.assertRaises(RuntimeError):
            fetch_synced_lyrics("FailArtist", "FailTitle")

    @patch("sonora.services.lyrics.set_cached_api")
    @patch("sonora.services.lyrics.get_cached_api", return_value=None)
    @patch("sonora.services.lyrics.syncedlyrics")
    def test_fetch_synced_lyrics_negative_caching(
        self,
        mock_syncedlyrics: MagicMock,
        mock_get_cache: MagicMock,
        mock_set_cache: MagicMock,
    ) -> None:
        mock_syncedlyrics.search.return_value = None
        result = fetch_synced_lyrics("Artist Without Lyrics", "Instrumental Track")
        self.assertIsNone(result)
        self.assertEqual(mock_set_cache.call_args.args[1], "")
        mock_get_cache.return_value = ""
        mock_syncedlyrics.search.reset_mock()
        self.assertIsNone(
            fetch_synced_lyrics("Artist Without Lyrics", "Instrumental Track")
        )
        mock_syncedlyrics.search.assert_not_called()

    def test_synced_lyrics_empty_query_returns_none(self) -> None:
        self.assertIsNone(fetch_synced_lyrics("", ""))

    @patch("sonora.core.http.SESSION.get")
    def test_genius_rejects_lyrics_unavailable_text(self, mock_get: MagicMock) -> None:
        mock_search_response = MagicMock()
        mock_search_response.json.return_value = {
            "response": {
                "hits": [
                    {
                        "result": {
                            "api_path": "/songs/1",
                            "title": "Title",
                            "primary_artist": {"name": "Artist"},
                        }
                    }
                ]
            }
        }
        mock_song_response = MagicMock()
        mock_song_response.json.return_value = {
            "response": {
                "song": {
                    "id": 1,
                    "title": "Title",
                    "primary_artist": {"name": "Artist"},
                    "description": {"plain": "Lyrics for this song are unavailable"},
                }
            }
        }
        mock_get.side_effect = [mock_search_response, mock_song_response]

        details = fetch_genius_song_details("Artist", "Title", api_token="token")
        self.assertIsNotNone(details)
        assert details is not None
        self.assertIsNone(details["description"])

    def test_clean_lyrics_text(self) -> None:
        dirty = (
            "[00:12.34] Valid lyric line\n"
            "12 Contributors\n"
            "You might also like\n"
            "[00:15.00] Second valid line\n"
            "https://genius.com/some-url\n"
            "14Embed"
        )
        self.assertEqual(
            clean_lyrics_text(dirty),
            "[00:12.34] Valid lyric line\n[00:15.00] Second valid line",
        )
        self.assertIsNone(clean_lyrics_text(None))
        self.assertEqual(clean_lyrics_text(""), "")
        self.assertEqual(
            clean_lyrics_text("12 Contributors\nYou might also like\n14Embed"), ""
        )

    @patch("sonora.services.lyrics.fetch_synced_lyrics")
    def test_process_track_lyrics_quality_upgrade(self, mock_fetch: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_file = Path(tmp_dir) / "track.flac"
            audio_file.write_bytes(b"dummy")
            lrc_file = Path(tmp_dir) / "track.lrc"

            # 1. Enhanced (Quality 3) skips remote call when not force
            lrc_file.write_text("<00:01.00> Enhanced word synced", encoding="utf-8")
            lyrics, tag_type = process_track_lyrics(
                audio_file, "Artist", "Title", force=False, duration=180.0
            )
            self.assertEqual(lyrics, "<00:01.00> Enhanced word synced")
            self.assertEqual(tag_type, "existing")
            mock_fetch.assert_not_called()

            # 2. Plain text (Quality 1) upgrades to Synced (Quality 2)
            lrc_file.write_text("Just plain lyrics", encoding="utf-8")
            mock_fetch.return_value = "[00:01.00] Just plain lyrics"
            lyrics, tag_type = process_track_lyrics(
                audio_file, "Artist", "Title", force=False, duration=180.0
            )
            self.assertEqual(lyrics, "[00:01.00] Just plain lyrics")
            self.assertEqual(tag_type, "synced")
            self.assertEqual(
                lrc_file.read_text(encoding="utf-8"),
                "[00:01.00] Just plain lyrics",
            )

            # 3. Just plain lyrics (Quality 2) upgrades to Enhanced (Quality 3)
            mock_fetch.return_value = "<00:01.00> Just plain lyrics"
            lyrics, tag_type = process_track_lyrics(
                audio_file, "Artist", "Title", force=False, duration=180.0
            )
            self.assertEqual(lyrics, "<00:01.00> Just plain lyrics")
            self.assertEqual(tag_type, "enhanced")

            # 4. Downgrade protection: force=True with remote Plain text does NOT overwrite Synced
            mock_fetch.return_value = "Just plain text fallback from remote"
            lyrics, tag_type = process_track_lyrics(
                audio_file, "Artist", "Title", force=True, duration=180.0
            )
            self.assertEqual(lyrics, "<00:01.00> Just plain lyrics")
            self.assertEqual(tag_type, "existing")

    def test_clean_lyrics_extended_junk_and_watermarks(self) -> None:
        raw_dirty = (
            "[re:www.megalobiz.com/lrc/maker]\n"
            "[ve:v1.2.3]\n"
            "[00:00.00-1] 作词 : Pierre Bourne\n"
            "[00:00.00-1] 作曲 : Shayaa Joseph\n"
            "[00:05.86]가사 제작: megalobiz\n"
            "[00:10.00]\n"
            "[00:12.34] Valid lyric line\n"
            "This song is called test song\n"
            "[00:15.00] Second valid line\n"
            "[00:20.00]Instrumental"
        )
        cleaned = clean_lyrics_text(raw_dirty)
        self.assertEqual(
            cleaned,
            "[re:www.megalobiz.com/lrc/maker]\n[ve:v1.2.3]\n"
            "[00:12.34] Valid lyric line\nThis song is called test song\n"
            "[00:15.00] Second valid line\n[00:20.00]Instrumental",
        )

    def test_clean_lyrics_preserves_lrc_semantics_and_sung_words(self) -> None:
        text = (
            "[ar:Performer]\n[ti:Song]\n[offset:-250]\n[length:03:30]\n"
            "[by:Lyric editor]\n[Verse 1]\n[Instrumental]\n"
            "[00:01.00][00:03.00]<00:01.10>Written <00:01.50>in the stars\n"
            "[00:05.00]Recorded in my heart\n"
            "This song is called love\nÉtoiles <05> &amp; 心\nI will embed"
        )
        expected = text.replace("&amp;", "&")
        self.assertEqual(clean_lyrics_text(text), expected)
        self.assertEqual(clean_lyrics_text(expected), expected)

    def test_clean_lyrics_preserves_end_timestamp_but_not_empty_placeholder(
        self,
    ) -> None:
        text = "[offset:100]\n[00:00.00]\n[00:01.00]Words\n[00:04.00]"
        expected = "[offset:100]\n[00:01.00]Words\n[00:04.00]"
        self.assertEqual(clean_lyrics_text(text), expected)
        self.assertEqual(clean_lyrics_text(expected), expected)
        self.assertEqual(get_lyrics_quality("[au:Author]\n[offset:-100]"), 0)
        self.assertEqual(get_lyrics_quality("[00:00.00]"), 0)

    def test_clean_lyrics_removes_html_without_discarding_lyric_text(self) -> None:
        text = (
            "<!DOCTYPE html><html><head><title>Website title</title></head><body>"
            '<script type="text/javascript">\nwindow.ad = "not lyrics";\n</script>'
            '<style>p { color: red; }</style><div class="lyrics">'
            "[00:01.00]First <b>line</b><!--tracker--><br/>"
            "[00:02.00]<00:02.10>Second <span>line &amp; 心</span></div></body></html>"
        )
        expected = "[00:01.00]First line\n[00:02.00]<00:02.10>Second line & 心"
        self.assertEqual(clean_lyrics_text(text), expected)
        self.assertEqual(clean_lyrics_text(expected), expected)

    def test_lyrics_duration_validation_and_mismatch(self) -> None:
        lrc_text = (
            "[00:10.00] First line\n[01:30.50] Middle line\n[03:15.20] Final line\n"
        )
        max_ts = extract_max_timestamp(lrc_text)
        self.assertIsNotNone(max_ts)
        assert max_ts is not None
        self.assertAlmostEqual(max_ts, 195.2, places=1)

        # Song of 141s (Azteca) rejected against 195s lyrics (tolerance 15s)
        self.assertFalse(is_lyrics_duration_valid(lrc_text, 141.0))
        # Song of 190s accepted (195.2 <= 190.0 + 15.0 = 205.0)
        self.assertTrue(is_lyrics_duration_valid(lrc_text, 190.0))

    def test_get_lyrics_quality_filters_empty_headers(self) -> None:
        header_only = "[ar:Artist]\n[ti:Title]\n[al:Album]"
        self.assertEqual(get_lyrics_quality(header_only), 0)

        chinese_credits_only = (
            "[00:00.00-1] 作词 : Writer\n[00:00.00-1] 作曲 : Composer"
        )
        self.assertEqual(get_lyrics_quality(chinese_credits_only), 0)

        spoken_intro = "This song is called love"
        self.assertEqual(get_lyrics_quality(spoken_intro), 1)
        self.assertEqual(get_lyrics_quality("[Instrumental]"), 0)
