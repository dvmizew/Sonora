import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from sonora.core.config import SonoraConfig
from sonora.services.genius import fetch_genius_song_details
from sonora.services.lyrics import (
    clean_lyrics_text,
    detect_lrc_quality,
    extract_max_timestamp,
    fetch_synced_lyrics,
    get_lyrics_quality,
    is_lyrics_duration_valid,
    process_track_lyrics,
)
from tests.service_support import ServiceResponseTestCase


class TestLyricsResponses(ServiceResponseTestCase):
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
        # Verify negative cache sentinel was persisted
        mock_set_cache.assert_called_with(
            "lyrics:v4:True:artist without lyrics:instrumental track::[false, false, false, [], null, null, null]",
            "",
        )

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
            mock_fetch.return_value = "[00:01.00] Synced line lyrics"
            lyrics, tag_type = process_track_lyrics(
                audio_file, "Artist", "Title", force=False, duration=180.0
            )
            self.assertEqual(lyrics, "[00:01.00] Synced line lyrics")
            self.assertEqual(tag_type, "synced")
            self.assertEqual(
                lrc_file.read_text(encoding="utf-8"),
                "[00:01.00] Synced line lyrics",
            )

            # 3. Synced line lyrics (Quality 2) upgrades to Enhanced (Quality 3)
            mock_fetch.return_value = "<00:01.00> Word enhanced lyrics"
            lyrics, tag_type = process_track_lyrics(
                audio_file, "Artist", "Title", force=False, duration=180.0
            )
            self.assertEqual(lyrics, "<00:01.00> Word enhanced lyrics")
            self.assertEqual(tag_type, "enhanced")

            # 4. Downgrade protection: force=True with remote Plain text does NOT overwrite Synced
            mock_fetch.return_value = "Just plain text fallback from remote"
            lyrics, tag_type = process_track_lyrics(
                audio_file, "Artist", "Title", force=True, duration=180.0
            )
            self.assertEqual(lyrics, "<00:01.00> Word enhanced lyrics")
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
            "[00:12.34] Valid lyric line\n[00:15.00] Second valid line",
        )

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

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_audio = Path(tmp_dir) / "track.flac"
            test_audio.write_bytes(b"dummy")
            test_lrc = Path(tmp_dir) / "track.lrc"
            test_lrc.write_text(lrc_text, encoding="utf-8")
            # Mismatched duration triggers quality 0
            self.assertEqual(detect_lrc_quality(test_audio, audio_duration=120.0), 0)
            # Matching duration returns quality 2 (synced)
            self.assertEqual(detect_lrc_quality(test_audio, audio_duration=200.0), 2)

    def test_get_lyrics_quality_filters_empty_headers(self) -> None:
        header_only = "[ar:Artist]\n[ti:Title]\n[al:Album]"
        self.assertEqual(get_lyrics_quality(header_only), 0)

        chinese_credits_only = (
            "[00:00.00-1] 作词 : Writer\n[00:00.00-1] 作曲 : Composer"
        )
        self.assertEqual(get_lyrics_quality(chinese_credits_only), 0)

        placeholder_only = "This song is called placeholder"
        self.assertEqual(get_lyrics_quality(placeholder_only), 0)
