"""
Unit tests for Sonora external API service clients using mocks.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Guarantee src/ is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import unittest

from sonora.core.exceptions import APIServiceError
from sonora.services.acoustid import lookup_acoustid
from sonora.services.discogs import search_discogs_release
from sonora.services.genius import fetch_genius_description
from sonora.services.itunes import fetch_itunes_cover_art_url
from sonora.services.lastfm import fetch_lastfm_tags
from sonora.services.lyrics import fetch_synced_lyrics
from sonora.services.musicbrainz import fetch_track_mbid, search_musicbrainz_release


class TestServicesEngine(unittest.TestCase):
    @patch("urllib.request.urlopen")
    def test_fetch_itunes_cover_art_url(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = (
            b'{"results": [{"artworkUrl100": "https://is1-ssl.mzstatic.com/image/thumb/100x100bb.jpg"}]}'
        )
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        url = fetch_itunes_cover_art_url("Beyoncé", "Halo", resolution=1400)
        self.assertEqual(url, "https://is1-ssl.mzstatic.com/image/thumb/1400x1400bb.jpg")

    @patch("sonora.services.lyrics.syncedlyrics")
    def test_fetch_synced_lyrics_basic(self, mock_syncedlyrics):
        mock_syncedlyrics.search.return_value = "[00:12.34] Test lyric line"

        lyrics = fetch_synced_lyrics("Artist", "Title")
        self.assertEqual(lyrics, "[00:12.34] Test lyric line")
        mock_syncedlyrics.search.assert_called_once_with("Artist - Title", plain_only=False, synced_only=False, enhanced=True)

    @patch("sonora.services.lyrics.syncedlyrics")
    def test_fetch_synced_lyrics_with_options(self, mock_syncedlyrics):
        mock_syncedlyrics.search.return_value = "<00:12.34> Enhanced lyric line"

        lyrics = fetch_synced_lyrics(
            "Artist",
            "Title",
            synced_only=True,
            enhanced=True,
            providers=["Lrclib"],
            lang="en"
        )
        self.assertEqual(lyrics, "<00:12.34> Enhanced lyric line")
        mock_syncedlyrics.search.assert_called_once_with(
            "Artist - Title",
            plain_only=False,
            synced_only=True,
            enhanced=True,
            providers=["Lrclib"],
            lang="en"
        )

    @patch("sonora.services.lyrics.get_cached_api", return_value=None)
    @patch("sonora.services.lyrics.syncedlyrics")
    def test_fetch_synced_lyrics_raises_api_service_error_on_failure(self, mock_syncedlyrics, mock_cache):
        mock_syncedlyrics.search.side_effect = RuntimeError("Network timeout")
        with self.assertRaises(APIServiceError):
            fetch_synced_lyrics("FailArtist", "FailTitle")

    def test_synced_lyrics_empty_query_returns_none(self):
        self.assertIsNone(fetch_synced_lyrics("", ""))

    @patch("sonora.services.musicbrainz.musicbrainzngs")
    def test_fetch_track_mbid(self, mock_mb):
        mock_mb.search_recordings.return_value = {
            "recording-list": [{"id": "12345678-1234-1234-1234-123456789abc"}]
        }

        mbid = fetch_track_mbid("Artist", "Title")
        self.assertEqual(mbid, "12345678-1234-1234-1234-123456789abc")

    @patch("sonora.services.musicbrainz.musicbrainzngs")
    def test_search_musicbrainz_release(self, mock_mb):
        mock_mb.search_releases.return_value = {
            "release-list": [{"id": "album-mbid-123", "title": "Lemonade"}]
        }
        release = search_musicbrainz_release("Beyoncé", "Lemonade")
        self.assertIsNotNone(release)
        if release:
            self.assertEqual(release["title"], "Lemonade")

    @patch("sonora.services.discogs.discogs_client")
    def test_search_discogs_release(self, mock_discogs_mod):
        mock_client = MagicMock()
        mock_item = MagicMock()
        mock_item.id = 999
        mock_item.title = "Test Album"
        mock_item.year = 2024
        mock_client.search.return_value = [mock_item]
        mock_discogs_mod.Client.return_value = mock_client

        res = search_discogs_release("Artist", "Album", user_token="dummy_token")
        self.assertIsNotNone(res)
        if res:
            self.assertEqual(res["id"], 999)

    def test_search_discogs_without_token_returns_none(self):
        self.assertIsNone(search_discogs_release("Artist", "Album", user_token=None))

    @patch("sonora.services.acoustid.acoustid")
    def test_lookup_acoustid(self, mock_acoustid):
        mock_acoustid.fingerprint_file.return_value = (120.0, "fingerprint_data_str")
        mock_acoustid.lookup.return_value = {}
        mock_acoustid.parse_lookup_result.return_value = [(0.95, "rec-id-123", "Title", "Artist")]

        mbid = lookup_acoustid(Path(__file__), api_key="dummy_key")
        self.assertEqual(mbid, "rec-id-123")

    @patch("urllib.request.urlopen")
    def test_fetch_lastfm_tags(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = (
            b'{"toptags": {"tag": [{"name": "pop"}, {"name": "rnb"}]}}'
        )
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        tags = fetch_lastfm_tags("Beyoncé", "Halo", api_key="dummy_lastfm_key")
        self.assertEqual(tags, ["Pop", "Rnb"])

    @patch("urllib.request.urlopen")
    def test_fetch_genius_description(self, mock_urlopen):
        mock_resp_search = MagicMock()
        mock_resp_search.read.return_value = (
            b'{"response": {"hits": [{"result": {"api_path": "/songs/123"}}]}}'
        )
        mock_resp_search.__enter__.return_value = mock_resp_search

        mock_resp_song = MagicMock()
        mock_resp_song.read.return_value = (
            b'{"response": {"song": {"description": {"plain": "Song story description"}}}}'
        )
        mock_resp_song.__enter__.return_value = mock_resp_song

        mock_urlopen.side_effect = [mock_resp_search, mock_resp_song]

        desc = fetch_genius_description("Artist", "Title", api_token="dummy_genius_token")
        self.assertEqual(desc, "Song story description")

    def test_acoustid_no_api_key_returns_none(self):
        self.assertIsNone(lookup_acoustid(Path(__file__), api_key=""))

    @patch("sonora.services.acoustid.acoustid")
    def test_acoustid_low_score_returns_none(self, mock_acoustid):
        mock_acoustid.fingerprint_file.return_value = (100.0, "fp_data")
        mock_acoustid.lookup.return_value = {}
        mock_acoustid.parse_lookup_result.return_value = [(0.5, "rec-id-low", "Title", "Artist")]

        self.assertIsNone(lookup_acoustid(Path(__file__), api_key="dummy_key"))

    def test_lastfm_no_api_key_returns_empty(self):
        self.assertEqual(fetch_lastfm_tags("Artist", "Title", api_key=None), [])

    @patch("urllib.request.urlopen")
    def test_genius_rejects_lyrics_unavailable_text(self, mock_urlopen):
        mock_resp1 = MagicMock()
        mock_resp1.read.return_value = b'{"response": {"hits": [{"result": {"api_path": "/songs/1"}}]}}'
        mock_resp1.__enter__.return_value = mock_resp1

        mock_resp2 = MagicMock()
        mock_resp2.read.return_value = b'{"response": {"song": {"description": {"plain": "Lyrics for this song are unavailable"}}}}'
        mock_resp2.__enter__.return_value = mock_resp2

        mock_urlopen.side_effect = [mock_resp1, mock_resp2]
        self.assertIsNone(fetch_genius_description("Artist", "Title", api_token="token"))

    @patch("sonora.services.musicbrainz.musicbrainzngs")
    def test_musicbrainz_error_handling(self, mock_mb):
        mock_mb.search_releases.side_effect = Exception("MusicBrainz server 500")
        with self.assertRaises(APIServiceError):
            search_musicbrainz_release("Artist", "Album")

    @patch("sonora.services.discogs.discogs_client")
    def test_discogs_error_handling(self, mock_discogs_mod):
        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("Discogs 401 Unauthorized")
        mock_discogs_mod.Client.return_value = mock_client

        with self.assertRaises(APIServiceError):
            search_discogs_release("Artist", "Album", user_token="bad_token")

    def test_clean_lyrics_text(self):
        from sonora.services.lyrics import clean_lyrics_text

        dirty = (
            "[00:12.34] Valid lyric line\n"
            "12 Contributors\n"
            "You might also like\n"
            "[00:15.00] Second valid line\n"
            "https://genius.com/some-url\n"
            "14Embed"
        )
        cleaned = clean_lyrics_text(dirty)
        self.assertEqual(cleaned, "[00:12.34] Valid lyric line\n[00:15.00] Second valid line")


if __name__ == "__main__":
    unittest.main()
