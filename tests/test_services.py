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

    @patch("syncedlyrics.search")
    def test_fetch_synced_lyrics_basic(self, mock_search):
        mock_search.return_value = "[00:12.34] Test lyric line"

        lyrics = fetch_synced_lyrics("Artist", "Title")
        self.assertEqual(lyrics, "[00:12.34] Test lyric line")
        mock_search.assert_called_once_with("Artist - Title", plain_only=False, synced_only=False, enhanced=False)

    @patch("syncedlyrics.search")
    def test_fetch_synced_lyrics_with_options(self, mock_search):
        mock_search.return_value = "<00:12.34> Enhanced lyric line"

        lyrics = fetch_synced_lyrics(
            "Artist",
            "Title",
            synced_only=True,
            enhanced=True,
            providers=["Lrclib"],
            lang="en"
        )
        self.assertEqual(lyrics, "<00:12.34> Enhanced lyric line")
        mock_search.assert_called_once_with(
            "Artist - Title",
            plain_only=False,
            synced_only=True,
            enhanced=True,
            providers=["Lrclib"],
            lang="en"
        )

    @patch("syncedlyrics.search")
    def test_fetch_synced_lyrics_raises_api_service_error_on_failure(self, mock_search):
        mock_search.side_effect = RuntimeError("Network timeout")
        with self.assertRaises(APIServiceError):
            fetch_synced_lyrics("Artist", "Title")

    def test_synced_lyrics_empty_query_returns_none(self):
        self.assertIsNone(fetch_synced_lyrics("", ""))

    @patch("musicbrainzngs.search_recordings")
    def test_fetch_track_mbid(self, mock_search_rec):
        mock_search_rec.return_value = {
            "recording-list": [{"id": "12345678-1234-1234-1234-123456789abc"}]
        }

        mbid = fetch_track_mbid("Artist", "Title")
        self.assertEqual(mbid, "12345678-1234-1234-1234-123456789abc")

    @patch("musicbrainzngs.search_releases")
    def test_search_musicbrainz_release(self, mock_search_rel):
        mock_search_rel.return_value = {
            "release-list": [{"id": "album-mbid-123", "title": "Lemonade"}]
        }
        release = search_musicbrainz_release("Beyoncé", "Lemonade")
        self.assertIsNotNone(release)
        if release:
            self.assertEqual(release["title"], "Lemonade")

    @patch("discogs_client.Client")
    def test_search_discogs_release(self, mock_discogs_cls):
        mock_client = MagicMock()
        mock_item = MagicMock()
        mock_item.id = 999
        mock_item.title = "Test Album"
        mock_item.year = 2024
        mock_client.search.return_value = [mock_item]
        mock_discogs_cls.return_value = mock_client

        res = search_discogs_release("Artist", "Album", user_token="dummy_token")
        self.assertIsNotNone(res)
        if res:
            self.assertEqual(res["id"], 999)

    def test_search_discogs_without_token_returns_none(self):
        self.assertIsNone(search_discogs_release("Artist", "Album", user_token=None))

    @patch("acoustid.lookup")
    @patch("acoustid.fingerprint_file")
    def test_lookup_acoustid(self, mock_fp, mock_lookup):
        mock_fp.return_value = (120.0, "fingerprint_data_str")
        mock_lookup.return_value = {}

        with patch("acoustid.parse_lookup_result") as mock_parse:
            mock_parse.return_value = [(0.95, "rec-id-123", "Title", "Artist")]
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


if __name__ == "__main__":
    unittest.main()
