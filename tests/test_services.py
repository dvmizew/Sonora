import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from sonora.services.acoustid import lookup_acoustid
from sonora.services.deezer import (
    fetch_deezer_album_details,
    fetch_deezer_track_details,
)
from sonora.services.discogs import search_discogs_release
from sonora.services.genius import fetch_genius_description
from sonora.services.itunes import fetch_itunes_cover_art_url
from sonora.services.lastfm import fetch_lastfm_tags
from sonora.services.lyrics import clean_lyrics_text, fetch_synced_lyrics
from sonora.services.musicbrainz import (
    fetch_cover_art_archive_url,
    fetch_track_mbid,
    search_musicbrainz_release,
)


class TestServicesEngine(unittest.TestCase):
    @patch("sonora.services.itunes.get_cached_api", return_value=None)
    @patch("sonora.core.http.SESSION.get")
    def test_fetch_itunes_cover_art_url(self, mock_get, _mock_cache):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"collectionName": "Halo", "artworkUrl100": "https://is1-ssl.mzstatic.com/image/thumb/100x100bb.jpg"}]
        }
        mock_get.return_value = mock_response
        # No context manager or read() needed for requests

        url = fetch_itunes_cover_art_url("Beyoncé", "Halo", resolution=1400)
        self.assertEqual(url, "https://is1-ssl.mzstatic.com/image/thumb/1400x1400bb.jpg")

    @patch("sonora.services.lyrics.get_cached_api", return_value=None)
    @patch("sonora.services.lyrics.syncedlyrics")
    def test_fetch_synced_lyrics_basic(self, mock_syncedlyrics, _mock_cache):
        mock_syncedlyrics.search.return_value = "[00:12.34] Test lyric line"

        lyrics = fetch_synced_lyrics("Artist", "Title")
        self.assertEqual(lyrics, "[00:12.34] Test lyric line")
        mock_syncedlyrics.search.assert_called_once_with("artist - title", plain_only=False, synced_only=False, enhanced=False)

    @patch("sonora.services.lyrics.get_cached_api", return_value=None)
    @patch("sonora.services.lyrics.syncedlyrics")
    def test_fetch_synced_lyrics_with_options(self, mock_syncedlyrics, _mock_cache):
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
            "artist - title",
            plain_only=False,
            synced_only=True,
            enhanced=True,
            providers=["Lrclib"],
            lang="en"
        )

    @patch("sonora.services.lyrics.get_cached_api", return_value=None)
    @patch("sonora.services.lyrics.syncedlyrics")
    def test_fetch_synced_lyrics_raises_api_service_error_on_failure(self, mock_syncedlyrics, _mock_cache):
        mock_syncedlyrics.search.side_effect = RuntimeError("Network timeout")
        with self.assertRaises(RuntimeError):
            fetch_synced_lyrics("FailArtist", "FailTitle")

    def test_synced_lyrics_empty_query_returns_none(self):
        self.assertIsNone(fetch_synced_lyrics("", ""))

    @patch("sonora.services.musicbrainz.get_cached_api", return_value=None)
    @patch("sonora.services.musicbrainz.musicbrainzngs")
    def test_fetch_track_mbid(self, mock_mb, _mock_cache):
        mock_mb.search_recordings.return_value = {
            "recording-list": [
                {
                    "id": "12345678-1234-1234-1234-123456789abc",
                    "title": "Title",
                    "artist-credit": [{"artist": {"name": "Artist"}}],
                }
            ]
        }

        mbid = fetch_track_mbid("Artist", "Title")
        self.assertEqual(mbid, "12345678-1234-1234-1234-123456789abc")


    @patch("sonora.services.discogs.get_cached_api", return_value=None)
    @patch("sonora.services.discogs.SESSION.get")
    def test_search_discogs_release(self, mock_get, _mock_cache):
        # 1. Search response
        mock_search_resp = MagicMock()
        mock_search_resp.status_code = 200
        mock_search_resp.json.return_value = {
            "results": [
                {
                    "id": 999,
                    "title": "Artist - Test Album",
                    "year": 2024,
                }
            ]
        }
        # 2. Full release details response
        mock_rel_resp = MagicMock()
        mock_rel_resp.status_code = 200
        mock_rel_resp.json.return_value = {
            "id": 999,
            "title": "Test Album",
            "year": 2024,
            "released": "2024-05-10",
            "artists": [{"id": 12345, "name": "Artist"}],
            "genres": ["Hip Hop"],
            "styles": ["Trap", "Cloud Rap"],
            "country": "US",
            "labels": [{"name": "Test Label", "catno": "CAT-123"}],
            "formats": [{"name": "CD", "descriptions": ["Album", "Deluxe Edition"]}],
            "identifiers": [{"type": "Barcode", "value": "123456789012"}],
            "extraartists": [
                {"name": "Metro Boomin (2)", "role": "Producer"},
                {"name": "Composer Guy", "role": "Written-By"},
            ],
            "tracklist": [
                {
                    "position": "1",
                    "title": "Track One",
                    "extraartists": [{"name": "Southside", "role": "Producer"}],
                }
            ],
        }
        mock_get.side_effect = [mock_search_resp, mock_rel_resp]

        res = search_discogs_release("Artist", "Album", user_token="dummy_token")
        self.assertIsNotNone(res)
        if res:
            self.assertEqual(res["id"], 999)
            self.assertEqual(res["artist_id"], "12345")
            self.assertEqual(res["label"], "Test Label")
            self.assertEqual(res["catalog_number"], "CAT-123")
            self.assertEqual(res["barcode"], "123456789012")
            self.assertEqual(res["media"], "CD, Album, Deluxe Edition")
            self.assertEqual(res["styles"], ["Trap", "Cloud Rap"])
            self.assertEqual(res["producers"], "Metro Boomin")
            self.assertEqual(res["composer"], "Composer Guy")
            self.assertEqual(res["track_credits"]["1"]["producers"], "Southside")

    def test_search_discogs_without_token_returns_none(self):
        self.assertIsNone(search_discogs_release("Artist", "Album", user_token=None))

    @patch("sonora.services.acoustid.get_cached_api", return_value=None)
    @patch("sonora.services.acoustid.acoustid")
    def test_lookup_acoustid(self, mock_acoustid, _mock_cache):
        mock_acoustid.fingerprint_file.return_value = (120.0, "fingerprint_data_str")
        mock_acoustid.lookup.return_value = {}
        mock_acoustid.parse_lookup_result.return_value = [(0.95, "c8b03190-306c-4125-9b32-3f9d86d60a12", "Title", "Artist")]

        mbid = lookup_acoustid(Path(__file__), api_key="dummy_key")
        self.assertEqual(mbid, "c8b03190-306c-4125-9b32-3f9d86d60a12")

    @patch("sonora.services.lastfm.get_cached_api", return_value=None)
    @patch("sonora.core.http.SESSION.get")
    def test_fetch_lastfm_tags(self, mock_get, _mock_cache):
        mock_response = MagicMock()
        mock_response.json.return_value = {"toptags": {"tag": [{"name": "pop"}, {"name": "rnb"}]}}
        mock_get.return_value = mock_response

        tags = fetch_lastfm_tags("Beyoncé", "Halo", api_key="dummy_lastfm_key")
        self.assertEqual(tags, ["Pop", "Rnb"])

    @patch("sonora.core.http.SESSION.get")
    def test_fetch_genius_description(self, mock_get):
        mock_resp_search = MagicMock()
        mock_resp_search.json.return_value = {
            "response": {
                "hits": [
                    {
                        "result": {
                            "api_path": "/songs/123",
                            "title": "Title",
                            "primary_artist": {"name": "Artist"},
                        }
                    }
                ]
            }
        }
        mock_resp_song = MagicMock()
        mock_resp_song.json.return_value = {"response": {"song": {"description": {"plain": "Song story description"}}}}
        mock_get.side_effect = [mock_resp_search, mock_resp_song]

        desc = fetch_genius_description("Artist", "Title", api_token="dummy_genius_token")
        self.assertEqual(desc, "Song story description")

    def test_acoustid_no_api_key_returns_none(self):
        self.assertIsNone(lookup_acoustid(Path(__file__), api_key=""))

    @patch("sonora.services.acoustid.get_cached_api", return_value=None)
    @patch("sonora.services.acoustid.acoustid")
    def test_acoustid_low_score_returns_none(self, mock_acoustid, _mock_cache):
        mock_acoustid.fingerprint_file.return_value = (100.0, "fp_data")
        mock_acoustid.lookup.return_value = {}
        mock_acoustid.parse_lookup_result.return_value = [(0.5, "c8b03190-306c-4125-9b32-3f9d86d60a12", "Title", "Artist")]

        self.assertIsNone(lookup_acoustid(Path(__file__), api_key="dummy_key"))

    def test_lastfm_no_api_key_returns_empty(self):
        self.assertEqual(fetch_lastfm_tags("Artist", "Title", api_key=None), [])

    @patch("sonora.core.http.SESSION.get")
    def test_genius_rejects_lyrics_unavailable_text(self, mock_get):
        mock_resp1 = MagicMock()
        mock_resp1.json.return_value = {"response": {"hits": [{"result": {"api_path": "/songs/1"}}]}}
        mock_resp2 = MagicMock()
        mock_resp2.json.return_value = {"response": {"song": {"description": {"plain": "Lyrics for this song are unavailable"}}}}
        mock_get.side_effect = [mock_resp1, mock_resp2]

        self.assertIsNone(fetch_genius_description("Artist", "Title", api_token="token"))

    @patch("sonora.services.musicbrainz.get_cached_api", return_value=None)
    @patch("sonora.services.musicbrainz.musicbrainzngs")
    def test_musicbrainz_error_handling(self, mock_mb, _mock_cache):
        mock_mb.search_recordings.side_effect = Exception("MusicBrainz server 500")
        with self.assertRaises(RuntimeError):
            fetch_track_mbid("Artist", "Title")

    @patch("sonora.services.discogs.get_cached_api", return_value=None)
    @patch("sonora.services.discogs.SESSION.get")
    def test_discogs_error_handling(self, mock_get, _mock_cache):
        mock_get.side_effect = httpx.HTTPError("Discogs 401 Unauthorized")
        res = search_discogs_release("Artist", "Album", user_token="bad_token")
        self.assertIsNone(res)

    def test_clean_lyrics_text(self):
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

    def test_clean_lyrics_none(self):
        self.assertIsNone(clean_lyrics_text(None))

    def test_clean_lyrics_empty(self):
        self.assertEqual(clean_lyrics_text(""), "")

    def test_clean_lyrics_only_junk(self):
        self.assertEqual(clean_lyrics_text("12 Contributors\nYou might also like\n14Embed"), "")

    @patch("sonora.services.deezer.get_cached_api", return_value=None)
    @patch("sonora.services.deezer.SESSION.get")
    def test_fetch_deezer_album_details(self, mock_get, _mock_cache):
        mock_search = MagicMock()
        mock_search.status_code = 200
        mock_search.json.return_value = {"data": [{"id": 12345}]}

        mock_album = MagicMock()
        mock_album.status_code = 200
        mock_album.json.return_value = {
            "title": "Album Title",
            "label": "Universal Music",
            "upc": "123456789012",
            "release_date": "2024-01-01",
            "genres": {"data": [{"name": "Hip Hop"}]},
            "cover_xl": "https://e-cdns-images.dzcdn.net/images/cover/1000x1000.jpg",
        }
        mock_get.side_effect = [mock_search, mock_album]

        res = fetch_deezer_album_details("Artist", "Album Title")
        self.assertIsNotNone(res)
        if res:
            self.assertEqual(res["label"], "Universal Music")
            self.assertEqual(res["barcode"], "123456789012")
            self.assertEqual(res["genre"], "Hip Hop")

    @patch("sonora.services.deezer.get_cached_api", return_value=None)
    @patch("sonora.services.deezer.SESSION.get")
    def test_fetch_deezer_track_details(self, mock_get, _mock_cache):
        mock_search = MagicMock()
        mock_search.status_code = 200
        mock_search.json.return_value = {
            "data": [
                {
                    "id": 9876,
                    "title": "Track Title",
                    "artist": {"name": "Artist Name"},
                }
            ]
        }

        mock_track = MagicMock()
        mock_track.status_code = 200
        mock_track.json.return_value = {
            "isrc": "USUM71703861",
            "bpm": 128.0,
            "gain": -5.5,
            "explicit_lyrics": True,
        }
        mock_get.side_effect = [mock_search, mock_track]

        res = fetch_deezer_track_details("Artist Name", "Track Title")
        self.assertIsNotNone(res)
        if res:
            self.assertEqual(res["isrc"], "USUM71703861")
            self.assertEqual(res["bpm"], 128.0)

    @patch("sonora.services.musicbrainz.get_cached_api", return_value=None)
    @patch("sonora.services.musicbrainz.SESSION.head")
    def test_fetch_cover_art_archive_url(self, mock_head, _mock_cache):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.url = "https://ia8000.us.archive.org/cover.jpg"
        mock_head.return_value = mock_resp

        url = fetch_cover_art_archive_url("c8b03190-306c-4125-9b32-3f9d86d60a12")
        self.assertEqual(url, "https://ia8000.us.archive.org/cover.jpg")

    def test_fetch_cover_art_archive_invalid_uuid_returns_none(self):
        self.assertIsNone(fetch_cover_art_archive_url("invalid-uuid"))

    @patch("sonora.services.musicbrainz.get_cached_api", return_value=None)
    @patch("sonora.services.musicbrainz.musicbrainzngs")
    def test_search_musicbrainz_release(self, mock_mb, _mock_cache):
        mock_mb.search_releases.return_value = {
            "release-list": [
                {
                    "id": "c8b03190-306c-4125-9b32-3f9d86d60a12",
                    "title": "Album Title",
                    "date": "2024-01-01",
                    "country": "US",
                    "barcode": "123456789012",
                    "artist-credit": [{"artist": {"name": "Artist", "id": "a1b2c3d4-1234-1234-1234-123456789abc"}}],
                }
            ]
        }
        res = search_musicbrainz_release("Artist", "Album Title")
        self.assertIsNotNone(res)
        if res:
            self.assertEqual(res["id"], "c8b03190-306c-4125-9b32-3f9d86d60a12")
            self.assertEqual(res["country"], "US")


if __name__ == "__main__":
    unittest.main()
