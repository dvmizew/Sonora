import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import musicbrainzngs

from sonora.services.acoustid import fingerprint_audio_file, lookup_acoustid
from sonora.services.deezer import (
    fetch_deezer_album_details,
    fetch_deezer_track_details,
)
from sonora.services.discogs import (
    fetch_discogs_release_details,
    search_discogs_release,
)
from sonora.services.genius import fetch_genius_description
from sonora.services.itunes import fetch_itunes_cover_art_url
from sonora.services.lastfm import fetch_lastfm_tags
from sonora.services.lyrics import (
    clean_lyrics_text,
    fetch_synced_lyrics,
    process_track_lyrics,
)
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
            "results": [
                {
                    "collectionName": "Halo",
                    "artworkUrl100": "https://is1-ssl.mzstatic.com/image/thumb/100x100bb.jpg",
                }
            ]
        }
        mock_get.return_value = mock_response
        # No context manager or read() needed for requests

        url = fetch_itunes_cover_art_url("Beyoncé", "Halo", resolution=1400)
        self.assertEqual(
            url, "https://is1-ssl.mzstatic.com/image/thumb/1400x1400bb.jpg"
        )

    @patch("sonora.services.lyrics.get_cached_api", return_value=None)
    @patch("sonora.services.lyrics.syncedlyrics")
    def test_fetch_synced_lyrics_basic(self, mock_syncedlyrics, _mock_cache):
        mock_syncedlyrics.search.return_value = "[00:12.34] Test lyric line"

        lyrics = fetch_synced_lyrics("Artist", "Title")
        self.assertEqual(lyrics, "[00:12.34] Test lyric line")
        mock_syncedlyrics.search.assert_called_once_with(
            "artist - title", plain_only=False, synced_only=False, enhanced=False
        )

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
            lang="en",
        )
        self.assertEqual(lyrics, "<00:12.34> Enhanced lyric line")
        mock_syncedlyrics.search.assert_called_once_with(
            "artist - title",
            plain_only=False,
            synced_only=True,
            enhanced=True,
            providers=["Lrclib"],
            lang="en",
        )

    @patch("sonora.services.lyrics.get_cached_api", return_value=None)
    @patch("sonora.services.lyrics.syncedlyrics")
    def test_fetch_synced_lyrics_raises_api_service_error_on_failure(
        self, mock_syncedlyrics, _mock_cache
    ):
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
        mock_search_response = MagicMock()
        mock_search_response.status_code = 200
        mock_search_response.json.return_value = {
            "results": [
                {
                    "id": 999,
                    "title": "Artist - Test Album",
                    "year": 2024,
                }
            ]
        }
        # 2. Full release details response
        mock_release_response = MagicMock()
        mock_release_response.status_code = 200
        mock_release_response.json.return_value = {
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
        mock_get.side_effect = [mock_search_response, mock_release_response]

        release_result = search_discogs_release(
            "Artist", "Album", user_token="dummy_token"
        )
        self.assertIsNotNone(release_result)
        if release_result:
            self.assertEqual(release_result["id"], 999)
            self.assertEqual(release_result["artist_id"], "12345")
            self.assertEqual(release_result["label"], "Test Label")
            self.assertEqual(release_result["catalog_number"], "CAT-123")
            self.assertEqual(release_result["barcode"], "123456789012")
            self.assertEqual(release_result["media"], "CD, Album, Deluxe Edition")
            self.assertEqual(release_result["styles"], ["Trap", "Cloud Rap"])
            self.assertEqual(release_result["producers"], "Metro Boomin")
            self.assertEqual(release_result["composer"], "Composer Guy")
            self.assertEqual(
                release_result["track_credits"]["1"]["producers"], "Southside"
            )

    def test_search_discogs_without_token_returns_none(self):
        self.assertIsNone(search_discogs_release("Artist", "Album", user_token=None))

    @patch("sonora.services.discogs.get_cached_api", return_value=None)
    @patch("sonora.services.discogs.SESSION.get")
    def test_fetch_discogs_release_details_without_token(self, mock_get, _mock_cache):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": 12345,
            "title": "Public Release",
            "year": 2023,
            "artists": [{"id": 1, "name": "Artist"}],
        }
        mock_get.return_value = mock_response

        details = fetch_discogs_release_details(12345, user_token=None)
        self.assertIsNotNone(details)
        if details:
            self.assertEqual(details["id"], 12345)
            self.assertEqual(details["title"], "Public Release")
        mock_get.assert_called_once_with(
            "https://api.discogs.com/releases/12345", headers={}, timeout=10
        )

    def test_fetch_discogs_release_details_empty_id(self):
        self.assertIsNone(fetch_discogs_release_details("", user_token=None))

    @patch("sonora.services.acoustid.get_cached_api", return_value=None)
    @patch("sonora.services.acoustid.acoustid")
    def test_lookup_acoustid(self, mock_acoustid, _mock_cache):
        mock_acoustid.fingerprint_file.return_value = (120.0, "fingerprint_data_str")
        mock_acoustid.lookup.return_value = {}
        mock_acoustid.parse_lookup_result.return_value = [
            (0.95, "c8b03190-306c-4125-9b32-3f9d86d60a12", "Title", "Artist")
        ]

        mbid = lookup_acoustid(Path(__file__), api_key="dummy_key")
        self.assertEqual(mbid, "c8b03190-306c-4125-9b32-3f9d86d60a12")

    @patch("sonora.services.lastfm.get_cached_api", return_value=None)
    @patch("sonora.core.http.SESSION.get")
    def test_fetch_lastfm_tags(self, mock_get, _mock_cache):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "toptags": {"tag": [{"name": "pop"}, {"name": "rnb"}]}
        }
        mock_get.return_value = mock_response

        tags = fetch_lastfm_tags("Beyoncé", "Halo", api_key="dummy_lastfm_key")
        self.assertEqual(tags, ["Pop", "Rnb"])

    @patch("sonora.services.genius.get_cached_api", return_value=None)
    @patch("sonora.core.http.SESSION.get")
    def test_fetch_genius_description(self, mock_get, _mock_cache):
        mock_search_response = MagicMock()
        mock_search_response.json.return_value = {
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
        mock_song_response = MagicMock()
        mock_song_response.json.return_value = {
            "response": {
                "song": {
                    "id": 123,
                    "description": {"plain": "Song story description"},
                    "featured_artists": [{"name": "Feat Artist"}],
                    "producer_artists": [{"name": "Producer 1"}],
                    "writer_artists": [{"name": "Writer 1"}],
                    "release_date": "2023-01-01",
                }
            }
        }
        mock_get.side_effect = [mock_search_response, mock_song_response]

        song_description = fetch_genius_description(
            "Artist", "Title", api_token="dummy_genius_token"
        )
        self.assertEqual(song_description, "Song story description")

    @patch("sonora.services.genius.get_cached_api")
    def test_fetch_genius_song_details_cached(self, mock_cache):
        mock_cache.return_value = {
            "genius_song_id": 999,
            "description": "Cached story",
            "featured_artists": None,
            "producers": None,
            "writers": None,
            "release_date": None,
        }
        details = fetch_genius_description("Artist", "Title", api_token="dummy_token")
        self.assertEqual(details, "Cached story")

    def test_acoustid_no_api_key_returns_none(self):
        self.assertIsNone(lookup_acoustid(Path(__file__), api_key=""))

    @patch("sonora.services.acoustid.get_cached_api", return_value=None)
    @patch("sonora.services.acoustid.acoustid")
    def test_acoustid_low_score_returns_none(self, mock_acoustid, _mock_cache):
        mock_acoustid.fingerprint_file.return_value = (100.0, "fp_data")
        mock_acoustid.lookup.return_value = {}
        mock_acoustid.parse_lookup_result.return_value = [
            (0.5, "c8b03190-306c-4125-9b32-3f9d86d60a12", "Title", "Artist")
        ]

        self.assertIsNone(lookup_acoustid(Path(__file__), api_key="dummy_key"))

    def test_lastfm_no_api_key_returns_empty(self):
        self.assertEqual(fetch_lastfm_tags("Artist", "Title", api_key=None), [])

    @patch("sonora.services.genius.get_cached_api", return_value=None)
    @patch("sonora.core.http.SESSION.get")
    def test_genius_rejects_lyrics_unavailable_text(self, mock_get, _mock_cache):
        mock_search_response = MagicMock()
        mock_search_response.json.return_value = {
            "response": {"hits": [{"result": {"api_path": "/songs/1"}}]}
        }
        mock_song_response = MagicMock()
        mock_song_response.json.return_value = {
            "response": {
                "song": {
                    "description": {"plain": "Lyrics for this song are unavailable"}
                }
            }
        }
        mock_get.side_effect = [mock_search_response, mock_song_response]

        self.assertIsNone(
            fetch_genius_description("Artist", "Title", api_token="token")
        )

    @patch("sonora.services.musicbrainz.get_cached_api", return_value=None)
    @patch("sonora.services.musicbrainz.musicbrainzngs")
    def test_musicbrainz_error_handling(self, mock_mb, _mock_cache):
        mock_mb.search_recordings.side_effect = musicbrainzngs.MusicBrainzError(
            "MusicBrainz server 500"
        )
        with self.assertRaises(RuntimeError):
            fetch_track_mbid("Artist", "Title")

    @patch("sonora.services.discogs.get_cached_api", return_value=None)
    @patch("sonora.services.discogs.SESSION.get")
    def test_discogs_error_handling(self, mock_get, _mock_cache):
        mock_get.side_effect = httpx.HTTPError("Discogs 401 Unauthorized")
        release_result = search_discogs_release(
            "Artist", "Album", user_token="bad_token"
        )
        self.assertIsNone(release_result)

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
        self.assertEqual(
            cleaned, "[00:12.34] Valid lyric line\n[00:15.00] Second valid line"
        )

    def test_clean_lyrics_none(self):
        self.assertIsNone(clean_lyrics_text(None))

    def test_clean_lyrics_empty(self):
        self.assertEqual(clean_lyrics_text(""), "")

    def test_clean_lyrics_only_junk(self):
        self.assertEqual(
            clean_lyrics_text("12 Contributors\nYou might also like\n14Embed"), ""
        )

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

        release_result = fetch_deezer_album_details("Artist", "Album Title")
        self.assertIsNotNone(release_result)
        if release_result:
            self.assertEqual(release_result["label"], "Universal Music")
            self.assertEqual(release_result["barcode"], "123456789012")
            self.assertEqual(release_result["genre"], "Hip Hop")

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

        track_details = fetch_deezer_track_details("Artist Name", "Track Title")
        self.assertIsNotNone(track_details)
        if track_details:
            self.assertEqual(track_details["isrc"], "USUM71703861")
            self.assertEqual(track_details["bpm"], 128.0)

    @patch("sonora.services.musicbrainz.get_cached_api", return_value=None)
    @patch("sonora.services.musicbrainz.SESSION.head")
    def test_fetch_cover_art_archive_url(self, mock_head, _mock_cache):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.url = "https://ia8000.us.archive.org/cover.jpg"
        mock_head.return_value = mock_response

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
                    "artist-credit": [
                        {
                            "artist": {
                                "name": "Artist",
                                "id": "a1b2c3d4-1234-1234-1234-123456789abc",
                            }
                        }
                    ],
                }
            ]
        }
        release_result = search_musicbrainz_release("Artist", "Album Title")
        self.assertIsNotNone(release_result)
        if release_result:
            self.assertEqual(
                release_result["id"], "c8b03190-306c-4125-9b32-3f9d86d60a12"
            )
            self.assertEqual(release_result["country"], "US")

    @patch("sonora.services.lyrics.fetch_synced_lyrics")
    def test_process_track_lyrics_quality_upgrade(self, mock_fetch):
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_file = Path(tmp_dir) / "track.flac"
            audio_file.write_bytes(b"dummy")
            lrc_file = Path(tmp_dir) / "track.lrc"

            # 1. Enhanced (Quality 3) skips remote call when not force
            lrc_file.write_text("<00:01.00> Enhanced word synced", encoding="utf-8")
            lyrics, tag_type = process_track_lyrics(
                audio_file, "Artist", "Title", force=False
            )
            self.assertEqual(lyrics, "<00:01.00> Enhanced word synced")
            self.assertEqual(tag_type, "enhanced")
            mock_fetch.assert_not_called()

            # 2. Plain text (Quality 1) upgrades to Synced (Quality 2)
            lrc_file.write_text("Just plain lyrics", encoding="utf-8")
            mock_fetch.return_value = "[00:01.00] Synced line lyrics"
            lyrics, tag_type = process_track_lyrics(
                audio_file, "Artist", "Title", force=False
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
                audio_file, "Artist", "Title", force=False
            )
            self.assertEqual(lyrics, "<00:01.00> Word enhanced lyrics")
            self.assertEqual(tag_type, "enhanced")

    @patch("sonora.services.acoustid.acoustid.fingerprint_file")
    def test_fingerprint_in_memory_cache(self, mock_fp):
        mock_fp.return_value = (180.0, "AQADtEmSJEqiJE")
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_file = Path(tmp_dir) / "track.flac"
            audio_file.write_bytes(b"dummy audio data for fingerprint")

            dur1, fp1 = fingerprint_audio_file(audio_file)
            self.assertEqual(dur1, 180.0)
            self.assertEqual(fp1, "AQADtEmSJEqiJE")
            self.assertEqual(mock_fp.call_count, 1)

            # Second call hits in-memory cache without calling acoustid.fingerprint_file again
            dur2, fp2 = fingerprint_audio_file(audio_file)
            self.assertEqual(dur2, 180.0)
            self.assertEqual(fp2, "AQADtEmSJEqiJE")
            self.assertEqual(mock_fp.call_count, 1)


if __name__ == "__main__":
    unittest.main()
