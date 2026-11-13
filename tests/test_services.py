"""
Unit and integration tests for Sonora external metadata and API provider integrations:
MusicBrainz, Deezer, iTunes, Discogs, Genius, Shazam, AcoustID, TheAudioDB, Fanart.tv, and Last.fm.
"""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

from sonora.services.acoustid import lookup_acoustid
from sonora.services.deezer import (
    fetch_deezer_album_details,
    fetch_deezer_track_by_isrc,
)
from sonora.services.discogs import (
    fetch_discogs_release_details,
    search_discogs_release,
)
from sonora.services.fanart import (
    fetch_fanart_album,
)
from sonora.services.genius import (
    fetch_genius_song_details,
)
from sonora.services.itunes import (
    fetch_itunes_album_details,
    fetch_itunes_track_metadata,
)
from sonora.services.lastfm import fetch_lastfm_tags
from sonora.services.musicbrainz import (
    fetch_cover_art_archive_url,
    fetch_musicbrainz_release_details,
    fetch_track_mbid,
)
from sonora.services.theaudiodb import fetch_artist_images
from tests.fixtures import image_bytes
from tests.support import IsolatedTestCase

MBID = "11111111-2222-3333-4444-555555555555"


class TestMusicBrainzService(IsolatedTestCase):
    @patch("sonora.services.musicbrainz.musicbrainzngs")
    def test_fetch_track_mbid_uses_isrc_before_text_search(
        self, mock_mb: MagicMock
    ) -> None:
        mock_mb.search_recordings.return_value = {
            "recording-list": [
                {
                    "id": "12345678-1234-1234-1234-123456789abc",
                    "title": "Title",
                    "artist-credit": [{"artist": {"name": "Artist"}}],
                }
            ]
        }

        cache: dict[str, object] = {}
        with (
            patch("sonora.services.musicbrainz.get_cached_api", side_effect=cache.get),
            patch(
                "sonora.services.musicbrainz.set_cached_api",
                side_effect=lambda key, value, **_kwargs: cache.__setitem__(key, value),
            ),
        ):
            mbid = fetch_track_mbid("Artist", "Title", isrc="USABC2500001")
            cached_mbid = fetch_track_mbid("Artist", "Title", isrc="USABC2500001")

        self.assertEqual(mbid, "12345678-1234-1234-1234-123456789abc")
        self.assertEqual(cached_mbid, mbid)
        mock_mb.search_recordings.assert_called_once_with(isrc="USABC2500001", limit=5)

    @patch("sonora.services.musicbrainz.musicbrainzngs")
    def test_fetch_track_mbid_normalizes_display_isrc(self, mock_mb: MagicMock) -> None:
        mock_mb.search_recordings.return_value = {"recording-list": []}
        fetch_track_mbid("Artist", "Title", isrc="ISRC US-ABC-25-00001")
        self.assertEqual(
            mock_mb.search_recordings.call_args_list,
            [
                call(isrc="USABC2500001", limit=5),
                call(artist="Artist", recording="Title", limit=5),
            ],
        )

    @patch("sonora.services.musicbrainz.musicbrainzngs")
    def test_fetch_musicbrainz_release_details(self, mock_mb: MagicMock) -> None:
        mock_mb.get_release_by_id.return_value = {
            "release": {
                "id": MBID,
                "title": "Album Title",
                "date": "2024-05-15",
                "country": "US",
                "barcode": "012345678901",
                "artist-credit": [
                    {"artist": {"name": "Artist", "sort-name": "Artist, The"}}
                ],
                "label-info-list": [
                    {"label": {"name": "Record Label", "id": "label-id"}}
                ],
                "medium-list": [
                    {
                        "position": 1,
                        "track-count": 10,
                        "track-list": [
                            {
                                "number": "1",
                                "recording": {"id": "rec-1", "title": "Track One"},
                            }
                        ],
                    }
                ],
            }
        }
        details = fetch_musicbrainz_release_details(MBID)
        assert details is not None
        self.assertEqual(details["barcode"], "012345678901")
        self.assertEqual(details["label"], "Record Label")
        self.assertEqual(details["artist_sort"], "Artist, The")
        self.assertEqual(details["release_country"], "United States")

    @patch("sonora.core.http.SESSION.head")
    def test_fetch_cover_art_archive_url(self, mock_head: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.url = f"https://coverartarchive.org/release/{MBID}/front"
        mock_head.return_value = mock_resp
        url = fetch_cover_art_archive_url(MBID)
        self.assertEqual(url, f"https://coverartarchive.org/release/{MBID}/front")


class TestDeezerService(IsolatedTestCase):
    @patch("sonora.services.deezer.SESSION.get")
    def test_fetch_deezer_album_details(self, mock_get: MagicMock) -> None:
        mock_search = MagicMock()
        mock_search.status_code = 200
        mock_search.json.return_value = {
            "data": [
                {"id": 12345, "title": "Album Title", "artist": {"name": "Artist"}}
            ]
        }
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
        assert release_result is not None
        self.assertEqual(release_result["label"], "Universal Music")
        self.assertEqual(release_result["barcode"], "123456789012")
        self.assertEqual(release_result["genre"], "Hip Hop")

    @patch("sonora.services.deezer.SESSION.get")
    def test_fetch_deezer_track_by_isrc(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": 123456,
            "title": "Track Title",
            "bpm": 128.0,
            "duration": 210,
            "artist": {"name": "Artist"},
        }
        mock_get.return_value = mock_resp

        track_data = fetch_deezer_track_by_isrc("USABC2400001")
        assert track_data is not None
        self.assertEqual(track_data["title"], "Track Title")
        self.assertEqual(track_data["duration"], 210.0)


class TestItunesService(IsolatedTestCase):
    @patch("sonora.services.itunes.SESSION.get")
    @patch("sonora.services.itunes.search_itunes")
    def test_fetch_itunes_album_details(
        self, mock_search: MagicMock, mock_get: MagicMock
    ) -> None:
        mock_search.return_value = [
            {
                "collectionId": 987654321,
                "collectionName": "Album Title",
                "artistName": "Artist",
                "primaryGenreName": "Hip-Hop/Rap",
                "copyright": "Record Label",
                "artworkUrl100": "https://example.com/100x100bb.jpg",
                "trackCount": 12,
            }
        ]
        lookup_resp = MagicMock()
        lookup_resp.status_code = 200
        lookup_resp.json.return_value = {
            "results": [
                {
                    "wrapperType": "collection",
                    "collectionId": 987654321,
                    "collectionName": "Album Title",
                    "artistName": "Artist",
                    "primaryGenreName": "Hip-Hop/Rap",
                    "copyright": "Record Label",
                    "artworkUrl100": "https://example.com/100x100bb.jpg",
                    "trackCount": 12,
                }
            ]
        }
        mock_get.return_value = lookup_resp

        details = fetch_itunes_album_details("Artist", "Album Title")
        assert details is not None
        self.assertEqual(details["itunes_collectionid"], "987654321")
        self.assertEqual(details["genre"], "Hip-Hop/Rap")
        self.assertEqual(details["total_tracks"], 12)

    @patch("sonora.services.itunes.search_itunes")
    def test_fetch_itunes_track_details_advisory(self, mock_search: MagicMock) -> None:
        mock_search.return_value = [
            {
                "trackId": 1111,
                "trackName": "Track",
                "artistName": "Artist",
                "trackExplicitness": "explicit",
            }
        ]
        details = fetch_itunes_track_metadata("Artist", "Track")
        assert details is not None
        self.assertEqual(details["advisory"], "Explicit")


class TestDiscogsService(IsolatedTestCase):
    @patch("sonora.services.discogs.SESSION.get")
    def test_search_and_fetch_discogs_release(self, mock_get: MagicMock) -> None:
        search_resp = MagicMock()
        search_resp.status_code = 200
        search_resp.json.return_value = {
            "results": [
                {
                    "id": 55555,
                    "title": "Artist - Album Title",
                    "format": ["CD", "Album"],
                    "year": "2024",
                }
            ]
        }
        release_resp = MagicMock()
        release_resp.status_code = 200
        release_resp.json.return_value = {
            "id": 55555,
            "title": "Album Title",
            "artists": [{"name": "Artist"}],
            "genres": ["Hip Hop"],
            "styles": ["Boom Bap"],
            "labels": [{"name": "Def Jam"}],
            "tracklist": [
                {"position": "1", "title": "Intro"},
                {"position": "2", "title": "Track Two"},
            ],
        }
        mock_get.side_effect = [search_resp, release_resp]

        details = search_discogs_release(
            "Artist", "Album Title", user_token="test_token"
        )
        assert details is not None
        self.assertEqual(details["id"], 55555)

        full_details = fetch_discogs_release_details(55555, user_token="test_token")
        assert full_details is not None
        self.assertEqual(full_details["label"], "Def Jam")
        self.assertEqual(full_details["audio_track_count"], 2)
        self.assertIn("1", full_details["track_credits"])


class TestGeniusService(IsolatedTestCase):
    @patch("sonora.services.genius.SESSION.get")
    def test_fetch_genius_song_details(self, mock_get: MagicMock) -> None:
        search_resp = MagicMock()
        search_resp.status_code = 200
        search_resp.json.return_value = {
            "response": {
                "hits": [
                    {
                        "type": "song",
                        "result": {
                            "id": 78910,
                            "title": "Song Title",
                            "api_path": "/songs/78910",
                            "primary_artist": {"name": "Artist"},
                        },
                    }
                ]
            }
        }
        song_resp = MagicMock()
        song_resp.status_code = 200
        song_resp.json.return_value = {
            "response": {
                "song": {
                    "id": 78910,
                    "title": "Song Title",
                    "primary_artist": {"name": "Artist"},
                    "producer_artists": [{"name": "Metro Boomin"}],
                    "writer_artists": [{"name": "Artist"}, {"name": "Metro Boomin"}],
                }
            }
        }
        mock_get.side_effect = [search_resp, song_resp]

        details = fetch_genius_song_details(
            "Artist", "Song Title", api_token="test_token"
        )
        assert details is not None
        self.assertEqual(details["genius_song_id"], 78910)
        self.assertEqual(details["producers"], "Metro Boomin")
        assert isinstance(details["writers"], str)
        self.assertIn("Artist", details["writers"])


class TestAcoustidService(IsolatedTestCase):
    @patch("sonora.services.acoustid.fingerprint_audio_file")
    @patch("acoustid.lookup")
    def test_lookup_acoustid_fingerprint(
        self, mock_lookup: MagicMock, mock_fp: MagicMock
    ) -> None:
        mock_fp.return_value = (180.0, "dummy_fingerprint")
        mock_lookup.return_value = {
            "status": "ok",
            "results": [
                {
                    "id": "acoustid-uuid-1",
                    "score": 0.95,
                    "recordings": [{"id": MBID}],
                }
            ],
        }

        mbid = lookup_acoustid(Path("/tmp/audio.flac"), api_key="test_key")
        self.assertEqual(mbid, MBID)


class TestFanartAndTheAudioDBService(IsolatedTestCase):
    @patch("sonora.services.theaudiodb.SESSION.get")
    def test_fetch_artist_images(self, mock_get: MagicMock) -> None:
        meta_resp = MagicMock()
        meta_resp.status_code = 200
        meta_resp.json.return_value = {
            "artists": [
                {
                    "strArtist": "Artist",
                    "strArtistThumb": "https://theaudiodb.com/thumb.jpg",
                    "strArtistBanner": "https://theaudiodb.com/banner.jpg",
                }
            ]
        }
        img_resp = MagicMock()
        img_resp.status_code = 200
        img_resp.content = image_bytes()
        mock_get.side_effect = [meta_resp, img_resp, img_resp]

        thumb, banner = fetch_artist_images("Artist")
        self.assertEqual(thumb, image_bytes())
        self.assertEqual(banner, image_bytes())

    @patch("sonora.services.fanart.SESSION.get")
    def test_fetch_fanart_album_and_artist(self, mock_get: MagicMock) -> None:
        album_resp = MagicMock()
        album_resp.status_code = 200
        album_resp.json.return_value = {
            "albums": {
                MBID: {
                    "cdart": [{"url": "https://fanart.tv/cdart.png"}],
                }
            }
        }
        mock_get.return_value = album_resp
        album_artwork = fetch_fanart_album(MBID, api_key="test_key")
        assert album_artwork is not None
        self.assertEqual(album_artwork.cdart_urls, ("https://fanart.tv/cdart.png",))


class TestLastfmService(IsolatedTestCase):
    @patch("sonora.services.lastfm.SESSION.get")
    def test_fetch_lastfm_tags(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "toptags": {
                "tag": [
                    {"name": "hip-hop", "count": 100},
                    {"name": "rap", "count": 80},
                    {"name": "seen live", "count": 5},
                ]
            }
        }
        mock_get.return_value = mock_resp

        tags = fetch_lastfm_tags("Artist", title="Track", api_key="test_key")
        self.assertIn("Hip-Hop", tags)
        self.assertIn("Rap", tags)
        self.assertIn("Seen Live", tags)
