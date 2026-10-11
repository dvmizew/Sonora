from unittest.mock import MagicMock, patch

import httpx

from sonora.services.deezer import (
    fetch_deezer_album_details,
    fetch_deezer_track_by_isrc,
    fetch_deezer_track_details,
)
from tests.service_support import ServiceResponseTestCase


class TestDeezerResponses(ServiceResponseTestCase):
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
        self.assertIsNotNone(release_result)
        if release_result:
            self.assertEqual(release_result["label"], "Universal Music")
            self.assertEqual(release_result["barcode"], "123456789012")
            self.assertEqual(release_result["genre"], "Hip Hop")

    @patch("sonora.services.deezer.SESSION.get")
    def test_fetch_deezer_album_details_prefers_expected_tracks(
        self, mock_get: MagicMock
    ) -> None:
        mock_search = MagicMock()
        mock_search.status_code = 200
        mock_search.json.return_value = {
            "data": [
                {
                    "id": 111,
                    "title": "Revenge",
                    "artist": {"name": "XXXTENTACION"},
                    "nb_tracks": 1,
                    "record_type": "single",
                },
                {
                    "id": 777,
                    "title": "Revenge",
                    "artist": {"name": "XXXTENTACION"},
                    "nb_tracks": 7,
                    "record_type": "album",
                },
            ]
        }

        mock_album = MagicMock()
        mock_album.status_code = 200
        mock_album.json.return_value = {
            "title": "Revenge",
            "tracks": {
                "data": [
                    {"id": 1, "title": "Track 1", "track_position": 1, "disk_number": 1}
                ]
            },
        }
        mock_get.side_effect = [mock_search, mock_album]

        deezer_album_details = fetch_deezer_album_details(
            "XXXTENTACION", "Revenge", expected_track_count=7
        )
        self.assertIsNotNone(deezer_album_details)
        # Verify it requested album ID 777, not the 1-track single 111
        mock_get.assert_called_with("https://api.deezer.com/album/777", timeout=6)

    @patch("sonora.services.deezer.SESSION.get")
    def test_fetch_deezer_track_details(self, mock_get: MagicMock) -> None:
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
            "explicit_lyrics": True,
        }
        mock_get.side_effect = [mock_search, mock_track]

        track_details = fetch_deezer_track_details("Artist Name", "Track Title")
        self.assertIsNotNone(track_details)
        if track_details:
            self.assertEqual(track_details["isrc"], "USUM71703861")

    @patch("sonora.services.deezer.SESSION.get")
    def test_fetch_deezer_track_by_isrc(self, mock_get: MagicMock) -> None:
        # Happy path
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": 675779202,
            "title": "Cartieru' Pantelimon (feat. Nico)",
            "isrc": "ROCMA1911835",
            "explicit_lyrics": True,
            "track_position": 2,
            "disk_number": 1,
            "release_date": "2023-03-20",
            "contributors": [
                {"name": "B.U.G. Mafia", "role": "Main"},
                {"name": "Nico", "role": "Featured"},
            ],
            "artist": {"name": "B.U.G. Mafia"},
            "album": {"title": "Dupa Blocuri"},
        }
        mock_get.return_value = mock_response

        deezer_track_metadata = fetch_deezer_track_by_isrc("ROCMA1911835")
        self.assertIsNotNone(deezer_track_metadata)
        if deezer_track_metadata:
            self.assertEqual(
                deezer_track_metadata["title"], "Cartieru' Pantelimon (feat. Nico)"
            )
            self.assertEqual(deezer_track_metadata["isrc"], "ROCMA1911835")
            self.assertEqual(deezer_track_metadata["track_position"], 2)
            self.assertTrue(deezer_track_metadata["explicit_lyrics"])
            self.assertEqual(deezer_track_metadata["featured_artists"], "Nico")

        # Edge case: empty/whitespace returns None without network call
        self.assertIsNone(fetch_deezer_track_by_isrc(""))
        self.assertIsNone(fetch_deezer_track_by_isrc("   "))

        # Edge case: not found / 404
        mock_404 = MagicMock()
        mock_404.status_code = 404
        mock_get.return_value = mock_404
        self.assertIsNone(fetch_deezer_track_by_isrc("UNKNOWNISRC12"))

        # Edge case: network timeout/exception
        mock_get.side_effect = httpx.ConnectTimeout("Timeout")
        self.assertIsNone(fetch_deezer_track_by_isrc("TIMEOUTISRC12"))
