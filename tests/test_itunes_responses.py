from unittest.mock import MagicMock, patch

from sonora.services.itunes import (
    fetch_itunes_album_details,
    fetch_itunes_cover_art_url,
)
from tests.service_support import ServiceResponseTestCase


class TestItunesResponses(ServiceResponseTestCase):
    @patch("sonora.core.http.SESSION.get")
    def test_fetch_itunes_cover_art_url(self, mock_get: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "collectionName": "Halo",
                    "artistName": "Beyoncé",
                    "artworkUrl100": "https://is1-ssl.mzstatic.com/image/thumb/100x100bb.jpg",
                }
            ]
        }
        mock_get.return_value = mock_response

        url = fetch_itunes_cover_art_url("Beyoncé", "Halo", resolution=1400)
        self.assertEqual(
            url, "https://is1-ssl.mzstatic.com/image/thumb/1400x1400bb.jpg"
        )

    @patch("sonora.core.http.SESSION.get")
    def test_fetch_itunes_cover_art_url_primary_artist_fallback(
        self, mock_get: MagicMock
    ) -> None:
        empty_response = MagicMock()
        empty_response.json.return_value = {"results": []}

        success_response = MagicMock()
        success_response.json.return_value = {
            "results": [
                {
                    "artistName": "21 Savage",
                    "collectionName": "Savage Mode II",
                    "artworkUrl100": "https://is1-ssl.mzstatic.com/image/thumb/100x100-75.jpg",
                }
            ]
        }
        mock_get.side_effect = [empty_response, success_response]

        url = fetch_itunes_cover_art_url(
            "21 Savage & Metro Boomin", "Savage Mode II", resolution=1400
        )
        self.assertEqual(
            url, "https://is1-ssl.mzstatic.com/image/thumb/1400x1400bb.jpg"
        )

    @patch("sonora.services.itunes.SESSION.get")
    @patch("sonora.services.itunes.search_itunes")
    def test_fetch_itunes_album_details_prefers_expected_tracks(
        self, mock_search: MagicMock, mock_get: MagicMock
    ) -> None:
        mock_search.return_value = [
            {
                "collectionId": 111,
                "collectionName": "Revenge",
                "artistName": "XXXTENTACION",
                "trackCount": 1,
            },
            {
                "collectionId": 777,
                "collectionName": "Revenge",
                "artistName": "XXXTENTACION",
                "trackCount": 7,
            },
        ]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "wrapperType": "collection",
                    "collectionId": 777,
                    "collectionName": "Revenge",
                    "artistName": "XXXTENTACION",
                    "trackCount": 7,
                }
            ]
        }
        mock_get.return_value = mock_response

        itunes_album_details = fetch_itunes_album_details(
            "XXXTENTACION", "Revenge", expected_track_count=7
        )
        self.assertIsNotNone(itunes_album_details)
        if itunes_album_details:
            self.assertEqual(itunes_album_details["itunes_collectionid"], "777")
