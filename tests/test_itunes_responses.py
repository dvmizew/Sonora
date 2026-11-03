from unittest.mock import MagicMock, patch

from sonora.services.itunes import (
    fetch_itunes_album_details,
    fetch_itunes_cover_art_url,
)
from tests.service_support import ServiceResponseTestCase


class TestItunesResponses(ServiceResponseTestCase):
    def test_cached_album_artist_uses_only_uniform_track_credits(self) -> None:
        for collection_id, artists, expected in (
            ("101", ("Artist", "Artist"), "Artist"),
            ("102", ("Artist", "Guest"), None),
        ):
            cached = {
                "tracks_by_disc_and_position": {
                    (1, index): {"artist": artist}
                    for index, artist in enumerate(artists, 1)
                }
            }
            with (
                self.subTest(collection_id=collection_id),
                patch("sonora.services.itunes.get_cached_api", return_value=cached),
                patch("sonora.services.itunes.SESSION.get") as request,
            ):
                details = fetch_itunes_album_details(
                    "Artist", "Cached Album", collection_id=collection_id
                )
            request.assert_not_called()
            assert details is not None
            self.assertEqual(details.get("album_artist"), expected)

    @patch("sonora.services.itunes.SESSION.get")
    @patch("sonora.services.itunes.search_itunes")
    def test_known_collection_id_skips_album_search(
        self, mock_search: MagicMock, mock_get: MagicMock
    ) -> None:
        response = MagicMock()
        response.json.return_value = {
            "results": [
                {
                    "wrapperType": "collection",
                    "collectionId": 777,
                    "collectionName": "Revenge",
                    "artistName": "XXXTENTACION",
                    "trackCount": 1,
                },
                {
                    "wrapperType": "track",
                    "collectionId": 777,
                    "trackName": "Recording",
                    "artistName": "XXXTENTACION",
                    "trackNumber": 1,
                    "trackTimeMillis": 120000,
                },
            ]
        }
        mock_get.return_value = response

        details = fetch_itunes_album_details(
            "XXXTENTACION", "Revenge", expected_track_count=1, collection_id="777"
        )

        mock_search.assert_not_called()
        assert details is not None
        self.assertEqual(details["itunes_collectionid"], "777")
        self.assertEqual(details["total_tracks"], 1)
        self.assertEqual(mock_get.call_args.kwargs["params"]["id"], "777")

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
        assert itunes_album_details is not None
        self.assertEqual(itunes_album_details["itunes_collectionid"], "777")
        self.assertEqual(itunes_album_details["album_artist"], "XXXTENTACION")
