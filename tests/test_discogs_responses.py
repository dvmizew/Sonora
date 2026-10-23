from unittest.mock import MagicMock, patch

import httpx

from sonora.services.discogs import (
    fetch_discogs_release_details,
    search_discogs_release,
)
from tests.service_support import ServiceResponseTestCase


class TestDiscogsResponses(ServiceResponseTestCase):
    @patch("sonora.services.discogs.SESSION.get")
    def test_release_credits_respect_documented_track_scopes(
        self, mock_get: MagicMock
    ) -> None:
        cases: tuple[tuple[str, tuple[str, ...], set[str]], ...] = (
            ("A1, B1", ("A1", "A2", "B1"), {"A1", "B1"}),
            ("A2 to B1", ("A1", "A2", "B1"), {"A2", "B1"}),
            ("A1 to Missing", ("A1", "A2", "B1"), set()),
            ("A1", ("A1", "A1", "B1"), set()),
            ("B1 to A1", ("A1", "A2", "B1"), set()),
        )
        for scope, positions, expected_positions in cases:
            response = MagicMock(status_code=200)
            response.json.return_value = {
                "extraartists": [
                    {"name": "Album Producer", "role": "Producer", "tracks": ""},
                    {"name": "Scoped Composer", "role": "Written-By", "tracks": scope},
                ],
                "tracklist": [
                    {"position": position, "title": f"Recording {index}"}
                    for index, position in enumerate(positions, 1)
                ],
            }
            mock_get.return_value = response
            with self.subTest(scope=scope, positions=positions):
                details = fetch_discogs_release_details(1)
                self.assertIsNotNone(details)
                assert details is not None
                self.assertEqual(details["producers"], "Album Producer")
                self.assertIsNone(details["composer"])
                for position in positions:
                    self.assertEqual(
                        details["track_credits"][position].get("composer"),
                        "Scoped Composer" if position in expected_positions else None,
                    )

    @patch("sonora.services.discogs.SESSION.get")
    def test_track_credit_indexes_retain_provider_identity(
        self, mock_get: MagicMock
    ) -> None:
        title_cases: tuple[tuple[str, str], ...] = (
            ("Second Song", "First Song"),
            ("Recording", "Recording"),
            ("", "Recording"),
        )
        for titles in title_cases:
            response = MagicMock(status_code=200)
            response.json.return_value = {
                "id": 1,
                "tracklist": [
                    {
                        "position": str(position),
                        "title": title,
                        "extraartists": [
                            {"name": f"Producer {position}", "role": "Producer"}
                        ],
                    }
                    for position, title in enumerate(titles, start=1)
                ],
            }
            mock_get.return_value = response
            with self.subTest(titles=titles):
                details = fetch_discogs_release_details(1)
                self.assertIsNotNone(details)
                assert details is not None
                credits = details["track_credits"]
                for position, title in enumerate(titles, start=1):
                    self.assertEqual(credits[str(position)]["title"], title)
                    self.assertEqual(
                        credits[str(position)]["producers"], f"Producer {position}"
                    )
                if titles[0] and titles[0] != titles[1]:
                    self.assertIs(credits[titles[0].lower()], credits["1"])
                if titles[0] == titles[1]:
                    self.assertNotIn(titles[0].lower(), credits)

    @patch("sonora.services.discogs.SESSION.get")
    def test_search_discogs_release(self, mock_get: MagicMock) -> None:
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
            "Artist", "Test Album", user_token="dummy_token"
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

    def test_search_discogs_without_token_returns_none(self) -> None:
        self.assertIsNone(search_discogs_release("Artist", "Album", user_token=None))

    @patch("sonora.services.discogs.SESSION.get")
    def test_fetch_discogs_release_details_without_token(
        self, mock_get: MagicMock
    ) -> None:
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
        assert details is not None
        self.assertEqual(details["id"], 12345)
        self.assertEqual(details["title"], "Public Release")
        mock_get.assert_called_once_with(
            "https://api.discogs.com/releases/12345", headers={}, timeout=10
        )

    def test_fetch_discogs_release_details_empty_id(self) -> None:
        self.assertIsNone(fetch_discogs_release_details("", user_token=None))

    @patch("sonora.services.discogs.SESSION.get")
    def test_discogs_error_handling(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = httpx.HTTPError("Discogs 401 Unauthorized")
        release_result = search_discogs_release(
            "Artist", "Album", user_token="bad_token"
        )
        self.assertIsNone(release_result)
