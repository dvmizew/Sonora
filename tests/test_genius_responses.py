from unittest.mock import MagicMock, patch

from sonora.services.genius import fetch_genius_song_details
from tests.service_support import ServiceResponseTestCase


class TestGeniusResponses(ServiceResponseTestCase):
    @patch("sonora.core.http.SESSION.get")
    def test_fetch_genius_description(self, mock_get: MagicMock) -> None:
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
                    "title": "Title",
                    "primary_artist": {"name": "Artist"},
                    "description": {"plain": "Song story description"},
                    "featured_artists": [{"name": "Feat Artist"}],
                    "producer_artists": [{"name": "Producer 1"}],
                    "writer_artists": [{"name": "Writer 1"}],
                    "release_date": "2023-01-01",
                }
            }
        }
        mock_get.side_effect = [mock_search_response, mock_song_response]

        song_details = fetch_genius_song_details(
            "Artist", "Title", api_token="dummy_genius_token"
        )
        self.assertIsNotNone(song_details)
        self.assertEqual(mock_get.call_args.kwargs["params"], {"text_format": "plain"})
        if song_details:
            self.assertEqual(song_details["description"], "Song story description")
            self.assertEqual(song_details["producers"], "Producer 1")
            self.assertEqual(song_details["writers"], "Writer 1")

    @patch("sonora.services.genius.get_cached_api")
    def test_fetch_genius_song_details_cached(self, mock_cache: MagicMock) -> None:
        mock_cache.return_value = {
            "genius_song_id": 999,
            "description": "Cached story",
            "featured_artists": None,
            "producers": None,
            "writers": None,
            "release_date": None,
        }
        details = fetch_genius_song_details("Artist", "Title", api_token="dummy_token")
        self.assertIsNotNone(details)
        assert details is not None
        self.assertEqual(details["description"], "Cached story")
