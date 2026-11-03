from unittest.mock import MagicMock, patch

from sonora.services.theaudiodb import fetch_theaudiodb_track_details
from tests.service_support import ServiceResponseTestCase


class TestTheaudiodbResponses(ServiceResponseTestCase):
    @patch("sonora.services.theaudiodb.set_cached_api")
    @patch("sonora.services.theaudiodb.get_cached_api", return_value=None)
    @patch("sonora.services.theaudiodb.SESSION.get")
    def test_fetch_theaudiodb_track_details_negative_caching(
        self,
        mock_get: MagicMock,
        mock_get_cache: MagicMock,
        mock_set_cache: MagicMock,
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"track": None}
        mock_get.return_value = mock_resp

        result = fetch_theaudiodb_track_details("Unknown Artist", "Unknown Track")
        self.assertIsNone(result)
        mock_set_cache.assert_called_with(
            "theaudiodb_track:v3:unknown artist:unknown track::None", {}
        )

    @patch("sonora.services.theaudiodb.SESSION.get")
    def test_search_cache_still_filters_each_duration(
        self, mock_get: MagicMock
    ) -> None:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "track": [
                {
                    "strArtist": "Artist",
                    "strTrack": "Recording",
                    "intDuration": "100000",
                    "strGenre": "Electronic",
                }
            ]
        }
        mock_get.return_value = response
        cache: dict[str, object] = {}

        def set_cache(key: str, value: object, **_kwargs: object) -> None:
            cache[key] = "" if value is None else value

        with (
            patch("sonora.services.theaudiodb.get_cached_api", side_effect=cache.get),
            patch("sonora.services.theaudiodb.set_cached_api", side_effect=set_cache),
        ):
            self.assertEqual(
                fetch_theaudiodb_track_details("Artist", "Recording", duration=100),
                {
                    "music_video_url": None,
                    "mood": None,
                    "style": None,
                    "initial_key": None,
                    "rating": None,
                    "description": None,
                    "genre": "Electronic",
                },
            )
            self.assertIsNone(
                fetch_theaudiodb_track_details("Artist", "Recording", duration=200)
            )

        mock_get.assert_called_once()
