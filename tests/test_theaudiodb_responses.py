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
