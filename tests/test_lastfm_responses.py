from unittest.mock import MagicMock, patch

from sonora.services.lastfm import fetch_lastfm_tags
from tests.service_support import ServiceResponseTestCase


class TestLastfmResponses(ServiceResponseTestCase):
    @patch("sonora.core.http.SESSION.get")
    def test_fetch_lastfm_tags(self, mock_get: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "toptags": {"tag": [{"name": "pop"}, {"name": "rnb"}]}
        }
        mock_get.return_value = mock_response

        tags = fetch_lastfm_tags("Beyoncé", "Halo", api_key="dummy_lastfm_key")
        self.assertEqual(tags, ["Pop", "Rnb"])

    def test_lastfm_no_api_key_returns_empty(self) -> None:
        self.assertEqual(fetch_lastfm_tags("Artist", "Title", api_key=None), [])
