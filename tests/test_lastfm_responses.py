from unittest.mock import MagicMock, patch

import httpx

from sonora.core.http import SESSION
from sonora.services import lastfm
from sonora.services.lastfm import fetch_lastfm_tags
from tests.service_support import ServiceResponseTestCase


class TestLastfmResponses(ServiceResponseTestCase):
    def test_mbid_fallback_is_cached_under_the_original_identity(self) -> None:
        mbid = "11111111-2222-3333-4444-555555555555"
        cache: dict[str, object] = {}

        def save(key: str, value: object) -> None:
            cache[key] = value

        request = httpx.Request("GET", "https://ws.audioscrobbler.com/2.0/")
        responses: tuple[dict[str, object], ...] = (
            {"toptags": {"tag": []}},
            {"error": 6, "message": "Track not found"},
        )
        for first in responses:
            cache.clear()
            with (
                self.subTest(first=first),
                patch.object(lastfm, "get_cached_api", side_effect=cache.get),
                patch.object(lastfm, "set_cached_api", side_effect=save),
                patch.object(
                    SESSION,
                    "get",
                    side_effect=[
                        httpx.Response(200, json=first, request=request),
                        httpx.Response(
                            200,
                            json={"toptags": {"tag": [{"name": "rock"}]}},
                            request=request,
                        ),
                    ],
                ) as get,
            ):
                self.assertEqual(
                    fetch_lastfm_tags("Primary", "Recording", "key", mbid), ["Rock"]
                )
                self.assertEqual(
                    fetch_lastfm_tags("Primary", "Recording", "key", mbid), ["Rock"]
                )
                self.assertEqual(get.call_count, 2)

    def test_service_failure_is_not_cached_as_a_recording_miss(self) -> None:
        mbid = "11111111-2222-3333-4444-555555555555"
        request = httpx.Request("GET", "https://ws.audioscrobbler.com/2.0/")
        with (
            patch.object(lastfm, "get_cached_api", return_value=None),
            patch.object(lastfm, "set_cached_api") as save,
            patch.object(
                SESSION,
                "get",
                side_effect=[
                    httpx.Response(200, json={"toptags": {"tag": []}}, request=request),
                    httpx.Response(503, request=request),
                ],
            ),
        ):
            self.assertEqual(fetch_lastfm_tags("Primary", "Recording", "key", mbid), [])
        save.assert_not_called()

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
