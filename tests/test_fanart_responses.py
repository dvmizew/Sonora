from unittest.mock import MagicMock, patch

from sonora.services.fanart import (
    download_fanart_image_bytes,
    fetch_fanart_album,
    fetch_fanart_artist,
    fetch_fanart_label,
)
from tests.service_support import ServiceResponseTestCase


class TestFanartResponses(ServiceResponseTestCase):
    @patch("sonora.core.http.SESSION.get")
    def test_fetch_fanart_artist(self, mock_get: MagicMock) -> None:
        mbid = "11111111-2222-3333-4444-555555555555"

        # 1. Missing api key returns None
        with patch("sonora.services.fanart.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(
                fanart_api_key=None, fanart_client_key=None
            )
            self.assertIsNone(fetch_fanart_artist(mbid))

        # 2. Happy path with valid payload
        with patch("sonora.services.fanart.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(
                fanart_api_key="valid_key", fanart_client_key="client_key"
            )
            mock_resp = MagicMock(status_code=200)
            mock_resp.json.return_value = {
                "name": "Test Artist",
                "hdmusiclogo": [
                    {"url": "https://assets.fanart.tv/hdlogo.png", "likes": "5"}
                ],
                "musiclogo": [
                    {"url": "https://assets.fanart.tv/logo.png", "likes": "1"}
                ],
                "artistbackground": [
                    {"url": "https://assets.fanart.tv/fanart.jpg", "likes": "10"}
                ],
                "musicbanner": [
                    {"url": "https://assets.fanart.tv/banner.jpg", "likes": "2"}
                ],
                "artistthumb": [
                    {"url": "https://assets.fanart.tv/thumb.jpg", "likes": "3"}
                ],
            }
            mock_get.return_value = mock_resp

            artwork = fetch_fanart_artist(mbid)
            self.assertIsNotNone(artwork)
            if artwork:
                self.assertIn("https://assets.fanart.tv/hdlogo.png", artwork.logo_urls)
                self.assertIn(
                    "https://assets.fanart.tv/fanart.jpg", artwork.background_urls
                )
                self.assertIn(
                    "https://assets.fanart.tv/banner.jpg", artwork.banner_urls
                )

        # 3. 404 response returns None
        with patch("sonora.services.fanart.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(
                fanart_api_key="valid_key", fanart_client_key=None
            )
            mock_resp = MagicMock(status_code=404)
            mock_get.return_value = mock_resp
            self.assertIsNone(fetch_fanart_artist(mbid))

        # 4. Invalid mbid returns None
        self.assertIsNone(fetch_fanart_artist("invalid-uuid-format"))

    @patch("sonora.core.http.SESSION.get")
    def test_fetch_fanart_album(self, mock_get: MagicMock) -> None:
        rg_mbid = "11111111-2222-3333-4444-555555555555"

        with patch("sonora.services.fanart.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(
                fanart_api_key="valid_key", fanart_client_key=None
            )
            mock_resp = MagicMock(status_code=200)
            mock_resp.json.return_value = {
                "albums": {
                    rg_mbid: {
                        "cdart": [
                            {"url": "https://assets.fanart.tv/cdart.png", "likes": "4"}
                        ],
                        "albumcover": [
                            {"url": "https://assets.fanart.tv/cover.jpg", "likes": "2"}
                        ],
                    }
                }
            }
            mock_get.return_value = mock_resp

            artwork = fetch_fanart_album(rg_mbid)
            self.assertIsNotNone(artwork)
            if artwork:
                self.assertIn("https://assets.fanart.tv/cdart.png", artwork.cdart_urls)
                self.assertIn("https://assets.fanart.tv/cover.jpg", artwork.cover_urls)

    @patch("sonora.core.http.SESSION.get")
    def test_fetch_fanart_label(self, mock_get: MagicMock) -> None:
        label_mbid = "11111111-2222-3333-4444-555555555555"

        with patch("sonora.services.fanart.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(
                fanart_api_key="valid_key", fanart_client_key=None
            )
            mock_resp = MagicMock(status_code=200)
            mock_resp.json.return_value = {
                "name": "Atlantic Records",
                "id": label_mbid,
                "musiclabel": [
                    {"url": "https://assets.fanart.tv/label.png", "likes": "3"}
                ],
            }
            mock_get.return_value = mock_resp

            artwork = fetch_fanart_label(label_mbid)
            self.assertIsNotNone(artwork)
            if artwork:
                self.assertIn("https://assets.fanart.tv/label.png", artwork.label_urls)

        # Missing API key returns None
        with patch("sonora.services.fanart.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(
                fanart_api_key=None, fanart_client_key=None
            )
            self.assertIsNone(fetch_fanart_label(label_mbid))

    @patch("sonora.core.http.SESSION.get")
    def test_download_fanart_image_bytes(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock(status_code=200, content=b"FANART_IMAGE_BYTES")
        mock_get.return_value = mock_resp
        fanart_image_bytes = download_fanart_image_bytes(
            "https://assets.fanart.tv/test.png"
        )
        self.assertEqual(fanart_image_bytes, b"FANART_IMAGE_BYTES")

        mock_get.return_value = MagicMock(status_code=404)
        self.assertIsNone(
            download_fanart_image_bytes("https://assets.fanart.tv/missing.png")
        )
        self.assertIsNone(download_fanart_image_bytes(None))
