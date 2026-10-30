from typing import Any
from unittest.mock import MagicMock, patch

from sonora.audio.art import (
    process_artist_artwork,
    process_label_artwork,
)
from sonora.services.theaudiodb import fetch_artist_images
from tests.fixtures import image_bytes
from tests.module_support import ModuleTestCase


class TestArtworkWorkflows(ModuleTestCase):
    @patch("sonora.services.theaudiodb.SESSION.get")
    def test_theaudiodb_service(self, mock_get: Any) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "artists": [
                {
                    "strArtist": "21 Savage",
                    "strArtistThumb": "http://example.com/thumb.jpg",
                    "strArtistBanner": "http://example.com/banner.jpg",
                }
            ]
        }
        mock_image_response = MagicMock()
        mock_image_response.status_code = 200
        mock_image_response.content = image_bytes()

        mock_get.side_effect = [
            mock_response,
            mock_image_response,
            mock_image_response,
            mock_response,
            mock_image_response,
            mock_image_response,
        ]

        thumbnail_bytes, banner_bytes = fetch_artist_images("21 Savage")
        self.assertEqual(thumbnail_bytes, image_bytes())
        self.assertEqual(banner_bytes, image_bytes())

        artist_folder = self.tmp_path / "21 Savage" / "Album"
        artist_folder.mkdir(parents=True, exist_ok=True)
        process_artist_artwork(artist_folder, "21 Savage")
        self.assertTrue((self.tmp_path / "21 Savage" / "artist.jpg").exists())
        self.assertTrue((self.tmp_path / "21 Savage" / "banner.jpg").exists())

    @patch("sonora.audio.art.fetch_fanart_label")
    @patch("sonora.audio.art.download_fanart_image_bytes")
    def test_process_label_artwork(
        self, mock_download: MagicMock, mock_fetch: MagicMock
    ) -> None:
        album_dir = self.tmp_path / "Album Dir"
        album_dir.mkdir()
        label_mbid = "11111111-2222-3333-4444-555555555555"

        mock_fetch.return_value = MagicMock(
            label_urls=("https://assets.fanart.tv/label.png",)
        )
        mock_download.return_value = image_bytes()

        with patch("sonora.audio.art.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(fanart_api_key="valid_key")
            process_label_artwork(
                album_dir, label_mbid=label_mbid, label_name="Test Records"
            )
            label_file = album_dir / "label.png"
            self.assertTrue(label_file.exists())
            self.assertEqual(label_file.read_bytes(), image_bytes())

            # Calling again skips if already exists
            mock_fetch.reset_mock()
            process_label_artwork(
                album_dir, label_mbid=label_mbid, label_name="Test Records"
            )
            mock_fetch.assert_not_called()

        # No key skips download
        with patch("sonora.audio.art.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(fanart_api_key=None)
            other_dir = self.tmp_path / "Other Dir"
            other_dir.mkdir()
            process_label_artwork(other_dir, label_mbid=label_mbid)
            self.assertFalse((other_dir / "label.png").exists())
