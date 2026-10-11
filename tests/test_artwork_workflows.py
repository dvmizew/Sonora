import io
from typing import Any
from unittest.mock import MagicMock, patch

from PIL import Image, ImageDraw

from sonora.audio.art import (
    check_image_similarity,
    process_artist_artwork,
    process_label_artwork,
)
from sonora.services.theaudiodb import fetch_artist_images
from tests.module_support import ModuleTestCase


class TestArtworkWorkflows(ModuleTestCase):
    def test_image_similarity(self) -> None:
        # Image 1: White background with black square in top-left
        image_1 = Image.new("RGB", (64, 64), color="white")
        draw_1 = ImageDraw.Draw(image_1)
        draw_1.rectangle([5, 5, 25, 25], fill="black")

        # Image 2: Exact copy of image 1
        image_2 = Image.new("RGB", (64, 64), color="white")
        draw_2 = ImageDraw.Draw(image_2)
        draw_2.rectangle([5, 5, 25, 25], fill="black")

        # Image 3: Completely distinct pattern (large filled ellipse in bottom-right)
        image_3 = Image.new("RGB", (64, 64), color="white")
        draw_3 = ImageDraw.Draw(image_3)
        draw_3.ellipse([30, 30, 60, 60], fill="black")

        buffer_1, buffer_2, buffer_3 = io.BytesIO(), io.BytesIO(), io.BytesIO()

        image_1.save(buffer_1, format="JPEG")
        image_2.save(buffer_2, format="JPEG")
        image_3.save(buffer_3, format="JPEG")

        image_bytes_1 = buffer_1.getvalue()
        image_bytes_2 = buffer_2.getvalue()
        image_bytes_3 = buffer_3.getvalue()

        self.assertTrue(
            check_image_similarity(image_bytes_1, image_bytes_2, threshold=0.8)
        )
        self.assertFalse(
            check_image_similarity(image_bytes_1, image_bytes_3, max_distance=6)
        )
        self.assertFalse(check_image_similarity(b"", image_bytes_2))

    @patch("sonora.services.theaudiodb.SESSION.get")
    def test_theaudiodb_service(self, mock_get: Any) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "artists": [
                {
                    "strArtistThumb": "http://example.com/thumb.jpg",
                    "strArtistBanner": "http://example.com/banner.jpg",
                }
            ]
        }
        mock_image_response = MagicMock()
        mock_image_response.status_code = 200
        mock_image_response.content = b"fakeimage"

        mock_get.side_effect = [
            mock_response,
            mock_image_response,
            mock_image_response,
            mock_response,
            mock_image_response,
            mock_image_response,
        ]

        thumbnail_bytes, banner_bytes = fetch_artist_images("21 Savage")
        self.assertEqual(thumbnail_bytes, b"fakeimage")
        self.assertEqual(banner_bytes, b"fakeimage")

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
        mock_download.return_value = b"LABEL_PNG_BYTES"

        with patch("sonora.audio.art.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(fanart_api_key="valid_key")
            process_label_artwork(
                album_dir, label_mbid=label_mbid, label_name="Test Records"
            )
            label_file = album_dir / "label.png"
            self.assertTrue(label_file.exists())
            self.assertEqual(label_file.read_bytes(), b"LABEL_PNG_BYTES")

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
