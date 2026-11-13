"""
Unit and integration tests for Sonora artwork processing modules:
album cover art, artist banners/thumbnails, label logos, and CD art.
"""

import io
import os
from unittest.mock import MagicMock, patch

import httpx
from PIL import Image

from sonora.audio import art
from sonora.audio.art import (
    _find_artist_directory,
    process_album_cover_art,
    process_label_artwork,
)
from sonora.core.config import SonoraConfig
from sonora.core.http import SESSION
from sonora.services import fanart
from tests.fixtures import image_bytes
from tests.support import IsolatedTestCase

MBID = "11111111-2222-3333-4444-555555555555"


class TestAlbumCoverArt(IsolatedTestCase):
    def test_process_album_cover_art_disc_folder(self) -> None:
        album_dir = self.test_directory / "The Wall"
        disc_dir = album_dir / "CD1"
        disc_dir.mkdir(parents=True)
        cover_path = album_dir / "cover.jpg"
        Image.new("RGB", (600, 600)).save(cover_path)

        found = process_album_cover_art(disc_dir, "Pink Floyd", "The Wall")
        assert found is not None
        self.assertEqual(found.resolve(), cover_path.resolve())

    def test_process_album_cover_art_low_resolution_upgrade(self) -> None:
        album_dir = self.test_directory / "LowResAlbum"
        album_dir.mkdir(parents=True)
        cover_path = album_dir / "cover.jpg"

        low_res = Image.new("RGB", (300, 300), color="blue")
        low_res_buf = io.BytesIO()
        low_res.save(low_res_buf, format="JPEG")
        cover_path.write_bytes(low_res_buf.getvalue())

        hi_res = Image.new("RGB", (1400, 1400), color="blue")
        hi_res_buf = io.BytesIO()
        hi_res.save(hi_res_buf, format="JPEG")
        hi_res_bytes = hi_res_buf.getvalue()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = hi_res_bytes
        mock_resp.raise_for_status = MagicMock()

        with (
            patch(
                "sonora.audio.art.fetch_itunes_cover_art_url",
                return_value="https://itunes.com/art.jpg",
            ),
            patch("sonora.core.http.SESSION.get", return_value=mock_resp),
        ):
            found = process_album_cover_art(album_dir, "Artist", "Album", force=False)
            assert found is not None
            with Image.open(found) as img:
                self.assertEqual(img.size, (1400, 1400))

    def test_process_album_cover_art_quality_downgrade_protection(self) -> None:
        album_dir = self.test_directory / "HighResAlbum"
        album_dir.mkdir(parents=True)
        cover_path = album_dir / "cover.jpg"

        hi_res = Image.new("RGB", (1400, 1400), color="green")
        hi_res_buf = io.BytesIO()
        hi_res.save(hi_res_buf, format="JPEG")
        cover_path.write_bytes(hi_res_buf.getvalue())

        low_res = Image.new("RGB", (500, 500), color="green")
        low_res_buf = io.BytesIO()
        low_res.save(low_res_buf, format="JPEG")
        low_res_bytes = low_res_buf.getvalue()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = low_res_bytes
        mock_resp.raise_for_status = MagicMock()

        with (
            patch(
                "sonora.audio.art.fetch_itunes_cover_art_url",
                return_value="https://itunes.com/art_small.jpg",
            ),
            patch("sonora.core.http.SESSION.get", return_value=mock_resp),
        ):
            found = process_album_cover_art(album_dir, "Artist", "Album", force=True)
            assert found is not None
            with Image.open(found) as img:
                self.assertEqual(img.size, (1400, 1400))

    def test_candidate_artwork_chooses_best_low_resolution_fallback(self) -> None:
        for dimensions in (
            ((100, 100), (200, 200), (150, 150)),
            ((100, 100), (600, 600)),
        ):
            images = [image_bytes(size) for size in dimensions]
            responses = [
                httpx.Response(
                    200,
                    content=content,
                    request=httpx.Request("GET", "https://example.com/art.png"),
                )
                for content in images
            ]
            with (
                self.subTest(dimensions=dimensions),
                patch.object(
                    art,
                    "fetch_cover_art_archive_url",
                    return_value="https://example.com/caa.png",
                ),
                patch.object(
                    art,
                    "fetch_itunes_cover_art_url",
                    return_value="https://example.com/itunes.png",
                ),
                patch.object(
                    art,
                    "fetch_deezer_cover_art_url",
                    return_value="https://example.com/deezer.png",
                ),
                patch.object(SESSION, "get", side_effect=responses),
            ):
                candidate = art._fetch_candidate_artwork("Primary", "Record", MBID)
            assert candidate is not None
            self.assertEqual(
                candidate[1], max(dimensions, key=lambda size: size[0] * size[1])
            )
            self.assertEqual(candidate[0], images[dimensions.index(candidate[1])])

    def test_cover_preview_and_upgrade_preserve_original_until_a_real_write(
        self,
    ) -> None:
        cover = self.test_directory / "cover.jpg"
        original = image_bytes((100, 100))
        cover.write_bytes(original)
        with patch.object(
            art,
            "_fetch_candidate_artwork",
            return_value=(image_bytes((600, 600)), (600, 600)),
        ):
            self.assertEqual(
                art.process_album_cover_art(
                    self.test_directory, "Primary", "Record", dry_run=True
                ),
                cover,
            )
            self.assertEqual(cover.read_bytes(), original)
            self.assertEqual(
                art.process_album_cover_art(self.test_directory, "Primary", "Record"),
                cover,
            )
        with Image.open(cover) as image:
            self.assertEqual(image.size, (600, 600))

    def test_cover_permissions_corruption_and_no_candidate(self) -> None:
        path = self.test_directory / "cover.jpg"
        with patch.object(os, "access", return_value=False):
            self.assertIsNone(
                art.process_album_cover_art(self.test_directory, "Primary", "Record")
            )
            path.write_bytes(b"corrupt artwork")
            self.assertIsNone(
                art.process_album_cover_art(self.test_directory, "Primary", "Record"),
            )
        with patch.object(
            art, "_fetch_candidate_artwork", return_value=(image_bytes(), (600, 600))
        ):
            self.assertEqual(
                art.process_album_cover_art(self.test_directory, "Primary", "Record"),
                path,
            )
        with Image.open(path) as image:
            self.assertEqual(image.size, (600, 600))


class TestArtistArtwork(IsolatedTestCase):
    def test_find_artist_directory_singles_hierarchy(self) -> None:
        artist_dir = self.test_directory / "3 Doors Down"
        singles_dir = artist_dir / "Singles"
        track_folder = singles_dir / "3 Doors Down - Here Without You"
        track_folder.mkdir(parents=True)

        found = _find_artist_directory(track_folder, "3 Doors Down")
        self.assertEqual(found.resolve(), artist_dir.resolve())

    def test_find_artist_directory_fallback(self) -> None:
        folder = self.test_directory / "Other Artist" / "Album"
        folder.mkdir(parents=True)
        found = _find_artist_directory(folder, "Unknown Artist")
        self.assertEqual(found.resolve(), folder.resolve())

    def test_fanart_artist_downloads_complete_assets_and_skips_fallback(self) -> None:
        assets = fanart.FanartArtistArtwork(
            logo_urls=("logo",), background_urls=("background",), thumb_urls=("thumb",)
        )
        with (
            patch.object(
                art, "get_config", return_value=SonoraConfig(fanart_api_key="fixture")
            ),
            patch.object(art, "fetch_fanart_artist", return_value=assets),
            patch.object(
                art,
                "download_fanart_image_bytes",
                side_effect=lambda url: image_bytes(),
            ),
            patch.object(art, "fetch_artist_images") as fallback,
        ):
            art.process_artist_artwork(self.test_directory, "Primary", MBID)
        fallback.assert_not_called()

    def test_theaudiodb_fallback_when_fanart_missing(self) -> None:
        with (
            patch.object(
                art, "get_config", return_value=SonoraConfig(fanart_api_key="fixture")
            ),
            patch.object(art, "fetch_fanart_artist", return_value=None),
            patch.object(
                art, "fetch_artist_images", return_value=(image_bytes(), image_bytes())
            ) as fallback,
        ):
            art.process_artist_artwork(self.test_directory, "Primary", MBID)
        fallback.assert_called_once_with("Primary", artist_mbid=MBID)
        self.assertTrue((self.test_directory / "artist.jpg").exists())
        self.assertTrue((self.test_directory / "banner.jpg").exists())


class TestLabelAndCdArt(IsolatedTestCase):
    def test_process_label_artwork(self) -> None:
        album_dir = self.test_directory / "Album Dir"
        album_dir.mkdir()

        mock_artwork = fanart.FanartLabelArtwork(
            label_urls=("https://assets.fanart.tv/label.png",)
        )
        with (
            patch.object(
                art, "get_config", return_value=SonoraConfig(fanart_api_key="valid_key")
            ),
            patch.object(
                art, "fetch_fanart_label", return_value=mock_artwork
            ) as mock_fetch,
            patch.object(
                art, "download_fanart_image_bytes", return_value=image_bytes()
            ),
        ):
            process_label_artwork(album_dir, label_mbid=MBID, label_name="Test Records")
            label_file = album_dir / "label.png"
            self.assertTrue(label_file.exists())
            self.assertEqual(label_file.read_bytes(), image_bytes())

            mock_fetch.reset_mock()
            process_label_artwork(album_dir, label_mbid=MBID, label_name="Test Records")
            mock_fetch.assert_not_called()

    def test_process_label_artwork_no_key(self) -> None:
        with patch.object(
            art, "get_config", return_value=SonoraConfig(fanart_api_key=None)
        ):
            other_dir = self.test_directory / "Other Dir"
            other_dir.mkdir()
            process_label_artwork(other_dir, label_mbid=MBID)
            self.assertFalse((other_dir / "label.png").exists())

    def test_cdart_transport_failure_does_not_create_an_asset(self) -> None:
        with (
            patch.object(
                art, "get_config", return_value=SonoraConfig(fanart_api_key="fixture")
            ),
            patch.object(art, "_fetch_candidate_artwork", return_value=None),
            patch.object(art, "fetch_fanart_album", side_effect=OSError("offline")),
        ):
            self.assertIsNone(
                art.process_album_cover_art(
                    self.test_directory,
                    "Primary",
                    "Record",
                    musicbrainz_releasegroup_id=MBID,
                )
            )
        self.assertFalse((self.test_directory / "cdart.png").exists())
