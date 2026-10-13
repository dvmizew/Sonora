import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
from PIL import Image

from sonora.audio import art
from sonora.core.config import SonoraConfig
from sonora.core.http import SESSION
from sonora.services import fanart
from tests.fixtures import image_bytes
from tests.support import IsolatedTestCase

MBID = "11111111-2222-3333-4444-555555555555"


class TestArtworkContracts(IsolatedTestCase):
    def test_missing_fanart_downloads_do_not_create_empty_assets(self) -> None:
        with (
            patch.object(
                art, "get_config", return_value=SonoraConfig(fanart_api_key="fixture")
            ),
            patch.object(art, "_fetch_candidate_artwork", return_value=None),
            patch.object(
                art,
                "fetch_fanart_album",
                return_value=fanart.FanartAlbumArtwork(cdart_urls=("disc",)),
            ),
            patch.object(art, "download_fanart_image_bytes", return_value=None),
        ):
            self.assertIsNone(
                art.process_album_cover_art(
                    self.test_directory,
                    "Primary",
                    "Record",
                    musicbrainz_releasegroup_id=MBID,
                )
            )
        with (
            patch.object(
                art, "get_config", return_value=SonoraConfig(fanart_api_key="fixture")
            ),
            patch.object(art, "fetch_fanart_artist", return_value=None),
            patch.object(
                art, "fetch_artist_images", return_value=(None, None)
            ) as fallback,
        ):
            art.process_artist_artwork(self.test_directory, "Primary", MBID)
        fallback.assert_called_once_with("Primary", artist_mbid=MBID)
        self.assertEqual(list(self.test_directory.iterdir()), [])

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

    def test_invalid_candidate_images_and_transport_failures_are_skipped(self) -> None:
        for failure in (b"not an image", OSError("offline")):
            response = httpx.Response(
                200,
                content=b"not an image",
                request=httpx.Request("GET", "https://example.com/art.png"),
            )
            with (
                self.subTest(failure=failure),
                patch.object(art, "fetch_cover_art_archive_url", return_value=None),
                patch.object(
                    art,
                    "fetch_itunes_cover_art_url",
                    return_value="https://example.com/art.png",
                ),
                patch.object(art, "fetch_deezer_cover_art_url", return_value=None),
                patch.object(
                    SESSION,
                    "get",
                    return_value=response,
                    side_effect=failure if isinstance(failure, OSError) else None,
                ),
            ):
                self.assertIsNone(art._fetch_candidate_artwork("Primary", "Record"))

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

    def test_missing_label_assets_and_label_transport_errors_create_no_files(
        self,
    ) -> None:
        for artwork, download in (
            (None, None),
            (fanart.FanartLabelArtwork(label_urls=("logo",)), None),
            (OSError("offline"), None),
        ):
            with (
                self.subTest(artwork=artwork),
                patch.object(
                    art,
                    "get_config",
                    return_value=SonoraConfig(fanart_api_key="fixture"),
                ),
                patch.object(
                    art,
                    "fetch_fanart_label",
                    return_value=artwork,
                    side_effect=artwork if isinstance(artwork, OSError) else None,
                ),
                patch.object(art, "download_fanart_image_bytes", return_value=download),
            ):
                art.process_label_artwork(self.test_directory, MBID)
            self.assertFalse((self.test_directory / "label.png").exists())

    def test_missing_artist_downloads_allow_theaudiodb_fallback(self) -> None:
        assets = fanart.FanartArtistArtwork(
            logo_urls=("logo",),
            background_urls=("background",),
            thumb_urls=("thumb",),
            banner_urls=("banner",),
        )
        with (
            patch.object(
                art, "get_config", return_value=SonoraConfig(fanart_api_key="fixture")
            ),
            patch.object(art, "fetch_fanart_artist", return_value=assets),
            patch.object(art, "download_fanart_image_bytes", return_value=None),
            patch.object(
                art, "fetch_artist_images", return_value=(None, None)
            ) as fallback,
        ):
            art.process_artist_artwork(self.test_directory, "Primary", MBID)
        fallback.assert_called_once_with("Primary", artist_mbid=MBID)
        self.assertEqual(list(self.test_directory.iterdir()), [])

    def test_image_modes_transparency_and_invalid_content(self) -> None:
        for mode in ("RGB", "RGBA", "LA", "P", "L"):
            with (
                self.subTest(mode=mode),
                art._load_normalized_image(image_bytes((20, 20), mode)) as image,
            ):
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (20, 20))
        self.assertFalse(art.check_image_similarity(b"invalid", image_bytes()))
        self.assertFalse(art.check_image_similarity(image_bytes(), b"invalid"))

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

    def test_cover_permissions_corruption_and_no_candidate(self) -> None:
        path = self.test_directory / "cover.jpg"
        with patch.object(os, "access", return_value=False):
            self.assertIsNone(
                art.process_album_cover_art(self.test_directory, "Primary", "Record")
            )
            path.write_bytes(b"corrupt artwork")
            self.assertEqual(
                art.process_album_cover_art(self.test_directory, "Primary", "Record"),
                path,
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

    def test_cdart_download_skip_and_expected_io_errors(self) -> None:
        for payload in (
            None,
            fanart.FanartAlbumArtwork(),
            fanart.FanartAlbumArtwork(cdart_urls=("https://example.com/disc.png",)),
        ):
            with (
                self.subTest(payload=payload),
                patch.object(
                    art,
                    "get_config",
                    return_value=SonoraConfig(fanart_api_key="fixture"),
                ),
                patch.object(art, "_fetch_candidate_artwork", return_value=None),
                patch.object(art, "fetch_fanart_album", return_value=payload),
                patch.object(
                    art, "download_fanart_image_bytes", return_value=b"disc artwork"
                ),
            ):
                art.process_album_cover_art(
                    self.test_directory,
                    "Primary",
                    "Record",
                    musicbrainz_releasegroup_id=MBID,
                )
            cdart = self.test_directory / "cdart.png"
            self.assertEqual(
                cdart.exists(), payload is not None and bool(payload.cdart_urls)
            )
        with (
            patch.object(
                art, "get_config", return_value=SonoraConfig(fanart_api_key="fixture")
            ),
            patch.object(art, "_fetch_candidate_artwork", return_value=None),
            patch.object(art, "fetch_fanart_album", side_effect=OSError("offline")),
        ):
            art.process_album_cover_art(
                self.test_directory,
                "Primary",
                "Record",
                musicbrainz_releasegroup_id=MBID,
            )

    def test_artist_directory_empty_and_root_hierarchy(self) -> None:
        self.assertEqual(
            art._find_artist_directory(self.test_directory, ""), self.test_directory
        )
        self.assertEqual(art._find_artist_directory(Path("/"), "Primary"), Path("/"))
        artist_directory = self.test_directory / "Primary - Discography"
        album_directory = artist_directory / "Record"
        album_directory.mkdir(parents=True)
        self.assertEqual(
            art._find_artist_directory(album_directory, "Primary"), artist_directory
        )

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
                art, "download_fanart_image_bytes", side_effect=lambda url: url.encode()
            ),
            patch.object(art, "fetch_artist_images") as fallback,
        ):
            art.process_artist_artwork(self.test_directory, "Primary", MBID)
        fallback.assert_not_called()
        for filename, expected in (
            ("logo.png", b"logo"),
            ("fanart.jpg", b"background"),
            ("artist.jpg", b"thumb"),
        ):
            self.assertEqual((self.test_directory / filename).read_bytes(), expected)

    def test_artist_banner_and_theaudiodb_fallback(self) -> None:
        with (
            patch.object(
                art, "get_config", return_value=SonoraConfig(fanart_api_key="fixture")
            ),
            patch.object(
                art,
                "fetch_fanart_artist",
                return_value=fanart.FanartArtistArtwork(banner_urls=("banner",)),
            ),
            patch.object(art, "download_fanart_image_bytes", return_value=b"banner"),
            patch.object(
                art,
                "fetch_artist_images",
                return_value=(b"thumbnail", b"fallback banner"),
            ),
        ):
            art.process_artist_artwork(self.test_directory, "Primary", MBID)
        self.assertEqual((self.test_directory / "banner.jpg").read_bytes(), b"banner")
        self.assertEqual(
            (self.test_directory / "artist.jpg").read_bytes(), b"thumbnail"
        )

    def test_artist_artwork_io_failures_and_read_only_directory(self) -> None:
        with (
            patch.object(os, "access", return_value=False),
            patch.object(art, "fetch_artist_images") as fetch,
        ):
            art.process_artist_artwork(self.test_directory, "Primary")
        fetch.assert_not_called()
        for artist in ("", "Various Artists"):
            art.process_artist_artwork(self.test_directory, artist)
        with patch.object(art, "fetch_artist_images", side_effect=OSError("offline")):
            art.process_artist_artwork(self.test_directory, "Primary")
        with (
            patch.object(
                art, "fetch_artist_images", return_value=(b"portrait", b"banner")
            ),
            patch.object(Path, "write_bytes", side_effect=OSError("disk full")),
        ):
            art.process_artist_artwork(self.test_directory, "Primary")
        self.assertEqual(list(self.test_directory.iterdir()), [])
        with (
            patch.object(
                art, "get_config", return_value=SonoraConfig(fanart_api_key="fixture")
            ),
            patch.object(art, "fetch_fanart_artist", side_effect=OSError("offline")),
            patch.object(art, "fetch_artist_images", return_value=(None, None)),
        ):
            art.process_artist_artwork(self.test_directory, "Primary", MBID)

    def test_label_logo_download_skip_and_expected_error(self) -> None:
        with (
            patch.object(
                art, "get_config", return_value=SonoraConfig(fanart_api_key="fixture")
            ),
            patch.object(
                art,
                "fetch_fanart_label",
                return_value=fanart.FanartLabelArtwork(label_urls=("logo",)),
            ),
            patch.object(art, "download_fanart_image_bytes", return_value=b"logo"),
        ):
            art.process_label_artwork(self.test_directory, MBID)
        self.assertEqual((self.test_directory / "label.png").read_bytes(), b"logo")
        with (
            patch.object(
                art, "get_config", return_value=SonoraConfig(fanart_api_key="fixture")
            ),
            patch.object(art, "fetch_fanart_label") as fetch,
        ):
            art.process_label_artwork(self.test_directory, MBID)
        fetch.assert_not_called()


class TestFanartProviderContracts(IsolatedTestCase):
    def test_download_non_success_status_and_image_url_validation(self) -> None:
        with patch.object(SESSION, "get", return_value=httpx.Response(404)):
            self.assertIsNone(
                fanart.download_fanart_image_bytes("https://example.com/absent")
            )
        self.assertEqual(
            fanart._extract_image_urls(
                [
                    {"url": "file:///private", "likes": 2},
                    {"url": 12},
                    {"url": "https://example.com", "likes": "bad"},
                ]
            ),
            ["https://example.com"],
        )

    def test_missing_credentials_cached_metadata_and_http_failures(self) -> None:
        for fetch in (
            fanart.fetch_fanart_artist,
            fanart.fetch_fanart_label,
            fanart.fetch_fanart_album,
        ):
            with self.subTest(fetch=fetch.__name__):
                self.assertIsNone(fetch(""))
                self.assertIsNone(fetch(MBID))
            with (
                self.subTest(fetch=fetch.__name__, cached=True),
                patch.object(fanart, "get_cached_api", return_value={}),
            ):
                self.assertIsNotNone(fetch(MBID, api_key="fixture"))
            for response in (
                httpx.Response(404),
                httpx.Response(
                    503, request=httpx.Request("GET", "https://webservice.fanart.tv")
                ),
            ):
                with (
                    self.subTest(fetch=fetch.__name__, response=response),
                    patch.object(fanart, "get_cached_api", return_value=None),
                    patch.object(SESSION, "get", return_value=response),
                ):
                    self.assertIsNone(
                        fetch(MBID, api_key="fixture", client_key="client")
                    )

    def test_album_payload_shapes_preserve_disc_and_cover_urls(self) -> None:
        descriptor = [{"url": "https://example.com/disc.png"}]
        payloads: tuple[dict[str, Any], ...] = (
            {"albums": {MBID: {"cdart": descriptor}}},
            {
                "albums": [
                    {"release_group_id": "other"},
                    None,
                    {"release_group_id": MBID, "cdart": descriptor},
                ]
            },
            {"cdart": descriptor},
            {"albums": {}},
        )
        for payload in payloads:
            with (
                self.subTest(payload=payload),
                patch.object(fanart, "get_cached_api", return_value=None),
                patch.object(
                    SESSION,
                    "get",
                    return_value=httpx.Response(
                        200,
                        json=payload,
                        request=httpx.Request("GET", "https://webservice.fanart.tv"),
                    ),
                ),
            ):
                artwork = fanart.fetch_fanart_album(MBID, api_key="fixture")
            assert artwork is not None
            self.assertEqual(
                artwork.cdart_urls,
                () if payload == {"albums": {}} else ("https://example.com/disc.png",),
            )
