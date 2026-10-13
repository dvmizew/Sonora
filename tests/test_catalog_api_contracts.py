from typing import Any
from unittest.mock import patch

import httpx

from sonora.core.config import SonoraConfig
from sonora.core.http import SESSION
from sonora.services import deezer, discogs, itunes
from tests.support import IsolatedTestCase


class TestITunesCatalogContracts(IsolatedTestCase):
    def test_strict_album_token_policy_rejects_fuzzy_track_and_album_candidates(
        self,
    ) -> None:
        candidate = {
            "artistName": "Primary",
            "collectionName": "Long Album Recordings",
            "trackName": "Long Album Recordings",
            "collectionId": 1,
        }
        with (
            patch.object(itunes, "search_itunes", return_value=[candidate]),
            patch.object(
                itunes,
                "get_config",
                return_value=SonoraConfig(album_token_match_threshold=100),
            ),
        ):
            self.assertIsNone(
                itunes.fetch_itunes_track_metadata("Primary", "Long Album Recording")
            )
            self.assertIsNone(
                itunes.fetch_itunes_album_details("Primary", "Long Album Recording")
            )

    def test_cover_candidates_obey_configured_fuzzy_threshold_and_first_edition(
        self,
    ) -> None:
        candidate = {
            "artistName": "Primary",
            "collectionName": "Long Album Recordings",
            "artworkUrl100": "https://example.com/100x100bb.jpg",
        }
        with (
            patch.object(itunes, "search_itunes", return_value=[candidate]),
            patch.object(
                itunes,
                "get_config",
                return_value=SonoraConfig(album_match_threshold=100),
            ),
        ):
            self.assertIsNone(
                itunes.fetch_itunes_cover_art_url("Primary", "Long Album Recording")
            )
        candidate["collectionName"] = "Long Album Recording"
        with patch.object(itunes, "search_itunes", return_value=[candidate]):
            self.assertIsNone(
                itunes.fetch_itunes_cover_art_url("Primary", "Long Album Recording I")
            )

    def test_unknown_artist_queries_still_reject_wrong_titles_and_series(self) -> None:
        for candidate in ("Unrelated Album", "Record II"):
            with (
                self.subTest(candidate=candidate),
                patch.object(
                    itunes,
                    "search_itunes",
                    return_value=[
                        {
                            "collectionName": candidate,
                            "trackName": candidate,
                            "collectionId": 1,
                            "artworkUrl100": "https://example.com/100x100bb.jpg",
                        }
                    ],
                ),
            ):
                self.assertIsNone(itunes.fetch_itunes_cover_art_url("", "Record"))
                self.assertIsNone(itunes.fetch_itunes_track_metadata("", "Record"))
                self.assertIsNone(itunes.fetch_itunes_album_details("", "Record"))
        self.assertEqual(
            itunes._score_itunes_album(
                {**self.album, "collectionName": "Record (Deluxe)"},
                "Primary",
                "Record",
                "record",
            ),
            100,
        )

    def setUp(self) -> None:
        self.album = {
            "collectionId": 1,
            "artistName": "Primary",
            "collectionName": "Record",
            "trackCount": 10,
        }

    def test_album_count_and_explicitness_ranking(self) -> None:
        for changes, count, expected in (
            ({}, 10, 180),
            ({"trackCount": 9}, 10, 155),
            ({"trackCount": 1}, 10, 70),
            ({"collectionExplicitness": "explicit"}, None, 145),
            ({"trackCount": 6}, 10, 130),
        ):
            with self.subTest(changes=changes, count=count):
                self.assertEqual(
                    itunes._score_itunes_album(
                        {**self.album, **changes}, "Primary", "Record", "record", count
                    ),
                    expected,
                )

    def test_cached_search_and_album_metadata_require_no_http_call(self) -> None:
        with patch.object(itunes, "get_cached_api", return_value=[self.album]):
            self.assertEqual(itunes.search_itunes("Primary", "Record"), [self.album])
        with (
            patch.object(itunes, "search_itunes", return_value=[self.album]),
            patch.object(itunes, "get_cached_api", return_value={"genre": "Rock"}),
        ):
            self.assertEqual(
                itunes.fetch_itunes_album_details("Primary", "Record"),
                {"genre": "Rock"},
            )

    def test_empty_unmatched_or_identifierless_albums_are_rejected(self) -> None:
        for candidates in (
            [],
            [
                {
                    "collectionId": 1,
                    "artistName": "Unrelated",
                    "collectionName": "Record",
                }
            ],
            [{"artistName": "Primary", "collectionName": "Record"}],
            [{"artistName": "Primary", "collectionName": "Totally Different"}],
        ):
            with (
                self.subTest(candidates=candidates),
                patch.object(itunes, "search_itunes", return_value=candidates),
            ):
                self.assertIsNone(
                    itunes.fetch_itunes_album_details("Primary", "Record")
                )
                self.assertIsNone(
                    itunes.fetch_itunes_track_metadata("Primary", "Recording")
                )

    def test_cover_art_fallback_and_cdn_resolution_variants(self) -> None:
        for url in (
            "https://example.com/600x600.jpg",
            "https://example.com/original.jpg",
            None,
        ):
            with (
                self.subTest(url=url),
                patch.object(
                    itunes,
                    "search_itunes",
                    return_value=[{**self.album, "artworkUrl100": url}],
                ),
            ):
                expected = (
                    url.replace("600x600", "1400x1400bb") if url is not None else None
                )
                self.assertEqual(
                    itunes.fetch_itunes_cover_art_url("Primary", "Record"), expected
                )
        with patch.object(
            itunes,
            "search_itunes",
            side_effect=[
                [],
                [{**self.album, "artworkUrl100": "https://example.com/100x100bb.jpg"}],
            ],
        ) as search:
            self.assertEqual(
                itunes.fetch_itunes_cover_art_url("Primary feat. Guest", "Record"),
                "https://example.com/1400x1400bb.jpg",
            )
        self.assertEqual(search.call_count, 2)

    def test_lookup_handles_wrapper_records_unknown_positions_and_multiple_discs(
        self,
    ) -> None:
        tracks = [
            {"wrapperType": "collection"},
            {
                "wrapperType": "track",
                "trackName": "Recording",
                "trackNumber": 1,
                "discNumber": 1,
                "country": "US",
                "artistId": 2,
                "trackId": 3,
                "releaseDate": "2020-01-02T00:00:00Z",
                "trackExplicitness": "explicit",
            },
            {
                "wrapperType": "track",
                "trackName": "Recording Disc Two",
                "trackNumber": 1,
                "discNumber": 2,
            },
            {"wrapperType": "track", "trackName": ""},
        ]
        response = httpx.Response(
            200,
            json={"results": tracks},
            request=httpx.Request("GET", "https://itunes.apple.com/lookup"),
        )
        with (
            patch.object(itunes, "search_itunes", return_value=[self.album]),
            patch.object(itunes, "get_cached_api", return_value=None),
            patch.object(SESSION, "get", return_value=response),
        ):
            details = itunes.fetch_itunes_album_details("Primary", "Record")
        assert details is not None
        indexes = details["tracks_by_disc_and_position"]
        assert isinstance(indexes, dict)
        self.assertEqual(set(indexes), {(1, 1), (2, 1)})
        self.assertEqual(indexes[(1, 1)]["advisory"], "Explicit")
        self.assertEqual(indexes[(1, 1)]["date"], "2020-01-02")
        self.assertNotIn("release_country", indexes[(1, 1)])
        titles = details["tracks_by_title"]
        assert isinstance(titles, dict)
        self.assertEqual(set(titles), {"recording", "recording disc two"})
        with (
            patch.object(itunes, "search_itunes", return_value=[self.album]),
            patch.object(itunes, "get_cached_api", return_value=None),
            patch.object(SESSION, "get", side_effect=OSError("offline")),
        ):
            self.assertIsNone(itunes.fetch_itunes_album_details("Primary", "Record"))


class TestDeezerCatalogContracts(IsolatedTestCase):
    def test_cover_rank_ties_preserve_the_first_candidate(self) -> None:
        candidates = [
            {**self.album, "cover_xl": "https://example.com/first.jpg"},
            {**self.album, "cover_xl": "https://example.com/second.jpg"},
        ]
        with patch.object(
            SESSION, "get", return_value=httpx.Response(200, json={"data": candidates})
        ):
            self.assertEqual(
                deezer.fetch_deezer_cover_art_url("Primary", "Record"),
                "https://example.com/first.jpg",
            )

    def test_album_variant_and_duplicate_disc_positions_keep_distinct_recordings(
        self,
    ) -> None:
        candidate = {**self.album, "title": "Long Album Recordings"}
        tracks = [
            {"id": 1, "title": "First", "track_position": 1, "disk_number": 1},
            {"id": 2, "title": "Second", "track_position": 1, "disk_number": 2},
            {"id": 3, "title": "", "track_position": 2, "disk_number": 2},
        ]
        with patch.object(
            SESSION,
            "get",
            side_effect=[
                httpx.Response(200, json={"data": [candidate]}),
                httpx.Response(
                    200,
                    json={"title": "Long Album Recordings", "tracks": {"data": tracks}},
                ),
            ],
        ):
            details = deezer.fetch_deezer_album_details(
                "Primary", "Long Album Recording", expected_track_count=10
            )
        assert details is not None
        self.assertEqual(details["tracks_by_position"][1]["id"], 1)
        self.assertEqual(details["tracks_by_disc_and_position"][(2, 1)]["id"], 2)
        self.assertNotIn("", details["tracks_by_title"])
        self.assertEqual(details["tracks_by_disc_and_position"][(2, 2)]["id"], 3)

    def setUp(self) -> None:
        self.album = {
            "id": 1,
            "title": "Record",
            "artist": {"name": "Primary"},
            "nb_tracks": 10,
        }
        self.track = {"id": 2, "title": "Recording", "artist": {"name": "Primary"}}
        cache = patch.object(deezer, "get_cached_api", return_value=None)
        cache.start()
        self.addCleanup(cache.stop)

    def test_cover_art_empty_cached_fallback_and_provider_error(self) -> None:
        self.assertIsNone(deezer.fetch_deezer_cover_art_url("", "Record"))
        with patch.object(
            deezer, "get_cached_api", return_value="https://example.com/art.jpg"
        ):
            self.assertEqual(
                deezer.fetch_deezer_cover_art_url("Primary", "Record"),
                "https://example.com/art.jpg",
            )
        for fallback, expected in (
            (httpx.Response(404), None),
            (httpx.Response(200, json={"error": {"code": 4}}), None),
            (
                httpx.Response(
                    200,
                    json={
                        "data": [
                            {**self.album, "cover_big": "https://example.com/art.jpg"},
                            self.album,
                        ]
                    },
                ),
                "https://example.com/art.jpg",
            ),
        ):
            with (
                self.subTest(fallback=fallback),
                patch.object(
                    SESSION,
                    "get",
                    side_effect=[httpx.Response(200, json={"data": []}), fallback],
                ),
            ):
                self.assertEqual(
                    deezer.fetch_deezer_cover_art_url("Primary", "Record"), expected
                )
        for response in (
            httpx.Response(503),
            httpx.Response(200, json={"error": {"code": 4}}),
        ):
            with (
                self.subTest(response=response),
                patch.object(SESSION, "get", return_value=response),
            ):
                self.assertIsNone(
                    deezer.fetch_deezer_cover_art_url("Primary", "Record")
                )

    def test_album_search_rejects_empty_wrong_identity_and_missing_ids(self) -> None:
        self.assertIsNone(deezer.fetch_deezer_album_details("", "Record"))
        with patch.object(deezer, "get_cached_api", return_value=self.album):
            self.assertEqual(
                deezer.fetch_deezer_album_details("Primary", "Record"), self.album
            )
        scenarios: tuple[dict[str, Any], ...] = (
            {"data": []},
            {"error": {"code": 4}},
            {"data": [{"title": "Unrelated", "artist": {"name": "Other"}}]},
            {"data": [{"title": "Record", "artist": {"name": "Primary"}}]},
        )
        for payload in scenarios:
            with (
                self.subTest(payload=payload),
                patch.object(
                    SESSION, "get", return_value=httpx.Response(200, json=payload)
                ),
            ):
                self.assertIsNone(
                    deezer.fetch_deezer_album_details("Primary", "Record")
                )

    def test_album_count_ranking_prefers_full_album_over_single(self) -> None:
        for competing_count in (1, 8, 6):
            competitors = [
                {
                    **self.album,
                    "id": 9,
                    "nb_tracks": competing_count,
                    "record_type": "single",
                },
                self.album,
            ]
            with (
                self.subTest(competing_count=competing_count),
                patch.object(
                    SESSION,
                    "get",
                    side_effect=[
                        httpx.Response(200, json={"data": competitors}),
                        httpx.Response(
                            200,
                            json={"id": 1, "title": "Record", "tracks": {"data": []}},
                        ),
                    ],
                ) as get,
            ):
                details = deezer.fetch_deezer_album_details("Primary", "Record", 10)
            assert details is not None
            self.assertEqual(get.call_args.args[0], "https://api.deezer.com/album/1")

    def test_album_detail_status_and_provider_errors(self) -> None:
        for detail_response in (
            httpx.Response(404),
            httpx.Response(200, json={"error": {"code": 4}}),
        ):
            with (
                self.subTest(detail_response=detail_response),
                patch.object(
                    SESSION,
                    "get",
                    side_effect=[
                        httpx.Response(200, json={"data": [self.album]}),
                        detail_response,
                    ],
                ),
            ):
                self.assertIsNone(
                    deezer.fetch_deezer_album_details("Primary", "Record")
                )

    def test_track_contributor_roles_preserve_credits_and_deduplicate(self) -> None:
        payload = {
            **self.track,
            "album": {"title": "Record"},
            "track_position": "2",
            "disk_number": "1",
            "contributors": [
                {"name": "Guest", "role": "Featured"},
                {"name": "Producer", "role": "Producer"},
                {"name": "Composer", "role": "Composer"},
                {"name": "Writer", "role": "Author"},
                {"name": "Producer", "role": "Producer"},
                {"role": "Producer"},
                {"name": "Performer", "role": "Main"},
                None,
            ],
        }
        details = deezer._parse_deezer_track_payload(payload)
        self.assertEqual(
            (
                details["featured_artists"],
                details["producers"],
                details["composer"],
                details["lyricist"],
            ),
            ("Guest", "Producer", "Composer", "Writer"),
        )
        self.assertEqual(
            (details["track_position"], details["disk_number"], details["album"]),
            (2, 1, "Record"),
        )
        self.assertIsNone(
            deezer._parse_deezer_track_payload({"contributors": None})["producers"]
        )

    def test_track_search_and_isrc_lookup_failure_contracts(self) -> None:
        self.assertIsNone(deezer.fetch_deezer_track_details("", "Recording"))
        self.assertIsNone(deezer.fetch_deezer_track_by_isrc(" "))
        for function, arguments in (
            (deezer.fetch_deezer_track_details, ("Primary", "Recording")),
            (deezer.fetch_deezer_track_by_isrc, ("USAAA2100001",)),
        ):
            with (
                self.subTest(function=function.__name__),
                patch.object(deezer, "get_cached_api", return_value=self.track),
            ):
                self.assertEqual(function(*arguments), self.track)
            for response in (
                httpx.Response(503),
                httpx.Response(200, json={"error": {"code": 4}}),
                httpx.Response(200, json={"data": []}),
            ):
                with (
                    self.subTest(function=function.__name__, response=response),
                    patch.object(SESSION, "get", return_value=response),
                ):
                    self.assertIsNone(function(*arguments))
        for candidates in (
            [{"title": "Wrong", "artist": {"name": "Other"}}],
            [{"title": "Recording", "artist": {"name": "Primary"}}],
        ):
            with (
                self.subTest(candidates=candidates),
                patch.object(
                    SESSION,
                    "get",
                    return_value=httpx.Response(200, json={"data": candidates}),
                ),
            ):
                self.assertIsNone(
                    deezer.fetch_deezer_track_details("Primary", "Recording")
                )
        for response in (
            httpx.Response(404),
            httpx.Response(200, json={"error": {"code": 4}}),
        ):
            with (
                self.subTest(response=response),
                patch.object(
                    SESSION,
                    "get",
                    side_effect=[
                        httpx.Response(200, json={"data": [self.track]}),
                        response,
                    ],
                ),
            ):
                self.assertIsNone(
                    deezer.fetch_deezer_track_details("Primary", "Recording")
                )


class TestDiscogsCatalogContracts(IsolatedTestCase):
    def test_format_descriptions_require_strings_and_ignore_empty_descriptions(
        self,
    ) -> None:
        for descriptions in (None, "LP", [12]):
            with (
                self.subTest(descriptions=descriptions),
                patch.object(discogs, "get_cached_api", return_value=None),
                patch.object(
                    SESSION,
                    "get",
                    return_value=httpx.Response(
                        200,
                        json={
                            "formats": [{"name": "Vinyl", "descriptions": descriptions}]
                        },
                    ),
                ),
                self.assertRaisesRegex(TypeError, "array of strings"),
            ):
                discogs.fetch_discogs_release_details(1)
        with (
            patch.object(discogs, "get_cached_api", return_value=None),
            patch.object(
                SESSION,
                "get",
                return_value=httpx.Response(
                    200,
                    json={"formats": [{"name": "Vinyl", "descriptions": ["", "LP"]}]},
                ),
            ),
        ):
            release = discogs.fetch_discogs_release_details(1)
        assert release is not None
        self.assertEqual(release["media"], "Vinyl, LP")

    def test_format_ranking_penalizes_bootlegs_and_unrequested_editions(self) -> None:
        for formats, expected in (
            ("File", 130),
            (["CD"], 115),
            (["Vinyl"], 80),
            (["File", "Unofficial"], 95),
            (["File", "Instrumental"], 90),
            ([], 100),
        ):
            with self.subTest(formats=formats):
                self.assertEqual(
                    discogs._score_discogs_candidate(
                        {"title": "Record", "format": formats}, "Primary", "Record"
                    ),
                    expected,
                )
        self.assertEqual(
            discogs._score_discogs_candidate(
                {"title": "Wrong - Unrelated"}, "Primary", "Record"
            ),
            0,
        )

    def test_search_fallback_preserves_known_fields_when_details_unavailable(
        self,
    ) -> None:
        candidate = {
            "id": 1,
            "title": "Primary - Record",
            "label": ["Publisher"],
            "catno": "PUB-1",
            "barcode": ["123"],
            "format": ["File"],
            "country": "US",
            "year": 2020,
        }
        with (
            patch.object(discogs, "get_cached_api", return_value=None),
            patch.object(discogs, "fetch_discogs_release_details", return_value=None),
            patch.object(
                SESSION,
                "get",
                return_value=httpx.Response(200, json={"results": [candidate]}),
            ),
        ):
            release = discogs.search_discogs_release("Primary", "Record", "fixture")
        assert release is not None
        self.assertEqual(
            (
                release["label"],
                release["catalog_number"],
                release["barcode"],
                release["media"],
            ),
            ("Publisher", "PUB-1", "123", "File"),
        )

    def test_empty_cached_and_unmatched_search_contracts(self) -> None:
        self.assertIsNone(discogs.fetch_discogs_release_details(0))
        self.assertIsNone(
            discogs.search_discogs_release("Primary", "unknown", "fixture")
        )
        with patch.object(discogs, "get_cached_api", return_value={"id": 1}):
            self.assertEqual(discogs.fetch_discogs_release_details(1), {"id": 1})
            self.assertEqual(
                discogs.search_discogs_release("Primary", "Record", "fixture"),
                {"id": 1},
            )
        for candidates in (
            [],
            [{"title": "Wrong - Unrelated"}],
            [{"title": "Primary - Record"}],
        ):
            with (
                self.subTest(candidates=candidates),
                patch.object(discogs, "get_cached_api", return_value=None),
                patch.object(
                    SESSION,
                    "get",
                    return_value=httpx.Response(200, json={"results": candidates}),
                ),
            ):
                self.assertIsNone(
                    discogs.search_discogs_release("Primary", "Record", "fixture")
                )

    def test_release_and_track_credit_roles_are_complete(self) -> None:
        credits = [
            {"name": "Producer", "role": "Producer"},
            {"name": "Remixer", "role": "Remixed By"},
            {"name": "Composer", "role": "Written By"},
            {"name": "Performer", "role": "Vocals"},
            {},
            {"name": "Producer", "role": "Producer"},
        ]
        payload: dict[str, Any] = {
            "id": 1,
            "extraartists": credits,
            "tracklist": [
                {"position": "1", "title": "Recording", "extraartists": credits},
                {},
            ],
            "labels": [{"catno": "none"}],
            "formats": [{}],
            "identifiers": [
                {"type": "Matrix"},
                {"type": "Barcode", "value": ""},
                {"type": "Barcode", "value": "12-3 4"},
            ],
        }
        with (
            patch.object(discogs, "get_cached_api", return_value=None),
            patch.object(
                SESSION, "get", return_value=httpx.Response(200, json=payload)
            ),
        ):
            details = discogs.fetch_discogs_release_details(1)
        assert details is not None
        self.assertEqual(
            (details["producers"], details["remixer"], details["composer"]),
            ("Producer", "Remixer", "Composer"),
        )
        self.assertEqual(
            details["track_credits"]["1"],
            {"producers": "Producer", "remixer": "Remixer", "composer": "Composer"},
        )
        self.assertEqual(details["barcode"], "1234")
        self.assertIsNone(details["catalog_number"])
