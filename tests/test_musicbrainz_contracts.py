import urllib.error
from contextlib import closing
from email.message import Message
from typing import Any
from unittest.mock import patch

import httpx
import musicbrainzngs

from sonora.core.http import SESSION
from sonora.services import musicbrainz as service
from tests.support import IsolatedTestCase

MBID = "11111111-2222-3333-4444-555555555555"
OTHER_MBID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def credited_recording() -> dict[str, Any]:
    return {
        "id": MBID,
        "title": "Recording",
        "artist-credit-phrase": "Primary",
        "artist-credit": [{"artist": {"id": OTHER_MBID, "name": "Primary"}}],
        "isrc-list": ["USAAA2100001", "USAAA2100002"],
        "artist-relation-list": [
            {"type": "composer", "artist": {"name": "Composer"}},
            {"type": "lyricist", "artist": {"name": "Lyricist"}},
            {"type": "writer", "artist": {"name": "Writer"}},
            {"type": "producer", "artist": {"name": "Producer"}},
            {"type": "remixer", "artist": {"name": "Remixer"}},
            {"type": "performer", "artist": {"name": "Performer"}},
            {"type": "writer", "artist": {}},
            {"type": "producer", "artist": {"name": "Producer"}},
        ],
        "work-relation-list": [{"work": {"id": OTHER_MBID}}],
    }


class TestMusicBrainzCatalogContracts(IsolatedTestCase):
    def test_malformed_nested_release_artist_is_a_schema_error(self) -> None:
        with self.assertRaisesRegex(TypeError, "JSON object"):
            service._score_musicbrainz_candidate(
                {"title": "Record", "artist-credit": [{"artist": None}]},
                "Primary",
                "Record",
            )

    def test_lower_scoring_discography_and_alternative_candidates_cannot_replace_the_best_release(
        self,
    ) -> None:
        lower_rank = {**self.release, "status": "Bootleg"}
        with patch.object(
            service, "fetch_artist_discography", return_value=[self.release, lower_rank]
        ):
            self.assertEqual(
                service.search_musicbrainz_release("Primary", "Record"), self.release
            )
        with (
            patch.object(service, "fetch_artist_discography", return_value=[]),
            patch.object(
                service,
                "_call_musicbrainz",
                side_effect=[
                    {"release-list": []},
                    {"release-list": [self.release, lower_rank]},
                ],
            ),
        ):
            self.assertEqual(
                service.search_musicbrainz_release(
                    "Primary", "Record (Official Audio)"
                ),
                self.release,
            )

    def test_absent_work_and_label_identifiers_remain_absent(self) -> None:
        recording = credited_recording()
        recording["work-relation-list"] = []
        with patch.object(
            service, "_call_musicbrainz", return_value={"recording": recording}
        ):
            details = service.fetch_musicbrainz_recording_details(MBID)
        assert details is not None
        self.assertIsNone(details["musicbrainz_workid"])
        with patch.object(
            service,
            "_call_musicbrainz",
            return_value={
                "release": {
                    "title": "Record",
                    "label-info-list": [
                        {"label": {"name": "Publisher"}, "catalog-number": "ABC"}
                    ],
                }
            },
        ):
            release = service.fetch_musicbrainz_release_details(MBID)
        assert release is not None
        self.assertEqual(release["label"], "Publisher")
        self.assertIsNone(release["label_mbid"])

    def test_invalid_track_positions_are_not_invented_from_provider_text(self) -> None:
        with patch.object(
            service,
            "_call_musicbrainz",
            return_value={
                "release": {
                    "medium-list": [
                        {"track-list": [{"position": "bad", "recording": {"id": MBID}}]}
                    ]
                }
            },
        ):
            self.assertEqual(service.fetch_album_track_mbids(MBID), {})

    def setUp(self) -> None:
        cache_lookup = patch.object(service, "get_cached_api", return_value=None)
        cache_lookup.start()
        self.addCleanup(cache_lookup.stop)
        self.release = {
            "id": MBID,
            "title": "Record",
            "status": "Official",
            "artist-credit": [{"artist": {"name": "Primary"}}],
            "medium-track-count": "10",
        }

    def test_release_ranking_penalizes_unwanted_editions_and_track_counts(self) -> None:
        cases: tuple[tuple[dict[str, object], int | None, float], ...] = (
            ({}, None, 125),
            ({}, 10, 165),
            ({}, 9, 120),
            ({}, 5, 65),
            ({"disambiguation": "clean"}, None, 120),
            ({"disambiguation": "karaoke"}, None, 60),
            ({"disambiguation": "instrumental"}, None, 60),
            ({"status": "Bootleg"}, None, 90),
            ({"status": ""}, None, 115),
            ({"title": "Unrelated Record"}, None, 0),
            ({"medium-track-count": "unknown"}, 10, 125),
        )
        for changes, count, expected in cases:
            with self.subTest(changes=changes, count=count):
                self.assertEqual(
                    service._score_musicbrainz_candidate(
                        {**self.release, **changes}, "Primary", "Record", count
                    ),
                    expected,
                )

    def test_cached_discography_and_release_return_without_network(self) -> None:
        with patch.object(service, "get_cached_api", return_value=[self.release]):
            self.assertEqual(
                service.fetch_artist_discography("Primary"), [self.release]
            )
        with patch.object(service, "get_cached_api", return_value=self.release):
            self.assertEqual(
                service.search_musicbrainz_release("Primary", "Record"), self.release
            )
        self.assertIsNone(
            service.search_musicbrainz_release("Primary", "Unknown Album")
        )

    def test_discography_match_avoids_another_request(self) -> None:
        with (
            patch.object(
                service,
                "fetch_artist_discography",
                return_value=[{"title": "Other"}, self.release],
            ),
            patch.object(service, "_call_musicbrainz") as request,
        ):
            self.assertEqual(
                service.search_musicbrainz_release("Primary", "Record", 10),
                self.release,
            )
        request.assert_not_called()

    def test_discography_wrong_track_count_triggers_better_search(self) -> None:
        incomplete = {**self.release, "medium-track-count": "8"}
        with (
            patch.object(
                service, "fetch_artist_discography", return_value=[incomplete]
            ),
            patch.object(
                service,
                "_call_musicbrainz",
                return_value={"release-list": [self.release, {"title": "Unrelated"}]},
            ) as request,
        ):
            self.assertEqual(
                service.search_musicbrainz_release("Primary", "Record", 10),
                self.release,
            )
        request.assert_called_once()

    def test_alternative_title_search_and_expected_failures(self) -> None:
        for alternative in ([], [self.release]):
            with (
                self.subTest(alternative=alternative),
                patch.object(service, "fetch_artist_discography", return_value=[]),
                patch.object(
                    service,
                    "_call_musicbrainz",
                    side_effect=[{"release-list": []}, {"release-list": alternative}],
                ) as request,
            ):
                result = service.search_musicbrainz_release(
                    "Primary", "Record (Official Audio)"
                )
            self.assertEqual(result, self.release if alternative else None)
            self.assertEqual(request.call_count, 2)
            self.assertEqual(request.call_args.kwargs["release"], "Record")
        with (
            patch.object(service, "fetch_artist_discography", return_value=[]),
            patch.object(service, "_call_musicbrainz", side_effect=OSError("offline")),
        ):
            self.assertIsNone(service.search_musicbrainz_release("Primary", "Record"))

    def test_artist_search_preserves_provider_payload_and_handles_io_failure(
        self,
    ) -> None:
        artists = [{"id": MBID, "name": "Primary"}]
        with patch.object(
            service, "_call_musicbrainz", return_value={"artist-list": artists}
        ):
            self.assertEqual(service.search_musicbrainz_artists("Primary"), artists)
        with patch.object(service, "_call_musicbrainz", side_effect=OSError("offline")):
            self.assertEqual(service.search_musicbrainz_artists("Primary"), [])

    def test_response_retry_policy_uses_documented_http_statuses(self) -> None:
        self.assertTrue(
            service._is_retryable_mb_error(musicbrainzngs.NetworkError("offline"))
        )
        self.assertTrue(
            service._is_retryable_mb_error(musicbrainzngs.ResponseError("response"))
        )
        self.assertFalse(service._is_retryable_mb_error(ValueError("bug")))
        for status, retry in ((400, False), (404, False), (429, True), (503, True)):
            cause = urllib.error.HTTPError(
                "https://musicbrainz.org", status, "response", Message(), None
            )
            with self.subTest(status=status), closing(cause):
                self.assertEqual(
                    service._is_retryable_mb_error(
                        musicbrainzngs.ResponseError("response", cause=cause)
                    ),
                    retry,
                )
        with patch(
            "musicbrainzngs.set_useragent", side_effect=ValueError("invalid contact")
        ):
            service.init_musicbrainz()

    def test_cover_art_invalid_cached_and_redirected_responses(self) -> None:
        self.assertIsNone(service.fetch_cover_art_archive_url(""))
        with patch.object(
            service, "get_cached_api", return_value="https://archive.org/front.jpg"
        ):
            self.assertEqual(
                service.fetch_cover_art_archive_url(MBID),
                "https://archive.org/front.jpg",
            )
        with patch.object(
            SESSION,
            "head",
            return_value=httpx.Response(
                200, request=httpx.Request("HEAD", "https://archive.org/front.jpg")
            ),
        ):
            self.assertEqual(
                service.fetch_cover_art_archive_url(MBID),
                "https://archive.org/front.jpg",
            )
        with patch.object(SESSION, "head", return_value=httpx.Response(404)):
            self.assertIsNone(service.fetch_cover_art_archive_url(MBID))

    def test_recording_candidates_without_artist_or_valid_id_are_rejected(self) -> None:
        for candidates in (
            [{"id": MBID, "title": "Recording"}],
            [
                {
                    "id": "invalid",
                    "title": "Recording",
                    "artist-credit": [{"artist": {"name": "Primary"}}],
                }
            ],
        ):
            with (
                self.subTest(candidates=candidates),
                patch.object(
                    service,
                    "_call_musicbrainz",
                    return_value={"recording-list": candidates},
                ),
            ):
                self.assertIsNone(service.fetch_track_mbid("Primary", "Recording"))

    def test_recording_credits_preserve_all_isrcs_and_deduplicate_names(self) -> None:
        with patch.object(
            service,
            "_call_musicbrainz",
            return_value={"recording": credited_recording()},
        ):
            details = service.fetch_musicbrainz_recording_details(MBID)
        assert details is not None
        expected = {
            "composer": "Composer, Writer",
            "lyricist": "Lyricist, Writer",
            "producers": "Producer",
            "remixer": "Remixer",
            "musicbrainz_workid": OTHER_MBID,
            "musicbrainz_artistid": OTHER_MBID,
            "isrcs": ["USAAA2100001", "USAAA2100002"],
        }
        for field, value in expected.items():
            with self.subTest(field=field):
                self.assertEqual(details[field], value)

    def test_recording_empty_cached_and_invalid_requests(self) -> None:
        self.assertIsNone(service.fetch_musicbrainz_recording_details(""))
        with patch.object(
            service, "get_cached_api", return_value={"title": "Recording"}
        ):
            self.assertEqual(
                service.fetch_musicbrainz_recording_details(MBID),
                {"title": "Recording"},
            )
        with patch.object(service, "_call_musicbrainz", return_value={"recording": {}}):
            self.assertIsNone(service.fetch_musicbrainz_recording_details(MBID))

    def test_release_credits_and_multidisc_track_indexes(self) -> None:
        recording = credited_recording()
        release = {
            "title": "Record",
            "date": "2021",
            "country": "US",
            "barcode": "123",
            "artist-credit": [
                {
                    "artist": {
                        "id": OTHER_MBID,
                        "name": "Primary",
                        "sort-name": "Primary",
                    }
                }
            ],
            "release-group": {
                "id": MBID,
                "primary-type": "Album",
                "first-release-date": "2020",
            },
            "label-info-list": [
                {
                    "label": {"name": "Publisher", "id": OTHER_MBID},
                    "catalog-number": "PUB-1",
                }
            ],
            "medium-list": [
                {
                    "format": "CD",
                    "track-count": "1",
                    "title": "First disc",
                    "track-list": [
                        {"position": "1", "recording": recording},
                        {"position": "invalid"},
                    ],
                },
                {
                    "position": "2",
                    "track-count": "1",
                    "track-list": [
                        {"position": "1", "recording": {**recording, "id": OTHER_MBID}}
                    ],
                },
            ],
            "text-representation": {"language": "eng", "script": "Latn"},
        }
        with patch.object(
            service, "_call_musicbrainz", return_value={"release": release}
        ):
            details = service.fetch_musicbrainz_release_details(MBID)
            mbids = service.fetch_album_track_mbids(MBID)
        assert details is not None
        expected = {
            "label": "Publisher",
            "label_mbid": OTHER_MBID,
            "catalog_number": "PUB-1",
            "artist": "Primary",
            "total_tracks": 2,
            "total_discs": 2,
            "media": "CD",
            "language": "English",
            "script": "Latin",
            "release_country": "United States",
            "original_date": "2020",
        }
        for field, value in expected.items():
            with self.subTest(field=field):
                self.assertEqual(details[field], value)
        self.assertEqual(
            mbids,
            {(1, 1): MBID, "1-1": MBID, 1: MBID, (2, 1): OTHER_MBID, "2-1": OTHER_MBID},
        )
        indexes = details["tracks_by_disc_and_position"]
        assert isinstance(indexes, dict)
        self.assertEqual(indexes[(1, 1)]["disc_subtitle"], "First disc")
        self.assertEqual(indexes[(2, 1)]["musicbrainz_workid"], OTHER_MBID)
        self.assertEqual(indexes[(1, 1)]["composer"], "Composer, Writer")

    def test_release_empty_and_invalid_cached_requests(self) -> None:
        self.assertIsNone(service.fetch_musicbrainz_release_details(""))
        self.assertEqual(service.fetch_album_track_mbids(""), {})
        with patch.object(
            service, "get_cached_api", return_value={"tracks_by_disc_and_position": {}}
        ):
            self.assertEqual(
                service.fetch_musicbrainz_release_details(MBID),
                {"tracks_by_disc_and_position": {}},
            )
        with patch.object(service, "get_cached_api", return_value={(1, 1): MBID}):
            self.assertEqual(service.fetch_album_track_mbids(MBID), {(1, 1): MBID})
        with patch.object(service, "_call_musicbrainz", return_value={"release": {}}):
            self.assertIsNone(service.fetch_musicbrainz_release_details(MBID))
        with patch.object(
            service, "_call_musicbrainz", return_value={"release": {"medium-list": []}}
        ):
            details = service.fetch_musicbrainz_release_details(MBID)
        assert details is not None
        self.assertIsNone(details["total_tracks"])
        self.assertEqual(details["total_discs"], 1)

    def test_track_index_fallbacks_preserve_unknown_ids_and_missing_counts(
        self,
    ) -> None:
        for work_id in (None, "bad", MBID):
            release = {
                "medium-list": [
                    {
                        "track-list": [
                            {
                                "position": "1",
                                "recording": {
                                    "title": "Recording",
                                    "work-relation-list": [{"work": {"id": work_id}}],
                                },
                            }
                        ]
                    }
                ]
            }
            with self.subTest(work_id=work_id):
                details = service._release_track_indexes(release, "Primary", None)
                tracks = details["tracks_by_position"]
                assert isinstance(tracks, dict)
                self.assertEqual(tracks[1]["artist"], "Primary")
                self.assertEqual(
                    tracks[1]["musicbrainz_workid"], MBID if work_id == MBID else None
                )
                self.assertIsNone(tracks[1]["recording_mbid"])
                self.assertIsNone(details["total_tracks"])
