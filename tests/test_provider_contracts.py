import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx

from sonora.core.http import SESSION
from sonora.services import acoustid as fingerprint_service
from sonora.services import genius, lastfm, theaudiodb
from tests.support import IsolatedTestCase

MBID = "11111111-2222-3333-4444-555555555555"


class TestAcousticIdentityContracts(IsolatedTestCase):
    def test_json_duration_precision_preserves_a_valid_acoustic_candidate(self) -> None:
        recording_path = self.test_directory / "recording.flac"
        recording_path.touch()
        payload = {
            "status": "ok",
            "results": [
                {
                    "score": 1.0,
                    "recordings": [
                        {
                            "id": MBID,
                            "title": "Original Song",
                            "artists": [{"name": "Alpha"}],
                            "duration": 223.533,
                        }
                    ],
                }
            ],
        }
        with (
            patch.object(fingerprint_service, "_ACOUSTID_CACHE", {}),
            patch.object(
                fingerprint_service, "get_cached_api", side_effect=[None, payload]
            ) as cache_lookup,
            patch.object(fingerprint_service, "set_cached_api"),
            patch(
                "subprocess.run",
                return_value=subprocess.CompletedProcess(
                    [], 0, stdout=b'{"duration":221.54,"fingerprint":"123456"}'
                ),
            ) as execute,
        ):
            self.assertEqual(
                fingerprint_service.lookup_acoustid(
                    recording_path, "fixture", "Alpha", "Original Song"
                ),
                MBID,
            )
        self.assertIn("-json", execute.call_args.args[0])
        self.assertTrue(
            cache_lookup.call_args_list[0].args[0].startswith("fingerprint:v2:")
        )

    def test_invalid_json_is_reported_as_a_fingerprint_decoding_failure(self) -> None:
        for output in (b"", b"not JSON", b"\xff"):
            with (
                self.subTest(output=output),
                patch(
                    "subprocess.run",
                    return_value=subprocess.CompletedProcess([], 0, stdout=output),
                ),
                self.assertRaisesRegex(OSError, "invalid JSON"),
            ):
                fingerprint_service._fingerprint_file_with_timeout(
                    Path("recording.flac")
                )

    def test_json_fingerprint_rejects_nonfinite_or_empty_fields(self) -> None:
        for output in (
            b'{"duration":NaN,"fingerprint":"123456"}',
            b'{"duration":Infinity,"fingerprint":"123456"}',
            b'{"duration":-Infinity,"fingerprint":"123456"}',
            b'{"duration":0,"fingerprint":"123456"}',
            b'{"duration":10,"fingerprint":""}',
            b'{"duration":10,"fingerprint":"  "}',
            b'{"duration":10,"fingerprint":[1,2]}',
            b"{}",
        ):
            with (
                self.subTest(output=output),
                patch(
                    "subprocess.run",
                    return_value=subprocess.CompletedProcess([], 0, stdout=output),
                ),
                self.assertRaisesRegex(OSError, "incomplete fingerprint"),
            ):
                fingerprint_service._fingerprint_file_with_timeout(
                    Path("recording.flac")
                )

    def test_json_fingerprint_requires_an_object_envelope(self) -> None:
        with (
            patch(
                "subprocess.run",
                return_value=subprocess.CompletedProcess([], 0, stdout=b"[]"),
            ),
            self.assertRaisesRegex(TypeError, "fpcalc output must be a JSON object"),
        ):
            fingerprint_service._fingerprint_file_with_timeout(Path("recording.flac"))

    def test_lower_scoring_candidate_cannot_replace_the_best_supported_recording(
        self,
    ) -> None:
        payload = {
            "status": "ok",
            "results": [
                {
                    "score": score,
                    "recordings": [
                        {
                            "id": recording_id,
                            "title": "Recording",
                            "artists": [{"name": "Primary"}],
                        }
                    ],
                }
                for score, recording_id in (
                    (0.99, MBID),
                    (0.95, "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
                )
            ],
        }
        with (
            patch.object(
                fingerprint_service,
                "fingerprint_audio_file",
                return_value=(10, "fingerprint"),
            ),
            patch.object(fingerprint_service, "get_cached_api", return_value=None),
            patch.object(fingerprint_service._ACOUSTID_LIMITER, "wait"),
            patch("acoustid.lookup", return_value=payload),
        ):
            self.assertEqual(
                fingerprint_service.lookup_acoustid(
                    Path("recording.flac"), "fixture", "Primary", "Recording"
                ),
                MBID,
            )

    def test_fingerprint_output_requires_duration_and_signature(self) -> None:
        for output, expected in (
            (b'{"duration":10.5,"fingerprint":"123456"}', (10.5, "123456")),
            (b'{"fingerprint":"123456"}', None),
            (b'{"duration":"broken","fingerprint":"123456"}', None),
            (b'{"duration":10}', None),
        ):
            with (
                self.subTest(output=output),
                patch(
                    "subprocess.run",
                    return_value=subprocess.CompletedProcess([], 0, stdout=output),
                ) as execute,
            ):
                if expected is None:
                    with self.assertRaisesRegex(OSError, "incomplete fingerprint"):
                        fingerprint_service._fingerprint_file_with_timeout(
                            Path("recording.flac")
                        )
                else:
                    self.assertEqual(
                        fingerprint_service._fingerprint_file_with_timeout(
                            Path("recording.flac")
                        ),
                        expected,
                    )
            self.assertEqual(execute.call_args.kwargs["timeout"], 20)
        with self.assertRaises(FileNotFoundError):
            fingerprint_service.fingerprint_audio_file(
                self.test_directory / "missing.flac"
            )

    def test_fingerprint_cache_capacity_evicts_previous_entries(self) -> None:
        path = self.test_directory / "recording.flac"
        path.touch()
        old_entries = {
            (f"recording-{index}", 0, 0): (10.0, "old") for index in range(1024)
        }
        with (
            patch.object(fingerprint_service, "_ACOUSTID_CACHE", old_entries),
            patch.object(
                fingerprint_service,
                "_fingerprint_file_with_timeout",
                return_value=(10, "new"),
            ) as fingerprint,
        ):
            self.assertEqual(
                fingerprint_service.fingerprint_audio_file(path), (10, "new")
            )
            self.assertEqual(
                fingerprint_service.fingerprint_audio_file(path), (10, "new")
            )
            self.assertEqual(len(old_entries), 1)
        fingerprint.assert_called_once_with(path)

    def test_acoustid_candidate_identity_requires_supported_evidence(self) -> None:
        cases = (
            ("Recording", "Primary", 0.95, MBID),
            ("Different", "Someone Else", 0.99, None),
            ("Recording", None, 0.95, None),
            ("Different", None, 0.95, None),
            (None, None, 0.99, None),
            ("Recording", "Primary", 0.1, None),
        )
        for title, artist, score, expected in cases:
            recording: dict[str, Any] = {"id": MBID}
            if title is not None:
                recording["title"] = title
            if artist is not None:
                recording["artists"] = [{"name": artist}]
            payload = {
                "status": "ok",
                "results": [{"score": score, "recordings": [recording]}],
            }
            with (
                self.subTest(title=title, artist=artist, score=score),
                patch.object(
                    fingerprint_service,
                    "fingerprint_audio_file",
                    return_value=(10, "fingerprint"),
                ),
                patch.object(fingerprint_service, "get_cached_api", return_value=None),
                patch.object(fingerprint_service._ACOUSTID_LIMITER, "wait"),
                patch("acoustid.lookup", return_value=payload),
                patch.object(fingerprint_service, "set_cached_api") as store,
            ):
                self.assertEqual(
                    fingerprint_service.lookup_acoustid(
                        Path("recording.flac"), "fixture-key", "Primary", "Recording"
                    ),
                    expected,
                )
            self.assertEqual(store.call_args.args[0], "acoustid:raw:v1:10:fingerprint")
            self.assertEqual(store.call_args.args[1], payload)

    def test_short_title_candidates_require_near_exact_match(self) -> None:
        payload = {
            "status": "ok",
            "results": [
                {"score": 0.99, "recordings": [{"id": MBID, "title": "Outro"}]}
            ],
        }
        with (
            patch.object(
                fingerprint_service,
                "fingerprint_audio_file",
                return_value=(10, "fingerprint"),
            ),
            patch.object(fingerprint_service, "get_cached_api", return_value=None),
            patch.object(fingerprint_service._ACOUSTID_LIMITER, "wait"),
            patch("acoustid.lookup", return_value=payload),
        ):
            self.assertIsNone(
                fingerprint_service.lookup_acoustid(
                    Path("recording.flac"), "fixture", "Primary", "Intro"
                )
            )

    def test_cached_and_missing_fingerprint_results_never_query_provider(self) -> None:
        for expected in (None, MBID):
            cached = {
                "status": "ok",
                "results": (
                    [
                        {
                            "score": 0.99,
                            "recordings": [
                                {
                                    "id": MBID,
                                    "title": "Recording",
                                    "artists": [{"name": "Primary"}],
                                }
                            ],
                        }
                    ]
                    if expected
                    else []
                ),
            }
            with (
                self.subTest(cached=cached),
                patch.object(
                    fingerprint_service,
                    "fingerprint_audio_file",
                    return_value=(10, "fingerprint"),
                ),
                patch.object(
                    fingerprint_service, "get_cached_api", return_value=cached
                ),
                patch("acoustid.lookup") as lookup,
            ):
                self.assertEqual(
                    fingerprint_service.lookup_acoustid(
                        Path("recording.flac"), "fixture"
                    ),
                    expected,
                )
            lookup.assert_not_called()
        with patch.object(
            fingerprint_service,
            "fingerprint_audio_file",
            side_effect=OSError("bad recording"),
        ):
            self.assertIsNone(
                fingerprint_service.lookup_acoustid(Path("recording.flac"), "fixture")
            )


class TestSmallProviderContracts(IsolatedTestCase):
    def test_lastfm_cache_identity_fallback_and_api_errors(self) -> None:
        with patch.object(lastfm, "get_cached_api", return_value=["Rock"]):
            self.assertEqual(
                lastfm.fetch_lastfm_tags("Primary", "Recording", "fixture"), ["Rock"]
            )
        self.assertEqual(lastfm.fetch_lastfm_tags("", "", "fixture"), [])
        request = httpx.Request("GET", "https://ws.audioscrobbler.com")
        for first_response in (
            httpx.Response(200, json={"toptags": {"tag": []}}, request=request),
            httpx.Response(503, request=request),
        ):
            with (
                self.subTest(status=first_response.status_code),
                patch.object(lastfm, "get_cached_api", return_value=None),
                patch.object(
                    SESSION,
                    "get",
                    side_effect=[
                        first_response,
                        httpx.Response(
                            200,
                            json={
                                "toptags": {"tag": [{"name": "rock"}, {}, {"name": 12}]}
                            },
                            request=request,
                        ),
                    ],
                ) as get,
            ):
                self.assertEqual(
                    lastfm.fetch_lastfm_tags("Primary", "Recording", "fixture", MBID),
                    ["Rock"],
                )
            self.assertEqual(get.call_args_list[0].kwargs["params"]["mbid"], MBID)
            self.assertNotIn("mbid", get.call_args_list[1].kwargs["params"])
        with (
            patch.object(lastfm, "get_cached_api", return_value=None),
            patch.object(
                SESSION,
                "get",
                return_value=httpx.Response(
                    200,
                    json={"error": 6, "message": "Track not found"},
                    request=request,
                ),
            ),
        ):
            self.assertEqual(
                lastfm.fetch_lastfm_tags("Primary", "Recording", "fixture"), []
            )

    def test_theaudiodb_cached_absent_and_partial_artwork(self) -> None:
        self.assertEqual(theaudiodb.fetch_artist_images("Unknown Artist"), (None, None))
        self.assertIsNone(theaudiodb.fetch_theaudiodb_track_details("", "Recording"))
        with patch.object(
            theaudiodb, "get_cached_api", return_value=(b"portrait", None)
        ):
            self.assertEqual(
                theaudiodb.fetch_artist_images("Primary"), (b"portrait", None)
            )
        for cached in ({}, {"mood": "Chill"}):
            with (
                self.subTest(cached=cached),
                patch.object(theaudiodb, "get_cached_api", return_value=cached),
            ):
                self.assertEqual(
                    theaudiodb.fetch_theaudiodb_track_details("Primary", "Recording"),
                    cached or None,
                )
        artist_responses: tuple[list[dict[str, str]] | None, ...] = (
            None,
            [],
            [{}],
            [
                {
                    "strArtist": "Primary",
                    "strArtistThumb": "https://example.com/avatar.png",
                }
            ],
        )
        for artists in artist_responses:
            with (
                self.subTest(artists=artists),
                patch.object(theaudiodb, "get_cached_api", return_value=None),
                patch.object(
                    SESSION,
                    "get",
                    return_value=httpx.Response(200, json={"artists": artists}),
                ),
                patch.object(
                    theaudiodb, "_download_artwork_bytes", return_value=b"portrait"
                ),
            ):
                expected = (
                    (b"portrait", None) if artists and artists[0] else (None, None)
                )
                self.assertEqual(theaudiodb.fetch_artist_images("Primary"), expected)
        with patch.object(SESSION, "get", return_value=httpx.Response(404)):
            self.assertIsNone(
                theaudiodb._download_artwork_bytes("https://example.com/absent.png")
            )
        self.assertIsNone(theaudiodb._clean_theaudiodb_str(None))

    def test_genius_empty_unmatched_and_missing_path_hits_are_rejected(self) -> None:
        request = httpx.Request("GET", "https://api.genius.com")
        self.assertIsNone(genius.fetch_genius_song_details("", "Recording", "fixture"))
        for hits in (
            [],
            [{"result": {"title": "Unrelated", "primary_artist": {"name": "Wrong"}}}],
            [{"result": {"title": "Recording", "primary_artist": {"name": "Primary"}}}],
        ):
            with (
                self.subTest(hits=hits),
                patch.object(genius, "get_cached_api", return_value=None),
                patch.object(
                    SESSION,
                    "get",
                    return_value=httpx.Response(
                        200, json={"response": {"hits": hits}}, request=request
                    ),
                ) as get,
            ):
                self.assertIsNone(
                    genius.fetch_genius_song_details("Primary", "Recording", "fixture")
                )
            get.assert_called_once()
        with self.assertRaises(TypeError):
            genius._normalize_genius_description({"unexpected": "object"})
