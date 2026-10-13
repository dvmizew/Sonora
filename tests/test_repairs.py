import asyncio
import contextlib
import tempfile
from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import numpy as np

from sonora.audio.bpm import calculate_bpm
from sonora.core.api_payloads import require_api_object, require_api_records
from sonora.core.config import SonoraConfig, clear_config_cache, get_config
from sonora.core.rate_limit import RateLimiter
from sonora.core.titles import score_artist_title_match
from sonora.core.utils import is_interruption
from sonora.services.deezer import fetch_deezer_track_details
from sonora.services.discogs import search_discogs_release
from sonora.services.fanart import _extract_image_urls, fetch_fanart_artist
from sonora.services.genius import fetch_genius_song_details
from sonora.services.itunes import search_itunes
from sonora.services.lastfm import fetch_lastfm_tags
from sonora.services.lyrics import _query_lrclib
from sonora.services.musicbrainz import fetch_artist_discography
from sonora.services.shazam import _recognize_async, recognize_audio_track
from sonora.services.theaudiodb import fetch_theaudiodb_track_details
from tests.support import IsolatedTestCase


class TestCancellationChains(IsolatedTestCase):
    def test_keyboard_interrupt_and_ordinary_errors(self) -> None:
        self.assertTrue(is_interruption(KeyboardInterrupt()))
        self.assertFalse(is_interruption(RuntimeError("release unlocked lock")))
        self.assertFalse(is_interruption(RuntimeError("internal failure")))

    def test_nested_explicit_causes(self) -> None:
        worker_error = RuntimeError("worker cleanup")
        executor_error = RuntimeError("executor cleanup")
        worker_error.__cause__ = KeyboardInterrupt()
        executor_error.__cause__ = worker_error
        self.assertTrue(is_interruption(executor_error))

    def test_context_and_suppressed_context(self) -> None:
        worker_error = RuntimeError("worker cleanup")
        worker_error.__context__ = KeyboardInterrupt()
        self.assertTrue(is_interruption(worker_error))
        worker_error.__suppress_context__ = True
        self.assertFalse(is_interruption(worker_error))

    def test_cyclic_exception_chain(self) -> None:
        first_error = RuntimeError("first")
        second_error = RuntimeError("second")
        first_error.__cause__ = second_error
        second_error.__cause__ = first_error
        self.assertFalse(is_interruption(first_error))


class TestProviderContracts(IsolatedTestCase):
    def setUp(self) -> None:
        RateLimiter.set_disabled(True)
        self.addCleanup(RateLimiter.set_disabled, False)

    def test_valid_objects_and_empty_arrays(self) -> None:
        self.assertEqual(require_api_object({}, "response"), {})
        self.assertEqual(require_api_records([], "results"), [])
        self.assertEqual(require_api_records([{"id": 1}], "results"), [{"id": 1}])

    def test_invalid_envelope_types(self) -> None:
        payload: object
        for payload in (None, "wrong", [], 17):
            with self.subTest(payload=payload), self.assertRaises(TypeError):
                require_api_object(payload, "response")

    def test_invalid_record_arrays(self) -> None:
        payload: object
        for payload in (None, "wrong", {}, [None], [{"id": 1}, "invalid"]):
            with self.subTest(payload=payload), self.assertRaises(TypeError):
                require_api_records(payload, "results")

    def test_provider_boundaries_reject_malformed_roots_without_caching(self) -> None:
        payload: object
        boundaries: tuple[tuple[str, Callable[[], object]], ...] = (
            ("itunes", lambda: search_itunes("Artist", "Title", "song")),
            ("deezer", lambda: fetch_deezer_track_details("Artist", "Title")),
            ("discogs", lambda: search_discogs_release("Artist", "Album", "token")),
            ("genius", lambda: fetch_genius_song_details("Artist", "Title", "token")),
            ("lastfm", lambda: fetch_lastfm_tags("Artist", "Title", "key")),
            ("theaudiodb", lambda: fetch_theaudiodb_track_details("Artist", "Title")),
            (
                "fanart",
                lambda: fetch_fanart_artist(
                    "11111111-2222-3333-4444-555555555555", "key"
                ),
            ),
            ("lyrics", lambda: _query_lrclib("Artist - Title")),
        )
        for service, invoke in boundaries:
            for payload in ([], "schema changed", None):
                with (
                    self.subTest(service=service, payload=payload),
                    contextlib.ExitStack() as resources,
                ):
                    response = MagicMock(status_code=200)
                    response.json.return_value = payload
                    resources.enter_context(
                        patch(
                            f"sonora.services.{service}.SESSION.get",
                            return_value=response,
                        )
                    )
                    resources.enter_context(
                        patch(
                            f"sonora.services.{service}.get_cached_api",
                            return_value=None,
                        )
                    )
                    write_cache = resources.enter_context(
                        patch(f"sonora.services.{service}.set_cached_api")
                    )
                    with self.assertRaises(TypeError):
                        invoke()
                    write_cache.assert_not_called()

    def test_itunes_rejects_missing_results_and_invalid_records(self) -> None:
        for payload in ({}, {"results": "wrong"}, {"results": [None]}):
            with (
                self.subTest(payload=payload),
                patch("sonora.services.itunes.get_cached_api", return_value=None),
                patch("sonora.services.itunes.SESSION.get") as request,
            ):
                request.return_value.json.return_value = payload
                with self.assertRaises(TypeError):
                    search_itunes("Artist", "Title", "song")

    def test_musicbrainz_schema_and_programming_errors_surface(self) -> None:
        payload: object
        for payload in ([], {"release-list": "wrong"}, {"release-list": [None]}):
            with (
                self.subTest(payload=payload),
                patch(
                    "sonora.services.musicbrainz._call_musicbrainz",
                    return_value=payload,
                ),
                self.assertRaises(TypeError),
            ):
                fetch_artist_discography("Artist")
        with (
            patch(
                "sonora.services.musicbrainz._call_musicbrainz",
                side_effect=RuntimeError("internal bug"),
            ),
            self.assertRaisesRegex(RuntimeError, "internal bug"),
        ):
            fetch_artist_discography("Artist")

    def test_documented_empty_responses(self) -> None:
        with patch("sonora.services.theaudiodb.SESSION.get") as request:
            request.return_value.status_code = 200
            request.return_value.json.return_value = {"track": None}
            self.assertIsNone(fetch_theaudiodb_track_details("Artist", "Title"))
        with patch("sonora.services.lastfm.SESSION.get") as request:
            request.return_value.json.return_value = {"toptags": {"tag": []}}
            self.assertEqual(fetch_lastfm_tags("Artist", "Title", "key"), [])
        with patch("sonora.services.itunes.SESSION.get") as request:
            request.return_value.json.return_value = {"results": []}
            self.assertEqual(search_itunes("Artist", "Title", "song"), [])

    def test_fanart_descriptor_validation_and_like_order(self) -> None:
        self.assertEqual(_extract_image_urls(None), [])
        self.assertEqual(_extract_image_urls([]), [])
        with self.assertRaises(TypeError):
            _extract_image_urls({"url": "https://example.com/art.jpg"})
        with self.assertRaises(TypeError):
            _extract_image_urls([None])
        self.assertEqual(
            _extract_image_urls(
                [
                    {"url": "https://example.com/first.jpg", "likes": "1"},
                    {"url": "https://example.com/second.jpg", "likes": "9"},
                ]
            ),
            ["https://example.com/second.jpg", "https://example.com/first.jpg"],
        )

    def test_lrclib_get_rejects_invalid_response_types(self) -> None:
        for query, payload in (
            ("Artist - Title", {"syncedLyrics": 42}),
            ("Artist - Title", [None]),
            ("Artist - Title", []),
        ):
            with (
                self.subTest(query=query, payload=payload),
                patch("sonora.services.lyrics.SESSION.get") as request,
            ):
                request.return_value.status_code = 200
                request.return_value.json.return_value = payload
                with self.assertRaises(TypeError):
                    _query_lrclib(query)

    def test_theaudiodb_current_free_api_key(self) -> None:
        with patch("sonora.services.theaudiodb.SESSION.get") as request:
            request.return_value.status_code = 200
            request.return_value.json.return_value = {"track": None}
            fetch_theaudiodb_track_details("Artist", "Title")
            self.assertIn("/api/v1/json/123/", request.call_args.args[0])


class TestShazamContracts(IsolatedTestCase):
    def setUp(self) -> None:
        RateLimiter.set_disabled(True)
        self.addCleanup(RateLimiter.set_disabled, False)
        config_patch = patch(
            "sonora.services.shazam.get_config",
            return_value=SonoraConfig(enable_shazam=True),
        )
        config_patch.start()
        self.addCleanup(config_patch.stop)

    def test_client_closed_after_success(self) -> None:
        client = MagicMock()
        client.__aenter__.return_value = client
        client.recognize = AsyncMock(return_value={"matches": []})
        with patch("sonora.services.shazam.Shazam", return_value=client):
            self.assertEqual(
                asyncio.run(_recognize_async(Path("track.flac"))), {"matches": []}
            )
        client.__aexit__.assert_awaited_once()

    def test_programming_errors_propagate_and_client_closes(self) -> None:
        for error in (RuntimeError("internal bug"), ValueError("bad internal state")):
            with self.subTest(error=error):
                client = MagicMock()
                client.__aenter__.return_value = client
                client.recognize = AsyncMock(side_effect=error)
                with (
                    patch("sonora.services.shazam.Shazam", return_value=client),
                    self.assertRaises(type(error)),
                ):
                    asyncio.run(_recognize_async(Path("track.flac")))
                client.__aexit__.assert_awaited_once()

    def test_operational_errors_return_none_and_client_closes(self) -> None:
        for error in (
            aiohttp.ClientConnectionError("offline"),
            TimeoutError(),
            OSError("unreadable audio"),
        ):
            with self.subTest(error=error):
                client = MagicMock()
                client.__aenter__.return_value = client
                client.recognize = AsyncMock(side_effect=error)
                with patch("sonora.services.shazam.Shazam", return_value=client):
                    self.assertIsNone(asyncio.run(_recognize_async(Path("track.flac"))))
                client.__aexit__.assert_awaited_once()

    def test_invalid_recognition_envelope_raises(self) -> None:
        client = MagicMock()
        client.__aenter__.return_value = client
        client.recognize = AsyncMock(return_value=[])
        with (
            patch("sonora.services.shazam.Shazam", return_value=client),
            self.assertRaises(TypeError),
        ):
            asyncio.run(_recognize_async(Path("track.flac")))

    def test_shazam_key_is_never_an_apple_music_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "track.flac"
            audio_path.write_bytes(b"audio")
            with patch(
                "sonora.services.shazam._recognize_async",
                new=AsyncMock(
                    return_value={
                        "track": {
                            "title": "Title",
                            "subtitle": "Artist",
                            "key": "7654321",
                        }
                    }
                ),
            ):
                recognized = recognize_audio_track(audio_path)
            self.assertIsNotNone(recognized)
            assert recognized is not None
            self.assertIsNone(recognized.apple_music_id)

    def test_invalid_track_schema_is_not_cached(self) -> None:
        track: object
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "track.flac"
            audio_path.write_bytes(b"audio")
            for track in (
                [],
                {"title": None, "subtitle": "Artist"},
                {"title": "Title", "subtitle": ""},
            ):
                with (
                    self.subTest(track=track),
                    patch(
                        "sonora.services.shazam._recognize_async",
                        new=AsyncMock(return_value={"track": track}),
                    ),
                    patch("sonora.services.shazam.set_cached_api") as write_cache,
                ):
                    with self.assertRaises(TypeError):
                        recognize_audio_track(audio_path)
                    write_cache.assert_not_called()

    def test_cache_distinguishes_equal_basenames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio_paths = [
                Path(directory) / artist / "track.flac"
                for artist in ("First", "Second")
            ]
            for audio_path in audio_paths:
                audio_path.parent.mkdir()
                audio_path.write_bytes(b"audio")
            with patch(
                "sonora.services.shazam._recognize_async",
                new=AsyncMock(
                    side_effect=[
                        {"track": {"title": "First", "subtitle": "Artist"}},
                        {"track": {"title": "Second", "subtitle": "Artist"}},
                    ]
                ),
            ) as recognize:
                first_match = recognize_audio_track(audio_paths[0])
                second_match = recognize_audio_track(audio_paths[1])
            assert first_match is not None and second_match is not None
            self.assertEqual(
                (first_match.title, second_match.title), ("First", "Second")
            )
            self.assertEqual(recognize.await_count, 2)

    def test_missing_and_empty_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "missing.flac"
            self.assertIsNone(recognize_audio_track(audio_path))
            audio_path.touch()
            self.assertIsNone(recognize_audio_track(audio_path))


class TestConfigurablePolicies(IsolatedTestCase):
    def test_invalid_policy_bounds(self) -> None:
        invalid_configs: tuple[Callable[[], SonoraConfig], ...] = (
            lambda: SonoraConfig(title_match_threshold=84),
            lambda: SonoraConfig(short_title_match_threshold=94),
            lambda: SonoraConfig(short_title_length=7),
            lambda: SonoraConfig(acoustid_confidence_threshold=1.1),
            lambda: SonoraConfig(key_confidence_threshold=-0.1),
            lambda: SonoraConfig(bpm_search_min=0),
            lambda: SonoraConfig(bpm_search_max=float("inf")),
            lambda: SonoraConfig(bpm_search_min=200, bpm_search_max=60),
            lambda: SonoraConfig(bpm_octave_min=75),
            lambda: SonoraConfig(bpm_octave_min=75, bpm_octave_max=140),
        )
        for construct_config in invalid_configs:
            with (
                self.subTest(construct_config=construct_config),
                self.assertRaises(ValueError),
            ):
                construct_config()

    def test_configuration_loads_new_policies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.toml"
            config_path.write_text(
                "[sonora]\ntitle_match_threshold = 91\nbpm_search_min = 40\nbpm_search_max = 240\nbpm_octave_min = 75\nbpm_octave_max = 190\n",
                encoding="utf-8",
            )
            with patch(
                "sonora.core.config._discover_config_file", return_value=config_path
            ):
                clear_config_cache()
                config = get_config()
            self.assertEqual(config.title_match_threshold, 91)
            self.assertEqual((config.bpm_search_min, config.bpm_search_max), (40, 240))
            self.assertEqual((config.bpm_octave_min, config.bpm_octave_max), (75, 190))

    def test_invalid_numeric_configuration(self) -> None:
        for configured in (True, "nan", "not a number"):
            with (
                self.subTest(configured=configured),
                patch(
                    "sonora.core.config._discover_config_file",
                    return_value=Path("config.json"),
                ),
                patch(
                    "sonora.core.config._parse_config_data",
                    return_value={"bpm_search_min": configured},
                ),
            ):
                clear_config_cache()
                with self.assertRaises((TypeError, ValueError)):
                    get_config()

    def test_matching_policy_changes_acceptance(self) -> None:
        self.assertGreater(
            score_artist_title_match(
                "Artist", "An ordinary song", "Artist", "An ordinary songs"
            ),
            0,
        )
        with patch(
            "sonora.core.titles.get_config",
            return_value=SonoraConfig(title_match_threshold=99),
        ):
            self.assertEqual(
                score_artist_title_match(
                    "Artist", "An ordinary song", "Artist", "An ordinary songs"
                ),
                0,
            )
            self.assertEqual(
                score_artist_title_match(
                    "Artist", "An ordinary song", "Artist", "An ordinary song"
                ),
                100,
            )
        self.assertEqual(
            score_artist_title_match("Artist", "Intro", "Artist", "Outro"), 0
        )

    def test_slow_tempo_not_doubled_unless_configured(self) -> None:
        sample_rate = 22050
        pulse_samples = int(sample_rate * 60 / 70)
        audio_samples = np.zeros(sample_rate * 20)
        for pulse_start in range(0, len(audio_samples), pulse_samples):
            pulse_length = min(256, len(audio_samples) - pulse_start)
            audio_samples[pulse_start : pulse_start + pulse_length] = np.hanning(
                pulse_length
            )
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "70bpm.wav"
            audio_path.touch()
            with patch(
                "sonora.audio.bpm.load_audio", return_value=(audio_samples, sample_rate)
            ):
                measured_bpm = calculate_bpm(audio_path)
                assert measured_bpm is not None
                self.assertAlmostEqual(measured_bpm, 70, delta=2)
                with patch(
                    "sonora.audio.bpm.get_config",
                    return_value=SonoraConfig(bpm_octave_min=75, bpm_octave_max=190),
                ):
                    folded_bpm = calculate_bpm(audio_path)
                assert folded_bpm is not None
                self.assertAlmostEqual(folded_bpm, 140, delta=4)


class TestDryRunReports(IsolatedTestCase):
    def test_bpm_reports_computation_and_actual_writes_separately(self) -> None:
        from sonora.cli import audio as audio_cli

        audio_path = Path("track.flac")
        for dry_run in (True, False):
            with (
                self.subTest(dry_run=dry_run),
                patch(
                    "sonora.cli.audio._run_parallel_audio_task",
                    return_value=([(audio_path, 70.0, True)], False),
                ),
                patch("sonora.cli.audio._write_json_report") as report,
            ):
                self.assertEqual(
                    audio_cli.bpm(
                        audio_path, dry_run=dry_run, json_report=Path("report.json")
                    ),
                    0,
                )
                payload = report.call_args.args[1]
                self.assertEqual(payload["summary"]["bpm_calculated"], 1)
                self.assertEqual(payload["summary"]["bpm_tagged"], 0 if dry_run else 1)
                self.assertEqual(payload["details"][0]["modified"], not dry_run)

    def test_key_reports_detection_and_actual_writes_separately(self) -> None:
        from sonora.cli import audio as audio_cli

        audio_path = Path("track.flac")
        for dry_run in (True, False):
            with (
                self.subTest(dry_run=dry_run),
                patch(
                    "sonora.cli.audio._run_parallel_audio_task",
                    return_value=([(audio_path, "A minor (8A)", True)], False),
                ),
                patch("sonora.cli.audio._write_json_report") as report,
            ):
                self.assertEqual(
                    audio_cli.key(
                        audio_path, dry_run=dry_run, json_report=Path("report.json")
                    ),
                    0,
                )
                payload = report.call_args.args[1]
                self.assertEqual(payload["summary"]["key_detected"], 1)
                self.assertEqual(payload["summary"]["key_tagged"], 0 if dry_run else 1)
                self.assertEqual(payload["details"][0]["modified"], not dry_run)

    def test_failed_or_skipped_bpm_is_not_reported_as_modified(self) -> None:
        from sonora.cli import audio as audio_cli

        audio_path = Path("track.flac")
        with (
            patch(
                "sonora.cli.audio._run_parallel_audio_task",
                return_value=(
                    [(audio_path, None, True), (audio_path, 70.0, False)],
                    False,
                ),
            ),
            patch("sonora.cli.audio._write_json_report") as report,
        ):
            audio_cli.bpm(audio_path, json_report=Path("report.json"))
            payload = report.call_args.args[1]
            self.assertEqual(payload["summary"]["bpm_tagged"], 0)
            self.assertFalse(any(track["modified"] for track in payload["details"]))

    def test_dsp_workers_do_not_write_during_dry_run(self) -> None:
        from sonora.cli import audio as audio_cli
        from sonora.core.models import TrackInfo

        audio_path = Path("track.flac")
        with (
            patch(
                "sonora.cli.audio.read_track_metadata",
                return_value=TrackInfo(audio_path),
            ),
            patch("sonora.cli.audio.calculate_bpm", return_value=70.0),
            patch(
                "sonora.cli.audio.detect_key_details",
                return_value=("A minor", "8A", 0.8),
            ),
            patch("sonora.cli.audio.write_track_metadata") as write_tags,
        ):
            self.assertEqual(
                audio_cli._process_bpm_file(audio_path, force=True, dry_run=True)[1], 70
            )
            self.assertEqual(
                audio_cli._process_key_file(audio_path, force=True, dry_run=True)[1],
                "A minor (8A)",
            )
            write_tags.assert_not_called()

    def test_exact_identity_survives_maximum_threshold(self) -> None:
        with patch(
            "sonora.core.titles.get_config",
            return_value=SonoraConfig(
                title_match_threshold=100, short_title_match_threshold=100
            ),
        ):
            self.assertEqual(
                score_artist_title_match("Artist", "Intro", "Artist", "Intro"), 100
            )
            self.assertEqual(
                score_artist_title_match("Artist", "Intro", "Artist", "Outro"), 0
            )


class TestITunesIdentityBoundaries(IsolatedTestCase):
    def test_metadata_does_not_trust_candidates_without_an_artist(self) -> None:
        from sonora.services.itunes import fetch_itunes_track_metadata

        with patch(
            "sonora.services.itunes.search_itunes",
            return_value=[{"trackName": "Intro", "trackId": 123}],
        ):
            self.assertIsNone(fetch_itunes_track_metadata("Artist Alpha", "Intro"))

    def test_short_titles_cannot_bypass_identity_scoring(self) -> None:
        from sonora.services.itunes import fetch_itunes_track_metadata

        with patch(
            "sonora.services.itunes.search_itunes",
            return_value=[
                {"trackName": "Outro", "artistName": "Artist Alpha", "trackId": 123}
            ],
        ):
            self.assertIsNone(fetch_itunes_track_metadata("Artist Alpha", "Intro"))

    def test_cover_art_does_not_trust_candidates_without_an_artist(self) -> None:
        from sonora.services.itunes import fetch_itunes_cover_art_url

        with patch(
            "sonora.services.itunes.search_itunes",
            return_value=[
                {
                    "collectionName": "Record",
                    "artworkUrl100": "https://example.com/100x100bb.jpg",
                }
            ],
        ):
            self.assertIsNone(fetch_itunes_cover_art_url("Artist Alpha", "Record"))


class TestCacheErrorBoundaries(IsolatedTestCase):
    def test_cache_programming_errors_are_not_silenced(self) -> None:
        from sonora.core.cache import get_cached_api, set_cached_api

        for exception in (RuntimeError("internal bug"), ValueError("internal bug")):
            with (
                self.subTest(exception=exception),
                patch("sonora.core.cache.get_cache") as cache,
            ):
                cache.return_value.get.side_effect = exception
                cache.return_value.set.side_effect = exception
                with self.assertRaises(type(exception)):
                    get_cached_api("track")
                with self.assertRaises(type(exception)):
                    set_cached_api("track", {"title": "Title"})

    def test_unavailable_database_is_an_operational_error(self) -> None:
        import sqlite3

        from sonora.core.cache import get_cached_api, set_cached_api

        with patch("sonora.core.cache.get_cache") as cache:
            cache.return_value.get.side_effect = sqlite3.OperationalError("locked")
            cache.return_value.set.side_effect = sqlite3.OperationalError("locked")
            self.assertIsNone(get_cached_api("track"))
            set_cached_api("track", {"title": "Title"})

    def test_corrupted_serialized_cache_entry_is_an_operational_error(self) -> None:
        from sonora.core.cache import get_cached_api
        from sonora.core.cache_storage import CacheDecodeError

        with patch("sonora.core.cache.get_cache") as cache:
            cache.return_value.get.side_effect = CacheDecodeError("corrupted entry")
            self.assertIsNone(get_cached_api("track"))


class TestArtistResponseContracts(IsolatedTestCase):
    def test_artist_resolution_exposes_malformed_deezer_responses(self) -> None:
        from sonora.core.artists import resolve_artist_name

        payload: object
        for payload in ([], {"data": "wrong"}, {"data": [None]}):
            with (
                self.subTest(payload=payload),
                patch(
                    "sonora.services.musicbrainz.search_musicbrainz_artists",
                    return_value=[],
                ),
                patch("sonora.core.artists.SESSION.get") as request,
            ):
                resolve_artist_name.cache_clear()
                request.return_value.status_code = 200
                request.return_value.json.return_value = payload
                with self.assertRaises(TypeError):
                    resolve_artist_name("Artist Alpha")
