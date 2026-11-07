import json
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import syncedlyrics

from sonora.audio import metadata as metadata_module
from sonora.core import sidecars
from sonora.core.config import get_config
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.models import TrackInfo
from sonora.core.titles import clean_title
from sonora.modules import track_processing
from sonora.services import lyrics
from tests.support import IsolatedTestCase

SYNCED = "[00:01.00] First line\n[00:02.00] Second line"
ENHANCED = "[00:01.00] <00:01.00> First <00:01.50> line"


class TestLyricsTokenContracts(IsolatedTestCase):
    def test_provider_order_partitions_the_lyrics_cache(self) -> None:
        recorded_keys: list[str] = []
        with (
            patch.object(lyrics, "get_cached_api", return_value=None),
            patch.object(
                lyrics,
                "set_cached_api",
                side_effect=lambda key, value: recorded_keys.append(key),
            ),
            patch.object(lyrics, "_query_syncedlyrics", return_value=SYNCED),
        ):
            lyrics.fetch_synced_lyrics(
                "Alpha", "Recording", providers=["Genius", "NetEase"]
            )
            lyrics.fetch_synced_lyrics(
                "Alpha", "Recording", providers=["NetEase", "Genius"]
            )
        self.assertEqual(len(recorded_keys), 2)
        self.assertNotEqual(recorded_keys[0], recorded_keys[1])

    def test_shared_lookup_lock_is_not_held_during_provider_request(self) -> None:
        lookup_state: dict[str, Any] = {
            "lock": threading.Lock(),
            "failures": 0,
            "suspended": False,
        }

        def query(*args: Any, **kwargs: Any) -> str:
            self.assertTrue(lookup_state["lock"].acquire(blocking=False))
            lookup_state["lock"].release()
            return SYNCED

        with (
            patch.object(lyrics, "get_cached_api", return_value=None),
            patch.object(lyrics, "_query_syncedlyrics", side_effect=query),
            patch.object(lyrics, "set_cached_api"),
        ):
            self.assertEqual(
                lyrics.fetch_synced_lyrics(
                    "Primary", "Recording", duration=100, lookup_state=lookup_state
                ),
                SYNCED,
            )

    def test_lrclib_failure_does_not_block_explicitly_enabled_fallback(self) -> None:
        for fallback in (SYNCED, None):
            with (
                self.subTest(fallback=fallback),
                patch.object(lyrics, "_query_lrclib", side_effect=OSError("offline")),
                patch.object(syncedlyrics, "search", return_value=fallback) as search,
                patch.object(lyrics._LYRICS_LIMITER, "wait"),
            ):
                if fallback:
                    self.assertEqual(
                        lyrics._query_syncedlyrics(
                            "Primary - Recording",
                            False,
                            False,
                            False,
                            ["Lrclib", "Genius"],
                            None,
                        ),
                        fallback,
                    )
                else:
                    with self.assertRaises(OSError):
                        lyrics._query_syncedlyrics(
                            "Primary - Recording",
                            False,
                            False,
                            False,
                            ["Lrclib", "Genius"],
                            None,
                        )
                self.assertEqual(search.call_args.kwargs["providers"], ["Genius"])

    def test_authenticated_fallback_miss_does_not_hide_lrclib_failure(self) -> None:
        with (
            patch.object(lyrics, "_query_lrclib", side_effect=OSError("offline")),
            patch.object(lyrics, "_query_authenticated_musixmatch", return_value=None),
            patch.object(syncedlyrics, "search") as search,
            self.assertRaises(OSError),
        ):
            lyrics._query_syncedlyrics(
                "Primary - Recording",
                False,
                False,
                False,
                ["Lrclib", "Musixmatch"],
                None,
                musixmatch_token="fixture-token",
            )
        search.assert_not_called()

    def test_authenticated_lookup_passes_language_once(self) -> None:
        with patch.object(
            lyrics, "_query_authenticated_musixmatch", return_value=SYNCED
        ) as query:
            self.assertEqual(
                lyrics._query_syncedlyrics(
                    "Primary - Recording",
                    False,
                    False,
                    False,
                    ["Musixmatch"],
                    "en",
                    musixmatch_token="fixture-token",
                ),
                SYNCED,
            )
        query.assert_called_once_with(
            "Primary - Recording", "fixture-token", False, False, False, "en"
        )

    def test_raw_and_exported_tokens_parse_without_writing_sdk_state(self) -> None:
        tokens = (
            " fixture-token ",
            urllib.parse.quote(
                json.dumps({"tokens": {"web-desktop-app-v1.0": "fixture-token"}})
            ),
            json.dumps({"tokens": {"mxm-account-v1.0": "fixture-token"}}),
            json.dumps({"tokens": {"user_token": "fixture-token"}}),
            json.dumps({"token": "fixture-token"}),
            json.dumps({"user_token": "fixture-token"}),
        )
        with patch.object(
            Path,
            "write_text",
            side_effect=AssertionError("Credentials must not be written"),
        ):
            for raw_token in tokens:
                with self.subTest(raw_token=raw_token):
                    self.assertEqual(
                        lyrics.parse_musixmatch_token(raw_token), "fixture-token"
                    )
        self.assertEqual(list(self.test_directory.iterdir()), [])

    def test_absent_credentials_and_malformed_exports_are_distinct(self) -> None:
        for raw_token in (None, "", " "):
            with self.subTest(raw_token=raw_token):
                self.assertIsNone(lyrics.parse_musixmatch_token(raw_token))
        for raw_token in ('{"tokens":', '{"tokens": {}}', '{"token": null}'):
            with self.subTest(raw_token=raw_token), self.assertRaises(ValueError):
                lyrics.parse_musixmatch_token(raw_token)
        with self.assertRaises(TypeError):
            lyrics.parse_musixmatch_token('{"tokens": []}')

    def test_authenticated_client_honors_quality_options_and_closes_its_session(
        self,
    ) -> None:
        scenarios = (
            (True, False, SYNCED, "Plain", "Plain"),
            (False, True, SYNCED, "Plain", SYNCED),
            (False, True, None, "Plain", None),
            (False, False, None, "Plain", "Plain"),
        )
        for plain_only, synced_only, synced, unsynced, expected in scenarios:
            with (
                self.subTest(
                    plain_only=plain_only, synced_only=synced_only, synced=synced
                ),
                patch("syncedlyrics.providers.Musixmatch") as client_class,
                patch.object(lyrics._LYRICS_LIMITER, "wait"),
                patch.object(
                    Path,
                    "write_text",
                    side_effect=AssertionError("No SDK cache writes"),
                ),
            ):
                client = client_class.return_value
                client.get_lrc.return_value = SimpleNamespace(
                    synced=synced, unsynced=unsynced
                )
                self.assertEqual(
                    lyrics._query_authenticated_musixmatch(
                        "Alpha - Recording",
                        "fixture-token",
                        plain_only,
                        synced_only,
                        True,
                        "en",
                    ),
                    expected,
                )
                self.assertEqual(client.token, "fixture-token")
                client_class.assert_called_once_with(lang="en", enhanced=True)
                client.get_lrc.assert_called_once_with("Alpha - Recording")
                client.session.close.assert_called_once_with()

    def test_authenticated_client_closes_when_the_sdk_returns_no_result(self) -> None:
        with (
            patch("syncedlyrics.providers.Musixmatch") as client_class,
            patch.object(lyrics._LYRICS_LIMITER, "wait"),
        ):
            client = client_class.return_value
            client.get_lrc.return_value = None
            self.assertIsNone(
                lyrics._query_authenticated_musixmatch(
                    "Alpha - Recording", "fixture-token", False, False, False, None
                )
            )
            client.session.close.assert_called_once_with()

    def test_authenticated_client_closes_on_failure_and_propagates_programming_errors(
        self,
    ) -> None:
        for failure in (OSError("offline"), TypeError("SDK contract changed")):
            with (
                self.subTest(failure=type(failure).__name__),
                patch("syncedlyrics.providers.Musixmatch") as client_class,
                patch.object(lyrics._LYRICS_LIMITER, "wait"),
            ):
                client = client_class.return_value
                client.get_lrc.side_effect = failure
                with self.assertRaises(type(failure)):
                    lyrics._query_authenticated_musixmatch(
                        "Alpha - Recording",
                        "fixture-token",
                        False,
                        False,
                        False,
                        None,
                    )
                client.session.close.assert_called_once_with()

    def test_authenticated_client_retries_transient_network_errors(self) -> None:
        with (
            patch.object(
                lyrics,
                "_query_authenticated_musixmatch",
                side_effect=[OSError("offline"), SYNCED],
            ) as query,
            patch.object(lyrics._LYRICS_LIMITER, "wait"),
            patch("sonora.core.retry.time.sleep"),
        ):
            self.assertEqual(
                lyrics._query_syncedlyrics(
                    "Alpha - Recording",
                    False,
                    False,
                    False,
                    ["Musixmatch"],
                    None,
                    musixmatch_token="fixture-token",
                ),
                SYNCED,
            )
            self.assertEqual(query.call_count, 2)

    def test_authenticated_failure_falls_back_only_to_other_selected_providers(
        self,
    ) -> None:
        for providers, expected in (
            (["Musixmatch"], None),
            (["Musixmatch", "Genius"], SYNCED),
        ):
            with (
                self.subTest(providers=providers),
                patch("syncedlyrics.providers.Musixmatch") as client_class,
                patch.object(lyrics._LYRICS_LIMITER, "wait"),
                patch("syncedlyrics.search", return_value=SYNCED) as fallback,
            ):
                client = client_class.return_value
                client.get_lrc.return_value = SimpleNamespace(
                    synced=None, unsynced=None
                )
                self.assertEqual(
                    lyrics._query_syncedlyrics(
                        "Alpha - Recording",
                        False,
                        False,
                        False,
                        providers,
                        None,
                        musixmatch_token="fixture-token",
                    ),
                    expected,
                )
                client.session.close.assert_called_once_with()
                if expected is None:
                    fallback.assert_not_called()
                else:
                    self.assertEqual(fallback.call_args.kwargs["providers"], ["Genius"])

    def test_incompatible_quality_options_do_not_use_cache_or_providers(self) -> None:
        for token in (None, "fixture-token"):
            for cached in (None, SYNCED):
                with (
                    self.subTest(
                        authenticated=token is not None, cached=cached is not None
                    ),
                    patch.object(
                        lyrics, "get_cached_api", return_value=cached
                    ) as cache,
                    patch.object(lyrics, "_query_syncedlyrics") as query,
                ):
                    self.assertIsNone(
                        lyrics.fetch_synced_lyrics(
                            "Alpha",
                            "Recording",
                            plain_only=True,
                            synced_only=True,
                            musixmatch_token=token,
                        )
                    )
                    cache.assert_not_called()
                    query.assert_not_called()

    def test_tokens_are_forwarded_and_partition_cache_without_exposing_credentials(
        self,
    ) -> None:
        recorded_keys: list[str] = []
        with (
            patch.object(lyrics, "get_cached_api", return_value=None),
            patch.object(
                lyrics,
                "set_cached_api",
                side_effect=lambda key, value: recorded_keys.append(key),
            ),
            patch.object(lyrics, "_query_syncedlyrics", return_value=SYNCED) as query,
        ):
            for token in ("first-secret", "second-secret"):
                self.assertEqual(
                    lyrics.fetch_synced_lyrics(
                        "Alpha", "Recording", musixmatch_token=token
                    ),
                    SYNCED,
                )
                self.assertEqual(query.call_args.kwargs["musixmatch_token"], token)
        self.assertEqual(len(set(recorded_keys)), 2)
        self.assertTrue(all("secret" not in key for key in recorded_keys))


class TestLyricsParsingContracts(IsolatedTestCase):
    def test_remote_lyrics_preview_does_not_create_a_companion_file(self) -> None:
        path = self.test_directory / "recording.flac"
        with patch.object(lyrics, "fetch_synced_lyrics", return_value=SYNCED):
            self.assertEqual(
                track_processing.process_track_lyrics(
                    path, "Primary", "Recording", duration=100, dry_run=True
                ),
                (SYNCED, "synced"),
            )
        self.assertFalse(path.with_suffix(".lrc").exists())

    def test_syncedlyrics_duration_mismatch_rejects_provider_text(self) -> None:
        with (
            patch.object(lyrics._LYRICS_LIMITER, "wait"),
            patch("syncedlyrics.search", return_value="[09:00] Wrong"),
        ):
            self.assertIsNone(
                lyrics._query_syncedlyrics(
                    "Primary - Recording",
                    False,
                    True,
                    True,
                    ["Musixmatch"],
                    None,
                    duration=100,
                )
            )

    def test_valid_cached_lyrics_skip_provider_requests(self) -> None:
        with (
            patch.object(lyrics, "get_cached_api", return_value=ENHANCED),
            patch.object(lyrics, "_query_syncedlyrics") as query,
        ):
            self.assertEqual(
                lyrics.fetch_synced_lyrics("Primary", "Recording", duration=100),
                ENHANCED,
            )
        query.assert_not_called()

    def test_existing_lyrics_are_read_once_and_preserved_against_downgrades(
        self,
    ) -> None:
        audio_path = self.test_directory / "recording.flac"
        audio_path.with_suffix(".lrc").write_text(SYNCED)
        for replacement, force in (("Plain lyrics", True), (SYNCED, False)):
            with (
                self.subTest(replacement=replacement, force=force),
                patch.object(
                    Path,
                    "read_text",
                    side_effect=[SYNCED, OSError("unexpected reread")],
                ) as read_lyrics,
                patch.object(lyrics, "fetch_synced_lyrics", return_value=replacement),
            ):
                self.assertEqual(
                    track_processing.process_track_lyrics(
                        audio_path, "Primary", "Recording", duration=100, force=force
                    ),
                    (SYNCED, "existing"),
                )
                read_lyrics.assert_called_once()
        self.assertEqual(audio_path.with_suffix(".lrc").read_text(), SYNCED)

    def test_equal_quality_remote_lyrics_preserve_existing_local_text(self) -> None:
        audio_path = self.test_directory / "recording.flac"
        audio_path.with_suffix(".lrc").write_text(SYNCED)
        with patch.object(
            lyrics, "fetch_synced_lyrics", return_value="[00:01] Replacement"
        ):
            self.assertEqual(
                track_processing.process_track_lyrics(
                    audio_path, "Primary", "Recording", duration=100
                ),
                (SYNCED, "existing"),
            )

    def test_inline_embed_artifacts_are_removed_without_losing_lyrics(self) -> None:
        self.assertEqual(
            sidecars.clean_lyrics_text("First line\nSecond line12Embed"),
            "First line\nSecond line",
        )
        self.assertEqual(sidecars.clean_lyrics_text("First line\nEmbed"), "First line")

    def test_lrclib_checks_album_and_instrumental_status(self) -> None:
        for album, instrumental, accepted in (
            ("Record", False, True),
            ("record", False, True),
            ("Record (Live)", False, False),
            ("Another Record", False, False),
            (None, False, False),
            ("Record", True, False),
        ):
            with (
                self.subTest(album=album, instrumental=instrumental),
                patch.object(
                    SESSION,
                    "get",
                    return_value=httpx.Response(
                        200,
                        json={
                            "artistName": "Primary",
                            "trackName": "Recording",
                            "albumName": album,
                            "duration": 100,
                            "instrumental": instrumental,
                            "plainLyrics": "Song words",
                        },
                    ),
                ),
            ):
                result = lyrics._query_lrclib(
                    "Primary - Recording", album_name="Record", duration=100
                )
            self.assertEqual(result is not None, accepted)

    def test_lrclib_does_not_report_source_without_usable_lyrics(self) -> None:
        for plain_only, synced_only, payload in (
            (True, False, {"syncedLyrics": SYNCED}),
            (False, True, {"plainLyrics": "Song words"}),
        ):
            with self.subTest(plain_only=plain_only, synced_only=synced_only):
                details: dict[str, Any] = {}
                with patch.object(
                    SESSION,
                    "get",
                    return_value=httpx.Response(
                        200,
                        json={
                            "artistName": "Primary",
                            "trackName": "Recording",
                            "albumName": "Record",
                            "duration": 100,
                            "id": 123,
                            **payload,
                        },
                    ),
                ):
                    result = lyrics._query_lrclib(
                        "Primary - Recording",
                        plain_only=plain_only,
                        synced_only=synced_only,
                        album_name="Record",
                        duration=100,
                        details=details,
                    )
                self.assertIsNone(result)
                self.assertEqual(details, {})

    def test_cached_lyrics_with_wrong_headers_are_refetched(self) -> None:
        for header in ("[ar:Other Artist]", "[ti:Different Song]"):
            with (
                self.subTest(header=header),
                patch.object(
                    lyrics, "get_cached_api", return_value=f"{header}\n{SYNCED}"
                ),
                patch.object(
                    lyrics, "_query_syncedlyrics", return_value=SYNCED
                ) as query,
                patch.object(lyrics, "set_cached_api"),
            ):
                self.assertEqual(
                    lyrics.fetch_synced_lyrics("Primary", "Recording", duration=100),
                    SYNCED,
                )
                query.assert_called_once()

    def test_lrclib_request_preserves_fractional_duration(self) -> None:
        with (
            patch.object(
                SESSION,
                "get",
                return_value=httpx.Response(
                    200,
                    json={
                        "artistName": "Primary",
                        "trackName": "Recording",
                        "duration": 102.8,
                        "plainLyrics": "Song words",
                    },
                ),
            ) as request,
        ):
            self.assertEqual(
                lyrics._query_lrclib("Primary - Recording", duration=100.9),
                "Song words",
            )
        self.assertEqual(request.call_args.kwargs["params"]["duration"], 100.9)

    def test_fetched_lyrics_with_conflicting_headers_are_not_cached(self) -> None:
        with (
            patch.object(lyrics, "get_cached_api", return_value=None),
            patch.object(
                lyrics,
                "_query_syncedlyrics",
                return_value=f"[ti:Different Song]\n{SYNCED}",
            ),
            patch.object(lyrics, "set_cached_api") as cache,
        ):
            self.assertIsNone(
                lyrics.fetch_synced_lyrics("Primary", "Recording", duration=100)
            )
        self.assertEqual(cache.call_args.args[1], "")

    def test_lrclib_requires_matching_title_and_known_duration(self) -> None:
        for title, duration, accepted in (
            ("Original Story", 100, True),
            ("Original Story Again", 100, False),
            ("Original Story", None, False),
            ("Original Story", 0, False),
            ("Original Story", True, False),
            ("Original Story", 200, False),
        ):
            with (
                self.subTest(title=title, duration=duration),
                patch.object(
                    SESSION,
                    "get",
                    return_value=httpx.Response(
                        200,
                        json={
                            "artistName": "Primary",
                            "trackName": title,
                            "duration": duration,
                            "plainLyrics": "Verified words",
                        },
                    ),
                ),
            ):
                result = lyrics._query_lrclib("Primary - Original Story", duration=100)
            self.assertEqual(result is not None, accepted)

    def test_lrclib_transport_errors_and_cached_misses_skip_lyrics(self) -> None:
        with (
            patch.object(SESSION, "get", side_effect=OSError("offline")),
            self.assertRaisesRegex(OSError, "offline"),
        ):
            lyrics._query_lrclib("Primary - Recording")
        with (
            patch.object(lyrics, "get_cached_api", return_value=None),
            patch.object(lyrics, "set_cached_api") as cache,
            patch.object(lyrics, "_query_lrclib", side_effect=OSError("offline")),
            self.assertRaisesRegex(RuntimeError, "offline"),
        ):
            lyrics.fetch_synced_lyrics("Primary", "Recording")
        cache.assert_not_called()
        with (
            patch.object(lyrics, "get_cached_api", return_value=""),
            patch.object(lyrics, "_query_syncedlyrics") as query,
        ):
            self.assertIsNone(lyrics.fetch_synced_lyrics("Primary", "Recording"))
        query.assert_not_called()

    def test_empty_timestamps_and_trailing_embed_artifacts_are_removed(self) -> None:
        self.assertEqual(sidecars.clean_lyrics_text(" \n"), "")
        text = "\n[00:00.00]\nFirst line\n\n\nSecond line\n5Embed\n\n"
        self.assertEqual(sidecars.clean_lyrics_text(text), "First line\n\nSecond line")
        self.assertIsNone(sidecars.extract_max_timestamp(None))
        self.assertEqual(
            sidecars.extract_max_timestamp("[01:02] line\n[02:00:250] line"), 120.25
        )

    def test_lrclib_get_and_search_enforce_lyric_type_and_quality(self) -> None:
        scenarios: tuple[tuple[str, bool, bool, object, str | None], ...] = (
            (
                "Primary - Recording",
                False,
                False,
                {"syncedLyrics": SYNCED, "plainLyrics": "Plain"},
                SYNCED,
            ),
            (
                "Primary - Recording",
                True,
                False,
                {"syncedLyrics": SYNCED, "plainLyrics": "Plain"},
                "Plain",
            ),
            ("Recording", False, True, [{"plainLyrics": "Plain"}], None),
            ("Recording", False, False, [{"plainLyrics": "Plain"}], None),
            ("Recording", False, False, [], None),
            (
                "Primary - Recording",
                False,
                False,
                {"syncedLyrics": "[09:00] Wrong", "plainLyrics": "Plain"},
                "Plain",
            ),
            (
                "Primary - Recording",
                False,
                False,
                {"syncedLyrics": "[00:01] Produced by Someone", "plainLyrics": "Plain"},
                "Plain",
            ),
        )
        for query, plain, synced, payload, expected in scenarios:
            if isinstance(payload, dict):
                payload = {
                    "artistName": "Primary",
                    "trackName": "Recording",
                    "albumName": "Record",
                    "duration": 100,
                    **payload,
                }
            with (
                self.subTest(query=query, plain=plain, synced=synced, payload=payload),
                patch.object(
                    SESSION, "get", return_value=httpx.Response(200, json=payload)
                ) as request,
            ):
                self.assertEqual(
                    lyrics._query_lrclib(
                        query,
                        plain_only=plain,
                        synced_only=synced,
                        album_name="Record",
                        duration=100,
                    ),
                    expected,
                )
            if " - " in query:
                self.assertEqual(
                    request.call_args.kwargs["params"],
                    {
                        "artist_name": "Primary",
                        "track_name": "Recording",
                        "album_name": "Record",
                        "duration": 100,
                    },
                )
            else:
                request.assert_not_called()
        for field in ("syncedLyrics", "plainLyrics"):
            with (
                self.subTest(field=field),
                patch.object(
                    SESSION, "get", return_value=httpx.Response(200, json={field: 123})
                ),
                self.assertRaisesRegex(TypeError, "strings or null"),
            ):
                lyrics._query_lrclib("Primary - Recording")
        with (
            patch.object(SESSION, "get", return_value=httpx.Response(404)),
        ):
            self.assertIsNone(lyrics._query_lrclib("Recording"))

    def test_syncedlyrics_prefers_lrclib_and_passes_explicit_provider_options(
        self,
    ) -> None:
        with (
            patch.object(lyrics, "_query_lrclib", return_value=SYNCED),
            patch("syncedlyrics.search") as search,
        ):
            self.assertEqual(
                lyrics._query_syncedlyrics(
                    "Primary - Recording", False, True, False, None, None
                ),
                SYNCED,
            )
        search.assert_not_called()
        with (
            patch.object(lyrics._LYRICS_LIMITER, "wait"),
            patch("syncedlyrics.search", return_value=SYNCED) as search,
        ):
            self.assertEqual(
                lyrics._query_syncedlyrics(
                    "Primary - Recording",
                    False,
                    True,
                    True,
                    ["Musixmatch"],
                    "en",
                    duration=100,
                ),
                SYNCED,
            )
        search.assert_called_once_with(
            "Primary - Recording",
            plain_only=False,
            synced_only=True,
            enhanced=True,
            providers=["Musixmatch"],
            lang="en",
        )

    def test_cached_lyrics_respect_explicit_quality_and_duration_requirements(
        self,
    ) -> None:
        cases: tuple[tuple[str, dict[str, Any], str | None], ...] = (
            ("", {}, None),
            ("Plain", {"synced_only": True}, SYNCED),
            (SYNCED, {"plain_only": True}, "Plain"),
            (SYNCED, {"enhanced": True}, SYNCED),
            ("[09:00] Wrong", {"duration": 100}, SYNCED),
        )
        for cached, options, expected in cases:
            with (
                self.subTest(cached=cached, options=options),
                patch.object(lyrics, "get_cached_api", return_value=cached),
                patch.object(
                    lyrics, "_query_syncedlyrics", return_value=expected
                ) as query,
            ):
                self.assertEqual(
                    lyrics.fetch_synced_lyrics("Primary", "Recording", **options),
                    expected,
                )
            self.assertEqual(
                query.call_count, 0 if not cached or options.get("enhanced") else 1
            )

    def test_alternative_query_failure_is_reported_with_original_cause(self) -> None:
        failure = OSError("provider offline")
        with (
            patch.object(lyrics, "get_cached_api", return_value=None),
            patch.object(
                lyrics, "_query_syncedlyrics", side_effect=[None, failure]
            ) as query,
            self.assertRaisesRegex(RuntimeError, "provider offline") as raised,
        ):
            lyrics.fetch_synced_lyrics("Primary", "Recording (feat. Guest)")
        self.assertIs(raised.exception.__cause__, failure)
        self.assertEqual(query.call_count, 2)


class TestLyricsLookupState(IsolatedTestCase):
    def test_parallel_track_failures_stop_after_three_provider_requests(self) -> None:
        state: dict[str, Any] = {
            "lock": threading.Lock(),
            "request_slots": threading.BoundedSemaphore(3),
            "failures": 0,
            "suspended": False,
        }
        request = httpx.Request("GET", "https://lrclib.net/api/get")
        error = httpx.HTTPStatusError(
            "overloaded", request=request, response=httpx.Response(503, request=request)
        )
        active = 0
        peak_active = 0
        active_lock = threading.Lock()

        def fail_query(*args: Any, **kwargs: Any) -> None:
            nonlocal active, peak_active
            with active_lock:
                active += 1
                peak_active = max(peak_active, active)
            time.sleep(0.02)
            with active_lock:
                active -= 1
            raise error

        with (
            patch.object(lyrics, "get_cached_api", return_value=None),
            patch.object(
                lyrics, "_query_syncedlyrics", side_effect=fail_query
            ) as query,
            patch.object(lyrics, "set_cached_api"),
            ThreadPoolExecutor(max_workers=8) as executor,
        ):
            list(
                executor.map(
                    lambda index: track_processing.process_track_lyrics(
                        self.test_directory / f"track-{index}.flac",
                        "Primary",
                        f"Track {index}",
                        duration=100,
                        lookup_state=state,
                    ),
                    range(8),
                )
            )

        self.assertTrue(state["suspended"])
        self.assertEqual(query.call_count, 3)
        self.assertEqual(peak_active, 3)

    def test_line_synced_result_is_reused_for_the_same_enhanced_request(self) -> None:
        details: dict[str, Any] = {}
        with (
            patch.object(lyrics, "_query_syncedlyrics", return_value=SYNCED) as query,
        ):
            self.assertEqual(
                lyrics.fetch_synced_lyrics("Primary", "Recording", enhanced=True),
                SYNCED,
            )
            self.assertEqual(
                lyrics.fetch_synced_lyrics(
                    "Primary", "Recording", enhanced=True, details=details
                ),
                SYNCED,
            )
        query.assert_called_once()
        self.assertTrue(details["cache_hit"])
        self.assertEqual(details["lookup_status"], "cache")

    def test_cache_preserves_lrclib_source_and_record_id(self) -> None:
        details: dict[str, Any] = {}
        with (
            patch.object(
                SESSION,
                "get",
                return_value=httpx.Response(
                    200,
                    json={
                        "id": 123,
                        "artistName": "Primary",
                        "trackName": "Recording",
                        "albumName": "Record",
                        "duration": 100,
                        "syncedLyrics": SYNCED,
                    },
                ),
            ) as request,
        ):
            self.assertEqual(
                lyrics.fetch_synced_lyrics(
                    "Primary",
                    "Recording",
                    album_name="Record",
                    duration=100,
                    details=details,
                ),
                SYNCED,
            )
            self.assertEqual(details["lookup_status"], "found")
            details.clear()
            self.assertEqual(
                lyrics.fetch_synced_lyrics(
                    "Primary",
                    "Recording",
                    album_name="Record",
                    duration=100,
                    details=details,
                ),
                SYNCED,
            )
        request.assert_called_once()
        self.assertEqual(details["source"], "LRCLIB")
        self.assertEqual(details["lookup_source"], "LRCLIB")
        self.assertEqual(details["record_id"], 123)
        self.assertTrue(details["cache_hit"])

    def test_repeated_failures_suspend_network_but_do_not_cache_misses(self) -> None:
        state: dict[str, Any] = {
            "lock": threading.Lock(),
            "failures": 0,
            "suspended": False,
        }
        details: dict[str, Any] = {}
        request = httpx.Request("GET", "https://lrclib.net/api/get")
        error = httpx.HTTPStatusError(
            "overloaded", request=request, response=httpx.Response(503, request=request)
        )
        details["reason"] = "Local lyric headers conflict with the recording."
        with (
            patch.object(lyrics, "_query_syncedlyrics", side_effect=error) as query,
            patch.object(lyrics, "set_cached_api") as cache,
            patch.object(LOG, "warning") as warning,
        ):
            for index in range(3):
                with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
                    lyrics.fetch_synced_lyrics(
                        "Primary", str(index), lookup_state=state, details=details
                    )
            self.assertIsNone(
                lyrics.fetch_synced_lyrics(
                    "Primary", "Deferred", lookup_state=state, details=details
                )
            )
        self.assertTrue(state["suspended"])
        self.assertEqual(details["lookup_status"], "deferred")
        self.assertEqual(
            details["reason"], "Local lyric headers conflict with the recording."
        )
        self.assertIn("suspended", details["lookup_reason"])
        self.assertEqual(query.call_count, 3)
        cache.assert_not_called()
        warning.assert_called_once()
        with (
            patch.object(lyrics, "get_cached_api", return_value=SYNCED),
            patch.object(lyrics, "_query_syncedlyrics") as query,
        ):
            self.assertEqual(
                lyrics.fetch_synced_lyrics(
                    "Primary", "Cached", lookup_state=state, details=details
                ),
                SYNCED,
            )
        query.assert_not_called()
        self.assertEqual(details["lookup_status"], "cache")

    def test_rate_limit_suspends_further_requests_in_the_run(self) -> None:
        for status in (429, 502, 503):
            state: dict[str, Any] = {
                "lock": threading.Lock(),
                "failures": 0,
                "suspended": False,
            }
            request = httpx.Request("GET", "https://lrclib.net/api/get")
            response = httpx.Response(
                status, request=request, headers={"Retry-After": "120"}
            )
            error = httpx.HTTPStatusError(
                "retry later", request=request, response=response
            )
            with (
                self.subTest(status=status),
                patch.object(lyrics, "_query_syncedlyrics", side_effect=error) as query,
                patch.object(lyrics, "set_cached_api") as cache,
            ):
                with self.assertRaisesRegex(RuntimeError, f"HTTP {status}"):
                    lyrics.fetch_synced_lyrics("Primary", "Limited", lookup_state=state)
                self.assertIsNone(
                    lyrics.fetch_synced_lyrics("Primary", "Next", lookup_state=state)
                )
            self.assertTrue(state["suspended"])
            self.assertEqual(query.call_count, 1)
            cache.assert_not_called()

    def test_rate_limit_short_cooldown_resumes_after_expiry(self) -> None:
        state: dict[str, Any] = {
            "lock": threading.Lock(),
            "failures": 0,
            "suspended": False,
        }
        request = httpx.Request("GET", "https://lrclib.net/api/get")
        response_429 = httpx.Response(
            429, request=request, headers={"Retry-After": "5"}
        )
        error_429 = httpx.HTTPStatusError(
            "slow down", request=request, response=response_429
        )

        with (
            patch.object(lyrics, "_query_syncedlyrics", side_effect=error_429),
            patch.object(lyrics, "set_cached_api"),
            patch("time.monotonic", return_value=100.0),
            self.assertRaisesRegex(RuntimeError, "HTTP 429"),
        ):
            lyrics.fetch_synced_lyrics("Primary", "Limited", lookup_state=state)

        # Permanent suspension was NOT engaged because cooldown was <= 60s
        self.assertFalse(state["suspended"])
        self.assertGreater(state.get("suspended_until", 0.0), 100.0)

        # Immediate follow-up while still within cooldown window is deferred
        with patch("time.monotonic", return_value=102.0):
            details: dict[str, Any] = {}
            self.assertIsNone(
                lyrics.fetch_synced_lyrics(
                    "Primary", "Next", lookup_state=state, details=details
                )
            )
            self.assertEqual(details["lookup_status"], "deferred")

        # After cooldown timestamp expires, lookups proceed
        with (
            patch("time.monotonic", return_value=110.0),
            patch.object(lyrics, "_query_syncedlyrics", return_value=SYNCED),
        ):
            res = lyrics.fetch_synced_lyrics("Primary", "Recovered", lookup_state=state)
            self.assertEqual(res, SYNCED)

    def test_dns_lockout_does_not_count_towards_failure_threshold(self) -> None:
        state: dict[str, Any] = {
            "lock": threading.Lock(),
            "failures": 0,
            "suspended": False,
        }
        request = httpx.Request("GET", "https://example.com")
        dns_lockout_error = httpx.ConnectError(
            "DNS lookup temporarily suspended after a resolver failure",
            request=request,
        )

        with (
            patch.object(lyrics, "_query_syncedlyrics", side_effect=dns_lockout_error),
            patch.object(lyrics, "set_cached_api"),
        ):
            for _ in range(5):
                with self.assertRaisesRegex(
                    RuntimeError, "DNS lookup temporarily suspended"
                ):
                    lyrics.fetch_synced_lyrics("Primary", "Track", lookup_state=state)

        # 5 fast-fails during DNS suspension must NOT trip failures counter or suspended flag
        self.assertEqual(state["failures"], 0)
        self.assertFalse(state["suspended"])

    def test_definitive_miss_resets_consecutive_failure_count(self) -> None:
        state: dict[str, Any] = {
            "lock": threading.Lock(),
            "failures": 2,
            "suspended": False,
        }
        details: dict[str, Any] = {}
        with patch.object(lyrics, "_query_syncedlyrics", return_value=None):
            self.assertIsNone(
                lyrics.fetch_synced_lyrics(
                    "Primary", "Recording", lookup_state=state, details=details
                )
            )
        self.assertEqual(state["failures"], 0)
        self.assertEqual(details["lookup_status"], "unavailable")

    def test_authenticated_transport_failure_is_not_cached_as_a_miss(self) -> None:
        with (
            patch.object(
                lyrics,
                "_query_authenticated_musixmatch",
                side_effect=OSError("offline"),
            ),
            patch.object(lyrics, "set_cached_api") as cache,
            self.assertRaisesRegex(RuntimeError, "offline"),
        ):
            lyrics.fetch_synced_lyrics(
                "Primary",
                "Recording",
                providers=["Musixmatch"],
                musixmatch_token="fixture-token",
            )
        cache.assert_not_called()


class TestLocalLyricsContracts(IsolatedTestCase):
    def setUp(self) -> None:
        self.audio_path = self.test_directory / "recording.flac"
        self.audio_path.touch()
        self.lrc_path = self.audio_path.with_suffix(".lrc")

    def test_embedded_synchronized_lyrics_restore_companion_without_network(
        self,
    ) -> None:
        for content in (SYNCED, ENHANCED):
            self.lrc_path.unlink(missing_ok=True)
            with (
                self.subTest(content=content),
                patch.object(lyrics, "fetch_synced_lyrics") as search,
            ):
                self.assertEqual(
                    track_processing.process_track_lyrics(
                        self.audio_path,
                        "Primary",
                        "Recording",
                        duration=100,
                        existing_lyrics=content,
                    ),
                    (content, "existing"),
                )
            search.assert_not_called()
            self.assertEqual(self.lrc_path.read_text(), content)

    def test_dry_run_can_restore_lyrics_without_writing_companion(self) -> None:
        result = track_processing.process_track_lyrics(
            self.audio_path,
            "Primary",
            "Recording",
            duration=100,
            existing_lyrics=ENHANCED,
            dry_run=True,
        )
        self.assertEqual(result, (ENHANCED, "existing"))
        self.assertFalse(self.lrc_path.exists())

    def test_lookup_failure_preserves_existing_synchronized_lyrics(self) -> None:
        self.lrc_path.write_text(SYNCED)
        with patch.object(
            lyrics, "fetch_synced_lyrics", side_effect=OSError("offline")
        ):
            self.assertEqual(
                track_processing.process_track_lyrics(
                    self.audio_path, "Primary", "Recording", duration=100, force=True
                ),
                (SYNCED, "existing"),
            )
        self.assertEqual(self.lrc_path.read_text(), SYNCED)

    def test_missing_duration_and_mismatched_timestamps_are_rejected(self) -> None:
        for duration, content in ((None, SYNCED), (10, "[09:00] Wrong")):
            with (
                self.subTest(duration=duration),
                patch("sonora.audio.metadata.get_audio_duration", return_value=None),
                patch.object(lyrics, "fetch_synced_lyrics", return_value=content),
            ):
                self.assertEqual(
                    track_processing.process_track_lyrics(
                        self.audio_path, "Primary", "Recording", duration=duration
                    ),
                    (None, None),
                )
        self.assertFalse(self.lrc_path.exists())

    def test_unreadable_lyrics_are_replaced_only_after_successful_lookup(self) -> None:
        self.lrc_path.write_text(SYNCED)
        with (
            patch.object(
                Path, "read_text", side_effect=PermissionError("locked lyrics")
            ),
            patch.object(lyrics, "fetch_synced_lyrics", return_value=ENHANCED),
        ):
            self.assertEqual(
                track_processing.process_track_lyrics(
                    self.audio_path, "Primary", "Recording", duration=100
                ),
                (ENHANCED, "enhanced"),
            )
        self.assertEqual(self.lrc_path.read_text(), ENHANCED)

    def test_lyrics_duration_valid_default_tolerance_fifteen_seconds(self) -> None:
        """Verify default overrun tolerance is 15.0s conforming to GEMINI.md."""
        audio_duration = 180.0
        lyrics_14s_overrun = "[03:14.00] Final lyric within 15s overrun"
        self.assertTrue(
            sidecars.is_lyrics_duration_valid(lyrics_14s_overrun, audio_duration)
        )
        lyrics_exact_boundary = "[03:15.00] Exactly 15s overrun"
        self.assertTrue(
            sidecars.is_lyrics_duration_valid(lyrics_exact_boundary, audio_duration)
        )
        lyrics_15_1s_overrun = "[03:15.10] Over 15s overrun"
        self.assertFalse(
            sidecars.is_lyrics_duration_valid(lyrics_15_1s_overrun, audio_duration)
        )
        lyrics_20s_overrun = "[03:20.00] Severely overlong lyrics"
        self.assertFalse(
            sidecars.is_lyrics_duration_valid(lyrics_20s_overrun, audio_duration)
        )

    def test_lyrics_duration_valid_explicit_tolerance_override(self) -> None:
        """Verify passing an explicit tolerance_seconds overrides the 15.0s default."""
        audio_duration = 100.0
        lyrics_5s_overrun = "[01:45.00] 5s overrun"
        self.assertFalse(
            sidecars.is_lyrics_duration_valid(
                lyrics_5s_overrun, audio_duration, tolerance_seconds=2.0
            )
        )
        self.assertTrue(
            sidecars.is_lyrics_duration_valid(
                lyrics_5s_overrun, audio_duration, tolerance_seconds=10.0
            )
        )

    def test_lyrics_duration_valid_offset_adjustment_with_default_tolerance(
        self,
    ) -> None:
        """Verify [offset:ms] metadata adjusts effective timestamp against duration."""
        audio_duration = 100.0
        lyrics_pos_offset = "[offset:1000]\n[01:54.00] Words"
        self.assertTrue(
            sidecars.is_lyrics_duration_valid(lyrics_pos_offset, audio_duration)
        )
        lyrics_neg_offset = "[offset:-2000]\n[01:54.00] Words"
        self.assertFalse(
            sidecars.is_lyrics_duration_valid(lyrics_neg_offset, audio_duration)
        )

    def test_process_track_lyrics_accepts_companion_within_fifteen_second_overrun(
        self,
    ) -> None:
        """End-to-end processing preserves companion lyrics within 15s overrun without re-fetching."""
        lyrics_content = "[01:52.00] Words ending 12s after audio"
        self.lrc_path.write_text(lyrics_content)
        with patch.object(lyrics, "fetch_synced_lyrics") as lookup:
            text, status = track_processing.process_track_lyrics(
                self.audio_path, "Primary", "Recording", duration=100.0
            )
        lookup.assert_not_called()
        self.assertEqual(text, lyrics_content)
        self.assertEqual(status, "existing")


class TestLyricsHeaderIdentity(IsolatedTestCase):
    def test_matching_headers_features_and_missing_headers_are_accepted(self) -> None:
        for content in (
            "[ar:Alpha feat. Beta]\n[ti:Recording]\n[00:01] Words",
            "\ufeff[AR: Alpha][TI: Recording]\n[00:01] Words",
            "[al:Release][ar:Alpha][ti:Recording]\nWords",
            "[00:01] Words without identity headers",
        ):
            with self.subTest(content=content):
                self.assertTrue(
                    sidecars.lyrics_identity_matches(content, "Alpha", "Recording")
                )

    def test_missing_and_placeholder_targets_do_not_reject_available_headers(
        self,
    ) -> None:
        content = "[ar:Alpha]\n[ti:Recording]\nWords"
        for artist, title in (
            (None, None),
            ("Unknown Artist", "Unknown Title"),
            ("", "Track 12"),
            ("Alpha", "Untitled"),
        ):
            with self.subTest(artist=artist, title=title):
                self.assertTrue(
                    sidecars.lyrics_identity_matches(content, artist, title)
                )

    def test_conflicting_artist_title_and_duplicate_headers_are_rejected(self) -> None:
        for content in (
            "Wrong Song Lyrics\nWords",
            '[Versuri "Wrong Song"]\nWords',
            "[Versuri „Wrong Song”]\nWords",
            "[Versurile piesei „Wrong Song” cu Alpha]\nWords",
            "[Versuri pentru „Wrong Song” cu Alpha]\nWords",
            "[Versuri pentru „Wrong Song“ cu Alpha]\nWords",
            '[Paroles de  "Wrong Song"]\nWords',
        ):
            self.assertFalse(
                sidecars.lyrics_identity_matches(content, "Alpha", "Recording")
            )
        self.assertTrue(
            sidecars.lyrics_identity_matches(
                "Recording Lyrics\nWords", "Alpha", "Recording"
            )
        )
        for content in (
            "[ar:Another performer]\n[ti:Recording]\nWords",
            "[ar:Alpha]\n[ti:Unrelated song]\nWords",
            "[ar:Alpha]\n[ti:Recording]\n[ti:Unrelated song]\nWords",
        ):
            with self.subTest(content=content):
                self.assertFalse(
                    sidecars.lyrics_identity_matches(content, "Alpha", "Recording")
                )

    def test_configured_artist_aliases_are_resolved_offline(self) -> None:
        with patch(
            "sonora.core.artists._load_user_overrides",
            return_value={"former credit": "Current Credit"},
        ):
            self.assertTrue(
                sidecars.lyrics_identity_matches(
                    "[ar:Former Credit]\n[ti:Recording]", "Current Credit", "Recording"
                )
            )

    def test_literal_header_like_text_after_timestamp_does_not_set_identity(
        self,
    ) -> None:
        self.assertTrue(
            sidecars.lyrics_identity_matches(
                "[00:01] Singing [ti:Another song]", "Alpha", "Recording"
            )
        )

    def test_timestamped_lyric_ending_in_lyrics_is_not_a_title_header(self) -> None:
        self.assertTrue(
            sidecars.lyrics_identity_matches("[00:01.00] Lyrics", "Alpha", "Recording")
        )

    def test_bracket_stripped_candidate_title_header_matches_clean_title(
        self,
    ) -> None:
        """Candidate title headers containing bracketed subtitles match clean track titles."""
        scenarios = (
            "[ti:Capu' sus (Viața noastră)]\n[00:01.00] Versuri",
            "Capu' sus (Viața noastră) Lyrics\n[00:01.00] Versuri",
            "[Versurile piesei „Capu' sus (Viața noastră)” cu B.U.G. Mafia]\n[00:01.00] Versuri",
            "[Versuri „Capu' sus (Viața noastră)”]\n[00:01.00] Versuri",
            "[ti:Recording (Official Audio)]\n[00:01.00] Words",
            "[ti:Recording (Bonus Track)]\n[00:01.00] Words",
        )
        for content in scenarios:
            expected_title = "Capu' sus" if "Capu" in content else "Recording"
            with self.subTest(content=content, expected_title=expected_title):
                self.assertTrue(
                    sidecars.lyrics_identity_matches(
                        content, "B.U.G. Mafia", expected_title
                    )
                )

    def test_different_songs_are_strictly_rejected_even_with_brackets(self) -> None:
        """Prefixes and similar words representing distinct songs must never match."""
        conflicting_cases = (
            ("[ti:Homies]\n[00:01.00] Words", "Alpha", "Home"),
            ("[ti:Home]\n[00:01.00] Words", "Alpha", "Homies"),
            ("[ti:Homies (Remix)]\n[00:01.00] Words", "Alpha", "Home"),
            ("[ti:Different Song]\n[00:01.00] Words", "Alpha", "Recording"),
            ("[ti:Stand by You]\n[00:01.00] Words", "Alpha", "Stand by Me"),
        )
        for content, artist, title in conflicting_cases:
            with self.subTest(content=content, title=title):
                self.assertFalse(
                    sidecars.lyrics_identity_matches(content, artist, title)
                )


class TestExistingLyricsValidation(IsolatedTestCase):
    def setUp(self) -> None:
        self.audio_path = self.test_directory / "recording.flac"
        self.audio_path.touch()
        self.lrc_path = self.audio_path.with_suffix(".lrc")

    def test_missing_lyrics_does_not_scan_siblings_when_lookup_fails(self) -> None:
        with (
            patch.object(lyrics, "fetch_synced_lyrics", return_value=None),
            patch.object(track_processing, "find_audio_files") as find_audio_files,
        ):
            self.assertEqual(
                track_processing.process_track_lyrics(
                    self.audio_path, "Alpha", "Recording", duration=100
                ),
                (None, None),
            )
        find_audio_files.assert_not_called()

    def test_sidecar_duration_is_checked_once(self) -> None:
        from sonora.core import sidecars

        self.lrc_path.write_text(
            "[ar:Alpha]\n[ti:Recording]\nPlain lyric text without timestamps"
        )
        with (
            patch.object(lyrics, "fetch_synced_lyrics", return_value=None),
            patch.object(
                track_processing,
                "is_lyrics_duration_valid",
                wraps=sidecars.is_lyrics_duration_valid,
            ) as check_duration,
        ):
            self.assertEqual(
                track_processing.process_track_lyrics(
                    self.audio_path, "Alpha", "Recording", duration=100
                ),
                (
                    "[ar:Alpha]\n[ti:Recording]\nPlain lyric text without timestamps",
                    "existing",
                ),
            )
        check_duration.assert_called_once()

    def test_fetched_duplicate_lyrics_are_checked_against_siblings(self) -> None:
        body = "A complete verse with enough distinctive words to compare. " * 12
        sibling = self.audio_path.with_name("different.flac")
        sibling.touch()
        conflicts: dict[str, str] = {}
        with (
            patch.object(lyrics, "fetch_synced_lyrics", return_value=body),
            patch.object(
                metadata_module,
                "read_track_metadata",
                return_value=TrackInfo(
                    sibling, artist="Alpha", title="Different", lyrics=body
                ),
            ),
        ):
            text, _ = track_processing.process_track_lyrics(
                self.audio_path,
                "Alpha",
                "Recording",
                duration=100,
                conflicts=conflicts,
            )
        self.assertIsNone(text)
        self.assertIn("lyrics", conflicts)
        self.assertFalse(self.lrc_path.exists())

    def test_up_tempo_version_can_share_original_lyrics(self) -> None:
        body = "A complete verse with enough distinctive words to compare. " * 12
        sibling = self.audio_path.with_name("birthday-sex.flac")
        sibling.touch()
        self.lrc_path.write_text(f"[ar:Jeremih]\n[ti:Birthday Sex (Up-Tempo)]\n{body}")
        conflicts: dict[str, str] = {}
        with (
            patch.object(lyrics, "fetch_synced_lyrics", return_value=None),
            patch.object(
                metadata_module,
                "read_track_metadata",
                return_value=TrackInfo(
                    sibling,
                    artist="Jeremih",
                    title="Birthday Sex",
                    lyrics=body,
                ),
            ),
        ):
            text, status = track_processing.process_track_lyrics(
                self.audio_path,
                "Jeremih",
                "Birthday Sex (Up-Tempo)",
                duration=237,
                conflicts=conflicts,
            )

        self.assertIsNotNone(text)
        self.assertEqual(status, "existing")
        self.assertNotIn("lyrics", conflicts)

    def test_cached_sibling_metadata_avoids_rereading_album_files(self) -> None:
        body = "A complete verse with enough distinctive words to compare. " * 12
        sibling = self.audio_path.with_name("different.flac")
        sibling_metadata = {
            sibling: TrackInfo(sibling, artist="Alpha", title="Different", lyrics=body)
        }
        conflicts: dict[str, str] = {}
        with (
            patch.object(lyrics, "fetch_synced_lyrics", return_value=body),
            patch.object(track_processing, "find_audio_files") as find_audio_files,
            patch.object(metadata_module, "read_track_metadata") as read_track_metadata,
        ):
            text, status = track_processing.process_track_lyrics(
                self.audio_path,
                "Alpha",
                "Recording",
                duration=100,
                sibling_metadata=sibling_metadata,
                conflicts=conflicts,
            )

        self.assertIsNone(text)
        self.assertIsNone(status)
        self.assertIn("lyrics", conflicts)
        find_audio_files.assert_not_called()
        read_track_metadata.assert_not_called()

    def test_lyrics_run_reuses_sibling_metadata_between_tracks(self) -> None:
        paths = [self.audio_path]
        paths.extend(self.test_directory / f"track-{index}.flac" for index in (2, 3))
        body = "A complete verse with enough distinctive words to compare. " * 12
        metadata = {}
        for index, path in enumerate(paths, start=1):
            path.touch(exist_ok=True)
            title = f"Recording {index}"
            path.with_suffix(".lrc").write_text(f"[ar:Alpha]\n[ti:{title}]\n{body}")
            metadata[path] = TrackInfo(
                path, artist="Alpha", title=title, lyrics=f"[ti:{title}]\n{body}"
            )
        lookup_state: dict[str, Any] = {"lock": threading.Lock()}
        with (
            patch.object(lyrics, "fetch_synced_lyrics", return_value=None),
            patch.object(
                track_processing, "find_audio_files", return_value=paths
            ) as find,
            patch.object(
                metadata_module, "read_track_metadata", side_effect=metadata.get
            ) as read,
            patch(
                "sonora.modules.track_processing.clean_title", wraps=clean_title
            ) as clean_title_mock,
        ):
            for index, path in enumerate(paths, start=1):
                track_processing.process_track_lyrics(
                    path,
                    "Alpha",
                    f"Recording {index}",
                    duration=100,
                    force=True,
                    lookup_state=lookup_state,
                )

        find.assert_called_once_with(self.test_directory, recursive=False)
        self.assertEqual(read.call_count, len(paths))
        self.assertEqual(clean_title_mock.call_count, 3 * len(paths))

    def test_instrumentals_never_import_vocal_lyrics(self) -> None:
        for title, album in (
            ("Recording (Instrumental)", "Record"),
            ("Recording", "Record Instrumentals"),
        ):
            for force in (False, True):
                for existing in (False, True):
                    self.lrc_path.unlink(missing_ok=True)
                    if existing:
                        self.lrc_path.write_text("[00:01.00] Vocal words")
                    conflicts: dict[str, str] = {}
                    with (
                        self.subTest(title=title, force=force, existing=existing),
                        patch.object(lyrics, "fetch_synced_lyrics") as lookup,
                    ):
                        text, status = track_processing.process_track_lyrics(
                            self.audio_path,
                            "Alpha",
                            title,
                            album_name=album,
                            duration=100,
                            force=force,
                            conflicts=conflicts,
                        )
                    lookup.assert_not_called()
                    self.assertIsNone(text)
                    self.assertEqual(status, "review" if existing else None)
                    self.assertEqual("lyrics" in conflicts, existing)
                    if existing:
                        self.assertEqual(
                            self.lrc_path.read_text(), "[00:01.00] Vocal words"
                        )

    def test_existing_enhanced_lyrics_are_cleaned_without_lookup(self) -> None:
        self.lrc_path.write_text(
            "[offset:100]\n[00:01.00]<00:01.00> Words\n"
            "<script>tracking()</script>\nYou might also like\n42Embed"
        )
        with patch.object(lyrics, "fetch_synced_lyrics") as lookup:
            text, status = track_processing.process_track_lyrics(
                self.audio_path, "Alpha", "Recording", duration=100
            )
        lookup.assert_not_called()
        self.assertEqual(status, "cleaned")
        assert text is not None
        self.assertIn("[offset:100]", text)
        self.assertIn("<00:01.00> Words", text)
        self.assertNotIn("tracking", text)
        self.assertEqual(self.lrc_path.read_text(), text)

    def test_same_artist_different_song_body_requires_review(self) -> None:
        body = "A complete verse with enough distinctive words to compare. " * 12
        self.lrc_path.write_text(body)
        sibling = self.audio_path.with_name("different.flac")
        sibling.touch()
        conflicts: dict[str, str] = {}
        with (
            patch.object(lyrics, "fetch_synced_lyrics", return_value=body),
            patch.object(
                metadata_module,
                "read_track_metadata",
                return_value=TrackInfo(
                    sibling, artist="Alpha", title="Different", lyrics=body
                ),
            ),
        ):
            text, _ = track_processing.process_track_lyrics(
                self.audio_path, "Alpha", "Recording", duration=100, conflicts=conflicts
            )
        self.assertIsNone(text)
        self.assertIn("lyrics", conflicts)
        self.assertEqual(self.lrc_path.read_text(), body)

        replacement = "Distinct words for the verified recording. " * 12
        with (
            patch.object(lyrics, "fetch_synced_lyrics", return_value=replacement),
            patch.object(
                metadata_module,
                "read_track_metadata",
                return_value=TrackInfo(
                    sibling, artist="Alpha", title="Different", lyrics=body
                ),
            ),
        ):
            text, _ = track_processing.process_track_lyrics(
                self.audio_path,
                "Alpha",
                "Recording",
                duration=100,
                conflicts=conflicts,
                force=True,
            )
        self.assertEqual(text, replacement.strip())
        self.assertNotIn("lyrics", conflicts)

    def test_collaboration_remix_cannot_import_original_versions_lyrics(self) -> None:
        original_body = (
            "The original singer performs the original verse and chorus. " * 12
        )
        different_body = (
            "A guest performs a new verse with different words and meaning. " * 12
        )
        remix = self.audio_path.with_name("Song (Remix).flac")
        remix.touch()
        for force in (False, True):
            for candidate, accepted in (
                (original_body + "!", False),
                (different_body, True),
            ):
                sidecar = remix.with_suffix(".lrc")
                sidecar.unlink(missing_ok=True)
                conflicts: dict[str, str] = {}
                with (
                    self.subTest(force=force, accepted=accepted),
                    patch.object(lyrics, "fetch_synced_lyrics", return_value=candidate),
                    patch.object(
                        metadata_module,
                        "read_track_metadata",
                        return_value=TrackInfo(
                            self.audio_path,
                            artist="Primary",
                            title="Song",
                            lyrics=original_body,
                        ),
                    ),
                ):
                    text, _quality = track_processing.process_track_lyrics(
                        remix,
                        "Primary & Guest",
                        "Song (Remix)",
                        duration=100,
                        force=force,
                        conflicts=conflicts,
                    )
                self.assertEqual(text is not None, accepted)
                self.assertEqual(sidecar.exists(), accepted)
                self.assertEqual("lyrics" in conflicts, not accepted)

    def test_duplicate_original_body_is_not_copied_from_embedding_to_remix_sidecar(
        self,
    ) -> None:
        body = "The original singer performs this entire original verse. " * 12
        remix = self.audio_path.with_name("Song (Remix).flac")
        remix.touch()
        conflicts: dict[str, str] = {}
        with (
            patch.object(lyrics, "fetch_synced_lyrics", return_value=None),
            patch.object(
                metadata_module,
                "read_track_metadata",
                return_value=TrackInfo(
                    self.audio_path,
                    artist="Primary",
                    title="Song",
                    lyrics=body,
                ),
            ),
        ):
            result = track_processing.process_track_lyrics(
                remix,
                "Primary & Guest",
                "Song (Remix)",
                duration=100,
                existing_lyrics=body,
                conflicts=conflicts,
            )
        self.assertEqual(result, (None, None))
        self.assertFalse(remix.with_suffix(".lrc").exists())
        self.assertIn("lyrics", conflicts)

    def test_conflicting_plain_synced_and_enhanced_lyrics_are_preserved_but_not_reused(
        self,
    ) -> None:
        for body in ("Words", "[00:01] Words", "[00:01]<00:01.00> Words"):
            payload = ("[ar:Alpha]\n[ti:Unrelated song]\n" + body).encode()
            self.lrc_path.write_bytes(payload)
            with (
                self.subTest(body=body),
                patch.object(
                    lyrics, "fetch_synced_lyrics", return_value=None
                ) as search,
                patch.object(LOG, "warning") as warning,
            ):
                self.assertEqual(
                    track_processing.process_track_lyrics(
                        self.audio_path, "Alpha", "Recording", duration=100
                    ),
                    (None, None),
                )
            search.assert_called_once()
            warning.assert_called()
            self.assertEqual(self.lrc_path.read_bytes(), payload)

    def test_conflicting_embedded_text_cannot_restore_rejected_sidecar(self) -> None:
        payload = "[ar:Alpha]\n[ti:Unrelated song]\n[00:01]<00:01.00> Words"
        for embedded in (payload, "[00:01]<00:01.00> Words"):
            self.lrc_path.write_text(payload)
            with (
                self.subTest(embedded=embedded),
                patch.object(lyrics, "fetch_synced_lyrics", return_value=None),
            ):
                self.assertEqual(
                    track_processing.process_track_lyrics(
                        self.audio_path,
                        "Alpha",
                        "Recording",
                        duration=100,
                        existing_lyrics=embedded,
                    ),
                    (None, None),
                )
            self.assertEqual(self.lrc_path.read_text(), payload)

    def test_matching_replacement_can_replace_conflicting_higher_quality_asset(
        self,
    ) -> None:
        self.lrc_path.write_text(
            "[ar:Other]\n[ti:Another song]\n[00:01]<00:01.00> Wrong"
        )
        replacement = "[ar:Alpha]\n[ti:Recording]\n[00:01] Correct"
        with patch.object(lyrics, "fetch_synced_lyrics", return_value=replacement):
            self.assertEqual(
                track_processing.process_track_lyrics(
                    self.audio_path, "Alpha", "Recording", duration=100
                ),
                (replacement, "synced"),
            )
        self.assertEqual(self.lrc_path.read_text(), replacement)

    def test_dry_run_replacement_leaves_rejected_original_unchanged(self) -> None:
        original = b"[ar:Other]\n[ti:Another song]\nWords"
        self.lrc_path.write_bytes(original)
        replacement = "[ar:Alpha]\n[ti:Recording]\n[00:01] Correct"
        with patch.object(lyrics, "fetch_synced_lyrics", return_value=replacement):
            self.assertEqual(
                track_processing.process_track_lyrics(
                    self.audio_path, "Alpha", "Recording", duration=100, dry_run=True
                ),
                (replacement, "synced"),
            )
        self.assertEqual(self.lrc_path.read_bytes(), original)

    def test_overlong_local_and_embedded_lyrics_are_not_returned(self) -> None:
        for content in (
            "[ar:Alpha]\n[ti:Recording]\n[09:00] Words",
            "[ar:Alpha]\n[ti:Recording]\n[00:01]<09:00.00> Words",
        ):
            self.lrc_path.write_text(content)
            with (
                self.subTest(content=content),
                patch.object(lyrics, "fetch_synced_lyrics", return_value=None),
            ):
                self.assertEqual(
                    track_processing.process_track_lyrics(
                        self.audio_path,
                        "Alpha",
                        "Recording",
                        duration=100,
                        existing_lyrics=content,
                    ),
                    (None, None),
                )
            self.assertEqual(self.lrc_path.read_text(), content)

    def test_conflicting_fetched_headers_do_not_replace_valid_local_lyrics(
        self,
    ) -> None:
        original = "[ar:Alpha]\n[ti:Recording]\n[00:01] Words"
        self.lrc_path.write_text(original)
        with patch.object(
            lyrics,
            "fetch_synced_lyrics",
            return_value="[ar:Alpha]\n[ti:Wrong song]\n[00:01]<00:01.00> Words",
        ):
            self.assertEqual(
                track_processing.process_track_lyrics(
                    self.audio_path, "Alpha", "Recording", duration=100, force=True
                ),
                (original, "existing"),
            )
        self.assertEqual(self.lrc_path.read_text(), original)

    def test_invalid_utf8_is_not_decoded_by_dropping_bytes(self) -> None:
        payload = b"[ar:Alpha]\n[ti:Recording]\n[00:01]caf\xe9"
        self.lrc_path.write_bytes(payload)
        conflicts: dict[str, str] = {}
        with patch.object(lyrics, "fetch_synced_lyrics", return_value=None) as lookup:
            self.assertEqual(
                track_processing.process_track_lyrics(
                    self.audio_path,
                    "Alpha",
                    "Recording",
                    duration=100,
                    conflicts=conflicts,
                ),
                (None, "review"),
            )
        self.assertEqual(self.lrc_path.read_bytes(), payload)

        self.assertIn("UTF-8", conflicts["lyrics"])
        lookup.assert_not_called()

    def test_unreadable_sidecar_remains_a_review_conflict(self) -> None:
        self.lrc_path.write_text("[ar:Alpha]\n[ti:Recording]\n[00:01] words")
        conflicts: dict[str, str] = {}
        original_read_text = Path.read_text

        def fail_sidecar(path: Path, *args: Any, **kwargs: Any) -> str:
            if path == self.lrc_path:
                raise PermissionError("unreadable")
            return original_read_text(path, *args, **kwargs)

        with (
            patch.object(Path, "read_text", fail_sidecar),
            patch.object(lyrics, "fetch_synced_lyrics", return_value=None),
        ):
            track_processing.process_track_lyrics(
                self.audio_path,
                "Alpha",
                "Recording",
                duration=100,
                conflicts=conflicts,
            )

        self.assertIn("lyrics", conflicts)


class TestLyricsHeaderRepair(IsolatedTestCase):
    def setUp(self) -> None:
        self.audio_path = self.test_directory / "recording.flac"
        self.audio_path.write_bytes(b"unchanged audio")
        self.lrc_path = self.audio_path.with_suffix(".lrc")

    def test_verified_words_preserve_enhanced_timing_and_repair_only_headers(
        self,
    ) -> None:
        original = (
            "[ar:Former Credit][ti:Unrelated song]\n"
            "[00:01.00] <00:01.00> Hello, <00:01.50> world!"
        )
        expected = original.replace("Former Credit", "Alpha").replace(
            "Unrelated song", "Recording"
        )
        for force in (False, True):
            self.lrc_path.write_text(original)
            with (
                self.subTest(force=force),
                patch.object(lyrics, "fetch_synced_lyrics", return_value="Hello world"),
            ):
                result = track_processing.process_track_lyrics(
                    self.audio_path, "Alpha", "Recording", duration=100, force=force
                )
            self.assertEqual(result, (expected, "enhanced"))
            self.assertEqual(self.lrc_path.read_text(), expected)
            self.assertEqual(self.audio_path.read_bytes(), b"unchanged audio")

    def test_different_words_or_invalid_timing_cannot_be_preserved(self) -> None:
        for body in (
            "[00:01.00] <00:01.00> Different words",
            "[09:00.00] <09:00.00> Correct words",
        ):
            original = "[ar:Other]\n[ti:Wrong song]\n" + body
            self.lrc_path.write_text(original)
            with (
                self.subTest(body=body),
                patch.object(
                    lyrics, "fetch_synced_lyrics", return_value="Correct words"
                ),
            ):
                result = track_processing.process_track_lyrics(
                    self.audio_path, "Alpha", "Recording", duration=100
                )
            self.assertEqual(result, ("Correct words", "plain"))
            self.assertEqual(self.lrc_path.read_text(), "Correct words")

    def test_dry_run_header_repair_preserves_original_bytes(self) -> None:
        original = b"[ar:Other]\r\n[ti:Wrong song]\r\n[00:01.00] Correct words"
        self.lrc_path.write_bytes(original)
        with patch.object(lyrics, "fetch_synced_lyrics", return_value="Correct words"):
            result = track_processing.process_track_lyrics(
                self.audio_path,
                "Alpha",
                "Recording",
                duration=100,
                dry_run=True,
                force=True,
            )
        self.assertEqual(
            result,
            ("[ar:Alpha]\n[ti:Recording]\n[00:01.00] Correct words", "synced"),
        )
        self.assertEqual(self.lrc_path.read_bytes(), original)
        self.assertEqual(self.audio_path.read_bytes(), b"unchanged audio")

    def test_conflicting_embedded_headers_can_retain_enhanced_timing(self) -> None:
        original = "[ar:Other]\n[00:01.00] <00:01.00> Correct words"
        with patch.object(lyrics, "fetch_synced_lyrics", return_value="Correct words"):
            result = track_processing.process_track_lyrics(
                self.audio_path,
                "Alpha",
                "Recording",
                duration=100,
                existing_lyrics=original,
            )
        expected = "[ar:Alpha]\n[00:01.00] <00:01.00> Correct words"
        self.assertEqual(result, (expected, "enhanced"))
        self.assertEqual(self.lrc_path.read_text(), expected)

    def test_unverified_provider_opt_in_cannot_establish_header_repair_identity(
        self,
    ) -> None:
        self.lrc_path.write_text("[ar:Other]\n[00:01.00] <00:01.00> Correct words")
        with (
            patch.object(lyrics, "fetch_synced_lyrics", return_value="Correct words"),
            patch.object(
                track_processing,
                "get_config",
                return_value=replace(get_config(), allow_unverified_lyrics=True),
            ),
        ):
            result = track_processing.process_track_lyrics(
                self.audio_path, "Alpha", "Recording", duration=100
            )
        self.assertEqual(result, ("Correct words", "plain"))
        self.assertEqual(self.lrc_path.read_text(), "Correct words")

    def test_stricter_duration_policy_does_not_reuse_looser_cached_lyrics(
        self,
    ) -> None:
        cached_lyrics: dict[str, str] = {}
        configuration = get_config()
        with (
            patch.object(lyrics, "get_cached_api", side_effect=cached_lyrics.get),
            patch.object(
                lyrics, "set_cached_api", side_effect=cached_lyrics.__setitem__
            ),
            patch.object(
                lyrics,
                "_query_syncedlyrics",
                side_effect=["Loosely matched words", None],
            ) as query,
        ):
            results = []
            for tolerance in (15.0, 2.0):
                with patch.object(
                    lyrics,
                    "get_config",
                    return_value=replace(
                        configuration,
                        single_duration_tolerance_seconds=tolerance,
                    ),
                ):
                    results.append(
                        lyrics.fetch_synced_lyrics("Alpha", "Recording", duration=100)
                    )
        self.assertEqual(results, ["Loosely matched words", None])
        self.assertEqual(query.call_count, 2)
        self.assertEqual(len(cached_lyrics), 2)
        self.assertEqual(
            {json.loads(key[key.index("[") :])[-2] for key in cached_lyrics},
            {15.0, 2.0},
        )

    def test_literal_bracket_in_artist_keeps_validated_remote_lyrics(self) -> None:
        self.lrc_path.write_text(
            "[ar:Other]\n[ti:Wrong song]\n[00:01.00] Correct words"
        )
        with patch.object(lyrics, "fetch_synced_lyrics", return_value="Correct words"):
            result = track_processing.process_track_lyrics(
                self.audio_path,
                "Alpha] Omega",
                "Recording",
                duration=100,
            )
        self.assertEqual(result, ("Correct words", "plain"))
        self.assertEqual(self.lrc_path.read_text(), "Correct words")

    def test_script_concordance_rejects_foreign_alphabets(self) -> None:
        self.assertFalse(
            sidecars.is_script_concordant("Samurai - Loial", "НЕМНОГО ГРУСТНЫЙ самураЙ")
        )
        self.assertFalse(
            sidecars.is_script_concordant("Shift - Vecina", "Вампирский щит")
        )
        self.assertFalse(
            sidecars.is_script_concordant("Augxst - LV", "[Август '20] (August '20)")
        )
        self.assertTrue(
            sidecars.is_script_concordant("Samurai - Loial", "Sunt loial pana la capat")
        )
        self.assertTrue(
            sidecars.is_script_concordant("Стресс - Интро", "Стресс начинается здесь")
        )
        self.assertTrue(
            sidecars.is_script_concordant(
                "21 Savage - Glock in My Lap",
                "Millionaire, still shakin' kill\u0435rs hand (woah)",
            )
        )
        self.assertTrue(
            sidecars.is_script_concordant(
                "$kinny - Tamam",
                "Opening the track with my squad Salaam Alaikum (\u200fاَلسَلامُ عَلَيْكُم) we ride out",
            )
        )
        self.assertTrue(
            sidecars.is_script_concordant(
                "Connect-R - Am Sosul Ăla",
                "\u0410m \u0455\u043e\u0455ul \u0430l\u0430, uuh-uuh",
            )
        )
        self.assertTrue(
            sidecars.is_script_concordant(
                "Andro - Bolen tvoey ulybkoy",
                "Факел разжег я во тьме... Болен твоей улыбкой",
                title="Bolen tvoey ulybkoy",
            )
        )
        self.assertTrue(
            sidecars.is_script_concordant(
                "Andro - Krasivaja",
                "Красивая моя девочка...",
                title="Krasivaja",
            )
        )
        self.assertFalse(
            sidecars.is_script_concordant(
                "B.U.G. Mafia - Un 2 și trei de 0",
                "baby 너여야만 해 너 없인 평소와 같ul 순 없어",
                title="Un 2 și trei de 0",
            )
        )
        self.assertFalse(
            sidecars.is_script_concordant(
                "Connect-R - Baby Don't Cry",
                "看着我们 困倦眼神",
                title="Baby Don't Cry",
            )
        )
        self.assertTrue(
            sidecars.lyrics_identity_matches(
                '[ti:Legea Gravitației]\nLegea gravitației Lyrics\n[Versuri "Secretul din atom (Remix)"]',
                "C.T.C.",
                "Legea Gravitației",
            )
        )

    def test_evaluate_lyrics_quality_scoring_and_rejections(self) -> None:
        self.assertEqual(sidecars.evaluate_lyrics_quality(None), 0.0)
        self.assertEqual(sidecars.evaluate_lyrics_quality("   "), 0.0)

        # Identity rejection (wrong song title)
        wrong_song = "[ar:2Scratch]\n[ti:Home]\nHomies Lyrics\n[00:10.00] Chilling with my homies"
        self.assertEqual(
            sidecars.evaluate_lyrics_quality(wrong_song, "2Scratch", "Home", 120.0),
            0.0,
        )

        # Script mismatch rejection
        cyrillic_lyrics = "[ar:Minzie]\n[ti:Rose]\n[00:10.00] Пиковару канон"
        self.assertEqual(
            sidecars.evaluate_lyrics_quality(cyrillic_lyrics, "Minzie", "Rose", 120.0),
            0.0,
        )

        # Bilingual track on Cyrillic album is concordant when album context is provided
        jony_lyrics = "[ar:JONY]\n[ti:Love Your Voice]\n[00:20.00] My baby I love your voice\n[00:40.00] Без тебя мне холодно"
        self.assertGreater(
            sidecars.evaluate_lyrics_quality(
                jony_lyrics,
                artist="JONY",
                title="Love Your Voice",
                album="Список твоих мыслей",
                audio_duration=150.0,
            ),
            0.0,
        )

        # Duration overrun rejection (> 15s overrun)
        overrun_lyrics = "[00:10.00] Line one\n[02:30.00] Line two"
        self.assertEqual(
            sidecars.evaluate_lyrics_quality(overrun_lyrics, "Artist", "Song", 100.0),
            0.0,
        )

        # Synced with good word density and coverage
        lines = [
            f"[{i // 60:02d}:{i % 60:02d}.00] Word line {i} with several lyrics tokens"
            for i in range(5, 95, 5)
        ]
        good_synced = "\n".join(lines)
        score = sidecars.evaluate_lyrics_quality(good_synced, "Artist", "Song", 100.0)
        self.assertGreaterEqual(score, 70.0)

    def test_lyrics_consensus_awards_bonus_for_independent_agreement(self) -> None:
        candidate = "[00:10.00] We will rock you and stand together all night long"
        matching = "[00:10.50] We will rock you and stand together all night long"
        different = "[00:10.00] Completely different lyrics from another track"

        self.assertGreater(
            sidecars.score_lyrics_consensus(candidate, [matching]),
            0.0,
        )
        self.assertEqual(
            sidecars.score_lyrics_consensus(candidate, [different]),
            0.0,
        )

    def test_force_purges_corrupt_mismatched_lyrics_when_no_remote_found(self) -> None:
        self.lrc_path.write_text(
            "[ar:2Scratch]\n[ti:Home]\nHomies Lyrics\n[00:10.00] Chilling with my homies"
        )
        track_info = TrackInfo(
            self.audio_path,
            artist="2Scratch",
            title="Home",
            lyrics=self.lrc_path.read_text(),
            duration=160.0,
        )
        with patch.object(lyrics, "fetch_synced_lyrics", return_value=None):
            result, status = track_processing.process_track_lyrics(
                self.audio_path,
                "2Scratch",
                "Home",
                force=True,
                duration=160.0,
                existing_lyrics=track_info.lyrics,
            )
            self.assertEqual(status, "purged")
            self.assertEqual(result, "")
            self.assertFalse(self.lrc_path.exists())

            track_processing._update_lyrics(
                track_info,
                self.audio_path,
                fetch_lyrics=True,
                force=True,
            )
            self.assertIsNone(track_info.lyrics)
