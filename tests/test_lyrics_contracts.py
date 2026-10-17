import json
import os
import urllib.parse
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx

from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.services import lyrics
from tests.support import IsolatedTestCase

SYNCED = "[00:01.00] First line\n[00:02.00] Second line"
ENHANCED = "[00:01.00] <00:01.00> First <00:01.50> line"


class TestLyricsTokenContracts(IsolatedTestCase):
    def test_malformed_cookie_json_remains_a_raw_token(self) -> None:
        raw_token = '{"tokens":'
        with patch(
            "syncedlyrics.utils.get_cache_path", return_value=self.test_directory
        ):
            self.assertTrue(lyrics.init_musixmatch_token(raw_token))
        self.assertEqual(
            json.loads((self.test_directory / "musixmatch_token.json").read_text())[
                "token"
            ],
            raw_token,
        )

    def test_raw_and_encoded_tokens_use_provider_cache_format(self) -> None:
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
        for raw_token in tokens:
            with (
                self.subTest(raw_token=raw_token),
                patch(
                    "syncedlyrics.utils.get_cache_path",
                    return_value=self.test_directory,
                ),
                patch("sonora.services.lyrics.time.time", return_value=100),
            ):
                self.assertTrue(lyrics.init_musixmatch_token(raw_token))
            stored = json.loads(
                (self.test_directory / "musixmatch_token.json").read_text()
            )
            self.assertEqual(
                stored, {"token": "fixture-token", "expiration_time": 31536100}
            )

    def test_missing_tokens_and_unwritable_cache_are_not_successes(self) -> None:
        for raw_token in (None, "", " ", '{"tokens": {}}', '{"token": null}'):
            with self.subTest(raw_token=raw_token):
                self.assertFalse(lyrics.init_musixmatch_token(raw_token))
        with (
            patch.dict(os.environ, {"MUSIXMATCH_USER_TOKEN": "fixture-token"}),
            patch(
                "syncedlyrics.utils.get_cache_path", return_value=self.test_directory
            ),
            patch.object(
                Path, "write_text", side_effect=PermissionError("read-only cache")
            ),
        ):
            self.assertFalse(lyrics.init_musixmatch_token())


class TestLyricsParsingContracts(IsolatedTestCase):
    def test_remote_lyrics_preview_does_not_create_a_companion_file(self) -> None:
        path = self.test_directory / "recording.flac"
        with patch.object(lyrics, "fetch_synced_lyrics", return_value=SYNCED):
            self.assertEqual(
                lyrics.process_track_lyrics(
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

    def test_existing_lyrics_that_disappear_after_quality_read_are_not_overwritten_by_downgrades(
        self,
    ) -> None:
        audio_path = self.test_directory / "recording.flac"
        audio_path.with_suffix(".lrc").write_text(SYNCED)
        for replacement, force in (("Plain lyrics", True), (SYNCED, False)):
            with (
                self.subTest(replacement=replacement, force=force),
                patch.object(
                    Path, "read_text", side_effect=[SYNCED, OSError("file locked")]
                ),
                patch.object(lyrics, "fetch_synced_lyrics", return_value=replacement),
            ):
                self.assertEqual(
                    lyrics.process_track_lyrics(
                        audio_path, "Primary", "Recording", duration=100, force=force
                    ),
                    (None, None),
                )
        self.assertEqual(audio_path.with_suffix(".lrc").read_text(), SYNCED)

    def test_equal_quality_remote_lyrics_preserve_existing_local_text(self) -> None:
        audio_path = self.test_directory / "recording.flac"
        audio_path.with_suffix(".lrc").write_text(SYNCED)
        with patch.object(
            lyrics, "fetch_synced_lyrics", return_value="[00:01] Replacement"
        ):
            self.assertEqual(
                lyrics.process_track_lyrics(
                    audio_path, "Primary", "Recording", duration=100
                ),
                (SYNCED, "existing"),
            )

    def test_inline_embed_artifacts_are_removed_without_losing_lyrics(self) -> None:
        self.assertEqual(
            lyrics.clean_lyrics_text("First line\nSecond line12Embed"),
            "First line\nSecond line",
        )
        self.assertEqual(lyrics.clean_lyrics_text("First line\nEmbed"), "First line")

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
                patch.object(lyrics._LRCLIB_LIMITER, "wait"),
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
            patch.object(lyrics._LRCLIB_LIMITER, "wait"),
            patch.object(SESSION, "get", side_effect=OSError("offline")),
        ):
            self.assertIsNone(lyrics._query_lrclib("Primary - Recording"))
        with (
            patch.object(lyrics, "get_cached_api", return_value=""),
            patch.object(lyrics, "_query_syncedlyrics") as query,
        ):
            self.assertIsNone(lyrics.fetch_synced_lyrics("Primary", "Recording"))
        query.assert_not_called()

    def test_empty_timestamps_and_trailing_embed_artifacts_are_removed(self) -> None:
        self.assertEqual(lyrics.clean_lyrics_text(" \n"), "")
        text = "\n[00:00.00]\nFirst line\n\n\nSecond line\n5Embed\n\n"
        self.assertEqual(lyrics.clean_lyrics_text(text), "First line\n\nSecond line")
        self.assertIsNone(lyrics.extract_max_timestamp(None))
        self.assertEqual(
            lyrics.extract_max_timestamp("[01:02] line\n[02:00:250] line"), 120.25
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
                    "duration": 100,
                    **payload,
                }
            with (
                self.subTest(query=query, plain=plain, synced=synced, payload=payload),
                patch.object(lyrics._LRCLIB_LIMITER, "wait"),
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
                patch.object(lyrics._LRCLIB_LIMITER, "wait"),
                patch.object(
                    SESSION, "get", return_value=httpx.Response(200, json={field: 123})
                ),
                self.assertRaisesRegex(TypeError, "strings or null"),
            ):
                lyrics._query_lrclib("Primary - Recording")
        with (
            patch.object(lyrics._LRCLIB_LIMITER, "wait"),
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
            (SYNCED, {"enhanced": True}, ENHANCED),
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
            self.assertEqual(query.call_count, 0 if not cached else 1)

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


class TestLocalLyricsContracts(IsolatedTestCase):
    def setUp(self) -> None:
        self.audio_path = self.test_directory / "recording.flac"
        self.audio_path.touch()
        self.lrc_path = self.audio_path.with_suffix(".lrc")

    def test_embedded_enhanced_lyrics_restore_missing_companion_without_network(
        self,
    ) -> None:
        with patch.object(lyrics, "fetch_synced_lyrics") as search:
            self.assertEqual(
                lyrics.process_track_lyrics(
                    self.audio_path,
                    "Primary",
                    "Recording",
                    duration=100,
                    existing_lyrics=ENHANCED,
                ),
                (ENHANCED, "existing"),
            )
        search.assert_not_called()
        self.assertEqual(self.lrc_path.read_text(), ENHANCED)

    def test_dry_run_can_restore_lyrics_without_writing_companion(self) -> None:
        result = lyrics.process_track_lyrics(
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
                lyrics.process_track_lyrics(
                    self.audio_path, "Primary", "Recording", duration=100, force=True
                ),
                (SYNCED, "existing"),
            )
        self.assertEqual(self.lrc_path.read_text(), SYNCED)

    def test_missing_duration_and_mismatched_timestamps_are_rejected(self) -> None:
        for duration, content in ((None, SYNCED), (10, "[09:00] Wrong")):
            with (
                self.subTest(duration=duration),
                patch.object(lyrics, "get_audio_duration", return_value=None),
                patch.object(lyrics, "fetch_synced_lyrics", return_value=content),
            ):
                self.assertEqual(
                    lyrics.process_track_lyrics(
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
                lyrics.process_track_lyrics(
                    self.audio_path, "Primary", "Recording", duration=100
                ),
                (ENHANCED, "enhanced"),
            )
        self.assertEqual(self.lrc_path.read_text(), ENHANCED)


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
                    lyrics.lyrics_identity_matches(content, "Alpha", "Recording")
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
                self.assertTrue(lyrics.lyrics_identity_matches(content, artist, title))

    def test_conflicting_artist_title_and_duplicate_headers_are_rejected(self) -> None:
        for content in (
            "[ar:Another performer]\n[ti:Recording]\nWords",
            "[ar:Alpha]\n[ti:Unrelated song]\nWords",
            "[ar:Alpha]\n[ti:Recording]\n[ti:Unrelated song]\nWords",
        ):
            with self.subTest(content=content):
                self.assertFalse(
                    lyrics.lyrics_identity_matches(content, "Alpha", "Recording")
                )

    def test_configured_artist_aliases_are_resolved_offline(self) -> None:
        with patch(
            "sonora.core.artists._load_user_overrides",
            return_value={"former credit": "Current Credit"},
        ):
            self.assertTrue(
                lyrics.lyrics_identity_matches(
                    "[ar:Former Credit]\n[ti:Recording]", "Current Credit", "Recording"
                )
            )

    def test_literal_header_like_text_after_timestamp_does_not_set_identity(
        self,
    ) -> None:
        self.assertTrue(
            lyrics.lyrics_identity_matches(
                "[00:01] Singing [ti:Another song]", "Alpha", "Recording"
            )
        )


class TestExistingLyricsValidation(IsolatedTestCase):
    def setUp(self) -> None:
        self.audio_path = self.test_directory / "recording.flac"
        self.audio_path.touch()
        self.lrc_path = self.audio_path.with_suffix(".lrc")

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
                    lyrics.process_track_lyrics(
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
                    lyrics.process_track_lyrics(
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
                lyrics.process_track_lyrics(
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
                lyrics.process_track_lyrics(
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
                    lyrics.process_track_lyrics(
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
                lyrics.process_track_lyrics(
                    self.audio_path, "Alpha", "Recording", duration=100, force=True
                ),
                (original, "existing"),
            )
        self.assertEqual(self.lrc_path.read_text(), original)

    def test_invalid_utf8_is_not_decoded_by_dropping_bytes(self) -> None:
        payload = b"[ar:Alpha]\n[ti:Recording]\n[00:01]caf\xe9"
        self.lrc_path.write_bytes(payload)
        with patch.object(lyrics, "fetch_synced_lyrics", return_value=None):
            self.assertEqual(
                lyrics.process_track_lyrics(
                    self.audio_path, "Alpha", "Recording", duration=100
                ),
                (None, None),
            )
        self.assertEqual(self.lrc_path.read_bytes(), payload)
