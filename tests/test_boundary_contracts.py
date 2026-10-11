import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx

from sonora.audio import checksum
from sonora.cli import audio
from sonora.cli.main import main
from sonora.core import cache, config
from sonora.core.config import SonoraConfig
from sonora.core.http import SESSION
from sonora.core.models import TrackInfo
from sonora.core.text import preserve_unicode_repertoire
from sonora.core.titles import score_artist_title_match
from sonora.services import (
    deezer,
    discogs,
    fanart,
    genius,
    itunes,
    musicbrainz,
    shazam,
    theaudiodb,
)
from tests.support import IsolatedTestCase

MBID = "11111111-2222-3333-4444-555555555555"


class TestConfigurationAndCommandBoundaries(IsolatedTestCase):
    def test_equal_accent_counts_preserve_repertoire_and_artist_articles_match(
        self,
    ) -> None:
        self.assertEqual(preserve_unicode_repertoire("Cáfe", "Café"), "Cáfe")
        self.assertEqual(preserve_unicode_repertoire("CAFE", "Cafe"), "Cafe")
        self.assertEqual(
            score_artist_title_match(
                "The Ensemble", "Recording", "Ensemble", "Recording"
            ),
            100,
        )

    def test_boolean_file_configuration_defaults_and_rejects_wrong_types(self) -> None:
        for settings, expected in (
            ({}, True),
            ({"enable_shazam": False}, False),
            ({"enable_shazam": "false"}, None),
        ):
            config.clear_config_cache()
            with (
                self.subTest(settings=settings),
                patch.dict(os.environ, {}, clear=True),
                patch.object(config, "load_app_environment"),
                patch.object(
                    config, "_discover_config_file", return_value=Path("config.json")
                ),
                patch.object(config, "_parse_config_data", return_value=settings),
            ):
                if expected is None:
                    with self.assertRaises(TypeError):
                        config.get_config()
                else:
                    self.assertEqual(config.get_config().enable_shazam, expected)

    def test_cache_directory_change_replaces_the_open_client(self) -> None:
        original = cache.get_cache()
        assert original is not None
        with patch.dict(
            os.environ, {"XDG_CACHE_HOME": str(self.test_directory / "other")}
        ):
            replacement = cache.get_cache()
            assert replacement is not None
            self.assertIsNot(original, replacement)
            self.assertNotEqual(original.directory, replacement.directory)

    def test_invalid_command_has_usage_exit_status(self) -> None:
        self.assertEqual(main(["nonexistent-command"]), 2)

    def test_existing_audio_analysis_tags_skip_decoding(self) -> None:
        path = self.test_directory / "recording.flac"
        recording = TrackInfo(path, bpm=123, initial_key="Am")
        with (
            patch.object(audio, "read_track_metadata", return_value=recording),
            patch.object(audio, "calculate_bpm") as tempo,
            patch.object(audio, "detect_key_details") as key,
        ):
            self.assertEqual(
                audio._process_bpm_file(path, False, False), (path, 123, False)
            )
            self.assertEqual(
                audio._process_key_file(path, False, False), (path, "Am", False)
            )
        tempo.assert_not_called()
        key.assert_not_called()
        with self.assertRaises(FileNotFoundError):
            checksum.verify_flac_checksum(path)


class TestProviderBoundaryFailures(IsolatedTestCase):
    def test_deezer_album_transport_status_and_cover_fallback_failure(self) -> None:
        with (
            patch.object(deezer, "get_cached_api", return_value=None),
            patch.object(SESSION, "get", return_value=httpx.Response(503)),
        ):
            self.assertIsNone(deezer.fetch_deezer_album_details("Primary", "Record"))
        with (
            patch.object(deezer, "get_cached_api", return_value=None),
            patch.object(
                SESSION,
                "get",
                side_effect=[
                    httpx.Response(200, json={"data": []}),
                    httpx.Response(503),
                ],
            ),
        ):
            self.assertIsNone(deezer.fetch_deezer_cover_art_url("Primary", "Record"))

    def test_shazam_ignores_unrelated_metadata_and_actions_but_keeps_lyrics(
        self,
    ) -> None:
        path = self.test_directory / "recording.flac"
        path.write_bytes(b"audio fixture")
        payload = {
            "track": {
                "title": "Recording",
                "subtitle": "Primary",
                "sections": [
                    {"type": "SONG", "metadata": [{"title": "Length", "text": "100"}]},
                    {"type": "VIDEO"},
                    {"type": "LYRICS", "text": ["", "Line", " "]},
                ],
                "hub": {"actions": [{"type": "uri", "uri": "https://example.com"}]},
            }
        }
        with (
            patch.object(
                shazam, "get_config", return_value=SonoraConfig(enable_shazam=True)
            ),
            patch.object(shazam, "get_cached_api", return_value=None),
            patch.object(shazam._SHAZAM_LIMITER, "wait"),
            patch.object(
                shazam, "_recognize_async", new=AsyncMock(return_value=payload)
            ),
        ):
            result = shazam.recognize_audio_track(path)
        assert result is not None
        self.assertEqual(result.lyrics, "Line")
        self.assertIsNone(result.apple_music_id)

    def test_theaudiodb_non_success_artist_response_has_no_artwork(self) -> None:
        with (
            patch.object(theaudiodb, "get_cached_api", return_value=None),
            patch.object(SESSION, "get", return_value=httpx.Response(503)),
        ):
            self.assertEqual(theaudiodb.fetch_artist_images("Primary"), (None, None))

    def test_theaudiodb_track_transport_failure_preserves_a_missing_result(
        self,
    ) -> None:
        with (
            patch.object(theaudiodb, "get_cached_api", return_value=None),
            patch.object(SESSION, "get", side_effect=OSError("offline")),
        ):
            self.assertIsNone(
                theaudiodb.fetch_theaudiodb_track_details("Primary", "Recording")
            )

    def test_itunes_cover_fuzzy_editions_and_series_are_distinct(self) -> None:
        for album, candidates, expected in (
            ("Record", [{"artistName": "Wrong", "collectionName": "Wrong"}], None),
            (
                "Record",
                [
                    {
                        "artistName": "Primary",
                        "collectionName": "Record (Deluxe)",
                        "artworkUrl100": "https://example.com/100x100bb.jpg",
                    }
                ],
                "https://example.com/1200x1200bb.jpg",
            ),
            (
                "Record",
                [
                    {
                        "artistName": "Primary",
                        "collectionName": "Record II",
                        "artworkUrl100": "https://example.com/100x100bb.jpg",
                    }
                ],
                None,
            ),
            (
                "Record",
                [
                    {
                        "artistName": "Primary",
                        "collectionName": "Record",
                        "artworkUrl100": "https://example.com/200x200bb.jpg",
                    }
                ],
                "https://example.com/1200x1200bb.jpg",
            ),
            ("Record", [{"artistName": "Primary", "collectionName": "Record"}], None),
        ):
            with (
                self.subTest(album=album, candidates=candidates),
                patch.object(itunes, "search_itunes", return_value=candidates),
            ):
                self.assertEqual(
                    itunes.fetch_itunes_cover_art_url(
                        "Primary", album, resolution=1200
                    ),
                    expected,
                )

    def test_shazam_empty_results_and_invalid_lyric_arrays(self) -> None:
        path = self.test_directory / "recording.flac"
        path.write_bytes(b"audio fixture")
        for payload in (
            None,
            {
                "track": {
                    "title": "Recording",
                    "subtitle": "Primary",
                    "sections": [{"type": "LYRICS", "text": ["valid", 12]}],
                }
            },
        ):
            with (
                self.subTest(payload=payload),
                patch.object(
                    shazam, "get_config", return_value=SonoraConfig(enable_shazam=True)
                ),
                patch.object(shazam, "get_cached_api", return_value=None),
                patch.object(shazam._SHAZAM_LIMITER, "wait"),
                patch.object(
                    shazam, "_recognize_async", new=AsyncMock(return_value=payload)
                ),
            ):
                if payload is None:
                    self.assertIsNone(shazam.recognize_audio_track(path))
                else:
                    with self.assertRaisesRegex(TypeError, "array of strings"):
                        shazam.recognize_audio_track(path)
        with (
            patch.object(
                shazam, "get_config", return_value=SonoraConfig(enable_shazam=True)
            ),
            patch.object(
                shazam,
                "get_cached_api",
                return_value={"title": "Recording", "artist": "Primary"},
            ),
        ):
            result = shazam.recognize_audio_track(path)
        assert result is not None
        self.assertEqual((result.title, result.artist), ("Recording", "Primary"))

    def test_theaudiodb_empty_tracks_and_normalized_track_fields(self) -> None:
        payloads: tuple[dict[str, Any], ...] = (
            {"track": None},
            {"track": []},
            {
                "track": [
                    {
                        "strMusicVid": "https://example.com",
                        "strMood": " Chill ",
                        "strOpenKey": "Am",
                        "intScore": "8.5",
                    }
                ]
            },
        )
        for payload in payloads:
            with (
                self.subTest(payload=payload),
                patch.object(theaudiodb, "get_cached_api", return_value=None),
                patch.object(
                    SESSION, "get", return_value=httpx.Response(200, json=payload)
                ),
            ):
                details = theaudiodb.fetch_theaudiodb_track_details(
                    "Primary", "Recording"
                )
            if payload["track"]:
                assert details is not None
                self.assertEqual(
                    (details["mood"], details["initial_key"], details["rating"]),
                    ("Chill", "Am", 8.5),
                )
            else:
                self.assertIsNone(details)
        self.assertIsNone(theaudiodb._clean_theaudiodb_str(" null "))

    def test_fanart_download_error_and_discogs_non_success_status(self) -> None:
        with patch.object(SESSION, "get", side_effect=OSError("offline")):
            self.assertIsNone(fanart.download_fanart_image_bytes("https://example.com"))
        with (
            patch.object(discogs, "get_cached_api", return_value=None),
            patch.object(SESSION, "get", return_value=httpx.Response(503)),
        ):
            self.assertIsNone(discogs.fetch_discogs_release_details(1))
            self.assertIsNone(
                discogs.search_discogs_release("Primary", "Record", "fixture")
            )
        self.assertIsNone(genius._normalize_genius_description(None))

    def test_musicbrainz_empty_results_cached_misses_and_discography_failure(
        self,
    ) -> None:
        with patch.object(musicbrainz, "get_cached_api", return_value=""):
            self.assertIsNone(musicbrainz.fetch_track_mbid("Primary", "Recording"))
        with (
            patch.object(musicbrainz, "get_cached_api", return_value=None),
            patch.object(
                musicbrainz, "_call_musicbrainz", return_value={"recording-list": []}
            ),
        ):
            self.assertIsNone(musicbrainz.fetch_track_mbid("Primary", "Recording"))
        with (
            patch.object(musicbrainz, "get_cached_api", return_value=None),
            patch.object(
                musicbrainz, "_call_musicbrainz", side_effect=OSError("offline")
            ),
        ):
            self.assertEqual(musicbrainz.fetch_artist_discography("Primary"), [])

    def test_itunes_track_metadata_and_unmatched_candidates(self) -> None:
        recording = {
            "trackName": "Recording",
            "artistName": "Primary",
            "trackId": 1,
            "artistId": 2,
            "collectionId": 3,
            "country": "US",
            "releaseDate": "2020-01-02T00:00:00Z",
            "trackExplicitness": "explicit",
            "primaryGenreName": "Rock",
        }
        with patch.object(
            itunes,
            "search_itunes",
            return_value=[{"trackName": "Other", "artistName": "Unrelated"}, recording],
        ):
            details = itunes.fetch_itunes_track_metadata("Primary", "Recording")
        assert details is not None
        self.assertEqual(
            (details["itunes_trackid"], details["advisory"], details["date"]),
            ("1", "Explicit", "2020-01-02"),
        )
        with patch.object(
            itunes,
            "search_itunes",
            return_value=[{"trackName": "Other", "artistName": "Unrelated"}],
        ):
            self.assertIsNone(
                itunes.fetch_itunes_track_metadata("Primary", "Recording")
            )
