from pathlib import Path
from unittest.mock import patch

import httpx

from sonora.audio.metadata import read_track_metadata, write_track_metadata
from sonora.core import artists
from sonora.core.cache import set_cached_api
from sonora.core.config import SonoraConfig
from sonora.core.http import SESSION
from sonora.core.models import TrackInfo
from sonora.modules import catalog_enrichment, musicbrainz_enrichment
from sonora.services import lyrics, musicbrainz, shazam
from sonora.services.shazam import ShazamTrackInfo
from tests.regression_support import AudioRegressionTestCase

MBID = "11111111-2222-3333-4444-555555555555"


class TestReportMismatchRegressions(AudioRegressionTestCase):
    def test_stored_isrc_cannot_replace_meaningful_standalone_identity(self) -> None:
        for artist, title, provider_artist, provider_title in (
            ("2Scratch", "Ice Cream", "Teenage Joans", "Ice Cream"),
            (
                "Bvcovia & Marko Glass",
                "BRR BRR",
                "Bvcovia & Marko Glass",
                "Nu Te Mai Recunosc",
            ),
            (
                "Daniel Di Angelo",
                "Ride for Me",
                "Longshot, Psalm One & Ka Di",
                "How You Like It?",
            ),
            ("Ivan B", "The Jump", "Ivan B", "Like the Way"),
        ):
            with self.subTest(artist=artist, title=title):
                recording = TrackInfo(
                    self.audio("recording.flac"),
                    artist=artist,
                    title=title,
                    isrc="USAAA2400001",
                    musicbrainz_trackid=MBID,
                    featured_artists=provider_artist,
                )
                provider_recording = {
                    "artist": provider_artist,
                    "title": provider_title,
                    "isrc": recording.isrc,
                    "composer": "Wrong Writer",
                }
                with (
                    patch.object(
                        musicbrainz,
                        "fetch_musicbrainz_recording_details",
                        return_value=provider_recording,
                    ),
                    patch.object(musicbrainz, "fetch_track_mbid", return_value=None),
                ):
                    self.assertIsNone(
                        musicbrainz_enrichment._fetch_standalone_recording(
                            recording, force=True
                        )
                    )
                self.assertIsNone(recording.musicbrainz_trackid)
                write_track_metadata(recording)
                saved = read_track_metadata(recording.file_path)
                self.assertEqual((saved.artist, saved.title), (artist, title))
                self.assertIsNone(saved.composer)

    def test_forced_research_revalidates_the_replacement_recording(self) -> None:
        recording = TrackInfo(
            self.audio("recording.flac"),
            artist="Alpha",
            title="Original Song",
            musicbrainz_trackid=MBID,
        )
        with (
            patch.object(
                musicbrainz,
                "fetch_musicbrainz_recording_details",
                return_value={"artist": "Unrelated", "title": "Wrong Song"},
            ) as fetch,
            patch.object(musicbrainz, "fetch_track_mbid", return_value=MBID) as search,
        ):
            self.assertIsNone(
                musicbrainz_enrichment._fetch_standalone_recording(
                    recording, force=True
                )
            )
        self.assertEqual(fetch.call_count, 2)
        search.assert_called_once_with("Alpha", "Original Song")
        self.assertIsNone(recording.musicbrainz_trackid)

    def test_standalone_matching_identity_and_missing_identity_with_isrc_are_accepted(
        self,
    ) -> None:
        provider = {"artist": "Alpha", "title": "Original Song", "isrc": "USAAA2400001"}
        for title in ("Original Song", "Untitled"):
            with self.subTest(title=title):
                recording = TrackInfo(
                    Path("recording.flac"),
                    artist="Alpha",
                    title=title,
                    isrc="USAAA2400001",
                    musicbrainz_trackid=MBID,
                )
                with patch.object(
                    musicbrainz,
                    "fetch_musicbrainz_recording_details",
                    return_value=provider,
                ):
                    self.assertEqual(
                        musicbrainz_enrichment._fetch_standalone_recording(recording),
                        provider,
                    )

    def test_forced_shazam_cannot_hijack_a_complete_standalone_track(self) -> None:
        recording = TrackInfo(
            self.audio("recording.flac"),
            artist="Alpha",
            title="Original Song",
            album="Original Album",
            isrc="USAAA2400001",
        )
        for recognized in (
            ShazamTrackInfo("Original Song", "Unrelated", genre="Wrong"),
            ShazamTrackInfo("Wrong Song", "Alpha", genre="Wrong"),
        ):
            with (
                self.subTest(recognized=recognized),
                patch.object(
                    catalog_enrichment,
                    "get_config",
                    return_value=SonoraConfig(enable_shazam=True),
                ),
                patch.object(shazam, "recognize_audio_track", return_value=recognized),
            ):
                catalog_enrichment._enrich_shazam(
                    recording, recording.file_path, force=True
                )
            self.assertEqual(
                (recording.artist, recording.title, recording.album),
                ("Alpha", "Original Song", "Original Album"),
            )
            self.assertIsNone(recording.genre)

    def test_lrclib_rejects_wrong_artist_title_and_recording_duration(self) -> None:
        for overrides in (
            {"artistName": "Unrelated"},
            {"trackName": "Wrong Song"},
            {"duration": 90},
            {},
        ):
            payload = {
                "artistName": "Alpha",
                "trackName": "Original Song",
                "duration": 180,
                "syncedLyrics": "[00:01.00] Verified",
                **overrides,
            }
            with (
                self.subTest(overrides=overrides),
                patch.object(lyrics._LRCLIB_LIMITER, "wait"),
                patch.object(
                    SESSION,
                    "get",
                    return_value=httpx.Response(200, json=payload),
                ),
            ):
                self.assertEqual(
                    lyrics._query_lrclib("Alpha - Original Song", duration=180),
                    None if overrides else "[00:01.00] Verified",
                )

    def test_release_base_title_does_not_remove_existing_edition(self) -> None:
        for album in (
            "Encore (Deluxe Version)",
            "Original Song (Remixes)",
            "Original Song (Radio Edit)",
        ):
            with self.subTest(album=album):
                recording = TrackInfo(
                    Path("recording.flac"),
                    artist="Alpha",
                    album_artist="Alpha",
                    album=album,
                    musicbrainz_albumid=MBID,
                )
                release = {"title": album.split(" (")[0], "album_artist": "Alpha"}
                with patch.object(
                    artists,
                    "resolve_artist_name",
                    return_value="Alpha",
                ):
                    musicbrainz_enrichment._apply_mb_release_metadata(
                        recording, MBID, release, force=True
                    )
                self.assertEqual(recording.album, album)

    def test_lyrics_cached_before_identity_validation_are_not_reused(self) -> None:
        set_cached_api(
            "lyrics:alpha:original song::[false, false, false, [], null, null, null]",
            "[00:01.00] Wrong Song",
        )
        with patch.object(
            lyrics, "_query_syncedlyrics", return_value="[00:01.00] Verified"
        ) as lookup:
            self.assertEqual(
                lyrics.fetch_synced_lyrics("Alpha", "Original Song"),
                "[00:01.00] Verified",
            )
        lookup.assert_called_once()

    def test_sdk_cannot_retry_a_rejected_lrclib_match_without_identity_checks(
        self,
    ) -> None:
        for providers in (None, ["Lrclib"]):
            with (
                self.subTest(providers=providers),
                patch.object(lyrics, "_query_lrclib", return_value=None),
                patch.object(lyrics._LYRICS_LIMITER, "wait"),
                patch("syncedlyrics.search", return_value=None) as search,
            ):
                self.assertIsNone(
                    lyrics._query_syncedlyrics(
                        "Alpha - Original Song", False, False, False, providers, None
                    )
                )
            search.assert_not_called()
