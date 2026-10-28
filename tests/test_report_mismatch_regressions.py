from pathlib import Path
from unittest.mock import patch

import httpx

from sonora.audio.metadata import read_track_metadata, write_track_metadata
from sonora.core.cache import set_cached_api
from sonora.core.config import SonoraConfig
from sonora.core.http import SESSION
from sonora.core.models import TrackInfo
from sonora.modules import (
    catalog_metadata,
    matching,
    metadata_merge,
    musicbrainz_metadata,
)
from sonora.services import artist_resolution, itunes, lyrics, musicbrainz, shazam
from sonora.services.itunes import _parse_itunes_track
from sonora.services.shazam import ShazamTrackInfo
from tests.regression_support import AudioRegressionTestCase
from tests.support import IsolatedTestCase

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
                        musicbrainz_metadata._fetch_standalone_recording(
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
                musicbrainz_metadata._fetch_standalone_recording(
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
                        musicbrainz_metadata._fetch_standalone_recording(recording),
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
                    catalog_metadata,
                    "get_config",
                    return_value=SonoraConfig(enable_shazam=True),
                ),
                patch.object(shazam, "recognize_audio_track", return_value=recognized),
            ):
                catalog_metadata._update_shazam(
                    recording, recording.file_path, force=True
                )
            self.assertEqual(
                (recording.artist, recording.title, recording.album),
                ("Alpha", "Original Song", "Original Album"),
            )
            self.assertIsNone(recording.genre)
            self.assertIn("identity", recording.metadata_conflicts)

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
                    artist_resolution,
                    "resolve_artist_name",
                    return_value="Alpha",
                ):
                    musicbrainz_metadata._apply_mb_release_metadata(
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


CONFLICTING_RECORDING_ID = "11111111-1111-4111-8111-111111111111"

MATCHING_RECORDING_ID = "22222222-2222-4222-8222-222222222222"


class TestRecordingIdentifierValidation(IsolatedTestCase):
    def test_validated_album_slot_repairs_stored_mbid_without_force(self) -> None:
        for stored_id in (CONFLICTING_RECORDING_ID, MATCHING_RECORDING_ID, None):
            with self.subTest(stored_id=stored_id):
                track = TrackInfo(
                    Path("song.flac"),
                    artist="Alpha",
                    title="Original Song",
                    duration=180,
                    track_number=2,
                    musicbrainz_trackid=stored_id,
                )
                candidate = {
                    "title": "Original Song",
                    "artist": "Alpha",
                    "duration": 180,
                    "recording_mbid": MATCHING_RECORDING_ID,
                }
                with patch.object(
                    musicbrainz,
                    "fetch_musicbrainz_recording_details",
                    return_value={"artist": "Other Performer", "title": "Other Song"},
                ):
                    musicbrainz_metadata._resolve_musicbrainz_track_id(
                        track, {(1, 2): MATCHING_RECORDING_ID}, candidate
                    )
                self.assertEqual(track.musicbrainz_trackid, MATCHING_RECORDING_ID)
                self.assertEqual(
                    track.metadata_sources.get("musicbrainz_trackid"), "MusicBrainz"
                )

    def test_album_mbid_assignment_rejects_conflicting_index_or_duration(self) -> None:
        for candidate_id, duration in (
            (CONFLICTING_RECORDING_ID, 180),
            (MATCHING_RECORDING_ID, 240),
            ("invalid", 180),
        ):
            with self.subTest(candidate_id=candidate_id, duration=duration):
                track = TrackInfo(
                    Path("song.flac"),
                    artist="Alpha",
                    title="Original Song",
                    duration=180,
                    track_number=2,
                    musicbrainz_trackid=CONFLICTING_RECORDING_ID,
                )
                candidate = {
                    "artist": "Alpha",
                    "title": "Original Song",
                    "duration": duration,
                    "recording_mbid": candidate_id,
                }
                musicbrainz_metadata._resolve_musicbrainz_track_id(
                    track, {(1, 2): MATCHING_RECORDING_ID}, candidate
                )
                self.assertEqual(track.musicbrainz_trackid, CONFLICTING_RECORDING_ID)
                self.assertNotIn("musicbrainz_trackid", track.metadata_sources)

    def test_release_extraction_and_standalone_validation_record_provenance(
        self,
    ) -> None:
        candidate = {
            "artist": "Alpha",
            "title": "Original Song",
            "duration": 180,
            "recording_mbid": MATCHING_RECORDING_ID,
        }
        track = TrackInfo(
            Path("song.flac"),
            artist="Alpha",
            title="Original Song",
            duration=180,
            musicbrainz_trackid=CONFLICTING_RECORDING_ID,
        )
        with patch.object(
            musicbrainz,
            "fetch_musicbrainz_recording_details",
            return_value={"artist": "Other Performer", "title": "Other Song"},
        ):
            self.assertEqual(
                musicbrainz_metadata._extract_release_recording(
                    track, {"tracks_by_mbid": {}}, candidate
                ),
                candidate,
            )
        self.assertEqual(track.musicbrainz_trackid, MATCHING_RECORDING_ID)
        self.assertEqual(
            track.metadata_sources.get("musicbrainz_trackid"), "MusicBrainz"
        )
        track.metadata_sources.clear()
        with patch.object(
            musicbrainz, "fetch_musicbrainz_recording_details", return_value=candidate
        ):
            self.assertEqual(
                musicbrainz_metadata._fetch_standalone_recording(track), candidate
            )
        self.assertEqual(
            track.metadata_sources.get("musicbrainz_trackid"), "MusicBrainz"
        )

    def test_placeholder_catalog_match_preserves_known_artist_and_duration(
        self,
    ) -> None:
        for artist, duration, expected in (
            ("Alpha", 180, True),
            ("Beta", 180, False),
            ("Alpha", 240, False),
        ):
            with self.subTest(artist=artist, duration=duration):
                self.assertEqual(
                    matching._catalog_recording_matches_track(
                        {
                            "artist": artist,
                            "title": "Original Song",
                            "duration": duration,
                            "isrc": "KNOWN",
                        },
                        "Untitled",
                        "Alpha",
                        "KNOWN",
                        None,
                        180,
                    ),
                    expected,
                )
        self.assertFalse(
            matching._catalog_recording_matches_track(
                {"artist": "Alpha", "title": "Original Song"},
                "Untitled",
                "Alpha",
                None,
                None,
                180,
            )
        )

    def test_complete_itunes_tags_still_validate_existing_id_without_force(
        self,
    ) -> None:
        track = TrackInfo(
            Path("song.flac"),
            artist="Alpha",
            title="Original Song",
            duration=180,
            itunes_trackid="123",
            itunes_collectionid="456",
            genre="Pop",
            advisory="Explicit",
        )
        with (
            patch.object(
                itunes,
                "lookup_itunes_track",
                return_value=_parse_itunes_track(
                    {
                        "artistName": "Alpha",
                        "trackName": "Wrong Song",
                        "trackTimeMillis": 180000,
                    }
                ),
            ) as lookup,
            patch.object(
                itunes, "fetch_itunes_track_metadata", return_value=None
            ) as search,
        ):
            catalog_metadata._update_itunes(track)
        lookup.assert_called_once_with("123")
        self.assertIsNone(track.itunes_trackid)
        self.assertIsNone(track.itunes_collectionid)
        search.assert_called_once()
        self.assertEqual(track.genre, "Pop")

    def test_itunes_payload_conflict_blocks_all_fields(self) -> None:
        for title, duration in (("Wrong Song", 180), ("Original Song", 240)):
            with self.subTest(title=title, duration=duration):
                track = TrackInfo(
                    Path("song.flac"),
                    artist="Alpha",
                    title="Original Song",
                    duration=180,
                    genre="Pop",
                )
                payload = {
                    "artist": "Alpha",
                    "title": title,
                    "duration": duration,
                    "genre": "Rock",
                    "advisory": "Explicit",
                    "itunes_trackid": "123",
                    "itunes_collectionid": "456",
                }
                with patch.object(
                    itunes, "fetch_itunes_track_metadata", return_value=payload
                ):
                    catalog_metadata._update_itunes(track, force=True)
                self.assertEqual(track.genre, "Pop")
                self.assertIsNone(track.itunes_trackid)
                self.assertIsNone(track.advisory)

    def test_unavailable_itunes_lookup_preserves_existing_identifier(self) -> None:
        track = TrackInfo(
            Path("song.flac"),
            artist="Alpha",
            title="Original Song",
            itunes_trackid="123",
            genre="Pop",
            advisory="Explicit",
        )
        with (
            patch.object(itunes, "lookup_itunes_track", return_value=None),
            patch.object(itunes, "fetch_itunes_track_metadata") as search,
        ):
            catalog_metadata._update_itunes(track)
        self.assertEqual(track.itunes_trackid, "123")
        self.assertNotIn("itunes_trackid", track.metadata_sources)
        search.assert_not_called()
        with (
            patch.object(
                itunes,
                "lookup_itunes_track",
                return_value=_parse_itunes_track(
                    {
                        "trackId": 123,
                        "artistName": "Alpha",
                        "trackName": "Original Song",
                    }
                ),
            ),
            patch.object(itunes, "fetch_itunes_track_metadata") as search,
        ):
            catalog_metadata._update_itunes(track)
        self.assertEqual(track.metadata_sources.get("itunes_trackid"), "iTunes")
        self.assertNotIn("itunes_collectionid", track.metadata_sources)
        search.assert_not_called()

    def test_structural_counts_cannot_be_below_existing_positions(self) -> None:
        for field, position_field, total, position in (
            ("total_tracks", "track_number", 10, 12),
            ("total_discs", "disc_number", 1, 2),
        ):
            with self.subTest(field=field):
                track = TrackInfo(Path("song.flac"))
                setattr(track, position_field, position)
                setattr(track, field, 20)
                metadata_merge.apply_metadata_mapping(
                    track,
                    {field: total},
                    {field: field},
                    force=True,
                    source="MusicBrainz",
                )
                self.assertEqual(getattr(track, field), 20)
                self.assertIn(field, track.metadata_conflicts)
        track = TrackInfo(Path("song.flac"), track_number=0, total_tracks=12)
        metadata_merge.apply_metadata_mapping(
            track, {"total": 12}, {"total": "total_tracks"}, force=True
        )
        self.assertEqual(track.track_number, 0)
        self.assertFalse(track.metadata_conflicts)
