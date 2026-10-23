from pathlib import Path
from typing import Any

from sonora.core.models import TrackInfo
from sonora.modules import matching, musicbrainz_enrichment
from tests.support import IsolatedTestCase

RELEASE_ID = "11111111-2222-3333-4444-555555555555"


class TestReleaseSequenceSafety(IsolatedTestCase):
    def test_flat_position_alias_does_not_move_recording_to_another_disc(self) -> None:
        track = TrackInfo(
            Path("03 - Original Song.flac"),
            artist="Alpha",
            title="Original Song",
            disc_number=1,
            total_discs=2,
        )
        recording = {"artist": "Alpha", "title": "Original Song", "disc_number": 2}
        release = {
            "total_discs": 2,
            "tracks_by_disc_and_position": {(2, 3): recording},
        }
        self.assertEqual(
            matching._resolve_album_track_position(
                track, track.file_path, album_mb_release_details=release
            ),
            3,
        )
        self.assertEqual(track.disc_number, 2)

    def test_filename_position_cannot_override_artist_or_duration(self) -> None:
        for wrong_artist, wrong_duration in (("Beta", 180), ("Alpha", 260)):
            with self.subTest(artist=wrong_artist, duration=wrong_duration):
                track = TrackInfo(
                    Path("01 - Original Song.flac"),
                    artist="Alpha",
                    title="Original Song",
                    duration=180,
                    track_number=1,
                )
                release = {
                    "tracks_by_disc_and_position": {
                        (1, 1): {
                            "artist": wrong_artist,
                            "title": "Original Song",
                            "duration": wrong_duration,
                        },
                        (1, 2): {
                            "artist": "Alpha",
                            "title": "Original Song",
                            "duration": 180,
                        },
                    }
                }
                self.assertEqual(
                    matching._resolve_album_track_position(
                        track, track.file_path, album_mb_release_details=release
                    ),
                    2,
                )

    def test_title_search_rejects_only_recording_with_incompatible_duration(
        self,
    ) -> None:
        track = TrackInfo(
            Path("Original Song.flac"),
            artist="Alpha",
            title="Original Song",
            duration=180,
        )
        release = {
            "tracks_by_disc_and_position": {
                (1, 1): {"artist": "Alpha", "title": "Original Song", "duration": 260}
            }
        }
        self.assertIsNone(
            matching._resolve_album_track_position(
                track, track.file_path, album_mb_release_details=release
            )
        )

    def test_matching_position_keeps_sequence_when_duration_is_unavailable(
        self,
    ) -> None:
        track = TrackInfo(
            Path("01 - Original Song.flac"),
            artist="Alpha",
            title="Original Song",
            duration=180,
            track_number=1,
        )
        self.assertEqual(
            matching._resolve_album_track_position(
                track,
                track.file_path,
                album_mb_release_details={
                    "tracks_by_disc_and_position": {
                        (1, 1): {"artist": "Alpha", "title": "Original Song"}
                    }
                },
            ),
            1,
        )

    def test_multidisc_release_uses_medium_total_and_can_correct_inflated_total(
        self,
    ) -> None:
        for disc, disc_total in ((1, 10), (2, 12)):
            with self.subTest(disc=disc):
                track = TrackInfo(
                    Path("Original Song.flac"),
                    artist="Alpha",
                    title="Original Song",
                    album="Record",
                    musicbrainz_albumid=RELEASE_ID,
                    track_number=3,
                    disc_number=disc,
                    total_tracks=22,
                    total_discs=2,
                )
                release: dict[str, Any] = {
                    "title": "Record",
                    "total_tracks": 22,
                    "total_discs": 2,
                    "tracks_by_disc_and_position": {
                        (disc, 3): {
                            "title": "Original Song",
                            "artist": "Alpha",
                            "disc_total_tracks": disc_total,
                        }
                    },
                }
                musicbrainz_enrichment._apply_mb_release_metadata(
                    track, RELEASE_ID, release, force=True
                )
                self.assertEqual(track.total_tracks, disc_total)

    def test_missing_medium_count_does_not_use_release_wide_total(self) -> None:
        for disc_total in (None, 2):
            with self.subTest(disc_total=disc_total):
                track = TrackInfo(
                    Path("Original Song.flac"),
                    artist="Alpha",
                    title="Original Song",
                    musicbrainz_albumid=RELEASE_ID,
                    track_number=3,
                    total_tracks=10,
                    total_discs=2,
                )
                musicbrainz_enrichment._apply_mb_release_metadata(
                    track,
                    RELEASE_ID,
                    {
                        "total_tracks": 22,
                        "total_discs": 2,
                        "tracks_by_disc_and_position": {
                            (1, 3): {
                                "title": "Original Song",
                                "disc_total_tracks": disc_total,
                            }
                        },
                    },
                    force=True,
                )
                self.assertEqual(track.total_tracks, 10)

    def test_single_disc_can_use_release_total(self) -> None:
        track = TrackInfo(
            Path("Original Song.flac"), musicbrainz_albumid=RELEASE_ID, track_number=3
        )
        musicbrainz_enrichment._apply_mb_release_metadata(
            track,
            RELEASE_ID,
            {"total_tracks": 10, "total_discs": 1},
            force=True,
        )
        self.assertEqual(track.total_tracks, 10)
