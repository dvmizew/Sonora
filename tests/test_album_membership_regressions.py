from pathlib import Path

from sonora.core.models import TrackInfo
from sonora.core.titles import score_artist_title_match
from sonora.modules.matching import (
    _compatible_release_layout,
    is_alien_album_track,
)
from tests.support import IsolatedTestCase


class TestAlbumMembershipRegressions(IsolatedTestCase):
    def test_collective_artist_without_a_tracklist_is_not_foreign_evidence(
        self,
    ) -> None:
        for compilation in (None, True):
            with self.subTest(compilation=compilation):
                track = TrackInfo(
                    Path("01 - Opening.flac"),
                    artist="Individual Performer",
                    title="Opening",
                    album="Collective Sessions",
                    album_artist="Collective",
                    compilation=compilation,
                )
                self.assertFalse(
                    is_alien_album_track(
                        track, track.file_path, "Collective", "Collective Sessions"
                    )
                )

    def test_different_artist_and_album_still_establish_an_outlier(self) -> None:
        track = TrackInfo(
            Path("01 - Foreign.flac"),
            artist="Unrelated Performer",
            title="Foreign",
            album="Another Album",
            album_artist="Unrelated Performer",
        )
        self.assertTrue(
            is_alien_album_track(track, track.file_path, "Collective", "Sessions")
        )

    def test_compilation_flag_does_not_override_a_conflicting_tracklist(self) -> None:
        track = TrackInfo(
            Path("01 - Foreign.flac"),
            artist="Unrelated Performer",
            title="Foreign",
            album="Sessions",
            compilation=True,
        )
        self.assertTrue(
            is_alien_album_track(
                track,
                track.file_path,
                "Collective",
                "Sessions",
                album_deezer_details={
                    "tracks_by_disc_and_position": {
                        (1, 1): {"artist": "Member", "title": "Opening"}
                    }
                },
            )
        )

    def test_compilation_title_without_performer_credits_is_not_a_conflict(
        self,
    ) -> None:
        track = TrackInfo(
            Path("01 - Opening.flac"),
            artist="Individual Performer",
            title="Opening",
            album="Sessions",
            compilation=True,
        )
        for recording_artist, expected in ((None, False), ("Guitar Trio", True)):
            with self.subTest(recording_artist=recording_artist):
                self.assertEqual(
                    is_alien_album_track(
                        track,
                        track.file_path,
                        "Collective",
                        "Sessions",
                        album_deezer_details={
                            "tracks_by_disc_and_position": {
                                (1, 1): {"title": "Opening", "artist": recording_artist}
                            }
                        },
                    ),
                    expected,
                )

    def test_disc_number_is_evidence_even_when_total_discs_is_missing(self) -> None:
        for total_discs in (None, 1, 2):
            with self.subTest(total_discs=total_discs):
                track = TrackInfo(
                    Path("2-01 - Opening.flac"),
                    disc_number=2,
                    total_discs=total_discs,
                )
                self.assertIsNone(_compatible_release_layout(track, {"total_discs": 1}))
                self.assertIn("album_sequence", track.metadata_conflicts)

    def test_clock_punctuation_and_production_credits_preserve_identity(self) -> None:
        for local, provider in (
            ('COOGIE 2"05', "COOGIE 205"),
            ("3:00 AM", "300 AM"),
            ("Vremea schimbarii prod. Criminalle", "Vremea schimbarii"),
            ("Long Summer Night produced by Producer", "Long Summer Night"),
        ):
            with self.subTest(local=local, provider=provider):
                self.assertEqual(
                    score_artist_title_match("Performer", local, "Performer", provider),
                    100,
                )

    def test_credit_and_clock_normalization_keeps_distinct_recordings_separate(
        self,
    ) -> None:
        for local, provider in (
            ('COOGIE 2"05', "COOGIE 206"),
            ("3:00 AM", "300 PM"),
            ("A1.2", "A12"),
            ("Long Summer Night (Live) prod. Producer", "Long Summer Night"),
            ("Long Summer Night Part 2 prod. Producer", "Long Summer Night Part 3"),
            (
                "Long Summer Night (Alpha Remix) prod. Producer",
                "Long Summer Night (Beta Remix)",
            ),
        ):
            with self.subTest(local=local, provider=provider):
                self.assertEqual(
                    score_artist_title_match("Performer", local, "Performer", provider),
                    0,
                )
