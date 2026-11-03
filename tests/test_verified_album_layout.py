from pathlib import Path
from unittest.mock import patch

from sonora.audio import metadata
from sonora.core.models import TrackInfo
from sonora.modules import tagger
from sonora.modules.matching import _recording_matches_track_identity
from tests.support import IsolatedTestCase


class TestVerifiedAlbumLayout(IsolatedTestCase):
    def test_album_layout_checks_expected_position_before_scanning_candidates(
        self,
    ) -> None:
        files = [Path(f"{index}.flac") for index in range(1, 4)]
        tracks: list[TrackInfo | None] = [
            TrackInfo(
                path,
                artist="Performer",
                title=f"Song {index}",
                track_number=index,
            )
            for index, path in enumerate(files, 1)
        ]
        release = {
            "tracks_by_disc_and_position": {
                (1, index): {"artist": "Performer", "title": f"Song {index}"}
                for index in range(1, 4)
            }
        }

        with patch(
            "sonora.modules.tagger._recording_matches_track_identity",
            wraps=_recording_matches_track_identity,
        ) as identity_match:
            self.assertTrue(
                tagger._release_matches_album_tracks(release, files, tracks)
            )

        self.assertEqual(identity_match.call_count, len(files))

    def test_verified_itunes_album_artist_repairs_album_credit_only(self) -> None:
        for verified, expected in (
            (False, "4 226"),
            (True, "4 226 & RAVA"),
        ):
            with self.subTest(verified=verified):
                track = TrackInfo(
                    Path("01 - Song.flac"),
                    artist="4 226",
                    album_artist="Old Artist",
                    title="Song",
                    album="Album",
                )
                tagger._apply_album_metadata(
                    track,
                    track.file_path,
                    True,
                    album_context=tagger._AlbumMetadataContext(
                        itunes_details={"album_artist": "4 226 & RAVA"},
                        itunes_layout_verified=verified,
                        sample_artist="4 226",
                    ),
                )
                self.assertEqual(track.album_artist, expected)
                self.assertEqual(track.artist, "4 226")
                self.assertEqual(
                    track.metadata_sources.get("album_artist"),
                    "iTunes" if verified else None,
                )

    def test_complete_verified_edition_repairs_totals_without_guessing_folder_count(
        self,
    ) -> None:
        positions = {
            (1, index): {"artist": "Performer", "title": f"Song {index}"}
            for index in range(1, 4)
        }
        for verified, conflict in (
            (False, "Stored numbering totals conflict; release verification required."),
            (True, "Stored numbering totals conflict; release verification required."),
            (True, "Provider disc count differs; sequence metadata withheld."),
        ):
            with self.subTest(verified=verified, conflict=conflict):
                track = TrackInfo(
                    Path("03 - Song 3.flac"),
                    artist="Performer",
                    title="Song 3",
                    album="Album",
                    track_number=3,
                    disc_number=1,
                    total_tracks=2,
                    total_discs=1,
                    metadata_conflicts={"album_sequence": conflict},
                )
                context = tagger._AlbumMetadataContext(
                    itunes_details={
                        "total_tracks": 3,
                        "tracks_by_disc_and_position": positions,
                    },
                    itunes_layout_verified=verified,
                )
                tagger._apply_album_metadata(
                    track, track.file_path, True, album_context=context
                )
                self.assertEqual(track.total_tracks, 3 if verified else 2)
                self.assertEqual(
                    "album_sequence" in track.metadata_conflicts,
                    not verified or conflict.startswith("Provider"),
                )

    def test_partial_or_conflicting_release_cannot_verify_a_layout(self) -> None:
        files = [Path("one.flac"), Path("two.flac")]
        tracks = [
            TrackInfo(path, artist="Performer", title=f"Song {index}")
            for index, path in enumerate(files, 1)
        ]
        positions = {
            (1, index): {"artist": "Performer", "title": f"Song {index}"}
            for index in (1, 2)
        }
        with patch.object(
            metadata,
            "read_track_metadata",
            side_effect=lambda path: tracks[files.index(path)],
        ):
            self.assertFalse(
                tagger._release_matches_album_tracks(
                    {"total_tracks": 3, "tracks_by_disc_and_position": positions}, files
                )
            )
            self.assertTrue(
                tagger._release_matches_album_tracks(
                    {"total_tracks": 2, "tracks_by_disc_and_position": positions}, files
                )
            )
            positions[(1, 2)] = {"artist": "Other Performer", "title": "Other Song"}
            self.assertFalse(
                tagger._release_matches_album_tracks(
                    {"total_tracks": 2, "tracks_by_disc_and_position": positions}, files
                )
            )
