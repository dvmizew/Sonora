from pathlib import Path
from typing import Any
from unittest.mock import patch

from sonora.core.models import TrackInfo
from sonora.modules import matching, musicbrainz_enrichment
from sonora.services import musicbrainz
from tests.support import IsolatedTestCase


class TestAlbumIdentityReview(IsolatedTestCase):
    def test_identifier_position_requires_remaining_identity_and_duration(self) -> None:
        track = TrackInfo(
            Path("song.flac"), artist="Alpha", title="Original Song", duration=180
        )
        for identifier in ("isrc", "itunes_trackid"):
            for artist, title, duration, expected in (
                ("Alpha", "Original Song", 180, 4),
                ("Beta", "Original Song", 180, None),
                ("Alpha", "Different Song", 180, None),
                ("Alpha", "Original Song", 240, None),
            ):
                with self.subTest(
                    identifier=identifier, artist=artist, title=title, duration=duration
                ):
                    release = {
                        "tracks_by_number": {
                            4: {
                                identifier: "KNOWN",
                                "artist": artist,
                                "title": title,
                                "duration": duration,
                            }
                        }
                    }
                    self.assertEqual(
                        matching._match_position_by_authoritative_id(
                            track, 1, identifier, "KNOWN", [release]
                        ),
                        expected,
                    )

    def test_placeholder_title_does_not_discard_known_artist(self) -> None:
        track = TrackInfo(
            Path("song.flac"), artist="Alpha", title="Untitled", duration=180
        )
        for artist, duration, expected in (
            ("Alpha", 180, True),
            ("Beta", 180, False),
            ("Alpha", 240, False),
        ):
            with self.subTest(artist=artist, duration=duration):
                self.assertEqual(
                    matching._recording_matches_track_identity(
                        track,
                        {
                            "title": "Original Song",
                            "artist": artist,
                            "duration": duration,
                        },
                    ),
                    expected,
                )
        track.artist = "Unknown Artist"
        self.assertTrue(
            matching._recording_matches_track_identity(
                track, {"title": "Original Song", "artist": "Beta", "duration": 180}
            )
        )
        self.assertFalse(matching._recording_matches_track_identity(track, {}))

    def test_title_position_search_respects_artist_and_rejects_ties(self) -> None:
        track = TrackInfo(Path("song.flac"), artist="Alpha", title="Original Song")
        release: dict[str, Any] = {
            "tracks_by_disc_and_position": {
                (2, 1): {"title": "Original Song", "artist": "Beta"},
                (2, 2): {"title": "Original Song", "artist": "Alpha"},
            }
        }
        self.assertEqual(
            matching._search_release_positions_by_title(
                track, [2], {2: {1, 2}}, release, None, None
            ),
            2,
        )
        self.assertEqual(track.disc_number, 2)
        release["tracks_by_disc_and_position"][(2, 1)]["artist"] = "Alpha"
        track.disc_number = 1
        self.assertIsNone(
            matching._search_release_positions_by_title(
                track, [2], {2: {1, 2}}, release, None, None
            )
        )
        self.assertEqual(track.disc_number, 1)
        self.assertIsNone(
            matching._search_release_positions_by_title(track, [], {}, None, None, None)
        )

    def test_standalone_isrc_heals_only_missing_identity_components(self) -> None:
        recording_id = "11111111-1111-4111-8111-111111111111"
        for title, artist, candidate_title, candidate_artist, expected in (
            ("Untitled", "Alpha", "Original Song", "Alpha", True),
            ("Untitled", "Alpha", "Original Song", "Beta", False),
            ("Original Song", "Unknown Artist", "Different Song", "Beta", False),
            ("Original Song", "Unknown Artist", "Original Song", "Beta", True),
        ):
            with self.subTest(
                title=title,
                artist=artist,
                candidate_title=candidate_title,
                candidate_artist=candidate_artist,
            ):
                track = TrackInfo(
                    Path("song.flac"),
                    title=title,
                    artist=artist,
                    isrc="KNOWN",
                    musicbrainz_trackid=recording_id,
                )
                recording = {
                    "title": candidate_title,
                    "artist": candidate_artist,
                    "isrc": "KNOWN",
                }
                with patch.object(
                    musicbrainz,
                    "fetch_musicbrainz_recording_details",
                    return_value=recording,
                ):
                    self.assertEqual(
                        musicbrainz_enrichment._fetch_standalone_recording(track)
                        is not None,
                        expected,
                    )

    def test_release_isrc_healing_preserves_meaningful_title_and_artist(self) -> None:
        for title, artist, candidate_artist, expected in (
            ("Untitled", "Alpha", "Alpha", True),
            ("Untitled", "Alpha", "Beta", False),
            ("Known Song", "Unknown Artist", "Beta", False),
        ):
            with self.subTest(
                title=title, artist=artist, candidate_artist=candidate_artist
            ):
                track = TrackInfo(
                    Path("song.flac"), title=title, artist=artist, isrc="KNOWN"
                )
                self.assertEqual(
                    musicbrainz_enrichment._heal_track_identity_from_release(
                        track,
                        {
                            "title": "Original Song",
                            "artist": candidate_artist,
                            "isrc": "KNOWN",
                        },
                    ),
                    expected,
                )
                self.assertEqual(track.title, "Original Song" if expected else title)

    def test_album_position_mbid_requires_candidate_identity_and_duration(self) -> None:
        recording_id = "11111111-1111-4111-8111-111111111111"
        for candidate, expected in (
            (None, None),
            (
                {
                    "title": "Original Song",
                    "artist": "Alpha",
                    "duration": 180,
                    "recording_mbid": recording_id,
                },
                recording_id,
            ),
            ({"title": "Original Song", "artist": "Alpha", "duration": 240}, None),
            ({"title": "Different Song", "artist": "Alpha", "duration": 180}, None),
        ):
            with self.subTest(candidate=candidate):
                track = TrackInfo(
                    Path("song.flac"),
                    artist="Alpha",
                    title="Original Song",
                    duration=180,
                    track_number=1,
                )
                musicbrainz_enrichment._resolve_musicbrainz_track_id(
                    track, {1: recording_id}, None, candidate, False
                )
                self.assertEqual(track.musicbrainz_trackid, expected)

    def test_repeated_recording_mbid_preserves_unique_release_slot(self) -> None:
        recording_id = "11111111-1111-4111-8111-111111111111"
        payload: dict[str, Any] = {
            "medium-list": [
                {
                    "position": "1",
                    "track-list": [
                        {
                            "position": str(position),
                            "title": "Original Song",
                            "length": "180000",
                            "recording": {"id": recording_id},
                        }
                        for position in (1, 2, 3)
                    ],
                }
            ]
        }
        indexes = musicbrainz._release_track_indexes(payload, "Alpha", None)
        indexed_recordings = indexes["tracks_by_mbid"]
        positions = indexes["tracks_by_disc_and_position"]
        assert isinstance(indexed_recordings, dict)
        assert isinstance(positions, dict)
        self.assertNotIn(recording_id, indexed_recordings)
        self.assertEqual(len(positions), 3)
        for current_position, expected in ((2, 2), (None, None), (9, None)):
            with self.subTest(current_position=current_position):
                track = TrackInfo(
                    Path("song.flac"),
                    artist="Alpha",
                    title="Original Song",
                    duration=180,
                    track_number=current_position,
                )
                self.assertEqual(
                    matching._match_position_by_mbid(
                        track, recording_id, None, indexes
                    ),
                    expected,
                )
        positions[(1, 1)]["duration"] = 300
        positions[(1, 3)]["duration"] = 300
        track.track_number = None
        self.assertEqual(
            matching._match_position_by_mbid(track, recording_id, None, indexes), 2
        )
