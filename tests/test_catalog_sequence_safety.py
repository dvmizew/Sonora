from pathlib import Path
from unittest.mock import patch

import httpx

from sonora.core.http import SESSION
from sonora.core.models import TrackInfo
from sonora.modules import catalog_enrichment, matching
from sonora.services import artist_resolution, deezer, itunes, musicbrainz
from tests.support import IsolatedTestCase


class TestCatalogSequenceSafety(IsolatedTestCase):
    def test_fallback_recording_does_not_choose_the_local_release_sequence(
        self,
    ) -> None:
        for has_album_context in (False, True):
            for position in (None, 4):
                with self.subTest(context=has_album_context, position=position):
                    recording = TrackInfo(
                        Path("recording.flac"),
                        artist="Performer",
                        title="Original Song",
                        track_number=position,
                        disc_number=position,
                    )
                    with (
                        patch.object(
                            deezer, "fetch_deezer_album_details", return_value=None
                        ),
                        patch.object(
                            deezer,
                            "fetch_deezer_track_details",
                            return_value={
                                "artist": "Performer",
                                "title": "Original Song",
                                "track_position": 8,
                                "disk_number": 2,
                                "producers": "Producer",
                            },
                        ),
                    ):
                        catalog_enrichment._enrich_deezer(
                            recording, None, has_album_context=has_album_context
                        )
                    self.assertEqual(recording.track_number, position)
                    self.assertEqual(recording.disc_number, position)
                    self.assertEqual(recording.producers, "Producer")

    def test_itunes_parser_preserves_release_counts_and_rating_meaning(self) -> None:
        for explicitness, advisory in (
            ("explicit", "Explicit"),
            ("cleaned", None),
            ("notExplicit", None),
            ("", None),
        ):
            with self.subTest(explicitness=explicitness):
                recording = itunes._parse_itunes_track(
                    {
                        "trackId": 123,
                        "collectionId": 456,
                        "trackName": "Original Song",
                        "artistName": "Performer",
                        "trackExplicitness": explicitness,
                        "trackNumber": 3,
                        "trackCount": 7,
                        "discNumber": 2,
                        "discCount": 2,
                        "trackTimeMillis": 180125,
                    }
                )
                self.assertEqual(recording["advisory"], advisory)
                self.assertEqual(recording["total_discs"], 2)
                self.assertEqual(recording["total_tracks"], 7)
                self.assertEqual(recording["duration"], 180.125)
                self.assertEqual(recording["itunes_trackid"], "123")

    def test_itunes_parser_does_not_invent_missing_metadata(self) -> None:
        recording = itunes._parse_itunes_track({})
        for field in (
            "duration",
            "advisory",
            "total_discs",
            "total_tracks",
            "itunes_trackid",
            "date",
        ):
            with self.subTest(field=field):
                self.assertIsNone(recording[field])


class TestArtistAliasAmbiguity(IsolatedTestCase):
    def test_alias_search_abstains_when_different_canonical_names_match(self) -> None:
        for candidates in (
            [
                {"name": "First Ensemble", "alias-list": [{"alias": "Shared Name"}]},
                {"name": "Second Ensemble", "alias-list": [{"alias": "Shared Name"}]},
            ],
            [
                {"name": "Second Ensemble", "alias-list": [{"alias": "Shared Name"}]},
                {"name": "First Ensemble", "alias-list": [{"alias": "Shared Name"}]},
            ],
        ):
            with (
                self.subTest(candidates=candidates),
                patch.object(artist_resolution, "get_cached_api", return_value=None),
                patch.object(artist_resolution, "set_cached_api"),
                patch.object(
                    musicbrainz, "search_musicbrainz_artists", return_value=candidates
                ),
                patch.object(SESSION, "get", return_value=httpx.Response(404)),
            ):
                self.assertEqual(
                    artist_resolution.resolve_artist_name("Shared Name"), "Shared Name"
                )

    def test_aliases_for_one_canonical_name_are_not_false_ambiguities(self) -> None:
        with (
            patch.object(artist_resolution, "get_cached_api", return_value=None),
            patch.object(artist_resolution, "set_cached_api"),
            patch.object(
                musicbrainz,
                "search_musicbrainz_artists",
                return_value=[
                    {
                        "name": "Ensemble",
                        "alias-list": [
                            {"alias": "Shared Name"},
                            {"alias": "Shared Name"},
                        ],
                    },
                    {"name": "Other Ensemble", "alias-list": [{"alias": "Other Name"}]},
                ],
            ),
            patch.object(SESSION, "get", return_value=httpx.Response(404)),
        ):
            self.assertEqual(
                artist_resolution.resolve_artist_name("Shared Name"), "Ensemble"
            )


class TestDeezerMediumTotals(IsolatedTestCase):
    def test_single_disc_release_cannot_replace_bonus_disc_total(self) -> None:
        recording = TrackInfo(
            Path("recording.flac"),
            artist="Performer",
            title="Original Song",
            track_number=1,
            total_tracks=1,
            disc_number=3,
            total_discs=3,
        )
        candidate = {
            "artist": recording.artist,
            "title": recording.title,
            "track_position": 19,
            "disk_number": 1,
        }
        album = {
            "nb_tracks": 19,
            "tracks_by_title": {"original song": candidate},
            "tracks_by_disc_and_position": {
                (1, position): candidate for position in range(1, 20)
            },
        }
        catalog_enrichment._enrich_deezer(
            recording, album, has_album_context=True, force=True
        )
        self.assertEqual(
            (
                recording.track_number,
                recording.total_tracks,
                recording.disc_number,
                recording.total_discs,
            ),
            (1, 1, 3, 3),
        )
        self.assertIn("album_sequence", recording.metadata_conflicts)

    def test_verified_multi_disc_album_uses_selected_medium_count(self) -> None:
        recording = TrackInfo(
            Path("recording.flac"),
            artist="Performer",
            title="Original Song",
            track_number=1,
            total_tracks=1,
            disc_number=2,
            total_discs=2,
        )
        candidate = {
            "artist": recording.artist,
            "title": recording.title,
            "track_position": 1,
            "disk_number": 2,
        }
        album = {
            "nb_tracks": 5,
            "tracks_by_disc_and_position": {
                (1, 1): {},
                (1, 2): {},
                (1, 3): {},
                (2, 1): candidate,
                (2, 2): {},
            },
        }
        catalog_enrichment._enrich_deezer(
            recording, album, has_album_context=True, force=True
        )
        self.assertEqual(recording.total_tracks, 2)
        self.assertEqual(recording.disc_number, 2)

    def test_totals_require_a_matching_album_recording(self) -> None:
        recording = TrackInfo(
            Path("recording.flac"),
            artist="Performer",
            title="Original Song",
            track_number=1,
            total_tracks=1,
            disc_number=1,
            total_discs=1,
        )
        album = {
            "nb_tracks": 19,
            "tracks_by_disc_and_position": {
                (1, 1): {
                    "artist": "Other Performer",
                    "title": "Different Song",
                    "track_position": 1,
                }
            },
        }
        with patch.object(deezer, "fetch_deezer_track_details", return_value=None):
            catalog_enrichment._enrich_deezer(
                recording, album, has_album_context=True, force=True
            )
        self.assertEqual(recording.total_tracks, 1)

    def test_incomplete_medium_indexes_do_not_establish_totals(self) -> None:
        recording = TrackInfo(Path("recording.flac"), disc_number=2, total_discs=2)
        index_cases: tuple[
            tuple[dict[tuple[int, int], dict[str, object]], int], ...
        ] = (
            ({(2, 1): {}, (2, 3): {}}, 2),
            ({(2, 1): {}, (2, 2): {}}, 5),
        )
        for positions, expected_tracks in index_cases:
            with self.subTest(positions=positions):
                self.assertIsNone(
                    catalog_enrichment._deezer_medium_track_count(
                        recording,
                        {"disk_number": 2},
                        {
                            "nb_tracks": expected_tracks,
                            "tracks_by_disc_and_position": positions,
                        },
                    )
                )

    def test_unknown_disc_requires_an_unambiguous_release_position(self) -> None:
        first_recording = {"title": "Song", "artist": "Performer"}
        second_recording = {"title": "Song", "artist": "Performer"}
        for positions, expected in (
            ({(2, 3): first_recording}, first_recording),
            ({(1, 3): first_recording, (2, 3): second_recording}, None),
            ({(1, 4): first_recording}, None),
        ):
            with self.subTest(positions=positions):
                recording, from_album = matching._find_track_in_album_mapping(
                    None, positions, "Song", 3, artist="Performer"
                )
                self.assertEqual(recording, expected)
                self.assertEqual(from_album, expected is not None)
