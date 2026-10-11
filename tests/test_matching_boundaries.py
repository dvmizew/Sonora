from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from sonora.core.models import TrackInfo
from sonora.modules import catalog_enrichment, matching
from tests.support import IsolatedTestCase

MBID = "11111111-2222-3333-4444-555555555555"


class TestAlbumPositionBoundaries(IsolatedTestCase):
    def test_discless_and_incomplete_position_indexes_have_explicit_results(
        self,
    ) -> None:
        self.assertEqual(
            matching._get_album_track_details(
                3, None, {"tracks_by_position": {3: {"title": "Recording"}}}
            ),
            (None, "Recording"),
        )
        self.assertEqual(
            matching._get_album_track_details(
                3, None, {"tracks_by_position": {3: {"artist": "Primary"}}}
            ),
            (None, None),
        )
        recording = TrackInfo(Path("recording.flac"))
        self.assertEqual(
            matching._match_position_by_mbid(
                recording,
                MBID,
                {1: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", 2: MBID},
                None,
            ),
            2,
        )

    def test_malformed_album_records_surface_schema_errors_before_healing(self) -> None:
        recording = TrackInfo(
            Path("recording.flac"), artist="Primary", title="Recording"
        )
        operations: tuple[Callable[[], object], ...] = (
            partial(
                matching._get_album_track_details,
                3,
                2,
                {"tracks_by_disc_and_position": {(2, 3): None}},
            ),
            partial(
                matching._match_position_by_mbid,
                recording,
                MBID,
                None,
                {"tracks_by_mbid": {MBID: []}},
            ),
            partial(
                matching._match_position_by_authoritative_id,
                recording,
                None,
                "isrc",
                "USAAA2100001",
                [{"tracks_by_position": {3: None}}],
            ),
            partial(
                matching._extract_known_album_entities,
                None,
                None,
                [{"tracks_by_position": {3: None}}],
            ),
            partial(
                catalog_enrichment._find_track_in_album_mapping,
                None,
                None,
                "Recording",
                3,
                2,
                {(2, 3): None},
            ),
            partial(
                catalog_enrichment._find_track_in_album_mapping,
                {"recording": []},
                None,
                "Recording",
                None,
            ),
            partial(
                catalog_enrichment._find_track_in_album_mapping,
                {"recording feat guest": []},
                None,
                "Recording (feat. Guest)",
                None,
            ),
            partial(
                catalog_enrichment._find_track_in_album_mapping, None, {3: []}, None, 3
            ),
        )
        for operation_number, operation in enumerate(operations):
            with (
                self.subTest(operation=operation_number),
                self.assertRaisesRegex(TypeError, "JSON object"),
            ):
                operation()
        self.assertEqual((recording.artist, recording.title), ("Primary", "Recording"))

    def test_filename_disc_and_cross_disc_title_search_recover_sequence(self) -> None:
        recording = TrackInfo(
            Path("CD 2/03 - Recording.flac"), title="Recording", disc_number=None
        )
        self.assertEqual(
            matching._resolve_album_track_position(
                recording,
                recording.file_path,
                album_mb_release_details={
                    "tracks_by_disc_and_position": {(2, 3): {"title": "Recording"}}
                },
            ),
            3,
        )
        self.assertEqual(recording.disc_number, 2)
        recording.file_path = Path("unknown.flac")
        recording.disc_number = None
        self.assertEqual(
            matching._resolve_album_track_position(
                recording,
                recording.file_path,
                album_mb_release_details={
                    "tracks_by_disc_and_position": {(3, 4): {"title": "Recording"}}
                },
            ),
            4,
        )
        self.assertEqual(recording.disc_number, 3)

    def test_itunes_identifier_can_heal_corrupt_title_and_artist(self) -> None:
        recording = TrackInfo(
            Path("03 - Wrong.flac"), title="Wrong", artist="Wrong", itunes_trackid="123"
        )
        release = {
            "tracks_by_position": {
                4: {"trackId": 123, "trackName": "Recording", "artist": "Primary"}
            }
        }
        self.assertEqual(
            matching._resolve_album_track_position(
                recording, recording.file_path, album_itunes_details=release
            ),
            4,
        )
        self.assertFalse(
            matching.is_alien_album_track(
                recording,
                recording.file_path,
                "Primary",
                "Record",
                album_itunes_details=release,
            )
        )

    def test_equal_titles_match_when_no_artist_is_known(self) -> None:
        self.assertTrue(
            matching._is_strong_track_match(None, "Recording", None, "Recording")
        )

    def test_mbid_resolves_position_before_filename_and_invalid_isrc_is_ignored(
        self,
    ) -> None:
        recording = TrackInfo(
            Path("03 - Recording.flac"),
            title="Recording",
            musicbrainz_trackid=MBID,
            isrc="null",
        )
        self.assertEqual(
            matching._resolve_album_track_position(
                recording, recording.file_path, album_track_mbids={(2, 4): MBID}
            ),
            4,
        )
        self.assertEqual(recording.disc_number, 2)
        recording.musicbrainz_trackid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        self.assertEqual(
            matching._resolve_album_track_position(
                recording,
                recording.file_path,
                album_mb_release_details={
                    "tracks_by_position": {3: {"title": "Recording"}}
                },
            ),
            3,
        )

    def test_tag_position_is_used_when_the_filename_has_no_index(self) -> None:
        recording = TrackInfo(Path("Recording.flac"), title="Recording", track_number=3)
        self.assertEqual(
            matching._resolve_album_track_position(
                recording,
                recording.file_path,
                album_mb_release_details={
                    "tracks_by_position": {3: {"title": "Recording"}}
                },
            ),
            3,
        )

    def test_numeric_titles_and_missing_identity_are_not_fuzzy_matches(self) -> None:
        for title in ("123", "123 [FLAC]", "Track 3", "Audio Track 3"):
            with self.subTest(title=title):
                self.assertTrue(matching._is_generic_title(title))
        self.assertFalse(
            matching._is_strong_track_match("Primary", None, "Primary", "Recording")
        )
        self.assertFalse(
            matching._is_strong_track_match("Primary", "Recording", "Primary", None)
        )

    def test_incomplete_disc_recording_falls_back_to_position_index(self) -> None:
        release = {
            "tracks_by_disc_and_position": {(2, 3): {"artist": "Primary"}},
            "tracks_by_position": {3: {"title": "Recording"}},
        }
        self.assertEqual(
            matching._get_album_track_details(3, 2, release), (None, "Recording")
        )
        self.assertEqual(matching._get_album_track_details(9, 2, release), (None, None))
        self.assertTrue(matching._pos_in_album(3, 2, {"2-3": MBID}))
        self.assertFalse(matching._pos_in_album(9, 2, {"2-3": MBID}))

    def test_missing_position_and_invalid_disc_do_not_invent_sequence(self) -> None:
        recording = TrackInfo(Path("recording.flac"))
        for index, release, expected in (
            ({("bad", "bad"): MBID}, None, None),
            ({(2, "bad"): MBID}, None, None),
            (None, {"tracks_by_mbid": {MBID: {"disc_number": "bad"}}}, None),
            (None, {"tracks_by_mbid": {MBID: {"position": 4}}}, 4),
            (
                None,
                {"tracks_by_mbid": {MBID: {"disc_number": 2, "position": "bad"}}},
                None,
            ),
        ):
            with self.subTest(index=index, release=release):
                self.assertEqual(
                    matching._match_position_by_mbid(recording, MBID, index, release),
                    expected,
                )
        self.assertEqual(recording.disc_number, 2)

    def test_empty_isrc_never_matches_missing_provider_codes(self) -> None:
        self.assertFalse(matching._recording_isrc_matches(None, {"isrcs": [None, ""]}))
        self.assertFalse(
            matching._recording_isrc_matches("USAAA2100001", {"isrcs": [None, ""]})
        )
        self.assertIsNone(
            matching._match_position_by_authoritative_id(
                TrackInfo(Path("recording.flac")),
                None,
                "isrc",
                "USAAA2100001",
                [{"tracks_by_position": {"bad": {"isrc": "USAAA2100001"}}}],
            )
        )

    def test_title_search_uses_best_variant_and_skips_incomplete_tracks(self) -> None:
        recording = TrackInfo(Path("recording.flac"), title="Long Recording Name")
        release = {
            "tracks_by_disc_and_position": {
                (2, 1): {},
                (2, 2): {"title": "Long Recording Names"},
                (2, 3): {"title": "Unrelated Song"},
            }
        }
        self.assertEqual(
            matching._search_release_positions_by_title(
                recording, [1, 2], {2: {1, 2, 3}}, release, None, None
            ),
            2,
        )
        self.assertEqual(recording.disc_number, 2)
        recording.title = "Unrelated Recording"
        self.assertIsNone(
            matching._search_release_positions_by_title(
                recording, [2], {2: {1, 3}}, release, None, None
            )
        )

    def test_generic_title_needs_an_available_release_position(self) -> None:
        for filename, expected in (("03 - Track.flac", 3), ("unknown.flac", None)):
            recording = TrackInfo(Path(filename), title="Track")
            with self.subTest(filename=filename):
                self.assertEqual(
                    matching._resolve_album_track_position(
                        recording, recording.file_path, album_track_mbids={3: MBID}
                    ),
                    expected,
                )
        recording = TrackInfo(Path("03 - Recording.flac"), title="Recording")
        self.assertEqual(
            matching._resolve_album_track_position(
                recording, recording.file_path, album_track_mbids={3: MBID}
            ),
            3,
        )

    def test_identifier_lookup_can_fall_back_to_title_when_ids_do_not_match(
        self,
    ) -> None:
        recording = TrackInfo(
            Path("unknown.flac"),
            title="Recording",
            isrc="USAAA2100001",
            itunes_trackid="999",
            disc_number=2,
        )
        release = {
            "tracks_by_disc_and_position": {
                (2, 3): {"title": "Recording", "trackId": "123"}
            }
        }
        self.assertEqual(
            matching._resolve_album_track_position(
                recording, recording.file_path, album_itunes_details=release
            ),
            3,
        )


class TestAlienClassificationBoundaries(IsolatedTestCase):
    def test_missing_titles_with_a_known_artist_do_not_imply_an_alien(self) -> None:
        recording = TrackInfo(Path("03.flac"), artist="Primary", title="Unknown Title")
        self.assertFalse(
            matching.is_alien_album_track(
                recording, recording.file_path, "Primary", "Record"
            )
        )

    def test_entity_indexes_preserve_all_codes_and_ignore_missing_positions(
        self,
    ) -> None:
        release: dict[str, Any] = {
            "artist": "Primary",
            "title": "Record",
            "tracks_by_number": {
                1: {
                    "trackName": "Recording",
                    "artist": "Guest",
                    "isrcs": [None, "USAAA2100001"],
                }
            },
            "tracks_by_position": {"bad": {}, 3: {}},
            "tracks_by_mbid": {MBID.upper(): {}},
        }
        entities = matching._extract_known_album_entities(None, None, [release])
        self.assertEqual(entities.artists, {"Primary", "Guest"})
        self.assertEqual(entities.titles, {"Recording"})
        self.assertEqual(entities.albums, {"Record"})
        self.assertEqual(entities.isrcs, {"USAAA2100001"})
        self.assertEqual(entities.mbids, {MBID})
        self.assertEqual(entities.positions, {3})

    def test_matching_recording_id_shields_corrupt_text(self) -> None:
        recording = TrackInfo(
            Path("recording.flac"),
            artist="Wrong",
            title="Wrong",
            musicbrainz_trackid=MBID,
        )
        self.assertFalse(
            matching.is_alien_album_track(
                recording,
                recording.file_path,
                "Primary",
                "Record",
                album_mb_release_details={"tracks_by_mbid": {MBID: {}}},
            )
        )

    def test_known_titles_can_heal_unknown_artists(self) -> None:
        recording = TrackInfo(
            Path("Recording.flac"), artist="Unknown Artist", title="Recording"
        )
        self.assertFalse(
            matching.is_alien_album_track(
                recording,
                recording.file_path,
                "Primary",
                "Record",
                album_mb_release_details={
                    "tracks_by_position": {1: {"title": "Recording"}}
                },
            )
        )

    def test_related_artist_names_and_bonus_tracks_preserve_membership(self) -> None:
        release = {
            "title": "Record",
            "tracks_by_position": {
                1: {"title": "First"},
                2: {"title": "Second"},
                3: {"title": "Third"},
            },
        }
        for artist, title, number, album, expected in (
            ("Primary Ensemble", "Unlisted", 4, "Record", False),
            ("Primery", "Unlisted Bonus", 2, "Record", False),
            ("Primary", "Unlisted", 2, "Record", True),
            ("Primary", "Unlisted Bonus", 4, "Other Album", True),
            ("Unknown Artist", "Unlisted", 4, "Record", True),
            ("Primary", "Unlisted", None, "Record", True),
        ):
            recording = TrackInfo(
                Path("recording.flac"),
                artist=artist,
                title=title,
                album=album,
                track_number=number,
            )
            with self.subTest(artist=artist, title=title, number=number, album=album):
                self.assertEqual(
                    matching.is_alien_album_track(
                        recording,
                        recording.file_path,
                        "Primary",
                        "Record",
                        album_mb_release_details=release,
                    ),
                    expected,
                )

    def test_a_known_title_with_an_unrelated_artist_remains_an_alien(self) -> None:
        recording = TrackInfo(
            Path("recording.flac"), artist="Unrelated", title="Recording"
        )
        self.assertTrue(
            matching.is_alien_album_track(
                recording,
                recording.file_path,
                "Primary",
                "Record",
                album_mb_release_details={
                    "tracks_by_position": {1: {"title": "Recording"}}
                },
            )
        )
