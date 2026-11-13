import dataclasses
import json
import os
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import httpx

from sonora.audio import art, metadata
from sonora.audio.metadata import read_track_metadata, write_track_metadata
from sonora.core import artists
from sonora.core.cache import set_cached_api
from sonora.core.config import SonoraConfig
from sonora.core.constants import DIRS
from sonora.core.http import SESSION
from sonora.core.models import CheckReport, TrackInfo
from sonora.core.titles import (
    is_version_or_remix,
    recording_identity_matches,
    score_artist_title_match,
)
from sonora.modules import (
    catalog_metadata,
    checker,
    matching,
    metadata_merge,
    musicbrainz_metadata,
    tagger,
)
from sonora.modules import catalog_metadata as catalog
from sonora.modules import musicbrainz_metadata as mb_metadata
from sonora.modules.matching import (
    _compatible_release_layout,
    _recording_matches_track_identity,
    _resolve_album_track_position,
    is_alien_album_track,
)
from sonora.modules.metadata_merge import (
    build_provider_proposal,
    merge_provider_proposals,
)
from sonora.modules.normalization import normalize_track_metadata
from sonora.services import (
    acoustid,
    artist_resolution,
    deezer,
    discogs,
    genius,
    itunes,
    lyrics,
    musicbrainz,
    shazam,
    theaudiodb,
)
from sonora.services import deezer as deezer_service
from sonora.services import discogs as discogs_service
from sonora.services import genius as genius_service
from sonora.services import itunes as itunes_service
from sonora.services import lastfm as lastfm_service
from sonora.services import shazam as shazam_service
from sonora.services import theaudiodb as theaudiodb_service
from sonora.services.itunes import _parse_itunes_track
from sonora.services.shazam import ShazamTrackInfo
from tests.fixtures import image_bytes
from tests.regression_support import AudioRegressionTestCase
from tests.service_support import ServiceResponseTestCase
from tests.support import IsolatedTestCase
from tools.evaluate_matching import MatchCase, evaluate_cases

MBID = "11111111-2222-3333-4444-555555555555"


class TestAlbumPositionBoundaries(IsolatedTestCase):
    def test_discless_and_incomplete_position_indexes_have_explicit_results(
        self,
    ) -> None:
        self.assertTrue(
            matching._pos_in_album(
                3,
                album_mb_release_details={
                    "tracks_by_disc_and_position": {(1, 3): {"title": "Recording"}}
                },
            )
        )
        self.assertFalse(
            matching._pos_in_album(
                3,
                album_mb_release_details={
                    "tracks_by_disc_and_position": {(1, 3): {"artist": "Primary"}}
                },
            )
        )
        recording = TrackInfo(Path("recording.flac"))
        self.assertIsNone(
            matching._match_position_by_mbid(
                recording,
                MBID,
                {(1, 1): "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", (1, 2): MBID},
                None,
            )
        )

    def test_malformed_album_records_surface_schema_errors_before_updates(self) -> None:
        recording = TrackInfo(
            Path("recording.flac"), artist="Primary", title="Recording"
        )
        operations: tuple[Callable[[], object], ...] = (
            partial(
                matching._pos_in_album,
                3,
                2,
                album_mb_release_details={
                    "tracks_by_disc_and_position": {(2, 3): None}
                },
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
                [{"tracks_by_disc_and_position": {(1, 3): None}}],
            ),
            partial(
                matching._extract_known_album_entities,
                None,
                None,
                [{"tracks_by_disc_and_position": {(1, 3): None}}],
            ),
            partial(
                matching._find_track_in_album_mapping,
                None,
                {(2, 3): None},
                "Recording",
                3,
                2,
            ),
            partial(
                matching._find_track_in_album_mapping,
                {"recording": []},
                None,
                "Recording",
                None,
            ),
            partial(
                matching._find_track_in_album_mapping,
                {"recording feat guest": []},
                None,
                "Recording (feat. Guest)",
                None,
            ),
            partial(matching._find_track_in_album_mapping, None, {(1, 3): []}, None, 3),
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

    def test_itunes_identifier_can_update_corrupt_title_and_artist(self) -> None:
        recording = TrackInfo(
            Path("03 - Unknown.flac"),
            title="Untitled",
            artist="Unknown Artist",
            itunes_trackid="123",
        )
        release = {
            "tracks_by_disc_and_position": {
                (1, 4): {
                    "itunes_trackid": 123,
                    "title": "Recording",
                    "artist": "Primary",
                }
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
                recording,
                recording.file_path,
                album_track_mbids={(2, 4): MBID},
                album_mb_release_details={
                    "tracks_by_mbid": {
                        MBID: {"title": "Recording", "disc_number": 2, "position": 4}
                    }
                },
            ),
            4,
        )
        self.assertEqual(recording.disc_number, 2)
        recording.musicbrainz_trackid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        recording.disc_number = None
        self.assertEqual(
            matching._resolve_album_track_position(
                recording,
                recording.file_path,
                album_mb_release_details={
                    "tracks_by_disc_and_position": {(1, 3): {"title": "Recording"}}
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
                    "tracks_by_disc_and_position": {(1, 3): {"title": "Recording"}}
                },
            ),
            3,
        )

    def test_numeric_titles_and_missing_identity_are_not_fuzzy_matches(self) -> None:
        for title in ("Track 3", "Audio Track 3"):
            with self.subTest(title=title):
                self.assertTrue(matching._is_generic_title(title))
        self.assertFalse(
            matching._is_strong_track_match("Primary", None, "Primary", "Recording")
        )
        self.assertFalse(
            matching._is_strong_track_match("Primary", "Recording", "Primary", None)
        )

    def test_disc_position_does_not_fall_back_to_another_medium(self) -> None:
        release = {"tracks_by_disc_and_position": {(2, 3): {"artist": "Primary"}}}
        self.assertFalse(matching._pos_in_album(3, 2, album_mb_release_details=release))
        self.assertFalse(matching._pos_in_album(9, 2, album_mb_release_details=release))
        self.assertTrue(matching._pos_in_album(3, 2, {(2, 3): MBID}))
        self.assertFalse(matching._pos_in_album(9, 2, {(2, 3): MBID}))

    def test_missing_position_and_invalid_disc_do_not_invent_sequence(self) -> None:
        recording = TrackInfo(Path("recording.flac"))
        for index, release, expected in (
            (
                None,
                {
                    "tracks_by_mbid": {
                        MBID: {"title": "Recording", "disc_number": "bad"}
                    }
                },
                None,
            ),
            (
                None,
                {"tracks_by_mbid": {MBID: {"title": "Recording", "position": 4}}},
                4,
            ),
            (
                None,
                {
                    "tracks_by_mbid": {
                        MBID: {
                            "title": "Recording",
                            "disc_number": 2,
                            "position": "bad",
                        }
                    }
                },
                None,
            ),
        ):
            with self.subTest(index=index, release=release):
                self.assertEqual(
                    matching._match_position_by_mbid(recording, MBID, index, release),
                    expected,
                )
        self.assertEqual(recording.disc_number, 1)

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
                [{"tracks_by_disc_and_position": {"bad": {"isrc": "USAAA2100001"}}}],
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
                        recording, recording.file_path, album_track_mbids={(1, 3): MBID}
                    ),
                    expected,
                )
        recording = TrackInfo(Path("03 - Recording.flac"), title="Recording")
        self.assertEqual(
            matching._resolve_album_track_position(
                recording, recording.file_path, album_track_mbids={(1, 3): MBID}
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
                (2, 3): {"title": "Recording", "itunes_trackid": "123"}
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
        release: dict[str, object] = {
            "artist": "Primary",
            "title": "Record",
            "tracks_by_disc_and_position": {
                (1, 1): {
                    "title": "Recording",
                    "artist": "Guest",
                    "isrcs": [None, "USAAA2100001"],
                }
            },
            "tracks_by_mbid": {MBID.upper(): {}},
        }
        entities = matching._extract_known_album_entities(None, None, [release])
        self.assertEqual(entities.artists, {"Primary", "Guest"})
        self.assertEqual(entities.titles, {"Recording"})
        self.assertEqual(entities.albums, {"Record"})
        self.assertEqual(entities.isrcs, {"USAAA2100001"})
        self.assertEqual(entities.mbids, {MBID})
        self.assertEqual(entities.positions, {1})

    def test_matching_recording_id_preserves_corrupt_text(self) -> None:
        recording = TrackInfo(
            Path("recording.flac"),
            artist="Wrong",
            title="Wrong",
            musicbrainz_trackid=MBID,
        )
        self.assertTrue(
            matching.is_alien_album_track(
                recording,
                recording.file_path,
                "Primary",
                "Record",
                album_mb_release_details={"tracks_by_mbid": {MBID: {}}},
            )
        )

    def test_known_titles_can_update_unknown_artists(self) -> None:
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
                    "tracks_by_disc_and_position": {(1, 1): {"title": "Recording"}}
                },
            )
        )

    def test_related_artist_names_and_bonus_tracks_preserve_membership(self) -> None:
        release = {
            "title": "Record",
            "tracks_by_disc_and_position": {
                (1, 1): {"title": "First"},
                (1, 2): {"title": "Second"},
                (1, 3): {"title": "Third"},
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
                    "tracks_by_disc_and_position": {(1, 1): {"title": "Recording"}}
                },
            )
        )

    def test_different_release_disc_layout_cannot_renumber_local_tracks(self) -> None:
        releases: tuple[dict[str, object], ...] = (
            {"total_discs": 1},
            {"tracks_by_disc_and_position": {(1, 9): {"title": "Recording"}}},
            {"total_discs": 2},
            {},
        )
        for release in releases:
            with self.subTest(release=release):
                recording = TrackInfo(
                    Path("2-01 - Recording.flac"),
                    artist="Primary",
                    title="Recording",
                    track_number=1,
                    disc_number=2,
                    total_discs=2,
                    total_tracks=6,
                )
                compatible = matching._compatible_release_layout(recording, release)
                if release.get("total_discs") == 2:
                    self.assertIs(compatible, release)
                else:
                    self.assertIsNone(compatible)
                if release:
                    release.setdefault(
                        "tracks_by_disc_and_position",
                        {
                            (1, 9): {
                                "title": "Recording",
                                "artist": "Primary",
                                "isrc": "USAAA2100001",
                            }
                        },
                    )
                recording.isrc = "USAAA2100001"
                if release.get("total_discs") != 2:
                    matching._resolve_album_track_position(
                        recording, recording.file_path, album_deezer_details=release
                    )
                    self.assertEqual(
                        (recording.disc_number, recording.track_number), (2, 1)
                    )


class TestRecordingIdentityNormalization(IsolatedTestCase):
    def test_canonical_equivalence_preserves_artist_and_title_variants(self) -> None:
        for artist, title, candidate_artist, candidate_title in (
            ("Adrian Sînă", "Hold On", "Adrian Sina", "Hold On"),
            ("B.U.G. Mafia", "... Și Cui Îi Pasă?", "B.U.G. Mafia", "…Si Cui Ii Pasa"),
            ("El Nino", "Păcat", "El Nino", "Pacat"),
            ("Rauf & Faik", "Мосты", "Rauf & Faik", "Mosty"),
            ("Rauf & Faik", "5 минут", "Rauf & Faik", "5 minut"),
        ):
            with self.subTest(artist=artist, title=title):
                self.assertEqual(
                    score_artist_title_match(
                        artist, title, candidate_artist, candidate_title
                    ),
                    100,
                )

    def test_canonical_text_equivalence_cannot_override_wrong_artist(self) -> None:
        self.assertEqual(
            score_artist_title_match("El Nino", "Păcat", "Someone Else", "Pacat"), 0
        )

    def test_version_words_do_not_match_fragments_of_unrelated_titles(self) -> None:
        for title, accented in (("Olive", "Olivé"), ("Demolition", "Démolition")):
            with self.subTest(title=title):
                self.assertFalse(is_version_or_remix(title))
                self.assertEqual(
                    score_artist_title_match("Alpha", title, "Alpha", accented), 100
                )
        for version in (
            "Live",
            "Remixes",
            "Remixed",
            "Extended Mix",
            "Sped-Up",
            "Acoustic",
        ):
            with self.subTest(version=version):
                self.assertTrue(is_version_or_remix(version))

    def test_version_remix_and_series_guards_survive_canonical_equivalence(
        self,
    ) -> None:
        for title, candidate_title in (
            ("Dărâmăm (Radio Edit)", "Daramam"),
            ("Dărâmăm (Alpha Remix)", "Daramam (Beta Remix)"),
            ("Înger, Part 1", "Inger, Part 2"),
            ("Intro", "Outro"),
        ):
            with self.subTest(title=title, candidate_title=candidate_title):
                self.assertEqual(
                    score_artist_title_match("Alpha", title, "Alpha", candidate_title),
                    0,
                )

    def test_empty_normalized_titles_do_not_become_equal(self) -> None:
        self.assertEqual(score_artist_title_match("Alpha", "!!!", "Alpha", "???"), 0)

    def test_canonical_text_equivalence_still_requires_compatible_duration(
        self,
    ) -> None:
        for duration, expected in ((180, True), (181.5, True), (240, False)):
            with self.subTest(duration=duration):
                self.assertEqual(
                    recording_identity_matches(
                        "Adrian Sînă",
                        "Păcat",
                        "Adrian Sina",
                        "Pacat",
                        expected_duration=180,
                        candidate_duration=duration,
                    ),
                    expected,
                )

    def test_feature_credits_are_compared_across_artist_and_title_fields(self) -> None:
        for artist, title, candidate_artist, candidate_title, expected in (
            (
                "Orion Quartet",
                "Night Signal (feat. Lyra)",
                "Orion Quartet",
                "Night Signal (feat. Vega)",
                False,
            ),
            (
                "Orion Quartet feat. Lyra",
                "Night Signal",
                "Orion Quartet",
                "Night Signal (feat. Vega)",
                False,
            ),
            (
                "Orion Quartet feat. Lyra",
                "Night Signal",
                "Orion Quartet",
                "Night Signal (feat. Lyra)",
                True,
            ),
            (
                "Orion Quartet",
                "Night Signal (feat. Lýra)",
                "Orion Quartet ft. Lyra",
                "Night Signal",
                True,
            ),
            (
                "Orion Quartet",
                "Night Signal (feat. Lyra & Vega)",
                "Orion Quartet feat. Vega, Lyra",
                "Night Signal",
                True,
            ),
            (
                "Orion Quartet",
                "Night Signal (feat. Lyra)",
                "Orion Quartet",
                "Night Signal",
                True,
            ),
        ):
            with self.subTest(
                artist=artist,
                title=title,
                candidate_artist=candidate_artist,
                candidate_title=candidate_title,
            ):
                self.assertEqual(
                    recording_identity_matches(
                        artist, title, candidate_artist, candidate_title
                    ),
                    expected,
                )
                self.assertEqual(
                    recording_identity_matches(
                        candidate_artist, candidate_title, artist, title
                    ),
                    expected,
                )


class TestRecordingDurationPolicy(IsolatedTestCase):
    def test_default_policy_rejects_shorter_edits_without_rejecting_rounding(
        self,
    ) -> None:
        self.assertEqual(SonoraConfig().single_duration_tolerance_seconds, 2.0)
        for candidate_duration, accepted in (
            (180.0, True),
            (179.0, True),
            (182.0, True),
            (182.1, False),
            (165.2, False),
        ):
            with self.subTest(candidate_duration=candidate_duration):
                self.assertEqual(
                    recording_identity_matches(
                        "Primary",
                        "Original Song",
                        "Primary",
                        "Original Song",
                        expected_duration=180.0,
                        candidate_duration=candidate_duration,
                    ),
                    accepted,
                )
                self.assertEqual(
                    _recording_matches_track_identity(
                        TrackInfo(
                            Path("recording.flac"),
                            artist="Primary",
                            title="Original Song",
                            duration=180.0,
                        ),
                        {
                            "artist": "Primary",
                            "title": "Original Song",
                            "duration": candidate_duration,
                        },
                    ),
                    accepted,
                )

    def test_missing_candidate_artist_does_not_verify_known_local_artist(self) -> None:
        track = TrackInfo(
            Path("recording.flac"), artist="Artist A", title="Intro", duration=30
        )
        self.assertFalse(
            _recording_matches_track_identity(track, {"title": "Intro", "duration": 30})
        )
        track.isrc = "USAAA0000001"
        self.assertTrue(
            _recording_matches_track_identity(
                track, {"title": "Intro", "duration": 30, "isrcs": ["USAAA0000001"]}
            )
        )


class TestMatchingEvaluation(IsolatedTestCase):
    def test_report_distinguishes_false_accepts_rejects_and_unresolved(self) -> None:
        cases = [
            MatchCase(str(index), "Alpha", "Song", "Alpha", "Song", expected, "Label")
            for index, expected in enumerate((True, True, False, False))
        ]
        with patch(
            "tools.evaluate_matching.recording_identity_matches",
            side_effect=(True, False, True, False),
        ):
            report = evaluate_cases(cases)
        self.assertEqual(report["true_accepts"], 1)
        self.assertEqual(report["false_accepts"], 1)
        self.assertEqual(report["false_rejects"], 1)
        self.assertEqual(report["true_rejects"], 1)
        self.assertEqual(report["unresolved"], 2)
        self.assertEqual(report["precision"], 0.5)
        self.assertEqual(report["recall"], 0.5)
        self.assertEqual(len(report["outcomes"]), 4)

    def test_empty_evaluation_does_not_claim_perfect_precision(self) -> None:
        report = evaluate_cases([])
        self.assertEqual(report["cases"], 0)
        self.assertEqual(report["unresolved"], 0)
        self.assertIsNone(report["precision"])
        self.assertIsNone(report["recall"])

    def test_committed_identity_corpus_matches_the_default_policy(self) -> None:
        corpus_path = Path(__file__).with_name("fixtures") / "matching_cases.json"
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        cases = [MatchCase(**case) for case in corpus["cases"]]
        self.assertTrue(any(case.expected_accept for case in cases))
        self.assertTrue(any(not case.expected_accept for case in cases))
        report = evaluate_cases(cases)
        self.assertEqual(report["false_accepts"], 0, report["outcomes"])
        self.assertEqual(report["false_rejects"], 0, report["outcomes"])


MBID = "11111111-2222-3333-4444-555555555555"


OTHER_MBID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


class TestArtistResolutionContracts(IsolatedTestCase):
    def test_empty_subtokens_do_not_become_feature_credits(self) -> None:
        self.assertEqual(artists.extract_featured_artist_tokens("& Guest &"), ["Guest"])

    def test_duplicate_alias_search_paths_are_loaded_once(self) -> None:
        config_root = self.test_directory / "home" / ".config"
        alias_path = config_root / "sonora" / "aliases.json"
        alias_path.parent.mkdir(parents=True)
        alias_path.write_text('{"Legal Name": "Stage"}')
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(config_root)}):
            self.assertEqual(artists._load_user_overrides(), {"legal name": "Stage"})

    def test_aliases_control_offline_resolution_and_registered_groups(self) -> None:
        alias_path = DIRS.user_config_path / "aliases.json"
        alias_path.parent.mkdir(parents=True, exist_ok=True)
        alias_path.write_text(
            json.dumps({"Legal Name": "Stage", "Alpha & Beta": "Ensemble"})
        )
        self.assertEqual(artists.normalize_artist_name("Legal Name"), "Stage")
        self.assertTrue(artists.is_single_group_artist("Alpha & Beta"))
        self.assertEqual(
            artists.extract_featured_artist_tokens("Legal Name"), ["Stage"]
        )
        with patch.object(
            Path, "home", return_value=DIRS.user_config_path.parent.parent
        ):
            artists._load_user_overrides.cache_clear()
            self.assertEqual(artists._load_user_overrides()["legal name"], "Stage")

    def test_artist_casing_retains_distinct_names_and_reference_acronyms(self) -> None:
        self.assertEqual(artists.harmonize_artist_casing("First", "Second"), "First")
        self.assertEqual(artists.harmonize_artist_casing("ABC", "abc"), "abc")
        self.assertEqual(artists.harmonize_artist_casing("Artist", "artist"), "artist")

    def test_algorithmic_artist_normalization_rules(self) -> None:
        cases = {
            "pnl": "PNL",
            "mgl": "MGL",
            "dmx": "DMX",
            "ctc": "CTC",
            "dzws": "DZWS",
            "m.g.l.": "M.G.L.",
            "m.g.l": "M.G.L.",
            "t.i.": "T.I.",
            "t.i": "T.I.",
            "d.o.c.": "D.O.C.",
            "asap rocky": "A$AP Rocky",
            "a$ap ferg": "A$AP Ferg",
            "asap mob": "A$AP Mob",
            "joey badass": "Joey Bada$$",
            "currensy": "Curren$y",
            "suicideboys": "$uicideboy$",
            "ty dolla sign": "Ty Dolla $ign",
            "satra benz": "ȘATRA B.E.N.Z.",
            "bug mafia": "B.U.G. Mafia",
            "parazitii": "Paraziții",
            "subcarpati": "Subcarpați",
            "sisu tudor": "Sișu Tudor",
            "dragonu ak47": "Dragonu'",
            "deliric 1": "Deliric",
            "albert nbn": "Albert Nbn",
            "6lack": "6LACK",
            "nf": "NF",
            "idk": "IDK",
            "atb": "ATB",
            "nosfe": "NOSFE",
            "rava": "RAVA",
            "renvto": "RENVTØ",
            "noua unspe": "NOUA UNSPE",
            "911": "NOUA UNSPE",
            "Kendrick Lamar": "Kendrick Lamar",
            "Mario": "Mario",
            "The Game": "The Game",
            "Eminem": "Eminem",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(artists.normalize_artist_name(raw), expected)

    def test_persisted_sqlite_artist_aliases_and_precedence(self) -> None:
        from sonora.core.state import get_library_state

        state_mgr = get_library_state()
        state_mgr.record_artist_alias("k-dot", "Kendrick Lamar", source="musicbrainz")
        artists.clear_artist_cache()
        self.assertEqual(artists.normalize_artist_name("K-Dot"), "Kendrick Lamar")
        alias_path = DIRS.user_config_path / "aliases.json"
        alias_path.parent.mkdir(parents=True, exist_ok=True)
        alias_path.write_text(json.dumps({"k-dot": "Custom King"}))
        artists.clear_artist_cache()
        self.assertEqual(artists.normalize_artist_name("K-Dot"), "Custom King")

    def test_artist_resolution_persists_discovered_aliases_to_library_state(
        self,
    ) -> None:
        from sonora.core.state import get_library_state

        mb_payload = [
            {
                "name": "Kendrick Lamar",
                "alias-list": [
                    {"alias": "K-Dot", "type": "Artist name"},
                    {"alias": "Kung Fu Kenny", "type": "Artist name"},
                ],
            }
        ]
        with (
            patch.object(
                musicbrainz, "search_musicbrainz_artists", return_value=mb_payload
            ),
            patch.object(artist_resolution, "SESSION") as mock_session,
        ):
            mock_session.get.return_value.status_code = 200
            mock_session.get.return_value.json.return_value = {"data": []}
            resolved = artist_resolution.resolve_artist_name("k-dot")
            self.assertEqual(resolved, "Kendrick Lamar")
            state_mgr = get_library_state()
            self.assertEqual(state_mgr.get_artist_alias("k-dot"), "Kendrick Lamar")
            self.assertEqual(
                state_mgr.get_artist_alias("kung fu kenny"), "Kendrick Lamar"
            )

    def test_balanced_features_preserve_unclosed_brackets(self) -> None:
        self.assertEqual(
            artists.extract_balanced_features("Recording (feat. Guest"),
            ("Recording (feat. Guest", [], "(", ")"),
        )

    def test_feature_token_cleanup_handles_delimiters_and_fuzzy_duplicates(
        self,
    ) -> None:
        self.assertEqual(
            artists.extract_featured_artist_tokens(
                [None, "", " ; Guest;;Singer / Ensemble;", "()", "Guestt", "!!!"]
            ),
            ["Guestt", "Singer", "Ensemble"],
        )
        self.assertEqual(artists.extract_featured_artist_tokens([None, ""]), [])
        policy = SonoraConfig(known_group_artists=frozenset({"Alpha & Beta"}))
        with patch("sonora.core.artists.get_config", return_value=policy):
            self.assertEqual(
                artists.extract_featured_artist_tokens("Guest; Alpha & Beta"),
                ["Guest", "Alpha & Beta"],
            )

    def test_user_alias_files_empty_malformed_and_nonobject_content(self) -> None:
        path = DIRS.user_config_path / "aliases.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        for content, expected in (
            ("", {}),
            ("broken", {}),
            ("[]", {}),
            (json.dumps({"Legal Name": "Stage Name"}), {"legal name": "Stage Name"}),
        ):
            artists._load_user_overrides.cache_clear()
            path.write_text(content)
            with self.subTest(content=content):
                self.assertEqual(artists._load_user_overrides(), expected)

    def test_artist_resolution_obeys_alias_and_punctuation_precedence(self) -> None:
        cases: tuple[
            tuple[str, list[dict[str, object]], list[dict[str, object]], str], ...
        ] = (
            ("Primary", [{"name": "Primary!"}], [], "Primary"),
            ("Primary", [], [{"name": "Primary!"}], "Primary"),
            ("Primary", [], [{"name": "Primary"}], "Primary"),
            ("Primary", [], [{"name": "Wrong"}], "Primary"),
            (
                "Legal Name",
                [
                    {
                        "name": "Ensemble",
                        "alias-list": [
                            {},
                            {"alias": "Unrelated"},
                            {"alias": "Legal Name"},
                        ],
                    }
                ],
                [],
                "Ensemble",
            ),
            ("Primary", [{"name": "Primary"}], [], "Primary"),
            ("Singer", [{"name": "SINGER"}], [], "Singer"),
            ("ABC", [{"name": "Abc"}], [], "ABC"),
            ("Primary", [{"name": "P.r.i.m.a.r.y"}], [], "Primary"),
            ("primary", [], [{"name": "Primary"}], "Primary"),
            ("Singer", [], [{"name": "SINGER"}], "Singer"),
            ("ABC", [], [{"name": "Abc"}], "ABC"),
            ("primary", [], [{"name": "P.r.i.m.a.r.y"}, {}], "primary"),
            (
                "Legal Name",
                [
                    {
                        "name": "Stage",
                        "alias-list": [{"alias": "Legal Name", "type": "Legal name"}],
                    }
                ],
                [],
                "Stage",
            ),
            (
                "Alias",
                [
                    {
                        "name": "Stage",
                        "alias-list": [{"alias": "Alias", "type": "Search hint"}],
                    }
                ],
                [],
                "Alias",
            ),
            (
                "Alias",
                [
                    {
                        "name": "Stage",
                        "alias-list": [{"alias": "Alias", "type": "Legal name"}],
                    }
                ],
                [],
                "Alias",
            ),
            (
                "Alias",
                [{"name": "Stage Name", "alias-list": [{"alias": "Alias"}]}],
                [],
                "Alias",
            ),
            (
                "Legal Name",
                [
                    {
                        "name": "Legal Name Ensemble",
                        "alias-list": [{"alias": "Legal Name"}],
                    }
                ],
                [],
                "Legal Name Ensemble",
            ),
            ("Alias", [{"name": "Stage", "alias-list": ["Alias"]}], [], "Stage"),
            ("Nobody", [{}, {"name": "Unrelated"}], [], "Nobody"),
        )
        for query, mb_candidates, deezer_candidates, expected in cases:
            with (
                self.subTest(
                    query=query, candidates=mb_candidates, deezer=deezer_candidates
                ),
                patch.object(artist_resolution, "get_cached_api", return_value=None),
                patch.object(
                    musicbrainz,
                    "search_musicbrainz_artists",
                    return_value=mb_candidates,
                ),
                patch.object(
                    SESSION,
                    "get",
                    return_value=httpx.Response(200, json={"data": deezer_candidates}),
                ),
            ):
                self.assertEqual(artist_resolution.resolve_artist_name(query), expected)

    def test_cached_corrupt_expansions_and_branding_preserve_existing_names(
        self,
    ) -> None:
        for query, cached, expected in (
            ("primary", "PRIMARY", "primary"),
            ("Alias", "Alias Ensemble", "Alias"),
            ("Singer", "SINGER", "Singer"),
            ("ABC", "ABC", "ABC"),
            ("Primary", "Primary", "Primary"),
        ):
            with (
                self.subTest(query=query, cached=cached),
                patch.object(artist_resolution, "get_cached_api", return_value=cached),
                patch.object(
                    musicbrainz, "search_musicbrainz_artists", return_value=[]
                ),
                patch.object(SESSION, "get", return_value=httpx.Response(404)),
            ):
                self.assertEqual(artist_resolution.resolve_artist_name(query), expected)
        self.assertEqual(artist_resolution.resolve_artist_name(" "), "Unknown Artist")
        self.assertEqual(artist_resolution.resolve_artist_name("!!!"), "!!!")

    def test_cached_allcaps_artist_names_skip_provider_lookups(self) -> None:
        for name in ("RAVA", "BLANCO", "ELIO"):
            with (
                self.subTest(artist=name),
                patch.object(artist_resolution, "get_cached_api", return_value=name),
                patch.object(
                    musicbrainz, "search_musicbrainz_artists", return_value=[]
                ) as search,
                patch.object(
                    SESSION, "get", return_value=httpx.Response(404)
                ) as request,
            ):
                self.assertEqual(artist_resolution.resolve_artist_name(name), name)
                search.assert_not_called()
                request.assert_not_called()

    def test_registered_groups_and_featured_subtokens_are_preserved(self) -> None:
        policy = SonoraConfig(known_group_artists=frozenset({"Alpha & Beta"}))
        with patch("sonora.core.artists.get_config", return_value=policy):
            self.assertTrue(artists.is_single_group_artist("Alpha & Beta"))
            self.assertEqual(
                artists.extract_artist_features("Alpha & Beta"), ("Alpha & Beta", [])
            )
            self.assertEqual(
                artists.extract_featured_artist_tokens(
                    ["Alpha", "Alpha & Beta", "Gamma"]
                ),
                ["Alpha & Beta", "Gamma"],
            )
        self.assertEqual(artists.extract_featured_artist_tokens([None, " "]), [])
        self.assertEqual(artists.extract_artist_features(None), ("", []))
        self.assertFalse(artists.is_artist_acronym(None))
        self.assertFalse(artists.is_single_group_artist("!!!"))

    def test_joint_main_artists_are_not_repeated_as_features(self) -> None:
        self.assertEqual(
            artists.extract_featured_artist_tokens(
                ["Dora Gaitanovici", "Dani Bușoi", "Guest"],
                primary_artist="Dora Gaitanovici & Dani Bușoi",
            ),
            ["Guest"],
        )
        self.assertEqual(
            artists.extract_featured_artist_tokens(
                ["Primary", "Guest"], primary_artist="Primary feat. Guest"
            ),
            ["Guest"],
        )
        policy = SonoraConfig(known_group_artists=frozenset({"Alpha & Beta"}))
        with patch("sonora.core.artists.get_config", return_value=policy):
            self.assertEqual(
                artists.extract_featured_artist_tokens(
                    ["Alpha & Beta", "Alpha", "Guest"], primary_artist="Alpha & Beta"
                ),
                ["Alpha", "Guest"],
            )

    def test_artist_provider_failures_preserve_clean_input(self) -> None:
        with (
            patch.object(artist_resolution, "get_cached_api", return_value=None),
            patch.object(
                musicbrainz,
                "search_musicbrainz_artists",
                side_effect=OSError("offline"),
            ),
            patch.object(SESSION, "get", side_effect=OSError("offline")),
        ):
            self.assertEqual(artist_resolution.resolve_artist_name("Singer"), "Singer")
            self.assertFalse(artists.is_single_group_artist("Alpha & Beta"))


class TestAlbumIdentityContracts(IsolatedTestCase):
    def test_album_position_indexes_use_disc_and_position(self) -> None:
        recording = {"title": "Recording", "artist": "Primary"}
        details = {"tracks_by_disc_and_position": {(2, 3): recording}}
        self.assertTrue(
            matching._pos_in_album(3, disc_number=2, album_mb_release_details=details)
        )
        self.assertFalse(
            matching._pos_in_album(
                1,
                1,
                album_mb_release_details={"tracks_by_disc_and_position": {(1, 1): {}}},
            )
        )
        self.assertTrue(matching._pos_in_album(1, album_track_mbids={(1, 1): MBID}))
        self.assertTrue(
            matching._pos_in_album(1, disc_number=2, album_track_mbids={(2, 1): MBID})
        )

    def test_recording_ids_require_release_identity_before_repairing_positions(
        self,
    ) -> None:
        for index in ({(2, 3): MBID}, {(1, 3): MBID}):
            recording = TrackInfo(Path("recording.flac"))
            with self.subTest(index=index):
                self.assertIsNone(
                    matching._match_position_by_mbid(recording, MBID, index, None)
                )
                self.assertEqual(recording.disc_number, 1)
        recording = TrackInfo(Path("recording.flac"))
        release = {
            "tracks_by_mbid": {
                MBID: {"title": "Recording", "disc_number": "2", "position": "3"}
            }
        }
        self.assertEqual(
            matching._match_position_by_mbid(recording, MBID, None, release), 3
        )
        self.assertEqual(recording.disc_number, 2)

    def test_track_positions_with_duplicate_identifiers_require_unique_context(
        self,
    ) -> None:
        recording = TrackInfo(Path("recording.flac"), disc_number=2)
        release = {
            "tracks_by_disc_and_position": {
                (1, 1): {"isrc": "USAAA2100001", "title": "Original"},
                (2, 3): {"isrc": "USAAA2100001", "title": "Original"},
            }
        }
        self.assertIsNone(
            matching._match_position_by_authoritative_id(
                recording, None, "isrc", "USAAA2100001", [None, release]
            )
        )
        self.assertEqual(
            matching._match_position_by_authoritative_id(
                recording, 3, "isrc", "USAAA2100001", [release]
            ),
            3,
        )
        single = {
            "tracks_by_disc_and_position": {
                (1, 3): {"itunes_trackid": "123", "disk_number": 2, "title": "Original"}
            }
        }
        self.assertEqual(
            matching._match_position_by_authoritative_id(
                recording, None, "itunes_trackid", "123", [single]
            ),
            3,
        )

    def test_disc_position_map_handles_missing_and_invalid_positions(self) -> None:
        self.assertEqual(
            matching._build_album_tracks_by_disc_map(
                None,
                {(2, 3): MBID, (1, 1): MBID},
                [{"tracks_by_disc_and_position": {(2, 4): {}, (3, 1): {}, (1, 2): {}}}],
            ),
            {1: {1, 2}, 2: {3, 4}, 3: {1}},
        )


class TestMusicBrainzMetadataContracts(IsolatedTestCase):
    def test_unmatched_fingerprints_do_not_create_a_musicbrainz_identifier(
        self,
    ) -> None:
        with patch.object(acoustid, "lookup_acoustid", return_value=None):
            mb_metadata._update_acoustid(
                self.recording, self.recording.file_path, "fixture"
            )
        self.assertIsNone(self.recording.musicbrainz_trackid)

    def test_malformed_release_recordings_are_not_silently_replaced_by_search_results(
        self,
    ) -> None:
        recording = TrackInfo(
            Path("recording.flac"), track_number=3, musicbrainz_trackid=MBID
        )
        with self.assertRaisesRegex(TypeError, "JSON object"):
            mb_metadata._find_album_release_recording(
                recording, {"tracks_by_disc_and_position": {(2, 3): None}}, 2
            )
        with self.assertRaisesRegex(TypeError, "JSON object"):
            mb_metadata._extract_release_recording(
                recording, {"tracks_by_mbid": {MBID: []}}, None
            )

    def test_verified_track_id_and_existing_date_do_not_trigger_reassignment(
        self,
    ) -> None:
        self.recording.musicbrainz_trackid = MBID
        self.recording.date = "2019"
        with patch.object(musicbrainz, "fetch_track_mbid") as fetch:
            mb_metadata._resolve_musicbrainz_track_id(self.recording, None, None)
        fetch.assert_not_called()
        with patch.object(
            musicbrainz,
            "search_musicbrainz_release",
            return_value={"id": OTHER_MBID, "date": "2020"},
        ):
            mb_metadata._resolve_musicbrainz_album_id(self.recording, None, None)
        self.assertEqual(
            (
                self.recording.musicbrainz_trackid,
                self.recording.musicbrainz_albumid,
                self.recording.date,
            ),
            (MBID, OTHER_MBID, "2019"),
        )

    def test_research_that_finds_an_id_without_recording_details_keeps_no_false_metadata(
        self,
    ) -> None:
        self.recording.musicbrainz_trackid = MBID
        with (
            patch.object(
                musicbrainz,
                "fetch_musicbrainz_recording_details",
                side_effect=[{"title": "Wrong", "artist": "Unrelated"}, None],
            ),
            patch.object(musicbrainz, "fetch_track_mbid", return_value=OTHER_MBID),
        ):
            self.assertIsNone(
                mb_metadata._fetch_standalone_recording(self.recording, force=True)
            )
        self.assertEqual(self.recording.musicbrainz_trackid, OTHER_MBID)
        self.assertEqual(self.recording.title, "Recording")

    def test_invalid_track_totals_and_conflicting_release_layout_are_withheld(
        self,
    ) -> None:
        self.recording.total_tracks = 10
        self.recording.total_discs = 2
        mb_metadata._apply_mb_recording_metadata(
            self.recording, {"disc_total_tracks": "bad"}, True, None, False, force=True
        )
        self.recording.musicbrainz_albumid = MBID
        mb_metadata._apply_mb_release_metadata(
            self.recording,
            MBID,
            {"title": "Record", "total_tracks": 5, "total_discs": 1},
        )
        self.assertEqual(
            (self.recording.total_tracks, self.recording.total_discs), (10, 2)
        )
        self.assertIn("album_sequence", self.recording.metadata_conflicts)

    def test_recording_artist_updates_require_album_credit_or_strong_identity(
        self,
    ) -> None:
        recording = TrackInfo(
            Path("recording.flac"), artist="Primary", title="Recording"
        )
        for candidate_artist, album_artist, force, expected in (
            ("PRIMARY", None, True, True),
            ("Guest", "Various Artists", False, True),
            ("Guest", "Guest", False, True),
            ("Guest", "Primary", False, False),
        ):
            with self.subTest(
                candidate_artist=candidate_artist,
                album_artist=album_artist,
                force=force,
            ):
                self.assertEqual(
                    mb_metadata._should_update_track_artist(
                        recording,
                        {"title": "Recording"},
                        candidate_artist,
                        True,
                        album_artist,
                        False,
                        force=force,
                    ),
                    expected,
                )

    def setUp(self) -> None:
        self.recording = TrackInfo(
            Path("recording.flac"), artist="Primary", title="Recording", album="Record"
        )
        self.details = {
            "title": "Recording",
            "artist": "Primary",
            "recording_mbid": MBID,
            "isrc": "USAAA2100001",
        }
        self.enterContext(
            patch.object(
                artist_resolution, "resolve_artist_name", side_effect=lambda name: name
            )
        )

    def test_any_recording_isrc_can_fill_a_missing_title(self) -> None:
        self.recording.title = "Untitled"
        self.recording.isrc = "USAAA2100002"
        candidate = {
            "title": "Recording",
            "isrc": "USAAA2100001",
            "isrcs": ["USAAA2100001", "USAAA2100002"],
        }
        self.assertTrue(mb_metadata._fill_missing_title(self.recording, candidate))
        self.assertEqual(self.recording.title, "Recording")
        self.assertFalse(mb_metadata._fill_missing_title(self.recording, candidate))
        self.recording.title = "Unrelated"
        self.recording.isrc = "USAAA2100003"
        self.assertFalse(mb_metadata._fill_missing_title(self.recording, candidate))
        self.assertEqual(self.recording.title, "Unrelated")

    def test_existing_mbid_skips_fingerprinting_without_force(self) -> None:
        self.recording.musicbrainz_trackid = MBID
        with patch.object(acoustid, "lookup_acoustid") as lookup:
            mb_metadata._update_acoustid(
                self.recording, self.recording.file_path, "fixture"
            )
        lookup.assert_not_called()

    def test_release_recording_lookup_uses_position_fallbacks(self) -> None:
        self.recording.track_number = 3
        self.assertEqual(
            mb_metadata._find_album_release_recording(
                self.recording,
                {"tracks_by_disc_and_position": {(2, 3): self.details}},
                2,
            ),
            self.details,
        )
        self.assertIsNone(
            mb_metadata._find_album_release_recording(
                self.recording, {"title": "Record"}, 2
            )
        )

    def test_track_id_assignment_requires_title_identity_when_release_is_known(
        self,
    ) -> None:
        self.recording.track_number = 3
        mb_metadata._resolve_musicbrainz_track_id(
            self.recording, {(1, 3): MBID}, self.details
        )
        self.assertEqual(self.recording.musicbrainz_trackid, MBID)
        self.recording.musicbrainz_trackid = None
        mb_metadata._resolve_musicbrainz_track_id(
            self.recording, {(1, 3): MBID}, {"title": "Unrelated", "artist": "Wrong"}
        )
        self.assertIsNone(self.recording.musicbrainz_trackid)
        mb_metadata._resolve_musicbrainz_track_id(self.recording, {(1, 3): MBID}, None)
        self.assertIsNone(self.recording.musicbrainz_trackid)

    def test_recording_extraction_prefers_mbid_and_rejects_unrelated_titles(
        self,
    ) -> None:
        self.recording.musicbrainz_trackid = MBID
        self.assertEqual(
            mb_metadata._extract_release_recording(
                self.recording, {"tracks_by_mbid": {MBID: self.details}}, None
            ),
            self.details,
        )
        self.assertIsNone(
            mb_metadata._extract_release_recording(
                self.recording,
                {"title": "Record"},
                {"title": "Other", "artist": "Wrong"},
            )
        )
        self.assertIsNone(
            mb_metadata._extract_release_recording(
                self.recording, {"title": "Record"}, None
            )
        )

    def test_standalone_recording_missing_or_failed_research_leaves_no_false_id(
        self,
    ) -> None:
        self.recording.musicbrainz_trackid = MBID
        with patch.object(
            musicbrainz, "fetch_musicbrainz_recording_details", return_value=None
        ):
            self.assertIsNone(mb_metadata._fetch_standalone_recording(self.recording))
        with (
            patch.object(
                musicbrainz,
                "fetch_musicbrainz_recording_details",
                return_value={"artist": "Wrong", "title": "Other"},
            ),
            patch.object(musicbrainz, "fetch_track_mbid", return_value=None),
        ):
            self.assertIsNone(
                mb_metadata._fetch_standalone_recording(self.recording, force=True)
            )
        self.assertIsNone(self.recording.musicbrainz_trackid)

    def test_valid_album_identifiers_dates_and_totals_are_preserved(self) -> None:
        self.recording.musicbrainz_albumid = MBID
        self.recording.date = "2019"
        mb_metadata._resolve_musicbrainz_album_id(self.recording, OTHER_MBID, None)
        self.assertEqual(
            (self.recording.musicbrainz_albumid, self.recording.date), (MBID, "2019")
        )
        with patch.object(
            musicbrainz, "fetch_musicbrainz_release_details", return_value=None
        ):
            mb_metadata._apply_mb_release_metadata(self.recording, None, None)
        self.assertEqual(self.recording.musicbrainz_albumid, MBID)
        self.recording.total_tracks = 5
        self.recording.total_discs = 1
        mb_metadata._apply_mb_release_metadata(
            self.recording,
            MBID,
            {"title": "Record", "total_tracks": "bad", "total_discs": "bad"},
            force=True,
        )
        self.assertEqual(
            (self.recording.total_tracks, self.recording.total_discs), (5, 1)
        )

    def test_acoustid_respects_verified_album_ids_and_text_identity(self) -> None:
        with (
            patch.object(acoustid, "lookup_acoustid", return_value=MBID) as lookup,
            patch.object(
                musicbrainz,
                "fetch_musicbrainz_recording_details",
                return_value=self.details,
            ),
        ):
            mb_metadata._update_acoustid(
                self.recording, self.recording.file_path, "fixture"
            )
            self.assertEqual(self.recording.musicbrainz_trackid, MBID)
            self.recording.track_number = 1
            self.recording.metadata_sources["musicbrainz_trackid"] = "MusicBrainz"
            mb_metadata._update_acoustid(
                self.recording,
                self.recording.file_path,
                "fixture",
                {(1, 1): MBID},
                force=True,
            )
        lookup.assert_called_once()
        for details in (None, {"artist": "Unrelated", "title": "Wrong"}):
            recording = dataclasses.replace(self.recording, musicbrainz_trackid=None)
            with (
                self.subTest(details=details),
                patch.object(acoustid, "lookup_acoustid", return_value=MBID),
                patch.object(
                    musicbrainz,
                    "fetch_musicbrainz_recording_details",
                    return_value=details,
                ),
            ):
                mb_metadata._update_acoustid(recording, recording.file_path, "fixture")
            self.assertIsNone(recording.musicbrainz_trackid)
        with patch.object(acoustid, "lookup_acoustid", side_effect=OSError("offline")):
            mb_metadata._update_acoustid(
                self.recording, self.recording.file_path, "fixture", force=True
            )

    def test_album_id_search_preserves_valid_dates_and_skips_invalid_identifiers(
        self,
    ) -> None:
        for release, expected_id, expected_date in (
            (None, None, None),
            ({"id": "bad", "date": "2020"}, None, "2020"),
            ({"id": MBID, "date": "2020"}, MBID, "2020"),
            ({"id": MBID, "date": "bad"}, MBID, None),
        ):
            recording = self.recording.copy()
            with (
                self.subTest(release=release),
                patch.object(
                    musicbrainz, "search_musicbrainz_release", return_value=release
                ),
            ):
                mb_metadata._resolve_musicbrainz_album_id(recording, None, None)
            self.assertEqual(
                (recording.musicbrainz_albumid, recording.date),
                (expected_id, expected_date),
            )

    def test_standalone_identity_conflict_discards_id_and_can_research_when_forced(
        self,
    ) -> None:
        self.recording.musicbrainz_trackid = MBID
        self.recording.metadata_sources["musicbrainz_trackid"] = "MusicBrainz"
        wrong = {"title": "Different Recording", "artist": "Unrelated"}
        with patch.object(
            musicbrainz, "fetch_musicbrainz_recording_details", return_value=wrong
        ):
            self.assertIsNone(mb_metadata._fetch_standalone_recording(self.recording))
        self.assertIsNone(self.recording.musicbrainz_trackid)
        self.assertNotIn("musicbrainz_trackid", self.recording.metadata_sources)
        self.recording.musicbrainz_trackid = MBID
        with (
            patch.object(
                musicbrainz,
                "fetch_musicbrainz_recording_details",
                side_effect=[wrong, self.details],
            ),
            patch.object(musicbrainz, "fetch_track_mbid", return_value=OTHER_MBID),
        ):
            self.assertEqual(
                mb_metadata._fetch_standalone_recording(self.recording, force=True),
                self.details,
            )
        self.assertEqual(self.recording.musicbrainz_trackid, OTHER_MBID)

    def test_unknown_and_alien_artists_can_be_updated_while_preserving_known_artists(
        self,
    ) -> None:
        for artist, alien, corrupt, expected in (
            ("", False, False, True),
            ("Unknown Artist", False, False, True),
            ("Primary", True, False, True),
            ("Primary", False, True, True),
            ("Primary", False, False, False),
        ):
            recording = dataclasses.replace(
                self.recording, artist=artist, is_alien=alien
            )
            with self.subTest(artist=artist, alien=alien, corrupt=corrupt):
                self.assertEqual(
                    mb_metadata._should_update_track_artist(
                        recording, {}, "Other", False, None, corrupt
                    ),
                    expected,
                )

    def test_release_artist_and_title_conflicts_clear_wrong_album_ids(self) -> None:
        for release in (
            {"title": "Record", "album_artist": "Unrelated"},
            {"title": "Different", "album_artist": "Primary"},
        ):
            recording = dataclasses.replace(
                self.recording,
                musicbrainz_albumid=MBID,
                musicbrainz_releasegroupid=OTHER_MBID,
            )
            with (
                self.subTest(release=release),
                patch.object(
                    musicbrainz,
                    "fetch_musicbrainz_release_details",
                    return_value=release,
                ),
            ):
                mb_metadata._apply_mb_release_metadata(recording, None, None)
            self.assertIsNone(recording.musicbrainz_albumid)
            self.assertEqual(
                recording.musicbrainz_releasegroupid,
                None if release["album_artist"] == "Unrelated" else OTHER_MBID,
            )

    def test_release_metadata_preserves_track_counts_and_normalizes_album_identity(
        self,
    ) -> None:
        release: dict[str, object] = {
            "title": "Record",
            "album_artist": "Primary",
            "musicbrainz_artistid": OTHER_MBID,
            "date": "2020",
            "original_date": "2019",
            "total_tracks": 10,
            "total_discs": 2,
            "tracks_by_disc_and_position": {
                (1, 3): {"title": "Recording", "disc_total_tracks": 10}
            },
        }
        self.recording.musicbrainz_albumid = MBID
        self.recording.album = "Unknown Album"
        self.recording.track_number = 3
        mb_metadata._apply_mb_release_metadata(
            self.recording, MBID, release, force=True
        )
        self.assertEqual(
            (
                self.recording.album,
                self.recording.album_artist,
                self.recording.total_tracks,
                self.recording.total_discs,
            ),
            ("Record", "Primary", 10, 2),
        )
        self.assertEqual(
            (
                self.recording.date,
                self.recording.original_date,
                self.recording.musicbrainz_artistid,
            ),
            ("2020", "2019", OTHER_MBID),
        )
        self.recording.track_number = 12
        self.recording.total_tracks = None
        mb_metadata._apply_mb_release_metadata(
            self.recording, MBID, release, force=True
        )
        self.assertIsNone(self.recording.total_tracks)

    def test_recording_metadata_updates_missing_identity_and_preserves_disc_totals(
        self,
    ) -> None:
        self.recording.artist = "Unknown Artist"
        self.recording.title = "Untitled"
        self.recording.isrc = "USAAA2100001"
        details = {
            **self.details,
            "disc_total_tracks": "10",
            "first-release-date": "2020",
        }
        mb_metadata._apply_mb_recording_metadata(
            self.recording, details, True, "Primary", False
        )
        self.assertEqual(
            (
                self.recording.artist,
                self.recording.title,
                self.recording.total_tracks,
                self.recording.date,
            ),
            ("Primary", "Recording", 10, "2020"),
        )

    def test_network_failure_does_not_clear_existing_identity(self) -> None:
        with patch.object(
            musicbrainz, "fetch_track_mbid", side_effect=OSError("offline")
        ):
            mb_metadata._update_musicbrainz(self.recording, None, None, None)
        self.assertEqual(self.recording.title, "Recording")


class TestArtistResponseContracts(IsolatedTestCase):
    def test_artist_resolution_exposes_malformed_deezer_responses(self) -> None:
        payload: object
        for payload in ([], {"data": "wrong"}, {"data": [None]}):
            with (
                self.subTest(payload=payload),
                patch(
                    "sonora.services.musicbrainz.search_musicbrainz_artists",
                    return_value=[],
                ),
                patch("sonora.services.artist_resolution.SESSION.get") as request,
            ):
                request.return_value.status_code = 200
                request.return_value.json.return_value = payload
                with self.assertRaises(TypeError):
                    artist_resolution.resolve_artist_name("Artist Alpha")


MBID = "11111111-2222-3333-4444-555555555555"


OTHER_MBID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


class TestRecordingSafety(IsolatedTestCase):
    def test_numeric_song_names_are_preserved_and_do_not_match_other_numbers(
        self,
    ) -> None:
        for title in ("23", "777", "007", "2020", "123 [FLAC]"):
            with self.subTest(title=title):
                self.assertFalse(matching._is_generic_title(title))
                self.assertTrue(
                    matching._is_strong_track_match("Alpha", title, "Alpha", title)
                )
                self.assertFalse(
                    matching._is_strong_track_match("Alpha", title, "Alpha", "9999")
                )

    def test_matching_isrc_cannot_override_a_meaningful_catalog_title(self) -> None:
        correct = {"title": "Original Song", "artist": "Alpha", "isrc": "USAAA2400001"}
        wrong = {**correct, "title": "Different Song"}
        self.assertEqual(
            matching._find_track_in_album_mapping(
                {"original song": correct},
                {(1, 1): wrong},
                "Original Song",
                1,
                disc_number=1,
                artist="Alpha",
                isrc="USAAA2400001",
            ),
            (correct, True),
        )
        self.assertEqual(
            matching._find_track_in_album_mapping(
                {},
                {(1, 1): wrong},
                "Original Song",
                1,
                disc_number=1,
                artist="Alpha",
                isrc="USAAA2400001",
            ),
            (None, False),
        )

    def test_itunes_direct_lookup_validates_identity_before_returning_metadata(
        self,
    ) -> None:
        for name, length, expected in (
            ("Original Song", 180000, "123"),
            ("Different Song", 180000, None),
            ("Original Song", 90000, None),
        ):
            payload = {
                "results": [
                    {
                        "trackId": 123,
                        "trackName": name,
                        "artistName": "Alpha",
                        "trackTimeMillis": length,
                    }
                ]
            }
            with (
                self.subTest(name=name, length=length),
                patch.object(itunes, "get_cached_api", return_value=None),
                patch.object(itunes, "set_cached_api"),
                patch.object(
                    SESSION,
                    "get",
                    return_value=httpx.Response(
                        200,
                        json=payload,
                        request=httpx.Request("GET", "https://itunes.apple.com/lookup"),
                    ),
                ) as request,
                patch.object(itunes, "search_itunes", return_value=[]),
            ):
                metadata = itunes.fetch_itunes_track_metadata(
                    "Alpha", "Original Song", track_id="123", duration=180
                )
            self.assertEqual(
                metadata.get("itunes_trackid") if metadata else None, expected
            )
            self.assertEqual(request.call_args.kwargs["params"]["id"], "123")

    def test_conflicting_stored_itunes_id_is_cleared_only_with_provider_evidence(
        self,
    ) -> None:
        for response, expected_id in (
            ({"artistName": "Alpha", "trackName": "Different Song"}, None),
            ({"artistName": "Alpha", "trackName": "Original Song"}, "123"),
            (None, "123"),
        ):
            track = TrackInfo(
                file_path=Path("recording.flac"),
                artist="Alpha",
                title="Original Song",
                itunes_trackid="123",
                itunes_collectionid="456",
            )
            with (
                self.subTest(response=response),
                patch.object(
                    itunes,
                    "lookup_itunes_track",
                    return_value=_parse_itunes_track(response)
                    if response is not None
                    else None,
                ),
                patch.object(itunes, "fetch_itunes_track_metadata", return_value=None),
            ):
                catalog_metadata._update_itunes(track, force=True)
            self.assertEqual(track.itunes_trackid, expected_id)
            self.assertEqual(track.itunes_collectionid, "456" if expected_id else None)

    def test_itunes_search_abstains_from_equally_supported_recordings(self) -> None:
        candidates = [
            {"trackId": identifier, "trackName": "Original Song", "artistName": "Alpha"}
            for identifier in (123, 456)
        ]
        with patch.object(itunes, "search_itunes", return_value=candidates):
            self.assertIsNone(
                itunes.fetch_itunes_track_metadata("Alpha", "Original Song")
            )
        with patch.object(itunes, "search_itunes", return_value=candidates[:1]):
            metadata = itunes.fetch_itunes_track_metadata("Alpha", "Original Song")
        assert metadata is not None
        self.assertEqual(metadata["itunes_trackid"], "123")

    def test_acoustic_near_ties_and_wrong_durations_do_not_select_an_id(self) -> None:
        for scores, length, expected in (
            ((0.99, 0.98), 180, None),
            ((0.99, 0.9), 180, MBID),
            ((0.99, 0.9), 90, None),
        ):
            payload = {
                "status": "ok",
                "results": [
                    {
                        "score": score,
                        "recordings": [
                            {
                                "id": identifier,
                                "title": "Original Song",
                                "artists": [{"name": "Alpha"}],
                                "duration": length,
                            }
                        ],
                    }
                    for score, identifier in zip(scores, (MBID, OTHER_MBID))
                ],
            }
            with (
                self.subTest(scores=scores, length=length),
                patch.object(
                    acoustid,
                    "fingerprint_audio_file",
                    return_value=(180, "fingerprint"),
                ),
                patch.object(acoustid, "get_cached_api", return_value=None),
                patch.object(acoustid, "set_cached_api"),
                patch.object(acoustid._ACOUSTID_LIMITER, "wait"),
                patch("acoustid.lookup", return_value=payload) as lookup,
            ):
                self.assertEqual(
                    acoustid.lookup_acoustid(
                        Path("recording.flac"), "fixture", "Alpha", "Original Song"
                    ),
                    expected,
                )
            self.assertEqual(lookup.call_args.args[2], 180)
            self.assertEqual(lookup.call_args.kwargs["meta"], ["recordings", "isrcs"])

    def test_theaudiodb_validates_recording_id_text_and_duration(self) -> None:
        for title, length, identifier, expected in (
            ("Original Song", 180000, MBID, "Rock"),
            ("Wrong Song", 180000, MBID, None),
            ("Original Song", 90000, MBID, None),
            ("Original Song", 180000, OTHER_MBID, None),
        ):
            raw_track = {
                "strArtist": "Alpha",
                "strTrack": title,
                "strMusicBrainzID": identifier,
                "intDuration": length,
                "strGenre": "Rock",
            }
            with (
                self.subTest(title=title, length=length, identifier=identifier),
                patch.object(theaudiodb, "get_cached_api", return_value=None),
                patch.object(theaudiodb, "set_cached_api"),
                patch.object(
                    SESSION,
                    "get",
                    side_effect=[
                        httpx.Response(200, json={"track": [raw_track]}),
                        httpx.Response(200, json={"track": None}),
                    ],
                ) as request,
            ):
                metadata = theaudiodb.fetch_theaudiodb_track_details(
                    "Alpha", "Original Song", recording_mbid=MBID, duration=180
                )
            self.assertEqual(metadata.get("genre") if metadata else None, expected)
            self.assertIn("track-mb.php", request.call_args_list[0].args[0])

    def test_check_flags_shared_ids_on_different_songs_and_accepts_repeat_recordings(
        self,
    ) -> None:
        for second_title, expected_issues in (
            ("Original Song", 0),
            ("Different Song", 2),
        ):
            first = TrackInfo(
                Path("first.flac"),
                artist="Alpha",
                title="Original Song",
                itunes_trackid="123",
                duration=180,
            )
            second = TrackInfo(
                Path("second.flac"),
                artist="Alpha",
                title=second_title,
                itunes_trackid="123",
                duration=180,
            )
            report = CheckReport()
            checker._check_identifier_conflicts(
                {Path("album"): [(first.file_path, first), (second.file_path, second)]},
                report,
            )
            self.assertEqual(len(report.issues), expected_issues)
        report = CheckReport()
        checker._check_identifier_conflicts({}, report)
        self.assertEqual(report.issues, {})

    def test_unverified_lyrics_fallback_requires_opt_in(self) -> None:
        for enabled in (False, True):
            with (
                self.subTest(enabled=enabled),
                patch.object(
                    lyrics,
                    "get_config",
                    return_value=SonoraConfig(allow_unverified_lyrics=enabled),
                ),
                patch.object(lyrics, "_query_lrclib", return_value=None),
                patch.object(lyrics._LYRICS_LIMITER, "wait"),
                patch(
                    "syncedlyrics.search", return_value="[00:01.00] Lyrics"
                ) as search,
            ):
                self.assertEqual(
                    lyrics._query_syncedlyrics(
                        "Alpha - Original Song", False, False, False, None, None
                    ),
                    "[00:01.00] Lyrics" if enabled else None,
                )
            self.assertEqual(search.call_count, int(enabled))

    def test_release_verification_requires_every_track_and_unique_assignment(
        self,
    ) -> None:
        paths = [Path("first.flac"), Path("second.flac")]
        originals = {
            paths[0]: TrackInfo(
                paths[0], artist="Alpha", title="First Song", track_number=1
            ),
            paths[1]: TrackInfo(
                paths[1], artist="Alpha", title="Second Song", track_number=2
            ),
        }
        for second_title, expected in (
            ("Second Song", True),
            ("Different Song", False),
        ):
            release = {
                "tracks_by_disc_and_position": {
                    (1, 1): {"artist": "Alpha", "title": "First Song"},
                    (1, 2): {"artist": "Alpha", "title": second_title},
                }
            }
            with patch.object(
                metadata, "read_track_metadata", side_effect=originals.__getitem__
            ):
                self.assertEqual(
                    tagger._release_matches_album_tracks(release, paths), expected
                )
                self.assertFalse(
                    tagger._release_matches_album_tracks(release, paths[:1])
                )
        self.assertFalse(tagger._release_matches_album_tracks({}, paths))

    def test_cached_acoustic_evidence_is_rechecked_against_current_policy(self) -> None:
        payload = {
            "status": "ok",
            "results": [
                {
                    "score": 0.95,
                    "recordings": [
                        {
                            "id": MBID,
                            "title": "Original Song",
                            "artists": [{"name": "Alpha"}],
                            "duration": 180,
                        }
                    ],
                }
            ],
        }
        for threshold, expected in ((0.8, MBID), (0.99, None)):
            with (
                self.subTest(threshold=threshold),
                patch.object(
                    acoustid,
                    "get_config",
                    return_value=SonoraConfig(acoustid_confidence_threshold=threshold),
                ),
                patch.object(
                    acoustid,
                    "fingerprint_audio_file",
                    return_value=(180, "fingerprint"),
                ),
                patch.object(acoustid, "get_cached_api", return_value=payload),
                patch.object(acoustid, "set_cached_api"),
                patch("acoustid.lookup") as request,
            ):
                self.assertEqual(
                    acoustid.lookup_acoustid(
                        Path("recording.flac"), "fixture", "Alpha", "Original Song"
                    ),
                    expected,
                )
            request.assert_not_called()


class TestProviderConflictHandling(IsolatedTestCase):
    def test_provider_conflict_preserves_disputed_field_and_allows_other_evidence(
        self,
    ) -> None:
        track = TrackInfo(
            Path("recording.flac"), artist="Artist", title="Song", total_tracks=14
        )
        disputed = dataclasses.replace(
            track,
            total_tracks=12,
            metadata_conflicts={"total_tracks": "Release count excludes this track"},
        )
        agreed = dataclasses.replace(
            track, genre="Pop", metadata_sources={"genre": "iTunes"}
        )
        merge_provider_proposals(
            track,
            [
                build_provider_proposal(track, candidate)
                for candidate in [disputed, agreed]
            ],
        )
        self.assertEqual(track.total_tracks, 14)
        self.assertEqual(
            track.metadata_conflicts,
            {"total_tracks": "Release count excludes this track"},
        )
        self.assertEqual(track.genre, "Pop")
        self.assertEqual(track.metadata_sources.get("genre"), "iTunes")

    def test_rejected_recording_cannot_inject_field_conflicts(self) -> None:
        track = TrackInfo(Path("recording.flac"), artist="Artist", title="Song")
        unrelated = dataclasses.replace(
            track,
            title="Entirely Different Recording",
            metadata_conflicts={"genre": "Unrelated conflict"},
        )
        merge_provider_proposals(track, [build_provider_proposal(track, unrelated)])
        self.assertIn("identity", track.metadata_conflicts)
        self.assertNotIn("genre", track.metadata_conflicts)

    def test_existing_conflict_cannot_be_overwritten_by_a_later_provider(self) -> None:
        track = TrackInfo(
            Path("recording.flac"),
            artist="Artist",
            title="Song",
            genre="Jazz",
            metadata_conflicts={"genre": "Album providers disagree"},
        )
        proposal = dataclasses.replace(
            track, genre="Pop", metadata_sources={"genre": "iTunes"}
        )
        merge_provider_proposals(track, [build_provider_proposal(track, proposal)])
        self.assertEqual(track.genre, "Jazz")
        self.assertNotIn("genre", track.metadata_sources)


class TestOfflineArtistNormalization(IsolatedTestCase):
    def test_normalization_preserves_configured_groups_in_artist_and_title(
        self,
    ) -> None:
        policy = SonoraConfig(known_group_artists=frozenset({"Example & Partners"}))
        for artist, title in (
            ("Primary feat. Example & Partners", "Recording"),
            ("Primary", "Recording (feat. Example & Partners)"),
            ("Primary", "Recording feat. Example & Partners"),
        ):
            with (
                self.subTest(artist=artist, title=title),
                patch.object(artists, "get_config", return_value=policy),
            ):
                recording = TrackInfo(Path("song.flac"), artist=artist, title=title)
                normalize_track_metadata(recording)
                self.assertEqual(recording.artist, "Primary")
                self.assertEqual(recording.title, "Recording")
                self.assertEqual(recording.featured_artists, "Example & Partners")

    def test_online_artist_resolution_uses_local_alias_before_providers(self) -> None:
        with (
            patch.object(
                artists, "_load_user_overrides", return_value={"legal name": "Stage"}
            ),
            patch.object(
                artist_resolution,
                "_load_user_overrides",
                return_value={"legal name": "Stage"},
            ),
            patch.object(artist_resolution, "get_cached_api") as cache_lookup,
            patch.object(SESSION, "get") as http_request,
            patch.object(musicbrainz, "search_musicbrainz_artists") as artist_search,
        ):
            self.assertEqual(
                artist_resolution.resolve_artist_name("Legal Name"), "Stage"
            )
            cache_lookup.assert_not_called()
            http_request.assert_not_called()
            artist_search.assert_not_called()

    def test_offline_artist_functions_ignore_external_identity_cache(self) -> None:
        for cached_identity in ("Unrelated Artist", True, False):
            with (
                self.subTest(cached_identity=cached_identity),
                patch.object(
                    artist_resolution, "get_cached_api", return_value=cached_identity
                ) as cache_lookup,
            ):
                self.assertEqual(artists.normalize_artist_name("Example"), "Example")
                self.assertEqual(
                    artists.get_primary_artist("Example & Partners"), "Example"
                )
                self.assertFalse(artists.is_single_group_artist("Example & Partners"))
                cache_lookup.assert_not_called()

    def test_configured_groups_preserve_primary_artist(self) -> None:
        policy = SonoraConfig(known_group_artists=frozenset({"Example & Partners"}))
        with patch.object(artists, "get_config", return_value=policy):
            self.assertEqual(
                artists.get_primary_artist("Example & Partners"), "Example & Partners"
            )
            self.assertEqual(artists.get_primary_artist(None), "Unknown")

    def test_local_aliases_work_without_online_artist_resolution(self) -> None:
        with (
            patch.object(
                artists, "_load_user_overrides", return_value={"legal name": "Stage"}
            ),
            patch.object(
                artist_resolution, "get_cached_api", return_value="Wrong Artist"
            ) as cache_lookup,
        ):
            self.assertEqual(artists.normalize_artist_name("Legal Name"), "Stage")
            self.assertEqual(artists.get_primary_artist("Legal Name"), "Stage")
            self.assertEqual(artists.normalize_artist_name("Other Name"), "Other Name")
            cache_lookup.assert_not_called()

    def test_normalization_does_not_resolve_identity_online(self) -> None:
        recording = TrackInfo(
            Path("song.flac"), artist="Legal Name", album_artist="Legal Name"
        )
        with patch.object(artist_resolution, "resolve_artist_name") as resolver:
            normalize_track_metadata(recording)
        self.assertEqual(recording.artist, "Legal Name")
        self.assertEqual(recording.album_artist, "Legal Name")
        resolver.assert_not_called()


MBID = "11111111-2222-3333-4444-555555555555"


class TestMetadataMappingContracts(IsolatedTestCase):
    def test_provider_placeholders_cannot_become_metadata_or_destroy_a_title(
        self,
    ) -> None:
        from sonora.modules.normalization import normalize_track_metadata

        track = TrackInfo(
            Path("Song.flac"), artist="Primary", title="?", mood="...", style="…"
        )
        normalize_track_metadata(track)
        self.assertEqual(track.title, "?")
        self.assertIsNone(track.mood)
        self.assertIsNone(track.style)
        metadata_merge.apply_metadata_mapping(
            track,
            {"mood": "...", "style": "…", "composer": "Unknown"},
            {"mood": "mood", "style": "style", "composer": "composer"},
            source="TheAudioDB",
        )
        self.assertIsNone(track.mood)
        self.assertIsNone(track.style)
        self.assertIsNone(track.composer)

    def test_mapping_normalizes_valid_metadata_and_records_provenance(self) -> None:
        recording = TrackInfo(Path("recording.flac"), artist="Primary")
        provider_fields = {
            "genre": "rock",
            "date": "2021-02-03",
            "original_date": "2020",
            "release_country": "US",
            "language": "eng",
            "script": "Latn",
            "title": "Song (feat. Guest)",
            "album": "The ’Album’",
            "album_artist": "Primary",
            "advisory": "explicit",
            "initial_key": "Am",
            "featured_artists": "Another Guest",
            "track_number": "2",
            "disc_number": "1",
            "total_tracks": "8",
            "total_discs": "1",
            "label": "Publisher",
            "isrc": "USAAA2100001",
        }
        metadata_merge.apply_metadata_mapping(
            recording,
            provider_fields,
            {name: name for name in provider_fields},
            source="Deezer",
        )
        expected = {
            "genre": "Rock",
            "date": "2021-02-03",
            "original_date": "2020",
            "release_country": "United States",
            "language": "English",
            "script": "Latin",
            "title": "Song",
            "album": "The 'Album'",
            "album_artist": "Primary",
            "advisory": "Explicit",
            "initial_key": "Am",
            "featured_artists": "Guest, Another Guest",
            "track_number": 2,
            "disc_number": 1,
            "total_tracks": 8,
            "total_discs": 1,
            "label": "Publisher",
            "isrc": "USAAA2100001",
        }
        for name, value in expected.items():
            with self.subTest(field=name):
                self.assertEqual(getattr(recording, name), value)
        self.assertEqual(recording.metadata_sources["label"], "Deezer")
        for name in ("track_number", "disc_number", "total_tracks", "total_discs"):
            self.assertEqual(recording.metadata_sources[name], "Deezer")

    def test_invalid_and_absent_provider_values_preserve_existing_tags(self) -> None:
        cases: tuple[tuple[str, object], ...] = (
            ("label", None),
            ("label", " "),
            ("label", "null"),
            ("label", "none"),
            ("genre", "unknown"),
            ("date", "not a date"),
            ("advisory", "clean"),
            ("initial_key", "not a key"),
            ("track_number", "unknown"),
            ("featured_artists", "Primary"),
        )
        for field, value in cases:
            recording = TrackInfo(Path("recording.flac"), artist="Primary")
            before = recording.copy()
            with self.subTest(field=field, value=value):
                metadata_merge.apply_metadata_mapping(
                    recording, {"value": value}, {"value": field}, force=True
                )
                self.assertEqual(recording, before)

    def test_force_preserves_richer_credits_and_equivalent_isrc(self) -> None:
        recording = TrackInfo(
            Path("recording.flac"),
            artist="Primary",
            label="Publisher Records",
            isrc="USAAA2100001",
            track_number=3,
            disc_number=1,
        )
        metadata_merge.apply_metadata_mapping(
            recording,
            {
                "label": "Publisher",
                "isrc": "usaaa2100001",
                "track_number": 2,
                "disc_number": 2,
            },
            {name: name for name in ("label", "isrc", "track_number", "disc_number")},
            force=True,
            source="Deezer",
        )
        self.assertEqual(recording.label, "Publisher Records")
        self.assertEqual(recording.isrc, "USAAA2100001")
        self.assertEqual((recording.track_number, recording.disc_number), (2, 2))

    def test_existing_tags_respect_force_and_source_precedence(self) -> None:
        recording = TrackInfo(
            Path("recording.flac"), label="Existing", track_number=3, disc_number=1
        )
        metadata_merge.apply_metadata_mapping(
            recording,
            {"label": "Replacement", "track_number": 2, "disc_number": 1},
            {name: name for name in ("label", "track_number", "disc_number")},
            source="Deezer",
        )
        self.assertEqual((recording.label, recording.track_number), ("Existing", 3))
        metadata_merge.apply_metadata_mapping(
            recording, {"label": "Existing"}, {"label": "label"}, source="Deezer"
        )
        self.assertEqual(recording.metadata_sources["label"], "Deezer")
        metadata_merge.apply_metadata_mapping(
            recording, {"label": "Existing"}, {"label": "label"}, source="Discogs"
        )
        self.assertEqual(recording.metadata_sources["label"], "Deezer")
        metadata_merge.apply_metadata_mapping(
            recording, {"label": "Different"}, {"label": "label"}, force=True
        )
        self.assertEqual(recording.label, "Different")

    def test_deezer_localized_genre_aliases_normalize(self) -> None:
        """Verify localized Deezer genre aliases map to canonical English genre names."""
        from sonora.core.text import normalize_genre

        expected_mappings = {
            "dans": "Dance",
            "Dans": "Dance",
            "DANS": "Dance",
            "latino": "Latin",
            "Latino": "Latin",
            "LATINO": "Latin",
            "muzică asiatică": "Asian Music",
            "muzica asiatica": "Asian Music",
            "Muzică Asiatică": "Asian Music",
            "muzică electronică": "Electronic",
            "muzica electronica": "Electronic",
            "Muzică Electronică": "Electronic",
        }
        for raw_genre, expected in expected_mappings.items():
            with self.subTest(raw_genre=raw_genre, expected=expected):
                self.assertEqual(normalize_genre(raw_genre), expected)

    def test_deezer_metadata_mapping_with_localized_genres(self) -> None:
        """Deezer provider proposals with localized genres normalize properly onto TrackInfo."""
        genres_to_test = (
            ("dans", "Dance"),
            ("latino", "Latin"),
            ("muzică asiatică", "Asian Music"),
            ("muzică electronică", "Electronic"),
        )
        for raw_genre, expected_genre in genres_to_test:
            with self.subTest(raw_genre=raw_genre):
                track = TrackInfo(Path("song.flac"), artist="Artist", title="Song")
                metadata_merge.apply_metadata_mapping(
                    track,
                    {"genre": raw_genre},
                    {"genre": "genre"},
                    source="Deezer",
                    force=True,
                )
                self.assertEqual(track.genre, expected_genre)
                self.assertEqual(track.metadata_sources["genre"], "Deezer")


class TestCatalogMetadataContracts(IsolatedTestCase):
    def test_shazam_cannot_invent_an_album_when_recognition_has_none(self) -> None:
        recording = TrackInfo(Path("recording.flac"))
        with (
            patch.object(
                catalog, "get_config", return_value=SonoraConfig(enable_shazam=True)
            ),
            patch.object(
                shazam_service,
                "recognize_audio_track",
                return_value=ShazamTrackInfo("Recording", "Primary"),
            ),
        ):
            catalog._update_shazam(
                recording, recording.file_path, has_album_context=True
            )
        self.assertEqual(
            (recording.title, recording.artist, recording.album),
            ("Recording", "Primary", "Unknown Album"),
        )

    def test_unmatched_discogs_track_credits_preserve_the_recording(self) -> None:
        recording = self.recording.copy()
        catalog._update_discogs(
            recording,
            "fixture",
            {"id": 1, "track_credits": {"other": {"producers": "Wrong"}}},
        )
        self.assertIsNone(recording.discogs_release_id)
        self.assertIsNone(recording.producers)

    def test_deezer_base_album_cannot_supply_deluxe_release_selectors(self) -> None:
        track = TrackInfo(
            Path("Song.flac"), artist="Primary", title="Song", album="Album (Deluxe)"
        )
        with patch.object(
            deezer_service, "fetch_deezer_track_details", return_value=None
        ):
            catalog._update_deezer(
                track,
                {
                    "title": "Album",
                    "nb_tracks": 18,
                    "barcode": "wrong-base-upc",
                    "release_date": "2016-11-25",
                    "label": "Base Edition Label",
                },
                has_album_context=True,
            )
        self.assertIsNone(track.barcode)
        self.assertIsNone(track.date)
        self.assertIsNone(track.label)
        self.assertEqual(track.album, "Album (Deluxe)")

    def test_deezer_preserves_authoritative_dates_and_consistent_album_totals(
        self,
    ) -> None:
        album = {
            "title": "Record",
            "nb_tracks": 5,
            "release_date": "2021",
            "tracks_by_disc_and_position": {(1, 3): {"title": "Song"}},
        }
        recording = dataclasses.replace(
            self.recording,
            track_number=3,
            total_tracks=10,
            date="2020",
            musicbrainz_albumid=MBID,
        )
        with patch.object(
            deezer_service, "fetch_deezer_track_details", return_value=None
        ):
            catalog._update_deezer(recording, album, has_album_context=True)
            self.assertEqual(
                (recording.album, recording.total_tracks, recording.date),
                ("Record", 10, "2020"),
            )
            recording.musicbrainz_albumid = None
            catalog._update_deezer(recording, album, has_album_context=True)
            catalog._update_deezer(recording, album, has_album_context=True, force=True)
        self.assertEqual(recording.total_tracks, 10)

    def test_deezer_preserves_conflicting_isrc_even_for_exact_duration_match(
        self,
    ) -> None:
        recording = TrackInfo(
            Path("i lied - sped up.flac"),
            artist="Cheyanne & Ex Habit",
            title="i lied - sped up",
            album="i lied",
            isrc="UKZGC2506039",
            duration=147.698,
            producers="Verified producer",
            musicbrainz_trackid=MBID,
        )
        conflicting = {
            "artist": "Cheyanne & Ex Habit",
            "title": "i lied",
            "duration": 165,
            "isrc": "UKZGC2506039",
        }
        exact_edit = {
            "artist": "Cheyanne & Ex Habit",
            "title": "i lied - sped up",
            "duration": 148,
            "isrc": "UKZGC2506040",
        }
        with (
            patch.object(
                deezer_service, "fetch_deezer_track_by_isrc", return_value=conflicting
            ),
            patch.object(
                deezer_service, "fetch_deezer_track_details", return_value=exact_edit
            ) as lookup,
        ):
            catalog._update_deezer(recording, None, has_album_context=True)
        lookup.assert_called_once_with(
            "Cheyanne & Ex Habit", "i lied - sped up", duration=147.698
        )
        self.assertEqual(recording.isrc, "UKZGC2506039")
        self.assertNotIn("isrc", recording.metadata_sources)
        self.assertIn("isrc", recording.metadata_conflicts)

    def test_deezer_keeps_conflicting_isrc_when_search_result_is_not_exact(
        self,
    ) -> None:
        recording = TrackInfo(
            Path("i lied - sped up.flac"),
            artist="Cheyanne & Ex Habit",
            title="i lied - sped up",
            isrc="UKZGC2506039",
            duration=147.698,
            producers="Verified producer",
        )
        with (
            patch.object(
                deezer_service,
                "fetch_deezer_track_by_isrc",
                return_value={
                    "artist": "Cheyanne & Ex Habit",
                    "title": "i lied",
                    "duration": 165,
                    "isrc": "UKZGC2506039",
                },
            ),
            patch.object(
                deezer_service,
                "fetch_deezer_track_details",
                return_value={
                    "artist": "Different Artist",
                    "title": "Other Song",
                    "duration": 148,
                    "isrc": "UKZGC2506040",
                },
            ),
        ):
            catalog._update_deezer(recording, None, has_album_context=True)
        self.assertEqual(recording.isrc, "UKZGC2506039")
        self.assertIn("isrc", recording.metadata_conflicts)

    def test_clean_title_index_matches_without_a_position(self) -> None:
        recording = {"title": "Recording"}
        self.assertEqual(
            matching._find_track_in_album_mapping(
                {"recording": recording}, None, "Recording (feat. Guest)", None
            ),
            (recording, True),
        )

    def test_album_lookup_supports_disc_positions_and_raw_title_indexes(self) -> None:
        recording = {"title": "Recording"}
        for title_index, position_index, disc_index, title, expected in (
            ({}, {}, {(2, 3): recording}, "Recording", recording),
            ({}, {(2, 3): recording}, None, "Recording", recording),
            (
                {"recording feat guest": recording},
                {},
                None,
                "Recording (feat. Guest)",
                recording,
            ),
            ({}, {}, None, "Untitled", None),
        ):
            with self.subTest(
                title_index=title_index,
                position_index=position_index,
                disc_index=disc_index,
            ):
                matched, from_album = matching._find_track_in_album_mapping(
                    title_index, disc_index or position_index, title, 3, 2
                )
                self.assertEqual(matched, expected)
                self.assertEqual(from_album, expected is not None)

    def test_album_shazam_without_an_album_keeps_known_identity_and_credits(
        self,
    ) -> None:
        recognized = ShazamTrackInfo("Recognized", "Primary")
        for has_context, title, artist, expected_title in (
            (True, "Song", "Primary", "Song"),
            (True, "", "", "Recognized"),
            (False, "Song", "Primary", "Song"),
        ):
            recording = TrackInfo(
                Path("recording.flac"),
                title=title,
                artist=artist,
                album="Record",
                genre="Rock",
                label="Publisher",
                lyrics="Verified",
                isrc="USAAA2100001",
                itunes_trackid="123",
                date="2020",
            )
            with (
                self.subTest(has_context=has_context, title=title),
                patch.object(
                    catalog, "get_config", return_value=SonoraConfig(enable_shazam=True)
                ),
                patch.object(
                    shazam_service, "recognize_audio_track", return_value=recognized
                ),
            ):
                catalog._update_shazam(
                    recording,
                    recording.file_path,
                    has_album_context=has_context,
                    force=True,
                )
            self.assertEqual(
                (recording.title, recording.artist, recording.album),
                (expected_title, "Primary", "Record"),
            )
            self.assertEqual(
                (recording.genre, recording.label, recording.lyrics),
                ("Rock", "Publisher", "Verified"),
            )

    def test_redundant_lastfm_styles_do_not_create_empty_style_tags(self) -> None:
        recording = dataclasses.replace(self.recording, mood="Verified")
        with patch.object(
            lastfm_service, "fetch_lastfm_tags", return_value=["rock", "Rock"]
        ):
            catalog._update_lastfm(recording, "fixture")
        self.assertEqual(recording.genre, "Rock")
        self.assertIsNone(recording.style)
        self.assertEqual(recording.mood, "Verified")

    @patch.object(itunes_service, "fetch_itunes_track_metadata", return_value=None)
    @patch.object(deezer_service, "fetch_deezer_track_details", return_value=None)
    def test_album_structure_preserves_existing_totals_and_authoritative_dates(
        self, _deezer_lookup: MagicMock, _itunes_lookup: MagicMock
    ) -> None:
        recording = dataclasses.replace(
            self.recording,
            track_number=3,
            disc_number=2,
            total_discs=2,
            total_tracks=10,
            date="2020",
        )
        catalog._update_itunes(
            recording,
            {
                "tracks_by_disc_and_position": {
                    (1, 3): {
                        "title": "Song",
                        "date": "2021",
                        "disc_number": 1,
                        "total_discs": 1,
                    }
                }
            },
        )
        self.assertEqual(
            (recording.disc_number, recording.total_discs, recording.date),
            (2, 2, "2020"),
        )
        catalog._update_deezer(
            recording,
            {
                "title": "Record",
                "nb_tracks": "bad",
                "tracks_by_disc_and_position": {(1, 3): {"title": "Song"}},
            },
            has_album_context=True,
            force=True,
        )
        self.assertEqual(recording.total_tracks, 10)

    def test_known_genius_dates_and_lyricists_are_not_replaced(self) -> None:
        recording = dataclasses.replace(
            self.recording, date="2020", lyricist="Verified"
        )
        with patch.object(
            genius_service,
            "fetch_genius_song_details",
            return_value={"writers": "Writer", "release_date": "2021"},
        ):
            catalog._update_genius(recording, "fixture")
        self.assertEqual((recording.date, recording.lyricist), ("2020", "Verified"))

    def test_theaudiodb_existing_rating_and_genre_survive_metadata_updates(
        self,
    ) -> None:
        recording = dataclasses.replace(self.recording, genre="Rock", rating=8.5)
        with patch.object(
            theaudiodb_service,
            "fetch_theaudiodb_track_details",
            return_value={"rating": 1, "genre": "Pop", "mood": "Chill"},
        ):
            catalog._update_theaudiodb(recording)
        self.assertEqual(
            (recording.genre, recording.rating, recording.mood), ("Rock", 8.5, "Chill")
        )

    def setUp(self) -> None:
        self.enterContext(
            patch(
                "sonora.services.artist_resolution.resolve_artist_name",
                side_effect=lambda artist: artist,
            )
        )
        self.recording = TrackInfo(
            Path("recording.flac"),
            artist="Primary",
            title="Song",
            album="Record",
            disc_number=None,
        )

    def test_shazam_recognition_fills_missing_metadata(self) -> None:
        recognized = ShazamTrackInfo(
            "Song",
            "Primary",
            "Album",
            "Rock",
            "123",
            isrc="USAAA2100001",
            release_date="2021",
            label="Publisher",
            lyrics="[00:01] Song",
        )
        recording = TrackInfo(Path("recording.flac"))
        with (
            patch.object(
                catalog, "get_config", return_value=SonoraConfig(enable_shazam=True)
            ),
            patch.object(
                shazam_service, "recognize_audio_track", return_value=recognized
            ),
        ):
            catalog._update_shazam(recording, recording.file_path)
        self.assertEqual(
            (recording.title, recording.artist, recording.album),
            ("Song", "Primary", "Album"),
        )
        self.assertEqual(
            (recording.genre, recording.itunes_trackid, recording.isrc),
            ("Rock", "123", "USAAA2100001"),
        )
        self.assertEqual(
            (recording.date, recording.label, recording.lyrics),
            ("2021", "Publisher", "[00:01] Song"),
        )

    def test_shazam_album_context_and_sample_hijacking_protection(self) -> None:
        cases = (
            (
                TrackInfo(Path("missing.flac")),
                None,
                None,
                "Recognized",
                "Primary",
                "Found Album",
            ),
            (
                TrackInfo(Path("missing.flac")),
                "Album Artist",
                "Album Title",
                "Recognized",
                "Album Artist",
                "Album Title",
            ),
            (
                TrackInfo(
                    Path("existing.flac"),
                    artist="Primary",
                    title="Song",
                    album="Record",
                    disc_number=None,
                ),
                "Primary",
                "Record",
                "Song",
                "Primary",
                "Record",
            ),
        )
        with (
            patch.object(
                catalog, "get_config", return_value=SonoraConfig(enable_shazam=True)
            ),
            patch.object(
                shazam_service,
                "recognize_audio_track",
                return_value=ShazamTrackInfo("Recognized", "Primary", "Found Album"),
            ),
        ):
            for (
                recording,
                artist,
                album,
                title,
                expected_artist,
                expected_album,
            ) in cases:
                with self.subTest(path=recording.file_path, album=album):
                    catalog._update_shazam(
                        recording,
                        recording.file_path,
                        artist,
                        album,
                        has_album_context=True,
                        force=True,
                    )
                    self.assertEqual(
                        (recording.title, recording.artist, recording.album),
                        (title, expected_artist, expected_album),
                    )
            with patch.object(
                shazam_service,
                "recognize_audio_track",
                return_value=ShazamTrackInfo("Wrong", "Unrelated Artist"),
            ):
                catalog._update_shazam(
                    self.recording, self.recording.file_path, "Primary", force=True
                )
                self.assertEqual(
                    (self.recording.artist, self.recording.title), ("Primary", "Song")
                )

    def test_shazam_skips_complete_metadata_and_empty_recognition(self) -> None:
        with (
            patch.object(
                catalog, "get_config", return_value=SonoraConfig(enable_shazam=True)
            ),
            patch.object(
                shazam_service, "recognize_audio_track", return_value=None
            ) as recognize,
        ):
            catalog._update_shazam(self.recording, self.recording.file_path)
            recognize.assert_not_called()
            catalog._update_shazam(self.recording, self.recording.file_path, force=True)
            recognize.assert_called_once()

    def test_itunes_album_structure_and_musicbrainz_date_precedence(self) -> None:
        track = {
            "genre": "Rock",
            "advisory": "Explicit",
            "date": "2020",
            "disc_number": 2,
            "total_discs": 2,
            "title": "Found",
            "artist": "Primary",
            "itunes_trackid": "123",
        }
        album = {"tracks_by_disc_and_position": {(1, 1): track}}
        self.recording.title = "Untitled"
        self.recording.itunes_trackid = "123"
        self.recording.track_number = 1
        with patch.object(itunes_service, "fetch_itunes_track_metadata") as lookup:
            catalog._update_itunes(self.recording, album, force=True)
        lookup.assert_not_called()
        self.assertEqual(
            (
                self.recording.title,
                self.recording.disc_number,
                self.recording.total_discs,
                self.recording.date,
            ),
            ("Found", 2, 2, "2020"),
        )
        self.recording.musicbrainz_albumid = MBID
        self.recording.date = "2019"
        with (
            patch.object(itunes_service, "lookup_itunes_track", return_value=None),
            patch.object(
                itunes_service, "fetch_itunes_track_metadata", return_value=None
            ),
        ):
            catalog._update_itunes(self.recording, album, force=True)
        self.assertEqual(self.recording.date, "2019")

    def test_itunes_single_fallback_does_not_copy_album_positions(self) -> None:
        with patch.object(
            itunes_service,
            "fetch_itunes_track_metadata",
            return_value={
                "genre": "Rock",
                "disc_number": 4,
                "total_discs": 4,
                "date": "2020",
                "artist": "Primary",
                "title": "Song",
            },
        ):
            catalog._update_itunes(self.recording)
        self.assertEqual(self.recording.genre, "Rock")
        self.assertIsNone(self.recording.disc_number)
        self.assertIsNone(self.recording.total_discs)

    def test_lastfm_genre_styles_and_mood(self) -> None:
        for tags, expected in (
            ([], (None, None, None)),
            (["rock"], ("Rock", None, None)),
            (["unknown", "rock", "chill"], (None, "rock, chill", "Chill")),
            (["rock", "Rock", "chill", "dark"], ("Rock", "chill, dark", "Chill")),
        ):
            recording = self.recording.copy()
            with (
                self.subTest(tags=tags),
                patch.object(lastfm_service, "fetch_lastfm_tags", return_value=tags),
            ):
                catalog._update_lastfm(recording, "fixture")
            self.assertEqual(
                (recording.genre, recording.style, recording.mood), expected
            )

    def test_discogs_release_and_track_credits(self) -> None:
        release: dict[str, object] = {
            "id": 123,
            "styles": ["Alternative", "Indie"],
            "genres": ["rock"],
            "year": 2020,
            "track_credits": {"2": {"title": "Found Song", "producers": "Producer"}},
            "country": "US",
        }
        self.recording.title = "Found Song"
        self.recording.track_number = 2
        self.recording.discogs_release_id = "123"
        catalog._update_discogs(
            self.recording, "fixture", release, has_album_context=True
        )
        self.assertEqual(
            (
                self.recording.discogs_release_id,
                self.recording.style,
                self.recording.genre,
                self.recording.date,
            ),
            ("123", "Alternative, Indie", "Rock", "2020"),
        )
        self.assertEqual(
            (self.recording.title, self.recording.producers), ("Found Song", "Producer")
        )
        self.recording.musicbrainz_albumid = MBID
        release["year"] = 2022
        release["track_credits"] = {
            "found song": {"title": "Found Song", "composer": "Composer"}
        }
        catalog._update_discogs(
            self.recording, "fixture", release, has_album_context=True, force=True
        )
        self.assertEqual(self.recording.date, "2020")
        self.assertEqual(self.recording.composer, "Composer")

    def test_discogs_album_match_cannot_invent_pressing_metadata(self) -> None:
        release = {
            "id": 123,
            "barcode": "001234567890",
            "catalog_number": "OTHER-1",
            "country": "JP",
            "media": "CD",
            "year": 2016,
            "label": "Other Label",
            "styles": ["House"],
        }
        for force in (False, True):
            for identifiers in (
                {},
                {"barcode": "999999999999"},
                {"discogs_release_id": "123", "barcode": "999999999999"},
            ):
                with self.subTest(force=force, identifiers=identifiers):
                    track = TrackInfo(
                        Path("Song.flac"),
                        artist="Primary",
                        album="Album",
                        barcode=identifiers.get("barcode"),
                        discogs_release_id=identifiers.get("discogs_release_id"),
                    )
                    catalog._update_discogs(track, "fixture", release, force=force)
                    self.assertEqual(track.style, "House")
                    self.assertIsNone(track.catalog_number)
                    self.assertIsNone(track.media)
                    self.assertIsNone(track.date)
                    self.assertIsNone(track.label)
                    self.assertEqual(track.barcode, identifiers.get("barcode"))
                    self.assertEqual(
                        track.discogs_release_id, identifiers.get("discogs_release_id")
                    )
                    self.assertEqual(
                        "discogs_release_id" in track.metadata_conflicts,
                        bool(identifiers),
                    )

    def test_discogs_matching_identifier_does_not_override_source_medium(self) -> None:
        track = TrackInfo(
            Path("Song.flac"),
            artist="Primary",
            album="Album",
            barcode="1234567890",
            media="Digital Media",
        )
        catalog._update_discogs(
            track,
            "fixture",
            {
                "id": 123,
                "barcode": "1234567890",
                "catalog_number": "CD-1",
                "media": "CD, Album",
            },
            force=True,
        )
        self.assertIsNone(track.discogs_release_id)
        self.assertIsNone(track.catalog_number)
        self.assertEqual(track.media, "Digital Media")
        self.assertIn("discogs_release_id", track.metadata_conflicts)

    def test_discogs_matching_barcode_allows_edition_metadata(self) -> None:
        track = TrackInfo(
            Path("Song.flac"), artist="Primary", album="Album", barcode="001234567890"
        )
        catalog._update_discogs(
            track,
            "fixture",
            {
                "id": 123,
                "barcode": "1234567890",
                "catalog_number": "KNOWN-1",
                "media": "CD",
            },
        )
        self.assertEqual(track.discogs_release_id, "123")
        self.assertEqual(track.catalog_number, "KNOWN-1")
        self.assertEqual(track.media, "CD")
        self.assertNotIn("discogs_release_id", track.metadata_conflicts)

    def test_discogs_does_not_search_other_albums_in_album_context(self) -> None:
        with patch.object(
            discogs_service, "search_discogs_release", return_value=None
        ) as search:
            catalog._update_discogs(
                self.recording, "fixture", None, has_album_context=True
            )
            search.assert_not_called()
            catalog._update_discogs(self.recording, "fixture", None)
            search.assert_called_once()

    def test_deezer_album_metadata_and_structure(self) -> None:
        recording = TrackInfo(
            Path("recording.flac"),
            artist="Primary",
            title="Untitled",
            isrc="USAAA2100001",
            track_number=1,
            disc_number=2,
        )
        track = {
            "title": "Found Song",
            "artist": "Primary",
            "track_position": 1,
            "disk_number": 2,
            "explicit_lyrics": True,
            "bpm": 100.15,
            "isrc": "USAAA2100001",
            "producers": "Producer",
        }
        album = {
            "title": "Found Album",
            "nb_tracks": 9,
            "genre": "Rock",
            "release_date": "2021",
            "tracks_by_disc_and_position": {
                (1, 1): {},
                **{(2, position): track for position in range(1, 9)},
            },
        }
        catalog._update_deezer(recording, album, has_album_context=True, force=True)
        self.assertEqual(
            (
                recording.album,
                recording.title,
                recording.total_tracks,
                recording.disc_number,
            ),
            ("Found Album", "Found Song", 8, 2),
        )
        self.assertEqual(
            (recording.advisory, recording.bpm, recording.isrc),
            ("Explicit", 100.2, "USAAA2100001"),
        )
        self.assertEqual(
            (recording.genre, recording.date, recording.producers),
            ("Rock", "2021", "Producer"),
        )

    def test_deezer_identity_lookup_and_context_isolation(self) -> None:
        self.recording.isrc = "USAAA2100001"
        for has_context in (False, True):
            recording = self.recording.copy()
            with (
                self.subTest(has_context=has_context),
                patch.object(
                    deezer_service, "fetch_deezer_album_details", return_value=None
                ),
                patch.object(
                    deezer_service,
                    "fetch_deezer_track_by_isrc",
                    return_value={
                        "artist": recording.artist,
                        "title": recording.title,
                        "isrc": recording.isrc,
                        "track_position": 7,
                        "disk_number": 3,
                    },
                ) as lookup,
            ):
                catalog._update_deezer(recording, None, has_album_context=has_context)
            lookup.assert_called_once_with(recording.isrc)
            self.assertIsNone(recording.track_number)
            self.assertIsNone(recording.disc_number)

    def test_deezer_single_fallback_preserves_authoritative_credits(self) -> None:
        self.recording.musicbrainz_trackid = MBID
        self.recording.producers = "Verified Producer"
        with (
            patch.object(
                deezer_service, "fetch_deezer_album_details", return_value=None
            ),
            patch.object(
                deezer_service,
                "fetch_deezer_track_details",
                return_value={
                    "artist": self.recording.artist,
                    "title": self.recording.title,
                    "producers": "Wrong Producer",
                    "release_date": "2021",
                    "bpm": "invalid",
                },
            ),
        ):
            catalog._update_deezer(self.recording, None, force=True)
        self.assertEqual(self.recording.producers, "Verified Producer")
        self.assertEqual(self.recording.date, "2021")
        self.assertIsNone(self.recording.bpm)

    def test_genius_credits_and_missing_rating(self) -> None:
        with patch.object(
            genius_service,
            "fetch_genius_song_details",
            return_value={
                "writers": "Writer",
                "producers": "Producer",
                "release_date": "2020",
            },
        ):
            catalog._update_genius(self.recording, "fixture")
        self.assertEqual(
            (
                self.recording.composer,
                self.recording.lyricist,
                self.recording.producers,
            ),
            ("Writer", "Writer", "Producer"),
        )
        with patch.object(
            theaudiodb_service,
            "fetch_theaudiodb_track_details",
            return_value={
                "rating": "8.5",
                "initial_key": "Am",
                "music_video_url": "https://example.com",
                "mood": "Chill",
                "genre": "rock",
                "description": "Description",
            },
        ):
            catalog._update_theaudiodb(self.recording)
        self.assertEqual(
            (self.recording.rating, self.recording.initial_key, self.recording.genre),
            (8.5, "Am", "Rock"),
        )

    def test_optional_services_skip_unnecessary_requests_and_empty_matches(
        self,
    ) -> None:
        self.recording.composer = "Composer"
        self.recording.producers = "Producer"
        with patch.object(
            genius_service, "fetch_genius_song_details", return_value=None
        ) as genius:
            catalog._update_genius(self.recording, "fixture")
            genius.assert_called_once()
        self.recording.music_video_url = "https://example.com"
        self.recording.mood = "Chill"
        self.recording.initial_key = "Am"
        self.recording.rating = 8.5
        self.recording.comment = "Description"
        with patch.object(
            theaudiodb_service, "fetch_theaudiodb_track_details", return_value=None
        ) as lookup:
            catalog._update_theaudiodb(self.recording)
            lookup.assert_not_called()
            catalog._update_theaudiodb(self.recording, force=True)
            lookup.assert_called_once()
        self.recording.rating = None
        with patch.object(
            theaudiodb_service,
            "fetch_theaudiodb_track_details",
            return_value={"rating": "invalid"},
        ):
            catalog._update_theaudiodb(self.recording)
        self.assertIsNone(self.recording.rating)

    def test_provider_network_errors_leave_tags_unchanged(self) -> None:
        providers: tuple[tuple[ModuleType, str, Callable[[TrackInfo], None]], ...] = (
            (itunes_service, "fetch_itunes_track_metadata", catalog._update_itunes),
            (
                lastfm_service,
                "fetch_lastfm_tags",
                partial(catalog._update_lastfm, lastfm_api_key="fixture"),
            ),
            (
                discogs_service,
                "search_discogs_release",
                partial(
                    catalog._update_discogs,
                    discogs_user_token="fixture",
                    album_discogs_release=None,
                ),
            ),
            (
                deezer_service,
                "fetch_deezer_album_details",
                partial(catalog._update_deezer, album_deezer_details=None),
            ),
            (
                genius_service,
                "fetch_genius_song_details",
                partial(catalog._update_genius, genius_api_token="fixture"),
            ),
            (
                theaudiodb_service,
                "fetch_theaudiodb_track_details",
                catalog._update_theaudiodb,
            ),
        )
        for service, name, update in providers:
            recording = self.recording.copy()
            with (
                self.subTest(provider=name),
                patch.object(
                    service, name, side_effect=httpx.ConnectTimeout("offline")
                ),
            ):
                update(recording)
            self.assertEqual(recording, self.recording)
        with (
            patch.object(
                catalog, "get_config", return_value=SonoraConfig(enable_shazam=True)
            ),
            patch.object(
                shazam_service,
                "recognize_audio_track",
                side_effect=OSError("decode failed"),
            ),
        ):
            catalog._update_shazam(self.recording, self.recording.file_path, force=True)
        self.assertEqual(self.recording.title, "Song")


class TestCatalogReview(ServiceResponseTestCase):
    def test_repeated_track_titles_do_not_select_an_arbitrary_edition(self) -> None:
        response = MagicMock(status_code=200)
        for titles in (
            ("Recording", "Recording"),
            ("Recording", "Second"),
            ("Recording", "Recording", "Recording"),
        ):
            with self.subTest(titles=titles):
                response.json.return_value = {
                    "results": [
                        {
                            "wrapperType": "track",
                            "trackNumber": position,
                            "trackName": title,
                            "trackId": position,
                        }
                        for position, title in enumerate(titles, 1)
                    ]
                }
                with (
                    patch.object(
                        itunes,
                        "search_itunes",
                        return_value=[
                            {
                                "collectionId": 1,
                                "collectionName": "Album",
                                "artistName": "Primary",
                            }
                        ],
                    ),
                    patch.object(SESSION, "get", return_value=response),
                ):
                    album = itunes.fetch_itunes_album_details("Primary", "Album")
                assert album is not None
                position_index = album["tracks_by_disc_and_position"]
                title_index = album["tracks_by_title"]
                assert isinstance(position_index, dict)
                assert isinstance(title_index, dict)
                self.assertEqual(len(position_index), len(titles))
                self.assertEqual(
                    "recording" in title_index, titles.count("Recording") == 1
                )
                search_response = MagicMock(status_code=200)
                search_response.json.return_value = {
                    "data": [{"id": 1, "title": "Album", "artist": {"name": "Primary"}}]
                }
                response.json.return_value = {
                    "tracks": {
                        "data": [
                            {"id": position, "track_position": position, "title": title}
                            for position, title in enumerate(titles, 1)
                        ]
                    }
                }
                with patch.object(
                    SESSION, "get", side_effect=[search_response, response]
                ):
                    album = deezer.fetch_deezer_album_details("Primary", "Album")
                assert album is not None
                title_index = album["tracks_by_title"]
                assert isinstance(title_index, dict)
                self.assertEqual(
                    "recording" in title_index, titles.count("Recording") == 1
                )
                response.json.return_value = {
                    "id": 1,
                    "tracklist": [
                        {"position": str(position), "title": title}
                        for position, title in enumerate(titles, 1)
                    ],
                }
                with patch.object(SESSION, "get", return_value=response):
                    release = discogs.fetch_discogs_release_details(1)
                assert release is not None
                self.assertEqual(
                    "recording" in release["track_credits"],
                    titles.count("Recording") == 1,
                )
                self.assertIn("1", release["track_credits"])

    def test_artist_artwork_requires_unambiguous_artist_identity(self) -> None:
        artist_mbid = "11111111-2222-3333-4444-555555555555"
        with patch.object(SESSION, "get") as request:
            for artist_name in ("", " "):
                self.assertEqual(
                    theaudiodb.fetch_artist_images(artist_name), (None, None)
                )
        request.assert_not_called()
        for artist_name, candidate_mbid, expected in (
            ("Primary", artist_mbid, (b"portrait", None)),
            ("Unrelated", artist_mbid, (None, None)),
            ("Primary", "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", (None, None)),
        ):
            response = MagicMock(status_code=200)
            response.json.return_value = {
                "artists": [
                    {
                        "strArtist": artist_name,
                        "strMusicBrainzID": candidate_mbid,
                        "strArtistThumb": "https://example.com/avatar.jpg",
                    }
                ]
            }
            with (
                self.subTest(artist_name=artist_name, candidate_mbid=candidate_mbid),
                patch.object(SESSION, "get", return_value=response) as request,
                patch.object(
                    theaudiodb, "_download_artwork_bytes", return_value=b"portrait"
                ),
            ):
                self.assertEqual(
                    theaudiodb.fetch_artist_images("Primary", artist_mbid=artist_mbid),
                    expected,
                )
            self.assertIn("artist-mb.php", request.call_args.args[0])
        response.json.return_value = {
            "artists": [{"strArtist": "Primary"}, {"strArtist": "Primary"}]
        }
        with (
            patch.object(SESSION, "get", return_value=response),
            patch.object(theaudiodb, "_download_artwork_bytes") as download,
        ):
            self.assertEqual(theaudiodb.fetch_artist_images("Primary"), (None, None))
        download.assert_not_called()

    def test_discogs_audio_count_checks_nested_tracks_and_rejects_wrong_editions(
        self,
    ) -> None:
        tracklist: list[dict[str, object]] = [
            {"type_": "heading", "title": "Side One"},
            {
                "type_": "index",
                "title": "Suite",
                "sub_tracks": [
                    {"type_": "track", "position": "1", "title": "First"},
                    {"type_": "track", "position": "2", "title": "Second"},
                ],
            },
            {"type_": "track", "position": "Video 1", "title": "Music Video"},
        ]
        self.assertEqual(discogs._count_discogs_audio_tracks(tracklist), 2)
        self.assertEqual(discogs._count_discogs_audio_tracks([]), 0)
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "results": [{"id": 1, "title": "Primary - Recording", "format": ["File"]}]
        }
        for expected_count, details, accepted in (
            (2, {"audio_track_count": 2}, True),
            (3, {"audio_track_count": 2}, False),
            (2, None, False),
        ):
            with (
                self.subTest(expected_count=expected_count, details=details),
                patch.object(SESSION, "get", return_value=response),
                patch.object(
                    discogs, "fetch_discogs_release_details", return_value=details
                ),
            ):
                self.assertEqual(
                    discogs.search_discogs_release(
                        "Primary", "Recording", "token", expected_count
                    )
                    is not None,
                    accepted,
                )

    def test_cover_upgrade_never_reduces_either_dimension(self) -> None:
        cover = self.test_directory / "cover.jpg"
        original = image_bytes((1000, 1000))
        for replacement_dimensions, accepted in (
            ((1200, 1200), True),
            ((2000, 600), False),
            ((600, 2000), False),
        ):
            cover.write_bytes(original)
            replacement = image_bytes(replacement_dimensions)
            with (
                self.subTest(dimensions=replacement_dimensions),
                patch.object(
                    art,
                    "_fetch_candidate_artwork",
                    return_value=(replacement, replacement_dimensions),
                ),
            ):
                art.process_album_cover_art(
                    self.test_directory, "Primary", "Recording", force=True
                )
            self.assertEqual(cover.read_bytes(), replacement if accepted else original)

    def test_equal_album_candidates_abstain_across_catalogs(self) -> None:
        for candidate_ids in ((1, 2), (2, 1)):
            with self.subTest(candidate_ids=candidate_ids):
                albums = [
                    {
                        "collectionId": identifier,
                        "collectionName": "Recording",
                        "artistName": "Primary",
                    }
                    for identifier in candidate_ids
                ]
                with (
                    patch.object(itunes, "search_itunes", return_value=albums),
                    patch.object(SESSION, "get") as request,
                ):
                    self.assertIsNone(
                        itunes.fetch_itunes_album_details("Primary", "Recording")
                    )
                request.assert_not_called()
                response = MagicMock(status_code=200)
                response.json.return_value = {
                    "data": [
                        {
                            "id": identifier,
                            "title": "Recording",
                            "artist": {"name": "Primary"},
                        }
                        for identifier in candidate_ids
                    ]
                }
                with patch.object(SESSION, "get", return_value=response) as request:
                    self.assertIsNone(
                        deezer.fetch_deezer_album_details("Primary", "Recording")
                    )
                self.assertEqual(request.call_count, 1)
                response.json.return_value = {
                    "results": [
                        {
                            "id": identifier,
                            "title": "Primary - Recording",
                            "format": ["File"],
                        }
                        for identifier in candidate_ids
                    ]
                }
                with (
                    patch.object(SESSION, "get", return_value=response),
                    patch.object(discogs, "fetch_discogs_release_details") as details,
                ):
                    self.assertIsNone(
                        discogs.search_discogs_release("Primary", "Recording", "token")
                    )
                details.assert_not_called()

    def test_genius_preserves_version_constraints_and_revalidates_details(self) -> None:
        search = MagicMock()
        search.json.return_value = {
            "response": {
                "hits": [
                    {
                        "result": {
                            "api_path": "/songs/1",
                            "title": "Recording",
                            "primary_artist": {"name": "Primary"},
                        }
                    }
                ]
            }
        }
        for title in ("Recording (Clean Version)", "Recording (Explicit Version)"):
            with (
                self.subTest(title=title),
                patch.object(SESSION, "get", return_value=search) as request,
            ):
                self.assertIsNone(
                    genius.fetch_genius_song_details("Primary", title, "token")
                )
            self.assertEqual(request.call_count, 1)
        for candidate_title, candidate_artist, accepted in (
            ("Recording", "Primary", True),
            ("Different", "Primary", False),
            ("Recording", "Unrelated", False),
        ):
            details = MagicMock()
            details.json.return_value = {
                "response": {
                    "song": {
                        "title": candidate_title,
                        "primary_artist": {"name": candidate_artist},
                        "id": 1,
                    }
                }
            }
            with (
                self.subTest(
                    candidate_title=candidate_title, candidate_artist=candidate_artist
                ),
                patch.object(SESSION, "get", side_effect=[search, details]),
            ):
                self.assertEqual(
                    genius.fetch_genius_song_details("Primary", "Recording", "token")
                    is not None,
                    accepted,
                )

    def test_lyrics_retry_preserves_editions_and_allows_generic_labels(self) -> None:
        for title in (
            "Recording (Clean Version)",
            "Recording (Album Version)",
            "Recording (feat. Guest)",
        ):
            with (
                self.subTest(title=title),
                patch.object(lyrics, "_query_syncedlyrics", return_value=None) as query,
            ):
                self.assertIsNone(lyrics.fetch_synced_lyrics("Primary", title))
            self.assertEqual(
                query.call_count,
                2 if "feat." in title or "Album Version" in title else 1,
            )


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
                        catalog_metadata._update_deezer(
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
        catalog_metadata._update_deezer(
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
        catalog_metadata._update_deezer(
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
            catalog_metadata._update_deezer(
                recording, album, has_album_context=True, force=True
            )
        self.assertEqual(recording.total_tracks, 1)

    def test_incomplete_medium_indexes_do_not_establish_totals(self) -> None:
        recording = TrackInfo(Path("recording.flac"), disc_number=2, total_discs=2)
        index_cases: tuple[
            tuple[dict[tuple[int, int], dict[str, object]], int], ...
        ] = (({(2, 1): {}, (2, 3): {}}, 2), ({(2, 1): {}, (2, 2): {}}, 5))
        for positions, expected_tracks in index_cases:
            with self.subTest(positions=positions):
                self.assertIsNone(
                    catalog_metadata._deezer_medium_track_count(
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
                        "tracks_by_disc_and_position": {
                            (1, 4): {
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
        release["tracks_by_disc_and_position"][2, 1]["artist"] = "Alpha"
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

    def test_standalone_isrc_updates_only_missing_identity_components(self) -> None:
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
                        musicbrainz_metadata._fetch_standalone_recording(track)
                        is not None,
                        expected,
                    )

    def test_release_isrc_updates_preserves_meaningful_title_and_artist(self) -> None:
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
                    musicbrainz_metadata._fill_missing_title(
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
                musicbrainz_metadata._resolve_musicbrainz_track_id(
                    track, {(1, 1): recording_id}, candidate
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
        positions[1, 1]["duration"] = 300
        positions[1, 3]["duration"] = 300
        track.track_number = None
        self.assertEqual(
            matching._match_position_by_mbid(track, recording_id, None, indexes), 2
        )


class TestAlbumIdentity(IsolatedTestCase):
    def test_album_positions_reject_recording_variants(self) -> None:
        pairs = (
            ("The Journey Part 2", "The Journey Part 3"),
            ("Long Summer Night (Alpha Remix)", "Long Summer Night (Omega Remix)"),
            ("Long Summer Night", "Long Summer Night (Live)"),
            ("Intro", "Outro"),
        )
        for query, candidate in pairs:
            for position in (1, 99):
                with self.subTest(query=query, candidate=candidate, position=position):
                    track = dataclasses.replace(
                        self._track(query),
                        track_number=position,
                        file_path=Path(f"/music/{position:02d} - Track.flac"),
                    )
                    self.assertIsNone(
                        _resolve_album_track_position(
                            track,
                            track.file_path,
                            album_deezer_details={
                                "tracks_by_disc_and_position": {
                                    (1, 1): {"title": candidate}
                                }
                            },
                        )
                    )

    def test_album_identifiers_still_update_corrupted_titles(self) -> None:
        track = dataclasses.replace(self._track("Untitled"), isrc="USAAA2400001")
        position = _resolve_album_track_position(
            track,
            track.file_path,
            album_deezer_details={
                "tracks_by_disc_and_position": {
                    (1, 3): {
                        "title": "Correct recording",
                        "isrc": "USAAA2400001",
                        "disc_number": 1,
                    }
                }
            },
        )
        self.assertEqual(position, 3)

    def test_variant_rejections_also_preserve_alien_tracks(self) -> None:
        for query, candidate in (
            ("The Journey Part 2", "The Journey Part 3"),
            ("Long Summer Night (Alpha Remix)", "Long Summer Night (Omega Remix)"),
        ):
            with self.subTest(query=query):
                track = dataclasses.replace(
                    self._track(query), album="Album", track_number=1
                )
                recordings = {
                    (1, 1): {"artist": "Artist", "title": candidate},
                    (1, 2): {"artist": "Artist", "title": "Different Recording"},
                    (1, 3): {"artist": "Artist", "title": "Another Recording"},
                }
                release = {
                    "artist": "Artist",
                    "title": "Album",
                    "tracks_by_disc_and_position": recordings,
                }
                self.assertTrue(
                    is_alien_album_track(
                        track,
                        track.file_path,
                        "Artist",
                        "Album",
                        album_deezer_details=release,
                    )
                )
                track.isrc = "USAAA2400001"
                recordings[1, 1]["isrc"] = track.isrc
                self.assertTrue(
                    is_alien_album_track(
                        track,
                        track.file_path,
                        "Artist",
                        "Album",
                        album_deezer_details=release,
                    )
                )

    def test_matching_title_does_not_override_a_conflicting_artist(self) -> None:
        track = dataclasses.replace(
            self._track("Correct recording"), artist="Damaged artist"
        )
        self.assertIsNone(
            _resolve_album_track_position(
                track,
                track.file_path,
                album_deezer_details={
                    "tracks_by_disc_and_position": {
                        (1, 1): {
                            "title": "Correct recording",
                            "artist": "Actual artist",
                        }
                    }
                },
            )
        )

    @staticmethod
    def _track(title: str) -> TrackInfo:
        return TrackInfo(
            file_path=Path("/music/01 - Track.flac"), artist="Artist", title=title
        )


class TestStandaloneAlbumContext(IsolatedTestCase):
    def setUp(self) -> None:
        self.path = self.test_directory / "01 - Recording.wav"
        self.path.touch()
        self.recording = TrackInfo(
            self.path,
            artist="Primary",
            title="Recording",
            album="Record",
            album_artist="Primary",
            track_number=1,
        )

    def test_standalone_preservation_does_not_clear_shared_album_context(self) -> None:
        context = tagger._AlbumMetadataContext(
            sample_artist="Primary",
            sample_album="Record",
            track_mbids={(1, 1): "recording"},
        )
        with (
            patch.object(metadata, "read_track_metadata", return_value=self.recording),
            patch.object(
                artist_resolution, "resolve_artist_name", return_value="Primary"
            ),
            patch.object(tagger, "is_alien_album_track", return_value=True),
            patch.object(tagger, "_update_metadata") as update,
            patch.object(
                tagger, "_process_audio_and_save_tags", return_value=self.recording
            ),
        ):
            tagger.process_single_track(self.path, album_context=context, dry_run=True)
        self.assertFalse(update.call_args.kwargs["album_context"].has_album_context)
        self.assertTrue(context.has_album_context)
        self.assertEqual(context.track_mbids, {(1, 1): "recording"})


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
                    Path("2-01 - Opening.flac"), disc_number=2, total_discs=total_discs
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


RECORDING_ID = "11111111-1111-4111-8111-111111111111"


OTHER_RECORDING_ID = "22222222-2222-4222-8222-222222222222"


RELEASE_ID = "33333333-3333-4333-8333-333333333333"


RELEASE_GROUP_ID = "44444444-4444-4444-8444-444444444444"


class TestReleaseIdentitySafety(IsolatedTestCase):
    def test_duration_disagreement_preserves_matching_stored_recording_and_release(
        self,
    ) -> None:
        for force in (False, True):
            track = TrackInfo(
                Path("song.flac"),
                artist="Performer",
                title="Song (Instrumental)",
                album="Album Instrumentals",
                album_artist="Performer",
                duration=192.4,
                musicbrainz_trackid=RECORDING_ID,
                musicbrainz_albumid=RELEASE_ID,
            )
            recording = {
                "artist": "Performer",
                "title": "Song (Instrumental)",
                "duration": 189,
            }
            release = {
                "title": "Album Instrumentals",
                "album_artist": "Performer",
                "label": "Unverified label",
                "tracks_by_mbid": {RECORDING_ID: recording},
            }
            with (
                self.subTest(force=force),
                patch.object(
                    musicbrainz,
                    "fetch_musicbrainz_recording_details",
                    return_value=recording,
                ),
                patch.object(
                    artist_resolution, "resolve_artist_name", return_value="Performer"
                ),
            ):
                self.assertIsNone(
                    musicbrainz_metadata._fetch_standalone_recording(track, force=force)
                )
                musicbrainz_metadata._apply_mb_release_metadata(
                    track, RELEASE_ID, release, force=force
                )
            self.assertEqual(track.musicbrainz_trackid, RECORDING_ID)
            self.assertEqual(track.musicbrainz_albumid, RELEASE_ID)
            self.assertIsNone(track.label)
            self.assertIn("musicbrainz_trackid", track.metadata_conflicts)
            self.assertNotIn("musicbrainz_albumid", track.metadata_conflicts)

    def test_clean_album_recording_requires_local_edition_evidence(self) -> None:
        for force in (False, True):
            for advisory in (None, "Explicit"):
                with self.subTest(force=force, advisory=advisory):
                    track = TrackInfo(
                        Path("01 - Song.flac"),
                        artist="Performer",
                        title="Song",
                        album="Album",
                        track_number=1,
                        duration=200,
                        advisory=advisory,
                        musicbrainz_trackid=RECORDING_ID,
                    )
                    recording = {
                        "artist": "Performer",
                        "title": "Song",
                        "duration": 200,
                        "recording_mbid": RECORDING_ID,
                        "disambiguation": "clean",
                    }
                    release = {
                        "title": "Album",
                        "album_artist": "Performer",
                        "tracks_by_disc_and_position": {(1, 1): recording},
                        "tracks_by_mbid": {RECORDING_ID: recording},
                    }
                    with patch.object(
                        musicbrainz, "fetch_musicbrainz_recording_details"
                    ) as fetch:
                        musicbrainz_metadata._update_musicbrainz(
                            track,
                            RELEASE_ID,
                            {(1, 1): RECORDING_ID},
                            release,
                            force=force,
                        )
                    fetch.assert_not_called()
                    self.assertEqual(track.musicbrainz_trackid, RECORDING_ID)
                    self.assertIsNone(track.musicbrainz_albumid)
                    self.assertIsNone(track.disambiguation)
                    self.assertEqual(track.advisory, advisory)
                    self.assertIn("identity", track.metadata_conflicts)
                    self.assertNotIn("musicbrainz_trackid", track.metadata_sources)

    def test_acoustid_clean_result_requires_edition_evidence(self) -> None:
        track = TrackInfo(
            Path("Song.flac"), artist="Performer", title="Song", duration=200
        )
        with (
            patch.object(acoustid, "lookup_acoustid", return_value=RECORDING_ID),
            patch.object(
                musicbrainz,
                "fetch_musicbrainz_recording_details",
                return_value={
                    "artist": "Performer",
                    "title": "Song",
                    "duration": 200,
                    "disambiguation": "clean",
                },
            ),
        ):
            musicbrainz_metadata._update_acoustid(track, track.file_path, "test-key")
        self.assertIsNone(track.musicbrainz_trackid)
        self.assertIn("identity", track.metadata_conflicts)

    def test_standalone_clean_recording_is_not_verified_by_stored_id(self) -> None:
        track = TrackInfo(
            Path("Song.flac"),
            artist="Performer",
            title="Song",
            duration=200,
            musicbrainz_trackid=RECORDING_ID,
        )
        with patch.object(
            musicbrainz,
            "fetch_musicbrainz_recording_details",
            return_value={
                "artist": "Performer",
                "title": "Song",
                "duration": 200,
                "disambiguation": "clean",
            },
        ):
            self.assertIsNone(musicbrainz_metadata._fetch_standalone_recording(track))
        self.assertEqual(track.musicbrainz_trackid, RECORDING_ID)
        self.assertIn("identity", track.metadata_conflicts)
        self.assertNotIn("musicbrainz_trackid", track.metadata_sources)

    def test_edition_guard_preserves_supported_clean_and_uncensored_recordings(
        self,
    ) -> None:
        for local, advisory, candidate, conflict in (
            ("clean", None, "cleaned version", False),
            ("clean", None, "explicit", True),
            (None, "Explicit", "uncensored", False),
            (None, None, "studio", False),
            ("clean", "Explicit", "clean", True),
        ):
            with self.subTest(local=local, advisory=advisory, candidate=candidate):
                track = TrackInfo(
                    Path("Song.flac"), disambiguation=local, advisory=advisory
                )
                self.assertEqual(
                    musicbrainz_metadata._release_edition_conflicts(
                        track, {"disambiguation": candidate}
                    ),
                    conflict,
                )

    def test_base_release_search_cannot_assign_deluxe_identifiers_or_dates(
        self,
    ) -> None:
        track = TrackInfo(Path("Song.flac"), artist="Performer", album="Album (Deluxe)")
        with patch.object(
            musicbrainz,
            "search_musicbrainz_release",
            return_value={"id": RELEASE_ID, "title": "Album", "date": "2016-11-25"},
        ):
            musicbrainz_metadata._resolve_musicbrainz_album_id(track, None, None)
        self.assertIsNone(track.musicbrainz_albumid)
        self.assertIsNone(track.date)
        for title, expected in (
            ("Album", False),
            ("Album (Deluxe)", True),
            ("Album (Deluxe Edition)", True),
            ("Album (Live)", False),
        ):
            with self.subTest(title=title):
                self.assertEqual(
                    musicbrainz_metadata._validate_mb_release_title(title, track.album),
                    expected,
                )

    def test_clean_qualifier_is_evidence_but_song_title_word_is_not(self) -> None:
        for title, album, expected in (
            ("Song (Clean)", "Album", False),
            ("Song", "Album [Clean Version]", False),
            ("Clean", "Album", True),
        ):
            with self.subTest(title=title, album=album):
                track = TrackInfo(Path("Song.flac"), title=title, album=album)
                self.assertEqual(
                    musicbrainz_metadata._release_edition_conflicts(
                        track, {"disambiguation": "clean"}
                    ),
                    expected,
                )

    def test_release_disambiguation_can_establish_deluxe_edition(self) -> None:
        track = TrackInfo(Path("Song.flac"), artist="Primary", album="Album (Deluxe)")
        with patch.object(
            musicbrainz,
            "search_musicbrainz_release",
            return_value={
                "id": RELEASE_ID,
                "title": "Album",
                "disambiguation": "deluxe, explicit",
            },
        ):
            musicbrainz_metadata._resolve_musicbrainz_album_id(track, None, None)
        self.assertEqual(track.musicbrainz_albumid, RELEASE_ID)
        self.assertFalse(
            musicbrainz_metadata._validate_mb_release_title(
                "Album", "Album (Deluxe)", "Japanese bonus track"
            )
        )

    def test_album_slot_cannot_replace_verified_or_unavailable_stored_recording(
        self,
    ) -> None:
        for stored_details in (
            {"artist": "Performer", "title": "Introduction", "duration": 103},
            None,
        ):
            with self.subTest(stored_details=stored_details):
                track = TrackInfo(
                    Path("01 - Introduction.flac"),
                    artist="Performer",
                    title="Introduction",
                    album="Album",
                    album_artist="Performer",
                    track_number=1,
                    duration=103,
                    musicbrainz_trackid=RECORDING_ID,
                    musicbrainz_albumid=RELEASE_ID,
                )
                candidate = {
                    "artist": "Performer",
                    "title": "Introduction",
                    "duration": 103,
                    "recording_mbid": OTHER_RECORDING_ID,
                }
                release = {
                    "title": "Album",
                    "album_artist": "Performer",
                    "tracks_by_disc_and_position": {(1, 1): candidate},
                    "tracks_by_mbid": {OTHER_RECORDING_ID: candidate},
                }
                with (
                    patch.object(
                        musicbrainz,
                        "fetch_musicbrainz_recording_details",
                        return_value=stored_details,
                    ),
                    patch.object(musicbrainz, "fetch_track_mbid") as search,
                    patch.object(
                        musicbrainz, "search_musicbrainz_release"
                    ) as release_search,
                    patch.object(
                        artist_resolution,
                        "resolve_artist_name",
                        return_value="Performer",
                    ),
                ):
                    musicbrainz_metadata._update_musicbrainz(
                        track,
                        RELEASE_ID,
                        {(1, 1): OTHER_RECORDING_ID},
                        release,
                        force=True,
                    )
                self.assertEqual(track.musicbrainz_trackid, RECORDING_ID)
                self.assertIsNone(track.musicbrainz_albumid)
                self.assertIn("musicbrainz_albumid", track.metadata_conflicts)
                search.assert_not_called()
                release_search.assert_not_called()

    def test_force_preserves_full_edition_beyond_fuzzy_base_title(self) -> None:
        base_title = "A Long Summer Evening With The Orchestra"
        for edition in ("Deluxe", "Live", "Remixes", "Radio Edit"):
            with self.subTest(edition=edition):
                track = TrackInfo(
                    Path("recording.flac"),
                    artist="Performer",
                    album_artist="Performer",
                    album=f"{base_title} ({edition})",
                    musicbrainz_albumid=RELEASE_ID,
                )
                with patch.object(
                    artist_resolution, "resolve_artist_name", return_value="Performer"
                ):
                    musicbrainz_metadata._apply_mb_release_metadata(
                        track,
                        RELEASE_ID,
                        {"title": base_title, "album_artist": "Performer"},
                        force=True,
                    )
                self.assertEqual(track.album, f"{base_title} ({edition})")

    def test_force_validates_stored_recording_before_searching_same_title(self) -> None:
        track = TrackInfo(
            Path("01 - Introduction.flac"),
            artist="Performer",
            title="Introduction",
            album="Second Album",
            duration=103,
            musicbrainz_trackid=RECORDING_ID,
        )
        recording = {"artist": "Performer", "title": "Introduction", "duration": 103}
        with (
            patch.object(
                musicbrainz, "fetch_track_mbid", return_value=OTHER_RECORDING_ID
            ) as search,
            patch.object(
                musicbrainz,
                "fetch_musicbrainz_recording_details",
                return_value=recording,
            ) as fetch,
        ):
            musicbrainz_metadata._resolve_musicbrainz_track_id(track, None, None)
            self.assertEqual(
                musicbrainz_metadata._fetch_standalone_recording(track, force=True),
                recording,
            )
        search.assert_not_called()
        fetch.assert_called_once_with(RECORDING_ID)
        self.assertEqual(track.musicbrainz_trackid, RECORDING_ID)

    def test_release_with_same_title_other_recording_does_not_match_verified_id(
        self,
    ) -> None:
        track = TrackInfo(
            Path("01 - Introduction.flac"),
            artist="Performer",
            title="Introduction",
            album="Album",
            album_artist="Performer",
            duration=103,
            musicbrainz_trackid=RECORDING_ID,
            musicbrainz_albumid=RELEASE_ID,
            musicbrainz_releasegroupid=RELEASE_GROUP_ID,
            metadata_sources={
                "musicbrainz_trackid": "MusicBrainz",
                "musicbrainz_albumid": "MusicBrainz",
                "musicbrainz_releasegroupid": "MusicBrainz",
            },
        )
        release = {
            "title": "Album",
            "album_artist": "Performer",
            "musicbrainz_releasegroupid": RELEASE_GROUP_ID,
            "tracks_by_mbid": {
                OTHER_RECORDING_ID: {
                    "artist": "Performer",
                    "title": "Introduction",
                    "duration": 103,
                }
            },
        }
        with patch.object(
            artist_resolution, "resolve_artist_name", return_value="Performer"
        ):
            musicbrainz_metadata._apply_mb_release_metadata(
                track, RELEASE_ID, release, force=True
            )
        self.assertEqual(track.musicbrainz_trackid, RECORDING_ID)
        self.assertEqual(track.metadata_sources["musicbrainz_trackid"], "MusicBrainz")
        self.assertIsNone(track.musicbrainz_albumid)
        self.assertEqual(track.musicbrainz_releasegroupid, RELEASE_GROUP_ID)
        self.assertNotIn("musicbrainz_albumid", track.metadata_sources)
        self.assertEqual(
            track.metadata_sources["musicbrainz_releasegroupid"], "MusicBrainz"
        )
        self.assertIn("musicbrainz_albumid", track.metadata_conflicts)

    def test_membership_failure_preserves_verified_recording_even_without_force(
        self,
    ) -> None:
        track = TrackInfo(
            Path("recording.flac"),
            artist="Performer",
            title="Original Song",
            album="Album",
            album_artist="Performer",
            musicbrainz_trackid=RECORDING_ID,
            musicbrainz_albumid=RELEASE_ID,
            metadata_sources={"musicbrainz_trackid": "AcoustID"},
        )
        musicbrainz_metadata._apply_mb_release_metadata(
            track,
            RELEASE_ID,
            {
                "title": "Album",
                "album_artist": "Performer",
                "tracks_by_mbid": {
                    OTHER_RECORDING_ID: {"artist": "Performer", "title": "Other Song"}
                },
            },
        )
        self.assertEqual(track.musicbrainz_trackid, RECORDING_ID)
        self.assertIsNone(track.musicbrainz_albumid)


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
        release = {"total_discs": 2, "tracks_by_disc_and_position": {(2, 3): recording}}
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
                musicbrainz_metadata._apply_mb_release_metadata(
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
                musicbrainz_metadata._apply_mb_release_metadata(
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
        musicbrainz_metadata._apply_mb_release_metadata(
            track, RELEASE_ID, {"total_tracks": 10, "total_discs": 1}, force=True
        )
        self.assertEqual(track.total_tracks, 10)


class TestVerifiedAlbumLayout(IsolatedTestCase):
    def test_album_layout_checks_expected_position_before_scanning_candidates(
        self,
    ) -> None:
        files = [Path(f"{index}.flac") for index in range(1, 4)]
        tracks: list[TrackInfo | None] = [
            TrackInfo(
                path, artist="Performer", title=f"Song {index}", track_number=index
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
        for verified, expected in ((False, "4 226"), (True, "4 226 & RAVA")):
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
            positions[1, 2] = {"artist": "Other Performer", "title": "Other Song"}
            self.assertFalse(
                tagger._release_matches_album_tracks(
                    {"total_tracks": 2, "tracks_by_disc_and_position": positions}, files
                )
            )


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
                musicbrainz_metadata._fetch_standalone_recording(recording, force=True)
            )
        self.assertEqual(fetch.call_count, 2)
        search.assert_called_once_with("Alpha", "Original Song", None)
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
                patch.object(
                    SESSION, "get", return_value=httpx.Response(200, json=payload)
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
                    artist_resolution, "resolve_artist_name", return_value="Alpha"
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

    def test_catalog_match_without_artist_requires_a_recording_identifier(self) -> None:
        self.assertFalse(
            matching._catalog_recording_matches_track(
                {"title": "Intro", "duration": 30}, "Intro", "Artist A", None, None, 30
            )
        )
        self.assertTrue(
            matching._catalog_recording_matches_track(
                {"title": "Intro", "duration": 30, "isrc": "KNOWN"},
                "Intro",
                "Artist A",
                "KNOWN",
                None,
                30,
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
