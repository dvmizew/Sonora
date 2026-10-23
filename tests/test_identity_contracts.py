import dataclasses
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx

from sonora.core import artists
from sonora.core.config import SonoraConfig
from sonora.core.constants import DIRS
from sonora.core.http import SESSION
from sonora.core.models import TrackInfo
from sonora.modules import matching
from sonora.modules import musicbrainz_enrichment as enrichment
from sonora.services import acoustid, artist_resolution, musicbrainz
from tests.support import IsolatedTestCase

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
            artists.extract_featured_artist_tokens("Legal Name"),
            ["Stage"],
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
            tuple[str, list[dict[str, Any]], list[dict[str, Any]], str], ...
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

    def test_cached_corrupt_expansions_and_branding_do_not_poison_names(self) -> None:
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

    def test_registered_groups_and_featured_subtokens_are_preserved(self) -> None:
        policy = SonoraConfig(known_group_artists=frozenset({"Alpha & Beta"}))
        with patch("sonora.core.artists.get_config", return_value=policy):
            self.assertTrue(artist_resolution.is_single_group_artist("Alpha & Beta"))
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
        self.assertFalse(artist_resolution.is_single_group_artist("!!!"))
        for candidate, expected in (
            ({"name": "Alpha & Beta", "type": "Group"}, True),
            ({"name": "Alpha & Beta", "ext:score": "95"}, True),
            ({"name": "Other", "type": "Group"}, False),
        ):
            artists.is_single_group_artist.cache_clear()
            with (
                self.subTest(candidate=candidate),
                patch.object(artist_resolution, "get_cached_api", return_value=None),
                patch.object(
                    musicbrainz, "search_musicbrainz_artists", return_value=[candidate]
                ),
            ):
                self.assertEqual(
                    artist_resolution.is_single_group_artist("Alpha & Beta"), expected
                )
        artists.is_single_group_artist.cache_clear()
        with patch.object(artist_resolution, "get_cached_api", return_value=True):
            self.assertTrue(artist_resolution.is_single_group_artist("Alpha & Beta"))

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
            self.assertFalse(artist_resolution.is_single_group_artist("Alpha & Beta"))


class TestAlbumIdentityContracts(IsolatedTestCase):
    def test_album_position_indexes_use_disc_and_position(self) -> None:
        recording = {"title": "Recording", "artist": "Primary"}
        for index_name, key in (("tracks_by_disc_and_position", (2, 3)),):
            with self.subTest(index_name=index_name, key=key):
                details = {index_name: {key: recording}}
                self.assertTrue(
                    matching._pos_in_album(
                        3, disc_number=2, album_mb_release_details=details
                    )
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
        for index in (
            {(2, 3): MBID},
            {(1, 3): MBID},
        ):
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
                [
                    {
                        "tracks_by_disc_and_position": {
                            (2, 4): {},
                            (3, 1): {},
                            (1, 2): {},
                        },
                    }
                ],
            ),
            {1: {1, 2}, 2: {3, 4}, 3: {1}},
        )


class TestMusicBrainzEnrichmentContracts(IsolatedTestCase):
    def test_unmatched_fingerprints_do_not_create_a_musicbrainz_identifier(
        self,
    ) -> None:
        with patch.object(acoustid, "lookup_acoustid", return_value=None):
            enrichment._enrich_acoustid(
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
            enrichment._find_album_release_recording(
                recording, {"tracks_by_disc_and_position": {(2, 3): None}}, 2
            )
        with self.assertRaisesRegex(TypeError, "JSON object"):
            enrichment._extract_release_recording(
                recording, {"tracks_by_mbid": {MBID: []}}, None, False
            )

    def test_verified_track_id_and_existing_date_do_not_trigger_reassignment(
        self,
    ) -> None:
        self.recording.musicbrainz_trackid = MBID
        self.recording.date = "2019"
        with patch.object(musicbrainz, "fetch_track_mbid") as fetch:
            enrichment._resolve_musicbrainz_track_id(
                self.recording, None, None, None, False
            )
        fetch.assert_not_called()
        with patch.object(
            musicbrainz,
            "search_musicbrainz_release",
            return_value={"id": OTHER_MBID, "date": "2020"},
        ):
            enrichment._resolve_musicbrainz_album_id(self.recording, None, None)
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
                enrichment._fetch_standalone_recording(self.recording, force=True)
            )
        self.assertEqual(self.recording.musicbrainz_trackid, OTHER_MBID)
        self.assertEqual(self.recording.title, "Recording")

    def test_invalid_track_totals_and_conflicting_release_layout_are_withheld(
        self,
    ) -> None:
        self.recording.total_tracks = 10
        self.recording.total_discs = 2
        enrichment._apply_mb_recording_metadata(
            self.recording, {"disc_total_tracks": "bad"}, True, None, False, force=True
        )
        self.recording.musicbrainz_albumid = MBID
        enrichment._apply_mb_release_metadata(
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
                    enrichment._should_update_track_artist(
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
        resolution = patch.object(
            artist_resolution,
            "resolve_artist_name",
            side_effect=lambda name: name,
        )
        resolution.start()
        self.addCleanup(resolution.stop)

    def test_any_recording_isrc_can_heal_a_corrupt_release_title(self) -> None:
        self.recording.title = "Untitled"
        self.recording.isrc = "USAAA2100002"
        candidate = {
            "title": "Recording",
            "isrc": "USAAA2100001",
            "isrcs": ["USAAA2100001", "USAAA2100002"],
        }
        self.assertTrue(
            enrichment._heal_track_identity_from_release(self.recording, candidate)
        )
        self.assertEqual(self.recording.title, "Recording")
        self.assertFalse(
            enrichment._heal_track_identity_from_release(self.recording, candidate)
        )
        self.recording.title = "Unrelated"
        self.recording.isrc = "USAAA2100003"
        self.assertFalse(
            enrichment._heal_track_identity_from_release(self.recording, candidate)
        )
        self.assertEqual(self.recording.title, "Unrelated")

    def test_existing_mbid_skips_fingerprinting_without_force(self) -> None:
        self.recording.musicbrainz_trackid = MBID
        with patch.object(acoustid, "lookup_acoustid") as lookup:
            enrichment._enrich_acoustid(
                self.recording, self.recording.file_path, "fixture"
            )
        lookup.assert_not_called()

    def test_release_recording_lookup_uses_position_fallbacks(self) -> None:
        self.recording.track_number = 3
        self.assertEqual(
            enrichment._find_album_release_recording(
                self.recording,
                {"tracks_by_disc_and_position": {(2, 3): self.details}},
                2,
            ),
            self.details,
        )
        self.assertIsNone(
            enrichment._find_album_release_recording(
                self.recording, {"title": "Record"}, 2
            )
        )

    def test_track_id_assignment_requires_title_identity_when_release_is_known(
        self,
    ) -> None:
        self.recording.track_number = 3
        release = {"title": "Record"}
        enrichment._resolve_musicbrainz_track_id(
            self.recording, {(1, 3): MBID}, release, self.details, False
        )
        self.assertEqual(self.recording.musicbrainz_trackid, MBID)
        self.recording.musicbrainz_trackid = None
        enrichment._resolve_musicbrainz_track_id(
            self.recording,
            {(1, 3): MBID},
            release,
            {"title": "Unrelated", "artist": "Wrong"},
            False,
        )
        self.assertIsNone(self.recording.musicbrainz_trackid)
        enrichment._resolve_musicbrainz_track_id(
            self.recording, {(1, 3): MBID}, None, None, False
        )
        self.assertIsNone(self.recording.musicbrainz_trackid)

    def test_recording_extraction_prefers_mbid_and_rejects_unrelated_titles(
        self,
    ) -> None:
        self.recording.musicbrainz_trackid = MBID
        self.assertEqual(
            enrichment._extract_release_recording(
                self.recording, {"tracks_by_mbid": {MBID: self.details}}, None, False
            ),
            self.details,
        )
        self.assertIsNone(
            enrichment._extract_release_recording(
                self.recording,
                {"title": "Record"},
                {"title": "Other", "artist": "Wrong"},
                False,
            )
        )
        self.assertIsNone(
            enrichment._extract_release_recording(
                self.recording, {"title": "Record"}, None, False
            )
        )

    def test_standalone_recording_missing_or_failed_research_leaves_no_false_id(
        self,
    ) -> None:
        self.recording.musicbrainz_trackid = MBID
        with patch.object(
            musicbrainz, "fetch_musicbrainz_recording_details", return_value=None
        ):
            self.assertIsNone(enrichment._fetch_standalone_recording(self.recording))
        with (
            patch.object(
                musicbrainz,
                "fetch_musicbrainz_recording_details",
                return_value={"artist": "Wrong", "title": "Other"},
            ),
            patch.object(musicbrainz, "fetch_track_mbid", return_value=None),
        ):
            self.assertIsNone(
                enrichment._fetch_standalone_recording(self.recording, force=True)
            )
        self.assertIsNone(self.recording.musicbrainz_trackid)

    def test_valid_album_identifiers_dates_and_totals_are_preserved(self) -> None:
        self.recording.musicbrainz_albumid = MBID
        self.recording.date = "2019"
        enrichment._resolve_musicbrainz_album_id(self.recording, OTHER_MBID, None)
        self.assertEqual(
            (self.recording.musicbrainz_albumid, self.recording.date), (MBID, "2019")
        )
        with patch.object(
            musicbrainz, "fetch_musicbrainz_release_details", return_value=None
        ):
            enrichment._apply_mb_release_metadata(self.recording, None, None)
        self.assertEqual(self.recording.musicbrainz_albumid, MBID)
        self.recording.total_tracks = 5
        self.recording.total_discs = 1
        enrichment._apply_mb_release_metadata(
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
            enrichment._enrich_acoustid(
                self.recording, self.recording.file_path, "fixture"
            )
            self.assertEqual(self.recording.musicbrainz_trackid, MBID)
            self.recording.track_number = 1
            self.recording.metadata_sources["musicbrainz_trackid"] = "MusicBrainz"
            enrichment._enrich_acoustid(
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
                enrichment._enrich_acoustid(recording, recording.file_path, "fixture")
            self.assertIsNone(recording.musicbrainz_trackid)
        with patch.object(acoustid, "lookup_acoustid", side_effect=OSError("offline")):
            enrichment._enrich_acoustid(
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
            recording = dataclasses.replace(self.recording)
            with (
                self.subTest(release=release),
                patch.object(
                    musicbrainz, "search_musicbrainz_release", return_value=release
                ),
            ):
                enrichment._resolve_musicbrainz_album_id(recording, None, None)
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
            self.assertIsNone(enrichment._fetch_standalone_recording(self.recording))
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
                enrichment._fetch_standalone_recording(self.recording, force=True),
                self.details,
            )
        self.assertEqual(self.recording.musicbrainz_trackid, OTHER_MBID)

    def test_unknown_and_alien_artists_can_be_healed_without_poisoning_known_artists(
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
                    enrichment._should_update_track_artist(
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
                enrichment._apply_mb_release_metadata(recording, None, None)
            self.assertIsNone(recording.musicbrainz_albumid)
            self.assertIsNone(recording.musicbrainz_releasegroupid)

    def test_release_metadata_preserves_track_counts_and_normalizes_album_identity(
        self,
    ) -> None:
        release: dict[str, Any] = {
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
        enrichment._apply_mb_release_metadata(self.recording, MBID, release, force=True)
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
        enrichment._apply_mb_release_metadata(self.recording, MBID, release, force=True)
        self.assertIsNone(self.recording.total_tracks)

    def test_recording_metadata_heals_corrupt_identity_and_preserves_disc_totals(
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
        enrichment._apply_mb_recording_metadata(
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
            enrichment._enrich_musicbrainz(self.recording, None, None, None)
        self.assertEqual(self.recording.title, "Recording")


class TestArtistResponseContracts(IsolatedTestCase):
    def test_artist_resolution_exposes_malformed_deezer_responses(self) -> None:
        from sonora.services.artist_resolution import resolve_artist_name

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
                    resolve_artist_name("Artist Alpha")
