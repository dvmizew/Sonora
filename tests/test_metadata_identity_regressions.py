import dataclasses
from pathlib import Path
from unittest.mock import patch

import taglib

from sonora.audio.metadata import (
    clear_metadata_cache,
    read_track_metadata,
    write_track_metadata,
)
from sonora.core.artists import get_primary_artist
from sonora.core.models import TrackInfo
from sonora.modules import matching, metadata_consensus
from sonora.modules import musicbrainz_enrichment as musicbrainz_enrichment_module
from sonora.modules import track_processing as track_processing_module
from sonora.modules.metadata_consensus import build_provider_proposal
from sonora.services import artist_resolution
from tests.regression_support import AudioRegressionTestCase
from tests.support import IsolatedTestCase


class TestMetadataIdentityRegressions(AudioRegressionTestCase):
    def test_same_title_different_artist_is_shielded(self) -> None:
        audio_path = self.root / "01 - Intro.flac"
        unrelated = TrackInfo(
            audio_path,
            artist="ZZZ",
            title="Intro",
            album="Different Album",
            track_number=1,
        )
        release = {
            "album_artist": "Alpha",
            "title": "Album",
            "tracks_by_disc_and_position": {
                (1, 1): {"artist": "Alpha", "title": "Intro", "isrc": "USAAA0000001"}
            },
        }
        self.assertTrue(
            matching.is_alien_album_track(
                unrelated,
                audio_path,
                "Alpha",
                "Album",
                album_mb_release_details=release,
            )
        )
        unrelated.isrc = "USAAA0000001"
        self.assertTrue(
            matching.is_alien_album_track(
                unrelated,
                audio_path,
                "Alpha",
                "Album",
                album_mb_release_details=release,
            )
        )

    def test_offline_artist_resolution_ignores_api_cache(self) -> None:
        with patch(
            "sonora.services.artist_resolution.get_cached_api", return_value=True
        ) as api_cache:
            self.assertEqual(get_primary_artist("Example & Partners"), "Example")
            api_cache.assert_not_called()

    def test_identifier_heals_corrupt_title_and_position(self) -> None:
        for id_field in ("isrc", "itunes_trackid"):
            with self.subTest(id_field=id_field):
                track = TrackInfo(
                    self.root / "99 - Corrupt.flac", title="Untitled", track_number=99
                )
                position = matching._match_position_by_authoritative_id(
                    track,
                    99,
                    id_field,
                    "KNOWN123",
                    [
                        {
                            "tracks_by_disc_and_position": {
                                (2, 4): {id_field: "known123", "title": "Original"}
                            }
                        }
                    ],
                )
                self.assertEqual(position, 4)
                self.assertEqual(track.disc_number, 2)

    def test_repeated_identifier_requires_unique_position(self) -> None:
        track = TrackInfo(self.root / "track.flac", title="Original", track_number=99)
        release = {
            "tracks_by_disc_and_position": {
                (1, 2): {"isrc": "KNOWN", "title": "Original"},
                (2, 4): {"isrc": "KNOWN", "title": "Original"},
            }
        }
        self.assertIsNone(
            matching._match_position_by_authoritative_id(
                track, 99, "isrc", "KNOWN", [release]
            )
        )
        track.disc_number = 2
        self.assertEqual(
            matching._match_position_by_authoritative_id(
                track, 4, "isrc", "KNOWN", [release]
            ),
            4,
        )

    def test_wrong_identifier_does_not_heal_from_title(self) -> None:
        track = TrackInfo(self.root / "track.flac", title="Original", track_number=2)
        release = {
            "tracks_by_disc_and_position": {
                (1, 2): {"itunes_trackid": "different", "title": "Original"}
            }
        }
        self.assertIsNone(
            matching._match_position_by_authoritative_id(
                track, 2, "itunes_trackid", "KNOWN", [release]
            )
        )

    def test_multiple_musicbrainz_isrcs_preserve_valid_existing_code(self) -> None:
        track = TrackInfo(self.root / "track.flac", isrc="SECOND")
        recording = {"title": "Original", "isrc": "FIRST", "isrcs": ["FIRST", "SECOND"]}
        metadata_consensus.apply_metadata_mapping(
            track, recording, {"isrc": "isrc"}, force=True, source="MusicBrainz"
        )
        self.assertEqual(track.isrc, "SECOND")
        self.assertTrue(matching._recording_isrc_matches("second", recording))
        self.assertFalse(matching._recording_isrc_matches("UNKNOWN", recording))
        self.assertFalse(matching._recording_isrc_matches(None, recording))
        self.assertFalse(matching._recording_isrc_matches("SECOND", {"isrcs": []}))
        self.assertEqual(
            matching._match_position_by_authoritative_id(
                track,
                None,
                "isrc",
                "SECOND",
                [{"tracks_by_disc_and_position": {(1, 3): recording}}],
            ),
            3,
        )

    def test_plain_diff_report_handles_add_remove_and_lyrics(self) -> None:
        original = TrackInfo(self.root / "track.flac", genre="Pop", composer="Composer")
        updated = dataclasses.replace(
            original,
            genre="Rock",
            composer=None,
            label="Label",
            lyrics="private lyric content",
        )
        descriptions = track_processing_module._describe_tag_diffs(
            track_processing_module._compute_tag_diffs(original, updated)
        )
        self.assertIn("genre: 'Pop' -> 'Rock'", descriptions)
        self.assertIn("composer: removed 'Composer'", descriptions)
        self.assertIn("label: added 'Label'", descriptions)
        self.assertTrue(
            any(
                "lyrics:" in description and "characters" in description
                for description in descriptions
            )
        )
        self.assertFalse(
            any("private lyric content" in description for description in descriptions)
        )
        self.assertEqual(
            track_processing_module._describe_tag_diffs(
                track_processing_module._compute_tag_diffs(original, original)
            ),
            [],
        )

    def test_album_verified_isrc_heals_corrupt_identity(self) -> None:
        recording = {
            "title": "Original Song",
            "artist": "Alpha",
            "isrc": "FIRST",
            "isrcs": ["FIRST", "SECOND"],
        }
        for isrc, expected_title in (
            ("SECOND", "Original Song"),
            ("UNKNOWN", "Original Song"),
            (None, "Original Song"),
        ):
            with (
                self.subTest(isrc=isrc),
                patch.object(
                    artist_resolution, "resolve_artist_name", return_value="Alpha"
                ),
            ):
                track = TrackInfo(
                    self.root / "track.flac",
                    artist="Unknown Artist",
                    title="Untitled",
                    isrc=isrc,
                    track_number=98,
                )
                musicbrainz_enrichment_module._apply_mb_recording_metadata(
                    track, recording, True, None, False
                )
                self.assertEqual(track.title, expected_title)
                self.assertEqual(track.track_number, 98)
                if isrc == "SECOND":
                    self.assertEqual(track.artist, "Alpha")
                    self.assertEqual(track.isrc, "SECOND")

    def test_dsp_write_preserves_multi_value_text_and_aliases(self) -> None:
        audio_path = self.audio("multi.flac")
        original = {
            "ARTIST": ["Alpha", "Beta"],
            "ALBUMARTIST": ["Alpha", "Beta"],
            "ALBUMARTISTS": ["Alpha", "Beta"],
            "TITLE": ["Song"],
            "ALBUM": ["Album"],
            "GENRE": ["Pop", "Rock"],
            "COMPOSER": ["Composer One", "Composer Two"],
            "PUBLISHER": ["Indie", "Major"],
        }
        with taglib.File(str(audio_path)) as audio_tags:
            audio_tags.tags.update(original)
            self.assertFalse(audio_tags.save())
        track = read_track_metadata(audio_path)
        self.assertEqual(track.featured_artists, "Beta")
        track.bpm = 120.0
        write_track_metadata(track)
        with taglib.File(str(audio_path)) as audio_tags:
            for tag_key in (
                "ARTIST",
                "ALBUMARTIST",
                "ALBUMARTISTS",
                "GENRE",
                "COMPOSER",
            ):
                self.assertEqual(audio_tags.tags[tag_key], original[tag_key])
            self.assertEqual(audio_tags.tags["LABEL"], original["PUBLISHER"])
            self.assertNotIn("PUBLISHER", audio_tags.tags)
            self.assertEqual(audio_tags.tags["BPM"], ["120.0"])

    def test_multi_value_preservation_allows_explicit_identity_change(self) -> None:
        from sonora.audio.metadata import _preserve_unchanged_multivalues

        original = {
            "ARTIST": ["Alpha", "Beta"],
            "GENRE": ["Pop", "Rock"],
            "CUSTOM": ["One", "Two"],
        }
        updated = {"ARTIST": ["Gamma"], "GENRE": ["Metal"], "CUSTOM": ["One", "Two"]}
        _preserve_unchanged_multivalues(original, updated)
        self.assertEqual(updated["ARTIST"], ["Gamma"])
        self.assertEqual(updated["GENRE"], ["Metal"])
        self.assertEqual(updated["CUSTOM"], ["One", "Two"])
        _preserve_unchanged_multivalues({}, {})

    def test_partial_credit_responses_do_not_downgrade_existing_credits(self) -> None:
        track = TrackInfo(
            self.root / "track.flac",
            label="Indie / Major Records",
            producers="Producer One, Producer Two",
            copyright="2024 Major Records",
        )
        metadata_consensus.apply_metadata_mapping(
            track,
            {
                "label": "Major Records",
                "producers": "Producer One",
                "copyright": "Major Records",
            },
            {"label": "label", "producers": "producers", "copyright": "copyright"},
            force=True,
            source="MusicBrainz",
        )
        self.assertEqual(track.label, "Indie / Major Records")
        self.assertEqual(track.producers, "Producer One, Producer Two")
        self.assertEqual(track.copyright, "2024 Major Records")
        self.assertNotIn("label", track.metadata_sources)
        blank = TrackInfo(track.file_path)
        metadata_consensus.apply_metadata_mapping(
            blank, {"label": "Major Records"}, {"label": "label"}, force=True
        )
        self.assertEqual(blank.label, "Major Records")
        metadata_consensus.apply_metadata_mapping(
            blank, {"label": "Different Records"}, {"label": "label"}, force=True
        )
        self.assertEqual(blank.label, "Different Records")

    def test_genre_aliases_and_noise_preserve_secondary_genres_on_dsp_write(
        self,
    ) -> None:
        for genres, expected in (
            (["RNB", "Rock"], ["R&B/Soul", "Rock"]),
            (["Other", "RNB", "Rock"], ["R&B/Soul", "Rock"]),
            (["Other", "Rock"], ["Rock"]),
        ):
            with self.subTest(genres=genres):
                audio_path = self.audio("genre-alias.flac")
                with taglib.File(str(audio_path)) as audio_tags:
                    audio_tags.tags["GENRE"] = genres
                    self.assertFalse(audio_tags.save())
                clear_metadata_cache()
                track = read_track_metadata(audio_path)
                self.assertEqual(track.genre, expected[0])
                track.bpm = 120.0
                write_track_metadata(track)
                with taglib.File(str(audio_path)) as audio_tags:
                    self.assertEqual(audio_tags.tags["GENRE"], expected)


class TestProviderDurationEvidence(IsolatedTestCase):
    def test_duration_conflict_is_checked_without_artist_or_title_changes(self) -> None:
        for duration, accepted in (
            (180.0, True),
            (182.0, True),
            (190.0, False),
            (60.0, False),
            (None, True),
        ):
            with self.subTest(duration=duration):
                recording = TrackInfo(
                    Path("recording.flac"),
                    artist="Alpha",
                    title="Original Song",
                    duration=180,
                )
                proposal = dataclasses.replace(
                    recording,
                    duration=duration,
                    genre="Rock",
                    metadata_sources={"genre": "iTunes"},
                )
                metadata_consensus.merge_provider_proposals(
                    recording,
                    [build_provider_proposal(recording, proposal)],
                )
                self.assertEqual(recording.genre, "Rock" if accepted else None)
                self.assertEqual(
                    "identity" in recording.metadata_conflicts, not accepted
                )
                self.assertEqual(recording.duration, 180)
