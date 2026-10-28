from pathlib import Path

from mutagen.flac import FLAC

from sonora.audio.metadata import read_track_metadata, write_track_metadata
from sonora.modules.track_processing import _compute_tag_diffs, normalize_single_track
from tests.regression_support import AudioRegressionTestCase


class TestNumberingRepairSafety(AudioRegressionTestCase):
    def test_verified_same_total_reconciles_conflicting_inline_encoding(self) -> None:
        path = self.recording(
            "01 - Song.flac",
            {"TRACKNUMBER": ["1/14"], "TRACKTOTAL": ["17"], "TOTALTRACKS": ["17"]},
        )
        original = read_track_metadata(path)
        verified = original.copy()
        self.assertEqual(verified.total_tracks, 17)
        self.assertFalse(_compute_tag_diffs(original, verified))
        verified.metadata_sources["total_tracks"] = "iTunes"
        self.assertTrue(
            any(
                field == "numbering"
                for field, _, _ in _compute_tag_diffs(original, verified)
            )
        )
        write_track_metadata(verified)
        self.assertEqual(FLAC(path)["TRACKNUMBER"], ["1/17"])

    def recording(self, name: str, tags: dict[str, list[str]]) -> Path:
        path = self.audio(name)
        audio = FLAC(path)
        audio.update({"ARTIST": ["Performer"], "ALBUM": ["Album"], "TITLE": ["Song"]})
        audio.update(tags)
        audio.save()
        return path

    def test_corroborating_totals_repair_stale_alias_without_force(self) -> None:
        path = self.recording(
            "11 - Song.flac",
            {
                "TRACKNUMBER": ["11/17"],
                "TRACKTOTAL": ["9"],
                "TOTALTRACKS": ["17"],
                "USER_DATA": ["one", "two"],
            },
        )
        original_digest = FLAC(path).info.md5_signature
        recording = read_track_metadata(path)
        self.assertEqual(recording.total_tracks, 17)
        self.assertTrue(recording._numbering_needs_normalization)
        self.assertNotIn("_numbering_needs_normalization", recording.to_dict())
        result = normalize_single_track(path)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(any("numbering" in diff for diff in result._diff_descriptions))
        audio = FLAC(path)
        self.assertEqual(audio["TRACKNUMBER"], ["11/17"])
        self.assertEqual(audio["TRACKTOTAL"], ["17"])
        self.assertEqual(audio["TOTALTRACKS"], ["17"])
        self.assertEqual(audio["USER_DATA"], ["one", "two"])
        self.assertEqual(audio.info.md5_signature, original_digest)
        repeated = normalize_single_track(path)
        assert repeated is not None
        self.assertEqual(repeated._diff_descriptions, [])

    def test_impossible_majority_does_not_override_only_valid_stored_total(
        self,
    ) -> None:
        path = self.recording(
            "11 - Song.flac",
            {"TRACKNUMBER": ["11/9"], "TRACKTOTAL": ["9"], "TOTALTRACKS": ["17"]},
        )
        recording = read_track_metadata(path)
        self.assertEqual(recording.total_tracks, 17)
        self.assertTrue(recording._numbering_needs_normalization)

    def test_ambiguous_totals_survive_unrelated_forced_normalization(self) -> None:
        tags = {
            "TRACKNUMBER": ["1/8"],
            "TRACKTOTAL": ["10"],
            "TOTALTRACKS": ["12"],
        }
        path = self.recording("01 - Song.flac", tags)
        recording = read_track_metadata(path)
        self.assertIn("album_sequence", recording.metadata_conflicts)
        self.assertFalse(recording._numbering_needs_normalization)
        normalize_single_track(path, force=True)
        audio = FLAC(path)
        for key, value in tags.items():
            self.assertEqual(audio[key], value)
        repeated = normalize_single_track(path)
        assert repeated is not None
        self.assertEqual(repeated._diff_descriptions, [])
        self.assertIn("album_sequence", repeated.metadata_conflicts)

    def test_redundant_aliases_cannot_outvote_another_plausible_edition(self) -> None:
        path = self.recording(
            "01 - Song.flac",
            {"TRACKNUMBER": ["1/17"], "TRACKTOTAL": ["9"], "TOTALTRACKS": ["17"]},
        )
        recording = read_track_metadata(path)
        self.assertEqual(recording.total_tracks, 9)
        self.assertFalse(recording._numbering_needs_normalization)
        self.assertIn("album_sequence", recording.metadata_conflicts)

    def test_consistently_impossible_count_requires_release_evidence(self) -> None:
        tags = {
            "TRACKNUMBER": ["13/12"],
            "TRACKTOTAL": ["12"],
            "TOTALTRACKS": ["12"],
        }
        path = self.recording("13 - Song.flac", tags)
        normalize_single_track(path, force=True)
        recording = read_track_metadata(path)
        self.assertEqual((recording.track_number, recording.total_tracks), (13, 12))
        self.assertIn("album_sequence", recording.metadata_conflicts)
        repeated = normalize_single_track(path)
        assert repeated is not None
        self.assertEqual(repeated._diff_descriptions, [])

    def test_explicit_disc_one_is_not_replaced_by_filename_disc_two(self) -> None:
        path = self.recording(
            "2-01 - Song.flac", {"TRACKNUMBER": ["1/10"], "DISCNUMBER": ["1/1"]}
        )
        recording = read_track_metadata(path)
        self.assertEqual((recording.disc_number, recording.total_discs), (1, 1))
        normalize_single_track(path, force=True)
        self.assertEqual(FLAC(path)["DISCNUMBER"], ["1/1"])

    def test_nonpositive_and_nonfinite_bpm_are_unknown_and_not_written(self) -> None:
        for value in ("0.0", "-1", "nan", "inf"):
            with self.subTest(value=value):
                path = self.recording(f"{value}.flac", {"BPM": [value]})
                recording = read_track_metadata(path)
                self.assertIsNone(recording.bpm)
                recording.bpm = float(value)
                write_track_metadata(recording)
                self.assertNotIn("BPM", FLAC(path))
