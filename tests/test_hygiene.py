import dataclasses
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from sonora.core.config import (
    get_balanced_feat_pattern,
    get_bracket_feat_pattern,
    get_config,
)
from sonora.core.models import TrackInfo
from sonora.core.text import normalize_language_name
from sonora.core.utils import retry_network_call
from sonora.modules.checker import check_file
from sonora.modules.track_processing import (
    _compute_tag_diffs,
    _describe_tag_diffs,
    _render_tag_diffs,
)
from sonora.services.itunes import fetch_itunes_cover_art_url
from tests.support import IsolatedTestCase


class TestCodeHygiene(IsolatedTestCase):
    def test_feature_patterns_accept_the_same_supported_tokens(self) -> None:
        for token in ("feat.", "ft", "w/", "w."):
            with self.subTest(token=token):
                self.assertIsNotNone(
                    get_balanced_feat_pattern().match(f"{token} Guest")
                )
                self.assertIsNotNone(
                    get_bracket_feat_pattern().search(f" ({token} Guest)")
                )

    def test_feature_patterns_preserve_non_feature_conjunctions(self) -> None:
        for token in ("and", "with", "vs", "w/o", "w/out"):
            with self.subTest(token=token):
                self.assertIsNone(get_balanced_feat_pattern().match(f"{token} Guest"))
                self.assertIsNone(
                    get_bracket_feat_pattern().search(f" ({token} Guest)")
                )

    def test_feature_patterns_share_custom_configuration(self) -> None:
        custom_config = dataclasses.replace(
            get_config(), featuring_conjunctions=("avec",)
        )
        with patch("sonora.core.config.get_config", return_value=custom_config):
            self.assertIsNotNone(get_balanced_feat_pattern().match("avec Guest"))
            self.assertIsNotNone(get_bracket_feat_pattern().search(" [avec Guest]"))
            self.assertIsNone(get_balanced_feat_pattern().match("feat. Guest"))
            self.assertIsNone(get_bracket_feat_pattern().search(" [feat. Guest]"))

    def test_itunes_exact_album_precedes_earlier_fuzzy_candidate(self) -> None:
        album_candidates = [
            {
                "artistName": "Artist Alpha",
                "collectionName": "Test Record Deluxe",
                "artworkUrl100": "https://example.com/fuzzy/100x100bb.jpg",
            },
            {
                "artistName": "Artist Alpha",
                "collectionName": "Test Record",
                "artworkUrl100": "https://example.com/exact/100x100bb.jpg",
            },
        ]
        with patch(
            "sonora.services.itunes.search_itunes", return_value=album_candidates
        ):
            self.assertEqual(
                fetch_itunes_cover_art_url("Artist Alpha", "Test Record"),
                "https://example.com/exact/1400x1400bb.jpg",
            )

    def test_itunes_rejects_wrong_artist_and_different_album_series(self) -> None:
        album_candidates = [
            {
                "artistName": "Unrelated Performer",
                "collectionName": "Test Record II",
                "artworkUrl100": "https://example.com/other/100x100bb.jpg",
            },
            {
                "artistName": "Artist Alpha",
                "collectionName": "Test Record III",
                "artworkUrl100": "https://example.com/series/100x100bb.jpg",
            },
        ]
        with patch(
            "sonora.services.itunes.search_itunes", return_value=album_candidates
        ):
            self.assertIsNone(
                fetch_itunes_cover_art_url("Artist Alpha", "Test Record II")
            )

    def test_retry_rejects_empty_attempt_budget_before_calling(self) -> None:
        for attempts in (0, -1):
            with self.subTest(attempts=attempts):
                provider_request = Mock(return_value="metadata")
                with self.assertRaisesRegex(ValueError, "max_retries must be positive"):
                    retry_network_call(provider_request, max_retries=attempts)
                provider_request.assert_not_called()

    def test_plain_and_rich_diffs_preserve_changes_and_hide_lyric_content(self) -> None:
        original = TrackInfo(Path("track.flac"), composer="Original composer")
        updated = dataclasses.replace(
            original, composer=None, label="[bold]", lyrics="private lyric content"
        )
        differences = _compute_tag_diffs(original, updated)
        descriptions = _describe_tag_diffs(differences)
        rendered = "".join(_render_tag_diffs(differences))
        self.assertIn("composer: removed 'Original composer'", descriptions)
        self.assertIn("label: added '[bold]'", descriptions)
        self.assertIn(r"\[bold]", rendered)
        self.assertIn("21 characters", rendered)
        self.assertNotIn("private lyric content", rendered)
        self.assertNotIn("private lyric content", "".join(descriptions))
        self.assertEqual(_describe_tag_diffs([]), [])
        self.assertEqual(_render_tag_diffs([]), [])

    def test_iso_language_codes_take_precedence_over_matching_names(self) -> None:
        for language_code, language_name in (
            ("en", "English"),
            ("EN", "English"),
            ("eng", "English"),
            ("fre", "French"),
            ("ger", "German"),
        ):
            with self.subTest(language_code=language_code):
                self.assertEqual(normalize_language_name(language_code), language_name)
        self.assertIsNone(normalize_language_name(None))
        self.assertEqual(normalize_language_name("unknown"), "unknown")


class TestCheckerMetadataFailures(IsolatedTestCase):
    def setUp(self) -> None:
        fixture_directory = tempfile.TemporaryDirectory()
        self.addCleanup(fixture_directory.cleanup)
        self.audio_path = Path(fixture_directory.name) / "01 - Song.wav"
        self.audio_path.write_bytes(b"unreadable audio fixture")

    def test_metadata_failure_still_reports_missing_lyrics(self) -> None:
        with patch(
            "sonora.modules.checker.read_track_metadata",
            side_effect=OSError("unreadable metadata"),
        ):
            issues = check_file(self.audio_path)
        self.assertIn("Metadata read error: unreadable metadata", issues)
        self.assertIn("Missing synchronized lyrics (.lrc) file.", issues)

    def test_metadata_failure_still_validates_existing_lyrics(self) -> None:
        lyrics_path = self.audio_path.with_suffix(".lrc")
        with patch(
            "sonora.modules.checker.read_track_metadata",
            side_effect=OSError("unreadable metadata"),
        ):
            lyrics_path.write_text("[00:01.00] A lyric line", encoding="utf-8")
            issues = check_file(self.audio_path)
            self.assertIn("Metadata read error: unreadable metadata", issues)
            self.assertNotIn("Missing synchronized lyrics (.lrc) file.", issues)
            self.assertNotIn("Corrupt 0-byte synchronized lyrics (.lrc) file.", issues)
            lyrics_path.write_bytes(b"")
            issues = check_file(self.audio_path)
            self.assertIn("Corrupt 0-byte synchronized lyrics (.lrc) file.", issues)

    def test_supplied_metadata_avoids_reopening_unreadable_audio(self) -> None:
        supplied_metadata = TrackInfo(
            self.audio_path,
            artist="Artist",
            title="Song",
            album="Album",
            track_number=1,
        )
        self.audio_path.with_suffix(".lrc").write_text(
            "[00:01.00] A lyric line", encoding="utf-8"
        )
        with patch(
            "sonora.modules.checker.read_track_metadata",
            side_effect=OSError("unreadable metadata"),
        ) as metadata_reader:
            issues = check_file(self.audio_path, track_metadata=supplied_metadata)
        metadata_reader.assert_not_called()
        self.assertFalse(
            any(issue.startswith("Metadata read error:") for issue in issues)
        )
        self.assertNotIn("Missing synchronized lyrics (.lrc) file.", issues)
