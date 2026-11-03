from pathlib import Path
from unittest.mock import patch

from sonora.cli import library, reports
from sonora.core.models import CheckReport, TrackInfo
from tests.support import IsolatedTestCase


class TestReadinessRepairs(IsolatedTestCase):
    def test_check_fails_for_audio_corruption_but_not_optional_metadata(self) -> None:
        for corrupt, expected in ((0, 0), (1, 1)):
            with (
                self.subTest(corrupt=corrupt),
                patch.object(
                    library,
                    "check_library",
                    return_value=CheckReport(
                        total_files=1, corrupt_files=corrupt, missing_metadata=1
                    ),
                ),
            ):
                self.assertEqual(library.check(Path("library")), expected)

    def test_bpm_coverage_counts_only_finite_positive_tempos(self) -> None:
        tracks = [
            TrackInfo(Path(f"{index}.flac"), bpm=value)
            for index, value in enumerate(
                (None, 0.0, -1.0, float("nan"), float("inf"), 120.0)
            )
        ]
        self.assertEqual(reports._tag_coverage_counts(tracks)["bpm"], 1)
