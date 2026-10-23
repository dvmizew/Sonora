import json
from pathlib import Path
from unittest.mock import patch

from tests.support import IsolatedTestCase
from tools.evaluate_matching import MatchCase, evaluate_cases


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
