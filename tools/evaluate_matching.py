"""Evaluate pairwise identity policy; this is not a real-library accuracy benchmark.

Run after installing Sonora: python -m tools.evaluate_matching --out report.json
"""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from sonora.core.config import get_config
from sonora.core.titles import recording_identity_matches


@dataclass(frozen=True)
class MatchCase:
    name: str
    expected_artist: str
    expected_title: str
    candidate_artist: str
    candidate_title: str
    expected_accept: bool
    reason: str
    expected_duration: float | None = None
    candidate_duration: float | None = None


class MatchOutcome(TypedDict):
    name: str
    expected_accept: bool
    accepted: bool
    reason: str


def evaluate_cases(cases: list[MatchCase]) -> dict[str, Any]:
    started = time.perf_counter()
    outcomes: list[MatchOutcome] = [
        {
            "name": case.name,
            "expected_accept": case.expected_accept,
            "accepted": recording_identity_matches(
                case.expected_artist,
                case.expected_title,
                case.candidate_artist,
                case.candidate_title,
                expected_duration=case.expected_duration,
                candidate_duration=case.candidate_duration,
            ),
            "reason": case.reason,
        }
        for case in cases
    ]
    true_accepts = sum(
        outcome["accepted"] and outcome["expected_accept"] for outcome in outcomes
    )
    false_accepts = sum(
        outcome["accepted"] and not outcome["expected_accept"] for outcome in outcomes
    )
    false_rejects = sum(
        not outcome["accepted"] and outcome["expected_accept"] for outcome in outcomes
    )
    return {
        "cases": len(cases),
        "true_accepts": true_accepts,
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "true_rejects": len(cases) - true_accepts - false_accepts - false_rejects,
        "unresolved": sum(not outcome["accepted"] for outcome in outcomes),
        "precision": true_accepts / (true_accepts + false_accepts)
        if true_accepts + false_accepts
        else None,
        "recall": true_accepts / (true_accepts + false_rejects)
        if true_accepts + false_rejects
        else None,
        "evaluation_seconds": time.perf_counter() - started,
        "outcomes": outcomes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "tests/fixtures/matching_cases.json",
    )
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args()
    corpus_bytes = arguments.cases.read_bytes()
    corpus = json.loads(corpus_bytes)
    cases = [MatchCase(**case) for case in corpus["cases"]]
    report = evaluate_cases(cases)
    config = get_config()
    report.update(
        {
            "scope": corpus["scope"],
            "policy_sources": corpus["policy_sources"],
            "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
            "python": sys.version,
            "policy": {
                name: getattr(config, name)
                for name in (
                    "artist_match_threshold",
                    "title_match_threshold",
                    "short_title_match_threshold",
                    "short_title_length",
                    "single_duration_tolerance_seconds",
                    "remix_match_threshold",
                    "version_equivalence_penalty",
                    "version_mismatch_penalty",
                    "remix_modifier_mismatch_penalty",
                    "title_score_weight",
                    "featuring_conjunctions",
                )
            },
            "peak_process_rss_bytes": None,
            "rss_scope": "Whole process high-water mark, including imports; not incremental matching memory.",
        }
    )
    if sys.platform in {"linux", "darwin"}:
        import resource

        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["peak_process_rss_bytes"] = int(
            peak_rss * 1024 if sys.platform == "linux" else peak_rss
        )
    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if arguments.out:
        arguments.out.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")
    return int(bool(report["false_accepts"] or report["false_rejects"]))


if __name__ == "__main__":
    raise SystemExit(main())
