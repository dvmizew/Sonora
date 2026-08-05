"""
Library auditor module for checking FLAC checksum integrity, bracket corruption,
genre compliance, missing LRC lyrics, and generating audit reports.
"""

import json
import re
from pathlib import Path

from sonora.audio.checksum import verify_flac_checksum
from sonora.audio.metadata import read_track_metadata
from sonora.audio.spectral import is_fake_lossless
from sonora.core.constants import GENRE_BLACKLIST, SUPPORTED_EXTS
from sonora.core.exceptions import AudioProcessingError, MetadataError
from sonora.core.models import AuditReport

BRACKET_PATTERN = re.compile(r"\[.*?\]|\(.*?\)")


def check_brackets_corruption(name: str) -> list[str]:
    """
    Check if a filename or tag contains corrupt/unwanted bracket metadata
    (e.g., [FLAC], (Official Video), [HQ]).
    """
    issues = []
    for match in BRACKET_PATTERN.findall(name):
        lowered = match.lower()
        if any(kw in lowered for kw in ["official", "video", "audio", "flac", "mp3", "320", "hq", "hd", "rip"]):
            issues.append(f"Corrupt bracket metadata: '{match}'")
    return issues


def audit_file(file_path: Path, check_spectral: bool = False) -> list[str]:
    """
    Perform audit checks on a single audio file.
    """
    issues: list[str] = []
    if not file_path.exists():
        return [f"File not found: {file_path}"]

    if file_path.suffix.lower() not in SUPPORTED_EXTS:
        return []

    # Check 1: FLAC stream MD5 integrity
    if file_path.suffix.lower() == ".flac":
        try:
            if not verify_flac_checksum(file_path):
                issues.append("FLAC audio stream MD5 checksum verification failed (corrupted FLAC).")
        except AudioProcessingError as e:
            issues.append(f"Checksum check failed: {e}")

        # Check 2: Fake lossless cutoff (optional, slow)
        if check_spectral:
            try:
                if is_fake_lossless(file_path):
                    issues.append("Possible fake lossless (spectral cutoff below 16kHz).")
            except AudioProcessingError:
                pass

    # Check 3: Read metadata tags
    try:
        track = read_track_metadata(file_path)

        # Check bracket corruption in artist or title
        issues.extend(check_brackets_corruption(track.artist))
        issues.extend(check_brackets_corruption(track.title))

        # Check genre blacklist
        if track.genre and any(bl.lower() in track.genre.lower() for bl in GENRE_BLACKLIST):
            issues.append(f"Blacklisted genre tag: '{track.genre}'")

        # Check missing basic tags
        if track.artist == "Unknown Artist":
            issues.append("Missing ARTIST tag.")
        if track.title == "Unknown Title":
            issues.append("Missing TITLE tag.")

    except MetadataError as e:
        issues.append(f"Metadata read error: {e}")

    # Check 4: Missing .lrc file
    lrc_path = file_path.with_suffix(".lrc")
    if not lrc_path.exists():
        issues.append("Missing synchronized lyrics (.lrc) file.")

    return issues


def audit_library(
    folder_path: Path,
    output_json: Path | None = None,
    check_spectral: bool = False,
) -> AuditReport:
    """
    Scan an entire music library folder recursively, audit all audio files,
    and generate an AuditReport dataclass (optionally writing to output_json).
    """
    if not folder_path.exists():
        raise AudioProcessingError(f"Directory not found: {folder_path}")

    report = AuditReport(total_files=0, corrupt_files=0, missing_metadata=0, missing_lrc=0)

    for path in folder_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            report.total_files += 1
            file_issues = audit_file(path, check_spectral=check_spectral)

            if file_issues:
                report.issues[str(path)] = file_issues
                if any("corrupt" in issue.lower() or "checksum" in issue.lower() for issue in file_issues):
                    report.corrupt_files += 1
                if any("missing" in issue.lower() and "tag" in issue.lower() for issue in file_issues):
                    report.missing_metadata += 1
                if any("missing" in issue.lower() and "lrc" in issue.lower() for issue in file_issues):
                    report.missing_lrc += 1

    if output_json:
        total = report.total_files
        corrupt = report.corrupt_files
        missing_meta = report.missing_metadata
        missing_lrc = report.missing_lrc
        issue_count = len(report.issues)

        llm_summary = (
            f"Sonora audited {total} files in '{folder_path}'. "
            f"Audit status: {corrupt} corrupted files, {missing_meta} missing metadata, "
            f"{missing_lrc} missing LRCs. Total files with issues: {issue_count}."
        )

        data = {
            "schema": "audit_report_v1",
            "generator": "Sonora",
            "llm_summary": llm_summary,
            "target_path": str(folder_path.resolve()),
            "summary": {
                "total_files": total,
                "corrupt_files": corrupt,
                "missing_metadata": missing_meta,
                "missing_lrc": missing_lrc,
                "files_with_issues": issue_count,
            },
            "issues": report.issues,
        }
        output_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    return report
