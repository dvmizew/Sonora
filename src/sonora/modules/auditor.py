import json
import re
from pathlib import Path

from sonora.audio.checksum import verify_flac_checksum
from sonora.audio.metadata import read_track_metadata
from sonora.audio.spectral import is_fake_lossless
from sonora.core.constants import GENRE_BLACKLIST, SUPPORTED_EXTS
from sonora.core.exceptions import AudioProcessingError, MetadataError
from sonora.core.logger import LOG
from sonora.core.models import AuditReport
from sonora.core.utils import normalize_str

BRACKET_PATTERN = re.compile(r"\[.*?\]|\(.*?\)")


def check_brackets_corruption(name: str) -> list[str]:
    """
    Check if a filename or tag contains corrupt/unwanted bracket metadata
    (e.g., [FLAC], (Official Video), [HQ]).
    """
    issues = []
    for match in BRACKET_PATTERN.findall(name):
        lowered = normalize_str(match)
        if any(kw in lowered for kw in ["official", "video", "audio", "flac", "mp3", "320", "hq", "hd", "rip"]):
            issues.append(f"Corrupt bracket metadata: '{match}'")
    return issues


def audit_file(file_path: Path, check_spectral: bool = False) -> list[str]:
    issues: list[str] = []
    if not file_path.exists():
        return [f"File not found: {file_path}"]

    if file_path.suffix.lower() not in SUPPORTED_EXTS:
        return []
    if file_path.suffix.lower() == ".flac":
        try:
            if not verify_flac_checksum(file_path):
                issues.append("FLAC audio stream MD5 checksum verification failed (corrupted FLAC).")
        except AudioProcessingError as e:
            issues.append(f"Checksum check failed: {e}")
        if check_spectral:
            try:
                if is_fake_lossless(file_path):
                    issues.append("Possible fake lossless (spectral cutoff below 16kHz).")
            except AudioProcessingError as e:
                LOG.debug(f"Spectral analysis failed for {file_path}: {e}")
    try:
        track = read_track_metadata(file_path)
        issues.extend(check_brackets_corruption(track.artist))
        issues.extend(check_brackets_corruption(track.title))

        if track.genre and any(normalize_str(bl) in normalize_str(track.genre) for bl in GENRE_BLACKLIST):
            issues.append(f"Blacklisted genre tag: '{track.genre}'")
        if track.artist == "Unknown Artist":
            issues.append("Missing ARTIST tag.")
        if track.title == "Unknown Title":
            issues.append("Missing TITLE tag.")

    except MetadataError as e:
        issues.append(f"Metadata read error: {e}")
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
    and generate an AuditReport dataclass.
    """
    if not folder_path.exists():
        raise AudioProcessingError(f"Directory not found: {folder_path}")

    report = AuditReport(total_files=0, corrupt_files=0, missing_metadata=0, missing_lrc=0)

    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeRemainingColumn,
    )
    from sonora.core.logger import CONSOLE

    files_to_process = [p for p in folder_path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=CONSOLE
    ) as progress:
        task = progress.add_task("[cyan]Auditing library...", total=len(files_to_process))
        for path in files_to_process:
            LOG.info(f"Auditing: {path.name}")
            report.total_files += 1
            file_issues = audit_file(path, check_spectral=check_spectral)

            if file_issues:
                report.issues[str(path)] = file_issues
                if any("corrupt" in normalize_str(issue) or "checksum" in normalize_str(issue) for issue in file_issues):
                    report.corrupt_files += 1
                if any("missing" in normalize_str(issue) and "tag" in normalize_str(issue) for issue in file_issues):
                    report.missing_metadata += 1
                if any("missing" in normalize_str(issue) and "lrc" in normalize_str(issue) for issue in file_issues):
                    report.missing_lrc += 1
            
            progress.advance(task)

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
