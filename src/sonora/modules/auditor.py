import re
from pathlib import Path

import orjson

from sonora.audio.checksum import verify_flac_checksum
from sonora.audio.metadata import read_track_metadata
from sonora.audio.spectral import is_fake_lossless
from sonora.core.constants import (
    FEAT_KEYWORDS,
    GENRE_BLACKLIST,
    PROTECTED_ARTISTS,
    SUPPORTED_EXTS,
)
from sonora.core.logger import LOG
from sonora.core.models import AuditReport
from sonora.core.utils import normalize_str

BRACKET_PATTERN = re.compile(r"\[.*?\]|\(.*?\)")
FEAT_PATTERN = re.compile(FEAT_KEYWORDS, re.IGNORECASE)


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
        except (OSError, ValueError, RuntimeError) as e:
            issues.append(f"Checksum check failed: {e}")
        if check_spectral:
            try:
                if is_fake_lossless(file_path):
                    issues.append("Possible fake lossless (spectral cutoff below 16kHz).")
            except (OSError, ValueError, RuntimeError) as e:
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
        if track.album == "Unknown Album":
            issues.append("Missing ALBUM tag.")
        if not track.album_artist:
            issues.append("Missing ALBUMARTIST tag (Risk of Split Album).")
        if track.track_number is None:
            issues.append("Missing TRACKNUMBER tag.")
        if not track.date:
            issues.append("Missing DATE (Year) tag.")
        if track.bpm is None:
            issues.append("Missing BPM tag.")
        if track.replaygain_track_gain is None:
            issues.append("Missing REPLAYGAIN_TRACK_GAIN tag.")
        if track.replaygain_track_peak is None:
            issues.append("Missing REPLAYGAIN_TRACK_PEAK tag.")
        if not track.musicbrainz_trackid:
            issues.append("Missing MUSICBRAINZ_TRACKID tag.")
        elif not re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", track.musicbrainz_trackid.strip(), re.IGNORECASE):
            issues.append(f"Invalid UUID format in MUSICBRAINZ_TRACKID: '{track.musicbrainz_trackid}'")

        if not track.musicbrainz_albumid:
            issues.append("Missing MUSICBRAINZ_ALBUMID tag.")
        elif not re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", track.musicbrainz_albumid.strip(), re.IGNORECASE):
            issues.append(f"Invalid UUID format in MUSICBRAINZ_ALBUMID: '{track.musicbrainz_albumid}'")

        if track.art_width and (track.art_width < 500 or (track.art_height and track.art_height < 500)):
            issues.append(f"Low resolution cover art: {track.art_width}x{track.art_height}")

        if not re.match(r'^\d{2,3}\s*[-._ ]\s*', file_path.name):
            issues.append(f"Filename does not start with track number: '{file_path.name}'")

        if FEAT_PATTERN.search(track.artist):
            issues.append(f"ARTIST entry '{track.artist}' contains 'feat' info (Rule: TITLE only)")
        
        # Check for unsplit artists (e.g. Artist A & Artist B)
        delimiters = [r"\s&\s", r"\s×\s", r"\sfeat\.?\s", r"\sft\.?\s"]
        is_protected = any(p.lower() in track.artist.lower() for p in PROTECTED_ARTISTS)
        if not is_protected:
            for d in delimiters:
                if re.search(d, track.artist, re.IGNORECASE):
                    issues.append(f"ARTIST tag seems unsplit: '{track.artist}' (Contains delimiter '{d.strip()}')")
        
        # Check Title feature duplicate markers
        feat_matches = re.findall(rf"[\(\[]\s*({FEAT_KEYWORDS})", track.title, re.IGNORECASE)
        if len(feat_matches) > 1:
            issues.append(f"Duplicate featuring markers detected in TITLE ({len(feat_matches)} markers found)")

        # Sync check filename vs title feat
        if FEAT_PATTERN.search(file_path.name) and not FEAT_PATTERN.search(track.title):
            issues.append("Filename contains 'feat' but TITLE tag does not")
        
        if track.sample_rate and track.sample_rate < 44100:
            issues.append(f"Sub-standard sample rate: {track.sample_rate}Hz")
        if track.bitrate and track.bitrate < 320000 and not track.is_lossless:
            issues.append(f"Sub-standard lossy bitrate: {round(track.bitrate / 1000)} kbps (Recommended: 320 kbps)")

    except (OSError, ValueError, RuntimeError) as e:
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
        raise FileNotFoundError(f"Directory not found: {folder_path}")

    report = AuditReport(total_files=0, corrupt_files=0, missing_metadata=0, missing_lrc=0)

    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    from sonora.core.logger import CONSOLE

    files_to_process = [p for p in folder_path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=CONSOLE
    ) as progress:
        task = progress.add_task("[cyan]Auditing library...", total=len(files_to_process))
        
        # Group by folder
        from collections import defaultdict
        folders = defaultdict(list)
        for path in files_to_process:
            folders[path.parent].append(path)
            
        for folder, paths in folders.items():
            album_names = set()
            album_artists = set()
            tracks_found = defaultdict(list)
            
            for path in paths:
                LOG.info(f"Auditing: {path.name}")
                report.total_files += 1
                
                # We need some info from the file for folder-level checks
                try:
                    track_info = read_track_metadata(path)
                    if track_info.album != "Unknown Album":
                        album_names.add(track_info.album)
                    if track_info.album_artist:
                        album_artists.add(track_info.album_artist)
                    
                    disc_no = track_info.disc_number or 1
                    track_no = track_info.track_number
                    if track_no is not None:
                        tracks_found[(disc_no, track_no)].append(path.name)
                        
                except (OSError, ValueError, RuntimeError) as e:
                    LOG.debug(f"Could not read metadata for folder check: {e}")

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
                
            # Folder-level checks
            folder_issues = []
            if len(album_names) > 1:
                folder_issues.append(f"Inconsistent ALBUM name in folder: {album_names}")
            if len(album_artists) > 1:
                folder_issues.append(f"Inconsistent ALBUMARTIST in folder: {album_artists}")
                
            for (dn, tn), f_list in tracks_found.items():
                if len(f_list) > 1:
                    folder_issues.append(f"Duplicate track number {tn} (Disc {dn}) found in files: {f_list}")
            
            discs = defaultdict(list)
            for (dn, tn) in tracks_found:
                discs[dn].append(tn)
            for dn, tns in discs.items():
                tns.sort()
                if tns:
                    max_t = max(tns)
                    missing = [t for t in range(1, max_t + 1) if t not in tns]
                    if missing:
                        folder_issues.append(f"Missing track numbers in sequence for Disc {dn}: {missing}")
            
            if folder_issues:
                report.issues[str(folder)] = folder_issues

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
        output_json.write_bytes(
            orjson.dumps(
                data,
                option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_DATACLASS,
            )
        )

    return report
