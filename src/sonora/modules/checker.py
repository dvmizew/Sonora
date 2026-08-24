import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import orjson
from mutagen._util import MutagenError
from mutagen.flac import FLAC, FLACNoHeaderError
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from sonora.audio.checksum import verify_flac_checksum
from sonora.audio.metadata import read_track_metadata
from sonora.audio.spectral import detect_fake_lossless
from sonora.core.constants import (
    FEAT_KEYWORDS,
    SUPPORTED_EXTS,
)
from sonora.core.logger import CONSOLE, LOG
from sonora.core.models import CheckReport
from sonora.core.utils import (
    find_audio_files,
    is_single_group_artist,
    is_valid_uuid,
    normalize_genre,
    normalize_str,
)

FEAT_PATTERN = re.compile(FEAT_KEYWORDS, re.IGNORECASE)

JUNK_BRACKET_KEYWORDS = {
    "official",
    "video",
    "audio",
    "flac",
    "mp3",
    "320",
    "320kbps",
    "hq",
    "hd",
    "rip",
    "cdrip",
    "webrip",
    "lossless",
    "remastered",
}

PAIRS = {"(": ")", "[": "]", "{": "}"}
CLOSING_TO_OPENING = {v: k for k, v in PAIRS.items()}


def extract_bracket_tokens(text: str) -> list[tuple[str, set[str]]]:
    """
    Extracts all bracketed substrings and their constituent alphanumeric word tokens
    using a character stack and pure tokenization.
    """
    results: list[tuple[str, set[str]]] = []
    stack: list[str] = []
    start = -1

    for index, char in enumerate(text):
        if char in PAIRS:
            if not stack:
                start = index
            stack.append(char)
        elif char in CLOSING_TO_OPENING and stack:
            expected_opening = CLOSING_TO_OPENING[char]
            if stack[-1] == expected_opening:
                stack.pop()
                if not stack and start != -1:
                    full_bracket = text[start : index + 1]
                    inner = full_bracket[1:-1].lower()
                    tokens = set(
                        "".join(c if c.isalnum() else " " for c in inner).split()
                    )
                    results.append((full_bracket, tokens))
                    start = -1
    return results


def is_valid_track_filename(filename: str) -> bool:
    """Check if filename starts with a clean track number prefix (e.g. '01 - ...' or '1-01 - ...')."""
    stem = filename.rsplit(".", 1)[0]
    for delim in [" - ", " _ ", ". ", "_", " "]:
        if delim in stem:
            prefix = stem.split(delim, 1)[0].strip()
            parts = prefix.split("-")
            if len(parts) in (1, 2) and all(
                part.isdigit() and 1 <= len(part) <= 4 for part in parts
            ):
                return True
    return False


def check_brackets_corruption(name: str) -> list[str]:
    """
    Check if a filename or tag contains corrupt/unwanted bracket metadata
    (e.g., [FLAC], (Official Video), [HQ]) using set intersection.
    """
    issues = []
    for full_bracket, tokens in extract_bracket_tokens(name):
        if tokens & JUNK_BRACKET_KEYWORDS:
            issues.append(f"Corrupt bracket metadata: '{full_bracket}'")
    return issues


def check_file(file_path: Path, check_spectral: bool = False) -> list[str]:
    issues: list[str] = []
    if not file_path.exists():
        return [f"File not found: {file_path}"]

    if file_path.suffix.lower() not in SUPPORTED_EXTS:
        return []
    if file_path.suffix.lower() == ".flac":
        try:
            if not verify_flac_checksum(file_path):
                issues.append(
                    "FLAC audio stream MD5 checksum verification failed (corrupted FLAC)."
                )
        except (OSError, ValueError, RuntimeError) as error:
            issues.append(f"Checksum check failed: {error}")

        try:
            audio_flac = FLAC(str(file_path))
            for index, picture in enumerate(audio_flac.pictures):
                if len(picture.data) == 0:
                    issues.append(f"Corrupt 0-byte picture block at index {index}.")
        except (
            OSError,
            ValueError,
            RuntimeError,
            AttributeError,
            KeyError,
            TypeError,
            FLACNoHeaderError,
            MutagenError,
        ) as error:
            LOG.debug(f"Mutagen picture check skipped for {file_path}: {error}")
        if check_spectral:
            try:
                is_fake, _, description = detect_fake_lossless(file_path)
                if is_fake:
                    issues.append(
                        description
                        or "Possible fake lossless (spectral cutoff below 16kHz)."
                    )
            except (OSError, ValueError, RuntimeError) as error:
                LOG.debug(f"Spectral analysis failed for {file_path}: {error}")
    try:
        track = read_track_metadata(file_path)
        issues.extend(check_brackets_corruption(track.artist))
        issues.extend(check_brackets_corruption(track.title))

        if track.genre and not normalize_genre(track.genre):
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
        elif not is_valid_uuid(track.musicbrainz_trackid):
            issues.append(
                f"Invalid UUID format in MUSICBRAINZ_TRACKID: '{track.musicbrainz_trackid}'"
            )

        if not track.musicbrainz_albumid:
            issues.append("Missing MUSICBRAINZ_ALBUMID tag.")
        elif not is_valid_uuid(track.musicbrainz_albumid):
            issues.append(
                f"Invalid UUID format in MUSICBRAINZ_ALBUMID: '{track.musicbrainz_albumid}'"
            )

        if track.musicbrainz_artistid and not is_valid_uuid(track.musicbrainz_artistid):
            issues.append(
                f"Invalid UUID format in MUSICBRAINZ_ARTISTID: '{track.musicbrainz_artistid}'"
            )

        if track.musicbrainz_albumartistid and not is_valid_uuid(
            track.musicbrainz_albumartistid
        ):
            issues.append(
                f"Invalid UUID format in MUSICBRAINZ_ALBUMARTISTID: '{track.musicbrainz_albumartistid}'"
            )

        if track.musicbrainz_releasegroupid and not is_valid_uuid(
            track.musicbrainz_releasegroupid
        ):
            issues.append(
                f"Invalid UUID format in MUSICBRAINZ_RELEASEGROUPID: '{track.musicbrainz_releasegroupid}'"
            )

        if track.musicbrainz_workid and not is_valid_uuid(track.musicbrainz_workid):
            issues.append(
                f"Invalid UUID format in MUSICBRAINZ_WORKID: '{track.musicbrainz_workid}'"
            )

        if track.art_width and (
            track.art_width < 500 or (track.art_height and track.art_height < 500)
        ):
            issues.append(
                f"Low resolution cover art: {track.art_width}x{track.art_height}"
            )

        if not is_valid_track_filename(file_path.name):
            issues.append(
                f"Filename does not start with track number: '{file_path.name}'"
            )

        if FEAT_PATTERN.search(track.artist):
            issues.append(
                f"ARTIST entry '{track.artist}' contains 'feat' info (Rule: TITLE only)"
            )

        # Check for unsplit artists (e.g. Artist A & Artist B)
        delimiters = [r"\s&\s", r"\s×\s", r"\sfeat\.?\s", r"\sft\.?\s"]
        if not is_single_group_artist(track.artist):
            for delimiter in delimiters:
                if re.search(delimiter, track.artist, re.IGNORECASE):
                    issues.append(
                        f"ARTIST tag seems unsplit: '{track.artist}' (Contains delimiter '{delimiter.strip()}')"
                    )

        # Check Title feature duplicate markers
        feat_matches = re.findall(
            rf"[\(\[]\s*({FEAT_KEYWORDS})", track.title, re.IGNORECASE
        )
        if len(feat_matches) > 1:
            issues.append(
                f"Duplicate featuring markers detected in TITLE ({len(feat_matches)} markers found)"
            )

        # Sync check filename vs title feat
        if FEAT_PATTERN.search(file_path.name) and not FEAT_PATTERN.search(track.title):
            issues.append("Filename contains 'feat' but TITLE tag does not")

        if track.sample_rate and track.sample_rate < 44100:
            issues.append(f"Sub-standard sample rate: {track.sample_rate}Hz")
        if track.bitrate and track.bitrate < 320000 and not track.is_lossless:
            issues.append(
                f"Sub-standard lossy bitrate: {round(track.bitrate / 1000)} kbps (Recommended: 320 kbps)"
            )

    except (OSError, ValueError, RuntimeError) as error:
        issues.append(f"Metadata read error: {error}")
    lrc_path = file_path.with_suffix(".lrc")
    if not lrc_path.exists():
        issues.append("Missing synchronized lyrics (.lrc) file.")

    return issues


def _check_single_file(
    path: Path, check_spectral: bool
) -> tuple[Path, list[str], str | None, str | None, int | None, int | None]:
    file_issues = check_file(path, check_spectral=check_spectral)
    album = None
    album_artist = None
    disc_number = None
    track_number = None
    try:
        track_info = read_track_metadata(path)
        if track_info.album != "Unknown Album":
            album = track_info.album
        if track_info.album_artist:
            album_artist = track_info.album_artist
        disc_number = track_info.disc_number or 1
        track_number = track_info.track_number
    except (OSError, ValueError, RuntimeError):
        pass
    return path, file_issues, album, album_artist, disc_number, track_number


def check_library(
    folder_path: Path,
    output_json: Path | None = None,
    check_spectral: bool = False,
    max_workers: int = 8,
) -> CheckReport:
    if not folder_path.exists():
        raise FileNotFoundError(f"Directory not found: {folder_path}")

    report = CheckReport(
        total_files=0, corrupt_files=0, missing_metadata=0, missing_lrc=0
    )

    files_to_process = find_audio_files(folder_path, recursive=True)

    # Map for folder-level checks
    folder_albums: dict[Path, set[str]] = defaultdict(set)
    folder_album_artists: dict[Path, set[str]] = defaultdict(set)
    folder_tracks_found: dict[Path, dict[tuple[int, int], list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        future_to_path = {
            executor.submit(_check_single_file, path, check_spectral): path
            for path in files_to_process
        }

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("[dim]/[/dim]"),
            TimeRemainingColumn(),
            console=CONSOLE,
        ) as progress:
            task = progress.add_task(
                "[cyan]Checking library...", total=len(files_to_process)
            )

            for future in as_completed(future_to_path):
                (
                    path,
                    file_issues,
                    album,
                    album_artist,
                    disc_number,
                    track_number,
                ) = future.result()
                report.total_files += 1
                folder = path.parent

                if album:
                    folder_albums[folder].add(album)
                if album_artist:
                    folder_album_artists[folder].add(album_artist)
                if track_number is not None:
                    disc = disc_number or 1
                    folder_tracks_found[folder][(disc, track_number)].append(path.name)

                if file_issues:
                    report.issues[str(path)] = file_issues
                    LOG.warning(f"🔍 [bold]{path.name}[/bold]")
                    for issue in file_issues:
                        LOG.warning(f"   ∟ ⚠️  {issue}")
                    if any(
                        "corrupt" in normalize_str(issue)
                        or "checksum" in normalize_str(issue)
                        for issue in file_issues
                    ):
                        report.corrupt_files += 1
                    if any(
                        "missing" in normalize_str(issue)
                        and "tag" in normalize_str(issue)
                        for issue in file_issues
                    ):
                        report.missing_metadata += 1
                    if any(
                        "missing" in normalize_str(issue)
                        and "lrc" in normalize_str(issue)
                        for issue in file_issues
                    ):
                        report.missing_lrc += 1

                progress.advance(task)
    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    # Folder-level checks after file checking completes
    for folder, albums in folder_albums.items():
        folder_issues = []
        if len(albums) > 1:
            folder_issues.append(f"Inconsistent ALBUM name in folder: {albums}")
        album_artists = folder_album_artists.get(folder, set())
        if len(album_artists) > 1:
            folder_issues.append(f"Inconsistent ALBUMARTIST in folder: {album_artists}")

        tracks_found = folder_tracks_found.get(folder, {})
        for (
            disc_idx,
            track_idx,
        ), found_files in tracks_found.items():
            if len(found_files) > 1:
                folder_issues.append(
                    f"Duplicate track number {track_idx} (Disc {disc_idx}) found in files: {found_files}"
                )

        discs: dict[int, list[int]] = defaultdict(list)
        for disc_idx, track_idx in tracks_found:
            discs[disc_idx].append(track_idx)
        for disc_idx, track_numbers in discs.items():
            track_numbers.sort()
            if track_numbers:
                max_track = max(track_numbers)
                missing = [
                    expected_track_number
                    for expected_track_number in range(1, max_track + 1)
                    if expected_track_number not in track_numbers
                ]
                if missing:
                    folder_issues.append(
                        f"Missing track numbers in sequence for Disc {disc_idx}: {missing}"
                    )

        if folder_issues:
            report.issues[str(folder)] = folder_issues
            LOG.warning(f"📁 [bold]{folder.name}[/bold]")
            for issue in folder_issues:
                LOG.warning(f"   ∟ ⚠️  {issue}")

    if output_json:
        total = report.total_files
        corrupt = report.corrupt_files
        missing_meta = report.missing_metadata
        missing_lrc = report.missing_lrc
        issue_count = len(report.issues)

        summary_text = (
            f"Sonora checked {total} files in '{folder_path}'. "
            f"Check status: {corrupt} corrupted files, {missing_meta} missing metadata, "
            f"{missing_lrc} missing LRCs. Total files with issues: {issue_count}."
        )

        data = {
            "schema": "check_report_v1",
            "generator": "Sonora",
            "summary_text": summary_text,
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
                option=orjson.OPT_INDENT_2
                | orjson.OPT_NON_STR_KEYS
                | orjson.OPT_SERIALIZE_DATACLASS,
            )
        )

    return report
