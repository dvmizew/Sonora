import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import orjson
from music_metadata_filter.functions import (
    remove_clean_explicit,
    remove_remastered,
    youtube,
)
from mutagen._util import MutagenError
from mutagen.flac import FLAC, FLACNoHeaderError
from rich.markup import escape

from sonora.audio.checksum import verify_flac_checksum
from sonora.audio.metadata import read_track_metadata
from sonora.audio.spectral import detect_fake_lossless
from sonora.core.config import get_config
from sonora.core.constants import FEAT_KEYWORDS, SUPPORTED_EXTS
from sonora.core.logger import (
    LOG,
    create_progress,
    interactive_pause_listener,
    wait_if_paused,
)
from sonora.core.models import CheckReport
from sonora.core.utils import (
    find_audio_files,
    find_companion_lyrics,
    is_single_group_artist,
    is_valid_uuid,
    is_version_or_remix,
    normalize_genre,
    normalize_str,
)

FEAT_PATTERN = re.compile(FEAT_KEYWORDS, re.IGNORECASE)

# [text], (text), {text}
_BRACKET_PATTERN = re.compile(r"[\(\[\{][^\(\)\[\]\{\}]+[\)\]\}]")


def extract_bracket_tokens(text: str) -> list[tuple[str, set[str]]]:
    """Extracts all bracketed substrings and their constituent alphanumeric word tokens."""
    results: list[tuple[str, set[str]]] = []
    for match in _BRACKET_PATTERN.finditer(text):
        full_bracket = match.group(0)
        inner = full_bracket[1:-1].lower()
        tokens = set("".join(c if c.isalnum() else " " for c in inner).split())
        results.append((full_bracket, tokens))
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


def _is_corrupt_bracket(full_bracket: str, tokens: set[str]) -> bool:
    if is_version_or_remix(full_bracket) or FEAT_PATTERN.search(full_bracket):
        return False

    dummy_title = f"Track {full_bracket}"
    if (
        youtube(dummy_title) == "Track"
        or remove_remastered(dummy_title) == "Track"
        or remove_clean_explicit(dummy_title) == "Track"
    ):
        return True

    return bool(tokens & get_config().codec_rip_keywords)


def check_brackets_corruption(name: str) -> list[str]:
    """
    Check if a filename or tag contains corrupt/unwanted bracket metadata
    (e.g., [FLAC], (Official Video), [HQ], (2011 Remaster), (320kbps)).
    """
    issues = []
    for full_bracket, tokens in extract_bracket_tokens(name):
        if _is_corrupt_bracket(full_bracket, tokens):
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
        if track.initial_key:
            from sonora.audio.key import key_to_camelot

            if key_to_camelot(track.initial_key) is None:
                issues.append(f"Invalid INITIALKEY tag format: '{track.initial_key}'")
        for field_name, tag_label in [
            ("musicbrainz_trackid", "MUSICBRAINZ_TRACKID"),
            ("musicbrainz_albumid", "MUSICBRAINZ_ALBUMID"),
        ]:
            val = getattr(track, field_name)
            if not val:
                issues.append(f"Missing {tag_label} tag.")
            elif not is_valid_uuid(val, allow_multivalue=True):
                issues.append(f"Invalid UUID format in {tag_label}: '{val}'")

        for field_name, tag_label in [
            ("musicbrainz_artistid", "MUSICBRAINZ_ARTISTID"),
            ("musicbrainz_albumartistid", "MUSICBRAINZ_ALBUMARTISTID"),
            ("musicbrainz_releasegroupid", "MUSICBRAINZ_RELEASEGROUPID"),
            ("musicbrainz_workid", "MUSICBRAINZ_WORKID"),
        ]:
            val = getattr(track, field_name)
            if val and not is_valid_uuid(val, allow_multivalue=True):
                issues.append(f"Invalid UUID format in {tag_label}: '{val}'")

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
        delimiters = [(" & ", "&"), (" × ", "×"), (" / ", "/"), (" + ", "+")]
        if not is_single_group_artist(track.artist):
            for delimiter_pattern, delimiter_name in delimiters:
                if delimiter_pattern in f" {track.artist} ":
                    issues.append(
                        f"ARTIST tag seems unsplit: '{track.artist}' (Contains delimiter '{delimiter_name}')"
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

    if not find_companion_lyrics(file_path):
        issues.append("Missing synchronized lyrics (.lrc) file.")

    return issues


def _check_single_file(
    path: Path, check_spectral: bool = False
) -> tuple[Path, list[str], str | None, str | None, int | None, int | None]:
    wait_if_paused()
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


LAST_CHECK_REPORT: CheckReport | None = None


def get_last_check_report() -> CheckReport | None:
    """Return the most recent or partially completed check report."""
    return LAST_CHECK_REPORT


def write_check_report_json(
    report: CheckReport,
    folder_path: Path,
    output_json: Path,
    aborted_by_user: bool = False,
) -> None:
    """Write check report results to JSON file."""
    total = report.total_files
    corrupt = report.corrupt_files
    missing_meta = report.missing_metadata
    missing_lrc = report.missing_lrc
    issue_count = len(report.issues)

    status_prefix = "PARTIAL Check (aborted)" if aborted_by_user else "Check completed"
    summary_text = (
        f"Sonora {status_prefix}: {total} files scanned in '{folder_path}'. "
        f"Check status: {corrupt} corrupted files, {missing_meta} missing metadata, "
        f"{missing_lrc} missing LRCs. Total files with issues: {issue_count}."
    )

    data = {
        "schema": "check_report_v1",
        "generator": "Sonora",
        "summary_text": summary_text,
        "aborted_by_user": aborted_by_user,
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
            | orjson.OPT_SERIALIZE_NUMPY,
        )
    )


def check_library(
    folder_path: Path,
    output_json: Path | None = None,
    check_spectral: bool = False,
    max_threads: int = 8,
) -> CheckReport:
    if not folder_path.exists():
        raise FileNotFoundError(f"Directory not found: {folder_path}")

    global LAST_CHECK_REPORT
    report = CheckReport(
        total_files=0, corrupt_files=0, missing_metadata=0, missing_lrc=0
    )
    LAST_CHECK_REPORT = report

    files_to_process = find_audio_files(folder_path, recursive=True)

    # Map for folder-level checks
    folder_albums: dict[Path, set[str]] = defaultdict(set)
    folder_album_artists: dict[Path, set[str]] = defaultdict(set)
    folder_tracks_found: dict[Path, dict[tuple[int, int], list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    executor = ThreadPoolExecutor(max_workers=max_threads)
    try:
        future_to_path = {
            executor.submit(_check_single_file, path, check_spectral): path
            for path in files_to_process
        }

        with create_progress() as progress:
            task = progress.add_task(
                "[cyan]Checking library...", total=len(files_to_process)
            )
            with interactive_pause_listener(progress, task):
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
                        folder_tracks_found[folder][(disc, track_number)].append(
                            path.name
                        )

                    if file_issues:
                        report.issues[str(path)] = file_issues
                        try:
                            display_name = str(path.relative_to(folder_path))
                        except ValueError:
                            display_name = path.name
                        LOG.warning(f"🔍 [bold]{escape(display_name)}[/bold]")
                        for issue in file_issues:
                            LOG.warning(f"   ∟ ⚠️  {escape(issue)}")
                        if any(
                            "corrupt" in normalize_str(issue)
                            or "checksum" in normalize_str(issue)
                            or "0-byte" in normalize_str(issue)
                            or "fake lossless" in normalize_str(issue)
                            for issue in file_issues
                        ):
                            report.corrupt_files += 1
                        if any(
                            "missing" in normalize_str(issue)
                            and "lrc" not in normalize_str(issue)
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

        # Check for missing track numbers in sequence per disc
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
            try:
                display_folder = str(folder.relative_to(folder_path))
            except ValueError:
                display_folder = folder.name
            LOG.warning(f"📁 [bold]{escape(display_folder)}[/bold]")
            for issue in folder_issues:
                LOG.warning(f"   ∟ ⚠️  {escape(issue)}")

    if output_json:
        write_check_report_json(report, folder_path, output_json, aborted_by_user=False)

    return report
