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
from mutagen.flac import FLAC
from rich.markup import escape

from sonora.audio.checksum import verify_flac_checksum
from sonora.audio.key import key_to_camelot
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
    InterruptedOperationError,
    find_audio_files,
    find_companion_lyrics,
    is_interruption,
    is_single_group_artist,
    is_valid_uuid,
    is_version_or_remix,
    normalize_genre,
    normalize_str,
)

FEAT_PATTERN = re.compile(FEAT_KEYWORDS, re.IGNORECASE)

# Matches balanced delimiters: square brackets, parentheses, and curly braces
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
        except OSError as error:
            issues.append(f"Checksum check failed: {error}")

        try:
            audio_flac = FLAC(str(file_path))
            for index, picture in enumerate(audio_flac.pictures):
                if len(picture.data) == 0:
                    issues.append(f"Corrupt 0-byte picture block at index {index}.")
        except (MutagenError, OSError) as error:
            LOG.debug(f"Mutagen picture check skipped for {file_path}: {error}")

    if check_spectral:
        try:
            is_fake, _, description = detect_fake_lossless(file_path)
            if is_fake:
                issues.append(
                    description
                    or "Possible fake lossless (spectral cutoff below 16kHz)."
                )
        except OSError as error:
            LOG.debug(f"Spectral analysis failed for {file_path}: {error}")

    try:
        track = read_track_metadata(file_path)
        issues.extend(check_brackets_corruption(track.artist))
        issues.extend(check_brackets_corruption(track.title))

        if track.genre and not normalize_genre(track.genre):
            issues.append(f"Blacklisted genre tag: '{track.genre}'")
        missing_checks = [
            (track.artist == "Unknown Artist", "Missing ARTIST tag."),
            (track.title == "Unknown Title", "Missing TITLE tag."),
            (track.album == "Unknown Album", "Missing ALBUM tag."),
            (not track.album_artist, "Missing ALBUMARTIST tag (Risk of Split Album)."),
            (track.track_number is None, "Missing TRACKNUMBER tag."),
            (not track.date, "Missing DATE (Year) tag."),
            (track.bpm is None, "Missing BPM tag."),
            (track.replaygain_track_gain is None, "Missing REPLAYGAIN_TRACK_GAIN tag."),
            (track.replaygain_track_peak is None, "Missing REPLAYGAIN_TRACK_PEAK tag."),
        ]
        for condition, msg in missing_checks:
            if condition:
                issues.append(msg)

        if track.initial_key and key_to_camelot(track.initial_key) is None:
            issues.append(f"Invalid INITIALKEY tag format: '{track.initial_key}'")

        for field_name, tag_label, required in [
            ("musicbrainz_trackid", "MUSICBRAINZ_TRACKID", True),
            ("musicbrainz_albumid", "MUSICBRAINZ_ALBUMID", True),
            ("musicbrainz_artistid", "MUSICBRAINZ_ARTISTID", False),
            ("musicbrainz_albumartistid", "MUSICBRAINZ_ALBUMARTISTID", False),
            ("musicbrainz_releasegroupid", "MUSICBRAINZ_RELEASEGROUPID", False),
            ("musicbrainz_workid", "MUSICBRAINZ_WORKID", False),
        ]:
            val = getattr(track, field_name)
            if not val and required:
                issues.append(f"Missing {tag_label} tag.")
            elif val and not is_valid_uuid(val, allow_multivalue=True):
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

        delimiters = [(" & ", "&"), (" \u00d7 ", "\u00d7"), (" / ", "/"), (" + ", "+")]
        if not is_single_group_artist(track.artist):
            for delimiter_pattern, delimiter_name in delimiters:
                if delimiter_pattern in f" {track.artist} ":
                    issues.append(
                        f"ARTIST tag seems unsplit: '{track.artist}' (Contains delimiter '{delimiter_name}')"
                    )

        feat_matches = re.findall(
            rf"[\(\[]\s*({FEAT_KEYWORDS})", track.title, re.IGNORECASE
        )
        if len(feat_matches) > 1:
            issues.append(
                f"Duplicate featuring markers detected in TITLE ({len(feat_matches)} markers found)"
            )

        if FEAT_PATTERN.search(file_path.name) and not FEAT_PATTERN.search(track.title):
            issues.append("Filename contains 'feat' but TITLE tag does not")

        if track.sample_rate and track.sample_rate < 44100:
            issues.append(f"Sub-standard sample rate: {track.sample_rate}Hz")
        if track.bitrate and track.bitrate < 320000 and not track.is_lossless:
            issues.append(
                f"Sub-standard lossy bitrate: {round(track.bitrate / 1000)} kbps (Recommended: 320 kbps)"
            )

    except OSError as error:
        issues.append(f"Metadata read error: {error}")

    companion_lrcs = find_companion_lyrics(file_path)
    if not companion_lrcs:
        issues.append("Missing synchronized lyrics (.lrc) file.")
    elif any(lrc.stat().st_size == 0 for lrc in companion_lrcs):
        issues.append("Corrupt 0-byte synchronized lyrics (.lrc) file.")

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
    except OSError:
        pass
    return path, file_issues, album, album_artist, disc_number, track_number


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

    report_payload = {
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
            report_payload,
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
    report: CheckReport | None = None,
) -> CheckReport:
    if not folder_path.exists():
        raise FileNotFoundError(f"Directory not found: {folder_path}")

    if report is None:
        report = CheckReport()

    files_to_process = find_audio_files(folder_path, recursive=True)

    folder_albums: dict[Path, set[str]] = defaultdict(set)
    folder_album_artists: dict[Path, set[str]] = defaultdict(set)
    folder_tracks_found: dict[Path, dict[tuple[int, int], list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    with (
        create_progress() as progress,
        ThreadPoolExecutor(max_workers=max_threads) as executor,
    ):
        future_to_path = {
            executor.submit(_check_single_file, path, check_spectral): path
            for path in files_to_process
        }
        task = progress.add_task(
            "[cyan]Checking library...", total=len(files_to_process)
        )
        with interactive_pause_listener(progress, task):
            try:
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
                        norm_issues = [normalize_str(issue) for issue in file_issues]
                        if any(
                            any(
                                k in ni
                                for k in (
                                    "corrupted flac",
                                    "checksum",
                                    "0-byte",
                                    "fake lossless",
                                )
                            )
                            for ni in norm_issues
                        ):
                            report.corrupt_files += 1
                        if any(
                            "missing" in ni and "lrc" not in ni for ni in norm_issues
                        ):
                            report.missing_metadata += 1
                        if any("missing" in ni and "lrc" in ni for ni in norm_issues):
                            report.missing_lrc += 1

                    progress.advance(task)
            except (KeyboardInterrupt, RuntimeError) as exc:
                if not is_interruption(exc):
                    raise
                executor.shutdown(wait=True, cancel_futures=True)
                raise InterruptedOperationError(report) from None

    for folder, albums in folder_albums.items():
        if get_config().is_generic_container(folder.name):
            continue
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

        discs: dict[int, set[int]] = defaultdict(set)
        for disc_idx, track_idx in tracks_found:
            discs[disc_idx].add(track_idx)
        for disc_idx, track_nums in discs.items():
            if track_nums:
                missing = [
                    i for i in range(1, max(track_nums) + 1) if i not in track_nums
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
