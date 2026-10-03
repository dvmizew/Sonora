import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import orjson
from mutagen._util import MutagenError
from mutagen.flac import FLAC
from rapidfuzz import fuzz
from rich.markup import escape

from sonora.audio.checksum import verify_flac_checksum
from sonora.audio.key import key_to_camelot
from sonora.audio.metadata import read_track_metadata
from sonora.audio.spectral import detect_fake_lossless
from sonora.core.config import get_config
from sonora.core.constants import (
    FEAT_KEYWORDS,
    MIN_COVER_ART_DIMENSION,
    SUPPORTED_EXTS,
)
from sonora.core.logger import (
    LOG,
    create_progress,
    interactive_pause_listener,
    wait_if_paused,
)
from sonora.core.models import CheckReport, TrackInfo
from sonora.core.utils import (
    InterruptedOperationError,
    _is_corrupt_bracket,
    clean_title,
    extract_artist_features,
    extract_bracket_tokens,
    extract_title_features,
    find_audio_files,
    find_companion_lyrics,
    get_primary_artist,
    is_in_singles_hierarchy,
    is_interruption,
    is_single_group_artist,
    is_valid_uuid,
    normalize_genre,
    normalize_str,
    parse_track_filename,
)

FEAT_PATTERN = re.compile(FEAT_KEYWORDS, re.IGNORECASE)


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


def check_file(
    file_path: Path,
    check_spectral: bool = False,
    track_metadata: TrackInfo | None = None,
) -> list[str]:
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
        track = track_metadata or read_track_metadata(file_path)
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
            track.art_width < MIN_COVER_ART_DIMENSION
            or (track.art_height and track.art_height < MIN_COVER_ART_DIMENSION)
        ):
            issues.append(
                f"Low resolution cover art: {track.art_width}x{track.art_height}"
            )

        _, filename_track_num, _ = parse_track_filename(file_path.name)
        if filename_track_num is None:
            issues.append(
                f"Filename does not start with track number: '{file_path.name}'"
            )

        _, artist_feats = extract_artist_features(track.artist, allow_network=False)
        if artist_feats:
            issues.append(
                f"ARTIST entry '{track.artist}' contains 'feat' info (Rule: clean artist, use FEATURED_ARTISTS tag)"
            )

        delimiters = [
            (" \u00d7 ", "\u00d7"),
            (" / ", "/"),
            (" + ", "+"),
            ("; ", ";"),
            (" ; ", ";"),
        ]
        norm_album_artist = (
            normalize_str(track.album_artist) if track.album_artist else ""
        )
        norm_artist = normalize_str(track.artist)
        is_album_level_collab = bool(
            norm_album_artist
            and (
                norm_artist == norm_album_artist
                or any(
                    normalize_str(part) in norm_album_artist
                    or norm_album_artist in normalize_str(part)
                    for part in re.split(r"[,/&×+;]", track.artist)
                    if part.strip()
                )
            )
        )
        if (
            not is_album_level_collab
            and not artist_feats
            and not is_single_group_artist(track.artist)
        ):
            for delimiter_pattern, delimiter_name in delimiters:
                if delimiter_pattern in f" {track.artist} ":
                    issues.append(
                        f"ARTIST tag seems unsplit: '{track.artist}' (Contains delimiter '{delimiter_name}')"
                    )

        _, title_features = extract_title_features(
            track.title, primary_artist=track.artist
        )
        if title_features:
            issues.append(
                f"TITLE tag '{track.title}' contains 'feat' info (Rule: clean title, use FEATURED_ARTISTS tag)"
            )

        _, filename_features = extract_title_features(
            file_path.stem, primary_artist=track.artist
        )
        if filename_features:
            issues.append(
                f"Filename contains 'feat' info: '{file_path.name}' (Rule: clean filename)"
            )

        if track.sample_rate and track.sample_rate < 44100:
            issues.append(f"Sub-standard sample rate: {track.sample_rate}Hz")
        if track.bitrate and track.bitrate < 320000 and not track.is_lossless:
            issues.append(
                f"Sub-standard lossy bitrate: {round(track.bitrate / 1000)} kbps (Recommended: 320 kbps)"
            )

    except OSError as error:
        issues.append(f"Metadata read error: {error}")

    companion_lrcs = find_companion_lyrics(
        file_path,
        track_number=track.track_number if "track" in locals() and track else None,
    )
    if not companion_lrcs:
        issues.append("Missing synchronized lyrics (.lrc) file.")
    elif any(lrc.stat().st_size == 0 for lrc in companion_lrcs):
        issues.append("Corrupt 0-byte synchronized lyrics (.lrc) file.")

    return issues


def _check_single_file(
    path: Path, check_spectral: bool = False
) -> tuple[Path, list[str], TrackInfo | None]:
    wait_if_paused()
    track_metadata: TrackInfo | None = None
    try:
        track_metadata = read_track_metadata(path)
    except OSError:
        pass
    file_issues = check_file(
        path, check_spectral=check_spectral, track_metadata=track_metadata
    )
    return path, file_issues, track_metadata


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


def _check_folder_album_consistency(
    folder: Path,
    tracks_in_folder: list[tuple[Path, TrackInfo]],
    folder_path: Path,
    report: CheckReport,
    folder_issues: list[str],
) -> tuple[str | None, str | None]:
    """
    Verify consistent ALBUM and ALBUMARTIST metadata across audio tracks in an album folder.
    Flags specific outlier files with mismatched album tags and records folder-level issues.
    Returns (dominant_album, dominant_artist) for downstream alien track detection.
    """
    total_tracks = len(tracks_in_folder)
    if total_tracks == 0:
        return None, None

    album_counts: Counter[str] = Counter()
    for _, track_metadata in tracks_in_folder:
        if track_metadata.album and track_metadata.album != "Unknown Album":
            album_counts[track_metadata.album] += 1

    dominant_album: str | None = None
    if album_counts:
        norm_folder = normalize_str(folder.name)
        folder_matched_album: str | None = None
        for candidate_alb in album_counts:
            norm_cand = normalize_str(candidate_alb)
            if norm_cand and (norm_cand in norm_folder or norm_folder in norm_cand):
                folder_matched_album = candidate_alb
                break

        top_alb, top_count = album_counts.most_common(1)[0]
        if folder_matched_album and album_counts[folder_matched_album] >= top_count:
            dominant_album = folder_matched_album
            dominant_album_count = album_counts[dominant_album]
        else:
            dominant_album = top_alb
            dominant_album_count = top_count

        if len(album_counts) > 1:
            folder_issues.append(
                f"Inconsistent ALBUM name in folder: {set(album_counts.keys())}"
            )
            # When an established consensus exists (>= 50% or >= 2 tracks in small directories),
            # flag each minority file diverging from the album consensus
            if dominant_album_count >= max(2, int(total_tracks * 0.5)):
                norm_dominant = normalize_str(dominant_album)
                for file_path, track_metadata in tracks_in_folder:
                    if (
                        not track_metadata.album
                        or track_metadata.album == "Unknown Album"
                    ):
                        continue
                    if normalize_str(track_metadata.album) != norm_dominant:
                        mismatch_issue = (
                            f"Mismatched ALBUM tag '{track_metadata.album}' "
                            f"(folder consensus album: '{dominant_album}')"
                        )
                        report.issues.setdefault(str(file_path), []).append(
                            mismatch_issue
                        )
                        report.missing_metadata += 1
                        try:
                            display_name = str(file_path.relative_to(folder_path))
                        except ValueError:
                            display_name = file_path.name
                        LOG.warning(f"🔍 [bold]{escape(display_name)}[/bold]")
                        LOG.warning(f"   ∟ ⚠️  {escape(mismatch_issue)}")

    album_artist_counts: Counter[str] = Counter()
    artist_counts: Counter[str] = Counter()
    for _, track_metadata in tracks_in_folder:
        if track_metadata.album_artist:
            album_artist_counts[track_metadata.album_artist] += 1
        primary_artist = get_primary_artist(track_metadata.artist)
        if primary_artist and primary_artist != "Unknown Artist":
            artist_counts[primary_artist] += 1

    dominant_artist: str | None = None
    if album_artist_counts:
        dominant_artist, _ = album_artist_counts.most_common(1)[0]
        if len(album_artist_counts) > 1:
            folder_issues.append(
                f"Inconsistent ALBUMARTIST in folder: {set(album_artist_counts.keys())}"
            )
            norm_dom_aa = normalize_str(dominant_artist)
            for file_path, track_metadata in tracks_in_folder:
                if not track_metadata.album_artist:
                    continue
                if normalize_str(track_metadata.album_artist) != norm_dom_aa:
                    aa_issue = (
                        f"Mismatched ALBUMARTIST tag '{track_metadata.album_artist}' "
                        f"(folder consensus: '{dominant_artist}')"
                    )
                    report.issues.setdefault(str(file_path), []).append(aa_issue)
                    try:
                        display_name = str(file_path.relative_to(folder_path))
                    except ValueError:
                        display_name = file_path.name
                    LOG.warning(f"🔍 [bold]{escape(display_name)}[/bold]")
                    LOG.warning(f"   ∟ ⚠️  {escape(aa_issue)}")
    elif artist_counts:
        dominant_artist, _ = artist_counts.most_common(1)[0]

    return dominant_album, dominant_artist


def _check_folder_alien_tracks(
    folder: Path,
    tracks_in_folder: list[tuple[Path, TrackInfo]],
    dominant_album: str | None,
    dominant_artist: str | None,
    folder_tracks_found: dict[tuple[int, int], list[str]],
    folder_path: Path,
    report: CheckReport,
    folder_issues: list[str],
) -> None:
    """
    Detect alien or displaced tracks inside an album directory without external network queries.
    Identifies:
    1. Unrelated tracks whose artist and album both mismatch the folder consensus.
    2. Alien single tracks colliding on track number with discordant album tags.
    3. Extreme sequence position outliers exceeding the album sequence boundary.
    """
    total_tracks = len(tracks_in_folder)
    if total_tracks < 2 or not dominant_album:
        return

    # Shield compilations and soundtrack releases where disparate artists are normal
    artist_lower = (dominant_artist or "").lower()
    album_lower = dominant_album.lower()
    folder_name_lower = folder.name.lower()
    if (
        artist_lower in ("various artists", "va", "soundtrack", "compilation")
        or album_lower.endswith("soundtrack")
        or "soundtrack" in folder_name_lower
        or "compilation" in folder_name_lower
    ):
        return

    norm_dominant_album = normalize_str(clean_title(dominant_album))
    norm_dominant_artist = normalize_str(get_primary_artist(dominant_artist or ""))

    disc_sequences: dict[int, list[int]] = defaultdict(list)
    for _, track_metadata in tracks_in_folder:
        if track_metadata.track_number is not None and track_metadata.track_number > 0:
            disc_idx = track_metadata.disc_number or 1
            disc_sequences[disc_idx].append(track_metadata.track_number)

    for file_path, track_metadata in tracks_in_folder:
        track_issues: list[str] = []
        norm_track_artist = normalize_str(get_primary_artist(track_metadata.artist))
        norm_track_album = normalize_str(clean_title(track_metadata.album))

        artist_matches = bool(norm_dominant_artist) and (
            norm_track_artist == norm_dominant_artist
            or (
                len(norm_track_artist) >= 4
                and len(norm_dominant_artist) >= 4
                and (
                    norm_track_artist in norm_dominant_artist
                    or norm_dominant_artist in norm_track_artist
                )
            )
            or fuzz.ratio(norm_track_artist, norm_dominant_artist) >= 85.0
        )
        album_matches = (
            norm_track_album == norm_dominant_album
            or fuzz.ratio(norm_track_album, norm_dominant_album) >= 80.0
        )

        # 1. Unrelated alien track: neither artist nor album matches consensus
        if not artist_matches and not album_matches and norm_track_artist:
            track_issues.append(
                f"Alien track detected in album folder: artist '{track_metadata.artist}' and "
                f"album '{track_metadata.album}' do not match album consensus "
                f"'{dominant_artist or 'Unknown'} - {dominant_album}'"
            )

        # 2. Alien single: shares duplicate track position on disc but possesses a different album tag
        disc_num = track_metadata.disc_number or 1
        pos_num = track_metadata.track_number or 0
        colliding_files = folder_tracks_found.get((disc_num, pos_num), [])
        if len(colliding_files) > 1 and not album_matches and norm_track_album:
            track_issues.append(
                f"Alien single detected in album folder: '{file_path.name}' "
                f"(album: '{track_metadata.album}') collides on track {pos_num} with '{dominant_album}'"
            )

        # 3. Displaced / extreme position outlier (e.g. track 15 on disc where sequence ends at 8)
        disc_nums = [n for n in disc_sequences.get(disc_num, []) if n != pos_num]
        if len(disc_nums) >= 3 and pos_num > (max(disc_nums) + 4):
            clean_track_title = normalize_str(track_metadata.title)
            is_labeled_bonus = any(
                b in track_metadata.title.lower()
                for b in ("bonus", "deluxe", "live", "remix")
            )
            if not is_labeled_bonus and (
                not album_matches or clean_track_title == norm_dominant_artist
            ):
                track_issues.append(
                    f"Displaced / alien track detected: '{file_path.name}' declares track {pos_num} "
                    f"on Disc {disc_num}, exceeding album sequence max ({max(disc_nums)})"
                )

        if track_issues:
            existing_issues = report.issues.setdefault(str(file_path), [])
            for issue_msg in track_issues:
                if issue_msg not in existing_issues:
                    existing_issues.append(issue_msg)
                    folder_issues.append(issue_msg)
                    report.missing_metadata += 1
                    try:
                        display_name = str(file_path.relative_to(folder_path))
                    except ValueError:
                        display_name = file_path.name
                    LOG.warning(f"🔍 [bold]{escape(display_name)}[/bold]")
                    LOG.warning(f"   ∟ ⚠️  {escape(issue_msg)}")


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

    folder_tracks: dict[Path, list[tuple[Path, TrackInfo]]] = defaultdict(list)
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
                    path, file_issues, track_metadata = future.result()
                    report.total_files += 1
                    folder = path.parent

                    if track_metadata is not None:
                        folder_tracks[folder].append((path, track_metadata))
                        if track_metadata.track_number is not None:
                            disc = track_metadata.disc_number or 1
                            folder_tracks_found[folder][
                                (disc, track_metadata.track_number)
                            ].append(path.name)

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

    for folder, tracks_in_folder in folder_tracks.items():
        if get_config().is_generic_container(folder.name):
            continue
        folder_issues: list[str] = []
        try:
            rel_parts = folder.relative_to(folder_path).parts
        except ValueError:
            rel_parts = (folder.name,)
        is_singles_container = any(
            get_config().is_generic_container(p) for p in rel_parts
        ) or is_in_singles_hierarchy(folder, folder_path)

        tracks_found = folder_tracks_found.get(folder, {})
        for (
            disc_idx,
            track_idx,
        ), found_files in tracks_found.items():
            if len(found_files) > 1:
                folder_issues.append(
                    f"Duplicate track number {track_idx} (Disc {disc_idx}) found in files: {found_files}"
                )

        if not is_singles_container:
            dominant_album, dominant_artist = _check_folder_album_consistency(
                folder=folder,
                tracks_in_folder=tracks_in_folder,
                folder_path=folder_path,
                report=report,
                folder_issues=folder_issues,
            )
            _check_folder_alien_tracks(
                folder=folder,
                tracks_in_folder=tracks_in_folder,
                dominant_album=dominant_album,
                dominant_artist=dominant_artist,
                folder_tracks_found=tracks_found,
                folder_path=folder_path,
                report=report,
                folder_issues=folder_issues,
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
