from pathlib import Path
from typing import Annotated

from cyclopts import Parameter
from rich.markup import escape

from sonora.cli.options import (
    ArtOpt,
    BpmOpt,
    DryRunOpt,
    ForceOpt,
    JsonReportOpt,
    KeyOpt,
    LyricsOpt,
    MusixmatchOpt,
    NoDspOpt,
    PathArg,
    ReplayGainOpt,
    ShazamOpt,
    ThreadsOpt,
)
from sonora.cli.reports import _emit_tag_summary_and_report, _write_json_report
from sonora.core.cache import set_ignore_cache
from sonora.core.cancellation import InterruptedOperationError, is_interruption
from sonora.core.config import clear_config_cache, get_config, load_app_environment
from sonora.core.logger import LOG
from sonora.core.models import CheckReport, NormalizeReport, RenameReport, TrackInfo
from sonora.modules.backup import backup_library_tags, restore_library_tags
from sonora.modules.checker import check_library, write_check_report_json
from sonora.modules.organizer import organize_library_singles
from sonora.modules.renamer import rename_directory_files
from sonora.modules.tagger import tag_album_folder
from sonora.modules.track_processing import normalize_library
from sonora.services.lyrics import init_musixmatch_token
from sonora.services.musicbrainz import init_musicbrainz


def tag(
    path: PathArg,
    fetch_bpm: BpmOpt = True,
    fetch_key: KeyOpt = True,
    fetch_replaygain: ReplayGainOpt = True,
    fetch_lyrics: LyricsOpt = True,
    fetch_artwork: ArtOpt = True,
    json_report: JsonReportOpt = None,
    force: ForceOpt = False,
    lastfm_api_key: Annotated[
        str | None,
        Parameter(
            name=["--lastfm-key"], help="Last.fm API key for genre and mood lookup"
        ),
    ] = None,
    acoustid_api_key: Annotated[
        str | None,
        Parameter(
            name=["--acoustid-key"], help="AcoustID API key for acoustic fingerprinting"
        ),
    ] = None,
    discogs_user_token: Annotated[
        str | None,
        Parameter(name=["--discogs-token"], help="Discogs personal user token"),
    ] = None,
    genius_api_token: Annotated[
        str | None,
        Parameter(
            name=["--genius-token"], help="Genius API token for song descriptions"
        ),
    ] = None,
    fanart_api_key: Annotated[
        str | None,
        Parameter(
            name=["--fanart-key"],
            help="Fanart.tv project API key for CD art, logos, and HD fanart",
        ),
    ] = None,
    fanart_client_key: Annotated[
        str | None,
        Parameter(
            name=["--fanart-client-key"],
            help="Fanart.tv personal VIP client key for immediate image updates",
        ),
    ] = None,
    enable_shazam: ShazamOpt = True,
    no_dsp: NoDspOpt = False,
    ignore_cache: Annotated[
        bool,
        Parameter(
            name=["--ignore-cache"],
            help="Bypass local API metadata cache and force re-fetching from remote services",
        ),
    ] = False,
    musixmatch_token: MusixmatchOpt = None,
    threads: ThreadsOpt = 4,
    dry_run: DryRunOpt = False,
) -> int:
    """
    Tag audio files and albums automatically with all metadata, artwork, BPM, ReplayGain & lyrics.
    """
    load_app_environment(path)
    clear_config_cache()
    init_musicbrainz()
    if not dry_run:
        init_musixmatch_token(musixmatch_token or get_config().musixmatch_token)

    if no_dsp:
        fetch_bpm = False
        fetch_key = False
        fetch_replaygain = False

    set_ignore_cache(ignore_cache)

    config = get_config()
    resolved_lastfm_key = lastfm_api_key or config.lastfm_api_key
    resolved_acoustid_key = acoustid_api_key or config.acoustid_api_key
    resolved_discogs_token = discogs_user_token or config.discogs_token
    resolved_genius_token = genius_api_token or config.genius_api_token
    resolved_fanart_key = fanart_api_key or config.fanart_api_key
    resolved_fanart_client_key = fanart_client_key or config.fanart_client_key
    has_musixmatch = bool(musixmatch_token or config.musixmatch_token)

    active_keys: list[str] = []
    if resolved_discogs_token:
        active_keys.append("[bold green]Discogs[/]")
    if resolved_acoustid_key:
        active_keys.append("[bold green]AcoustID[/]")
    if resolved_genius_token:
        active_keys.append("[bold green]Genius[/]")
    if resolved_lastfm_key:
        active_keys.append("[bold green]Last.fm[/]")
    if has_musixmatch:
        active_keys.append("[bold green]Musixmatch[/]")
    if resolved_fanart_key:
        active_keys.append("[bold green]Fanart.tv[/]")

    if active_keys:
        LOG.info(f"🔑 [bold]Active API Keys/Tokens:[/] {', '.join(active_keys)}")
    else:
        LOG.info(
            "🔑 [dim]Active API Keys/Tokens:[/] [yellow]None (using free unauthenticated tiers)[/]"
        )

    LOG.info(f"Tagging album directory: [bold]{escape(str(path))}[/bold]")
    interrupted = False
    failures: list[dict[str, str]] = []
    tagged_tracks: list[TrackInfo] = []
    try:
        tagged_tracks = tag_album_folder(
            path,
            max_threads=threads,
            fetch_bpm=fetch_bpm,
            fetch_key=fetch_key,
            fetch_replaygain=fetch_replaygain,
            fetch_lyrics=fetch_lyrics,
            fetch_itunes_art=fetch_artwork,
            lastfm_api_key=resolved_lastfm_key,
            acoustid_api_key=resolved_acoustid_key,
            discogs_user_token=resolved_discogs_token,
            genius_api_token=resolved_genius_token,
            fanart_api_key=resolved_fanart_key,
            fanart_client_key=resolved_fanart_client_key,
            enable_shazam=enable_shazam,
            force=force,
            dry_run=dry_run,
            failures=failures,
        )
    except (KeyboardInterrupt, RuntimeError) as exc:
        if not is_interruption(exc):
            raise
        interrupted = True
        if isinstance(exc, InterruptedOperationError) and isinstance(
            exc.partial_result, list
        ):
            tagged_tracks = exc.partial_result
        LOG.warning(
            "\n⏹️  [bold yellow]INTERRUPTED[/] - Tagging stopped by user (Ctrl+C). Generating summary for processed tracks..."
        )

    if not interrupted:
        action = "Previewed" if dry_run else "Processed"
        if failures:
            LOG.warning(
                f"{action} {len(tagged_tracks)} tracks with {len(failures)} failures."
            )
        else:
            LOG.success(f"{action} {len(tagged_tracks)} tracks.")
        conflict_count = sum(bool(track.metadata_conflicts) for track in tagged_tracks)
        if conflict_count:
            LOG.warning(
                f"{conflict_count} tracks have unresolved provider conflicts; review the tagging report."
            )
    else:
        LOG.warning(
            f"Partially processed {len(tagged_tracks)} tracks before interruption."
        )

    _emit_tag_summary_and_report(
        tagged_tracks,
        failures,
        interrupted,
        path,
        threads,
        json_report,
        dry_run=dry_run,
    )
    if interrupted:
        return 130
    return 1 if failures else 0


def check(
    path: PathArg,
    json_report: JsonReportOpt = None,
    spectral_analysis: Annotated[
        bool,
        Parameter(
            name=["--spectral"],
            negative="",
            help="Enable deep spectral cutoff analysis for fake lossless detection (slow)",
        ),
    ] = False,
    threads: ThreadsOpt = 8,
) -> int:
    """
    Check music library for FLAC integrity, bracket corruption & missing LRCs.
    """
    LOG.info(f"Checking music library: [bold]{escape(str(path))}[/bold]")
    interrupted = False
    check_report = CheckReport(
        total_files=0, corrupt_files=0, missing_metadata=0, missing_lrc=0
    )
    try:
        check_report = check_library(
            path,
            output_json=None,
            check_spectral=spectral_analysis,
            max_threads=threads,
            report=check_report,
        )
    except (KeyboardInterrupt, RuntimeError) as exc:
        if not is_interruption(exc):
            raise
        interrupted = True
        if isinstance(exc, InterruptedOperationError) and isinstance(
            exc.partial_result, CheckReport
        ):
            check_report = exc.partial_result
        LOG.warning(
            "\n⏹️  [bold yellow]INTERRUPTED[/] - Check stopped by user (Ctrl+C). Generating summary for scanned files..."
        )

    if not interrupted:
        LOG.success(
            f"Check completed: {check_report.total_files} files scanned, {check_report.files_with_issues} files with issues."
        )
    else:
        LOG.warning(
            f"Partially scanned {check_report.total_files} files before interruption."
        )

    total = check_report.total_files
    issue_count = check_report.files_with_issues
    passing_files = max(0, total - issue_count)
    passing_percentage = (passing_files / total * 100) if total > 0 else 0.0
    issues_pct = (issue_count / total * 100) if total > 0 else 0.0
    corrupt_pct = (check_report.corrupt_files / total * 100) if total > 0 else 0.0
    missing_meta_pct = (
        (check_report.missing_metadata / total * 100) if total > 0 else 0.0
    )
    missing_lrc_pct = (check_report.missing_lrc / total * 100) if total > 0 else 0.0

    validation_summary_rows = [
        ("Total Files Scanned", str(total), None),
        (
            "✅ Files Passing Checks",
            f"{passing_files}/{total} ({passing_percentage:.1f}%)",
            "green" if passing_files == total else None,
        ),
        (
            "⚠️ Files with Issues",
            f"{issue_count}/{total} ({issues_pct:.1f}%)",
            "red" if issue_count > 0 else "green",
        ),
        (
            "Missing Metadata Tags",
            f"{check_report.missing_metadata} files ({missing_meta_pct:.1f}%)",
            "yellow" if check_report.missing_metadata > 0 else "green",
        ),
        (
            "Missing Synced Lyrics (.lrc)",
            f"{check_report.missing_lrc} files ({missing_lrc_pct:.1f}%)",
            "yellow" if check_report.missing_lrc > 0 else "green",
        ),
        (
            "Corrupted / Damaged Audio",
            f"{check_report.corrupt_files} files ({corrupt_pct:.1f}%)",
            "red" if check_report.corrupt_files > 0 else "green",
        ),
    ]
    LOG.summary_table("Validation Summary", validation_summary_rows)

    if json_report:
        write_check_report_json(
            check_report, path, json_report, aborted_by_user=interrupted
        )
        LOG.info(
            f"Saved validation JSON report with all {issue_count} issue(s) to [bold]{escape(str(json_report))}[/bold]"
        )

    if interrupted:
        return 130
    return 0


def rename(
    path: PathArg,
    threads: ThreadsOpt = 4,
    dry_run: DryRunOpt = False,
    json_report: JsonReportOpt = None,
) -> int:
    """
    Rename audio files and sync .lrc metadata headers.
    """
    LOG.info(f"Renaming files in directory: [bold]{escape(str(path))}[/bold]")
    interrupted = False
    report = RenameReport()
    try:
        rename_directory_files(
            path, dry_run=dry_run, max_threads=threads, report=report
        )
    except (KeyboardInterrupt, RuntimeError) as exc:
        if not is_interruption(exc):
            raise
        interrupted = True
        if isinstance(exc, InterruptedOperationError) and isinstance(
            exc.partial_result, RenameReport
        ):
            report = exc.partial_result
        LOG.warning(
            "\n⏹️  [bold yellow]INTERRUPTED[/] - Renaming stopped by user (Ctrl+C). Generating summary..."
        )

    if not interrupted:
        LOG.success(
            f"Renaming completed: {report.files_renamed}/{report.total_files} files renamed on disk."
        )
    else:
        LOG.warning(
            f"Partially renamed {report.files_renamed}/{report.total_files} files before interruption."
        )

    renaming_summary_rows = [
        ("Total Files Scanned", str(report.total_files), None),
        (
            "Files Failed",
            str(report.failed_files),
            "red" if report.failed_files else None,
        ),
        (
            "Files Renamed on Disk",
            str(report.files_renamed),
            "green" if report.files_renamed > 0 else None,
        ),
        (
            "Album Folders Renamed",
            str(report.folders_renamed),
            "green" if report.folders_renamed > 0 else None,
        ),
        (
            "Already Compliant Files",
            str(report.unchanged_files),
            "green" if report.unchanged_files > 0 else None,
        ),
    ]
    LOG.summary_table("Renaming Summary", renaming_summary_rows)

    if json_report:
        _write_json_report(
            json_report,
            {
                "schema": "rename_report_v1",
                "generator": "Sonora",
                "aborted_by_user": interrupted,
                "target_path": str(path.resolve()),
                "summary": {
                    "total_scanned": report.total_files,
                    "files_renamed": report.files_renamed,
                    "folders_renamed": report.folders_renamed,
                    "lrc_synced": report.lrc_synced,
                    "unchanged_files": report.unchanged_files,
                    "failed_files": report.failed_files,
                },
            },
            "renaming",
        )
    if interrupted:
        return 130
    return 1 if report.failed_files else 0


def organize(
    path: PathArg,
    target_singles: Annotated[
        Path | None,
        Parameter(
            name=["--target-singles"],
            help="Destination directory for single tracks (default: <path>/Singles)",
        ),
    ] = None,
    dry_run: DryRunOpt = False,
    threads: ThreadsOpt = 4,
    json_report: JsonReportOpt = None,
) -> int:
    """
    Organize single tracks into a Singles directory structure.
    """
    destination_directory = target_singles or (path / "Singles")
    LOG.info(
        f"Organizing single tracks from {escape(str(path))} to {escape(str(destination_directory))}"
    )
    interrupted = False
    organized_count = 0
    failures: list[str] = []
    try:
        organized_count = organize_library_singles(
            path,
            destination_directory,
            dry_run=dry_run,
            max_threads=threads,
            failures=failures,
        )
    except (KeyboardInterrupt, RuntimeError) as exc:
        if not is_interruption(exc):
            raise
        interrupted = True
        if isinstance(exc, InterruptedOperationError) and isinstance(
            exc.partial_result, int
        ):
            organized_count = exc.partial_result
        LOG.warning(
            "\n⏹️  [bold yellow]INTERRUPTED[/] - Organization stopped by user (Ctrl+C). Generating summary..."
        )

    if not interrupted:
        LOG.success(f"Organized and moved {organized_count} single tracks.")
    else:
        LOG.warning(
            f"Partially organized {organized_count} single tracks before interruption."
        )

    organization_summary_rows = [
        ("Source Directory", str(path.resolve()), None),
        ("Target Directory", str(destination_directory.resolve()), None),
        (
            "Single Tracks Organized",
            str(organized_count),
            "green" if organized_count else "white",
        ),
    ]
    LOG.summary_table("Organization Summary", organization_summary_rows)

    if json_report:
        _write_json_report(
            json_report,
            {
                "schema": "organize_report_v1",
                "generator": "Sonora",
                "aborted_by_user": interrupted,
                "source_path": str(path.resolve()),
                "target_singles_path": str(destination_directory.resolve()),
                "summary": {
                    "single_tracks_organized": organized_count,
                    "failed_files": failures,
                },
            },
            "organization",
        )

    if interrupted:
        return 130
    return 1 if failures else 0


def backup(
    path: PathArg,
    output_file: Annotated[
        Path | None,
        Parameter(
            name=["--out", "-j", "--json"],
            help="Output JSON backup file path",
        ),
    ] = None,
    threads: ThreadsOpt = 4,
) -> int:
    """
    Create JSON backup of audio tags.
    """
    try:
        backup_path = backup_library_tags(
            path, output_file=output_file, max_threads=threads
        )
    except (KeyboardInterrupt, RuntimeError) as exc:
        if not is_interruption(exc):
            raise
        LOG.warning(
            "\n⏹️  [bold yellow]INTERRUPTED[/] - Backup stopped by user (Ctrl+C)."
        )
        return 130

    LOG.success(f"Backup created at: [bold]{escape(str(backup_path))}[/bold]")
    backup_summary_rows = [
        ("Source Directory", str(path.resolve()), None),
        ("Backup Archive", str(backup_path), "green"),
    ]
    LOG.summary_table("Backup Summary", backup_summary_rows)
    return 0


def restore(
    backup_file: Annotated[
        Path,
        Parameter(help="Path to JSON backup file"),
    ],
    target_directory: Annotated[
        Path | None, Parameter(name="--target", help="Relocated library root")
    ] = None,
    threads: ThreadsOpt = 4,
    json_report: JsonReportOpt = None,
) -> int:
    """
    Restore audio tags from JSON backup file.
    """
    interrupted = False
    restored_count = 0
    failures: list[str] = []
    try:
        restored_count = restore_library_tags(
            backup_file,
            target_directory=target_directory,
            max_threads=threads,
            failures=failures,
        )
    except (KeyboardInterrupt, RuntimeError) as exc:
        if not is_interruption(exc):
            raise
        interrupted = True
        if isinstance(exc, InterruptedOperationError) and isinstance(
            exc.partial_result, int
        ):
            restored_count = exc.partial_result
        LOG.warning(
            "\n⏹️  [bold yellow]INTERRUPTED[/] - Restoration stopped by user (Ctrl+C). Generating summary for restored files..."
        )

    if not interrupted:
        LOG.success(f"Restored metadata for {restored_count} tracks.")
    else:
        LOG.warning(f"Partially restored {restored_count} tracks before interruption.")

    restore_summary_rows = [
        ("Backup File", str(backup_file.resolve()), None),
        (
            "Tracks Restored",
            str(restored_count),
            "green" if restored_count else "white",
        ),
    ]
    LOG.summary_table("Restoration Summary", restore_summary_rows)

    if json_report:
        _write_json_report(
            json_report,
            {
                "schema": "restore_report_v1",
                "generator": "Sonora",
                "aborted_by_user": interrupted,
                "backup_file": str(backup_file.resolve()),
                "summary": {
                    "tracks_restored": restored_count,
                    "failed_files": failures,
                },
            },
            "restoration",
        )

    if interrupted:
        return 130
    return 1 if failures else 0


def normalize(
    path: PathArg,
    fetch_bpm: BpmOpt = True,
    fetch_key: KeyOpt = True,
    fetch_replaygain: ReplayGainOpt = True,
    no_dsp: NoDspOpt = False,
    force: ForceOpt = False,
    threads: ThreadsOpt = 4,
    dry_run: DryRunOpt = False,
    json_report: JsonReportOpt = None,
) -> int:
    """
    Locally clean tags, remove bracket noise, and calculate BPM/Key/ReplayGain (100% offline).
    """
    LOG.info(
        f"Normalizing audio tags in [bold]{escape(str(path))}[/bold] (offline mode)..."
    )
    if no_dsp:
        fetch_bpm = False
        fetch_key = False
        fetch_replaygain = False
    interrupted = False
    report = NormalizeReport([])
    try:
        report_or_list = normalize_library(
            path,
            fetch_bpm=fetch_bpm,
            fetch_key=fetch_key,
            fetch_replaygain=fetch_replaygain,
            force=force,
            dry_run=dry_run,
            max_threads=threads,
        )
        if isinstance(report_or_list, NormalizeReport):
            report = report_or_list
        else:
            report = NormalizeReport(report_or_list)
    except (KeyboardInterrupt, RuntimeError) as exc:
        if not is_interruption(exc):
            raise
        interrupted = True
        if isinstance(exc, InterruptedOperationError):
            if isinstance(exc.partial_result, NormalizeReport):
                report = exc.partial_result
            elif isinstance(exc.partial_result, list):
                report = NormalizeReport(exc.partial_result)
        LOG.warning(
            "\n⏹️  [bold yellow]INTERRUPTED[/] - Normalization stopped by user (Ctrl+C)."
        )

    total_scanned = report.total_scanned
    total_modified = report.total_modified
    total_unchanged = report.total_unchanged

    if not interrupted:
        LOG.success(
            f"Normalization completed: {total_modified} modified, {total_unchanged} already normalized ({total_scanned} total)."
        )
    else:
        LOG.warning(
            f"Partially processed {total_scanned} files before interruption ({total_modified} modified)."
        )

    summary_rows = [
        ("Target Directory", str(path.resolve()), None),
        ("Tracks Scanned", str(total_scanned), "cyan" if total_scanned else "white"),
        (
            "Tracks Modified",
            str(total_modified),
            "green" if total_modified else "white",
        ),
        (
            "Already Normalized",
            str(total_unchanged),
            "dim" if total_unchanged else "white",
        ),
        ("BPM Included", "Yes" if fetch_bpm else "No", None),
        ("Musical Key Included", "Yes" if fetch_key else "No", None),
        ("ReplayGain Included", "Yes" if fetch_replaygain else "No", None),
    ]
    LOG.summary_table("Normalization Summary", summary_rows)

    if json_report:
        report_data = {
            "summary": {
                "target_directory": str(path.resolve()),
                "total_scanned": total_scanned,
                "total_modified": total_modified,
                "total_unchanged": total_unchanged,
                "bpm_included": bool(fetch_bpm),
                "key_included": bool(fetch_key),
                "replaygain_included": bool(fetch_replaygain),
            },
            "modified_files": report.modified_files,
        }
        _write_json_report(json_report, report_data, "normalization")

    if interrupted:
        return 130
    return 0
