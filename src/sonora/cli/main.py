import contextlib
import dataclasses
import datetime
import functools
import signal
import socket
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, Any, TypeVar

import orjson
from cyclopts import App, Parameter
from cyclopts.exceptions import CycloptsError
from rich.markup import escape

from sonora import __version__
from sonora.audio.bpm import calculate_bpm
from sonora.audio.key import detect_key_details
from sonora.audio.metadata import read_track_metadata, write_track_metadata
from sonora.audio.replaygain import calculate_album_replaygain
from sonora.core.cache import (
    CacheStats,
    get_cache_stats,
    set_ignore_cache,
)
from sonora.core.cache import (
    clear_cache as perform_clear_cache,
)
from sonora.core.config import (
    clear_config_cache,
    get_config,
    load_app_environment,
)
from sonora.core.logger import (
    LOG,
    create_progress,
    interactive_pause_listener,
    wait_if_paused,
)
from sonora.core.models import CheckReport, RenameReport, TrackInfo
from sonora.core.utils import (
    InterruptedOperationError,
    find_audio_files,
    format_filesize,
    group_files_by_parent,
)
from sonora.modules.backup import (
    backup_library_tags,
    restore_library_tags,
)
from sonora.modules.checker import (
    check_library,
    write_check_report_json,
)
from sonora.modules.organizer import (
    organize_library_singles,
)
from sonora.modules.renamer import rename_directory_files
from sonora.modules.tagger import (
    normalize_library,
    tag_album_folder,
)
from sonora.services.lyrics import init_musixmatch_token, process_track_lyrics
from sonora.services.musicbrainz import init_musicbrainz

load_app_environment()
socket.setdefaulttimeout(15)

if hasattr(signal, "SIGCONT"):
    with contextlib.suppress(ValueError, OSError):
        signal.signal(
            signal.SIGCONT,
            lambda *_: LOG.info("▶️ [green]Resumed execution.[/]"),
        )

app = App(
    "sonora",
    version=__version__,
    version_flags=["--version", "-v"],
    help_flags=["--help", "-h"],
    help="Sonora - Music tagging, library checking, and file organization",
    result_action="return_value",
)

T = TypeVar("T")

PathArg = Annotated[Path, Parameter(help="Directory containing audio files")]
ThreadsOpt = Annotated[
    int, Parameter(name=["-t", "--threads"], help="Number of parallel threads")
]
DryRunOpt = Annotated[
    bool,
    Parameter(negative="", help="Simulate actions without modifying files on disk"),
]
ForceOpt = Annotated[
    bool, Parameter(negative="", help="Force reprocessing even if valid")
]
JsonReportOpt = Annotated[
    Path | None, Parameter(name=["-j", "--json"], help="Save report to JSON file")
]
BpmOpt = Annotated[bool, Parameter(name=["--bpm"], help="Calculate audio tempo (BPM)")]
KeyOpt = Annotated[
    bool,
    Parameter(name=["--key"], help="Detect musical key and Camelot wheel tonality"),
]
ReplayGainOpt = Annotated[
    bool,
    Parameter(
        name=["--replaygain"], help="Calculate ReplayGain loudness normalization tags"
    ),
]
LyricsOpt = Annotated[
    bool, Parameter(name=["--lyrics"], help="Fetch synchronized (.lrc) lyrics")
]
ArtOpt = Annotated[
    bool,
    Parameter(name=["--art"], help="Download high-resolution album and artist artwork"),
]
ShazamOpt = Annotated[
    bool,
    Parameter(
        name=["--shazam"],
        help="Enable acoustic recognition via Shazam for untagged tracks",
    ),
]


def _write_json_report(
    json_report: Path,
    report_data: dict[str, Any],
    label: str = "report",
) -> None:
    json_report.write_bytes(
        orjson.dumps(
            report_data,
            option=orjson.OPT_INDENT_2
            | orjson.OPT_NON_STR_KEYS
            | orjson.OPT_SERIALIZE_DATACLASS,
        )
    )
    LOG.info(f"Saved {label} JSON report to [bold]{json_report}[/bold]")


def _emit_tag_summary_and_report(
    tagged_tracks: list[TrackInfo],
    failures: list[dict[str, str]],
    interrupted: bool,
    path: Path,
    threads: int,
    json_report: Path | None,
) -> None:
    total_tracks = len(tagged_tracks)
    counts = {
        "mb": sum(1 for t in tagged_tracks if t.musicbrainz_trackid is not None),
        "genre": sum(1 for t in tagged_tracks if t.genre is not None),
        "lyrics": sum(
            1
            for t in tagged_tracks
            if t.lyrics is not None or t.synced_lyrics is not None
        ),
        "isrc": sum(1 for t in tagged_tracks if t.isrc is not None),
        "bpm": sum(1 for t in tagged_tracks if t.bpm is not None),
        "rg": sum(1 for t in tagged_tracks if t.replaygain_track_gain is not None),
        "key": sum(1 for t in tagged_tracks if t.initial_key is not None),
        "composer": sum(1 for t in tagged_tracks if t.composer is not None),
        "producers": sum(1 for t in tagged_tracks if t.producers is not None),
        "advisory": sum(1 for t in tagged_tracks if t.advisory is not None),
        "discogs": sum(1 for t in tagged_tracks if t.discogs_release_id is not None),
        "genius": sum(
            1
            for t in tagged_tracks
            if t.genius_song_id is not None or t.comment is not None
        ),
        "theaudiodb": sum(
            1
            for t in tagged_tracks
            if t.initial_key is not None
            or t.music_video_url is not None
            or t.mood is not None
        ),
        "lossless": sum(1 for t in tagged_tracks if t.is_lossless),
    }
    lossy_count = total_tracks - counts["lossless"]
    pcts = {
        k: (v / total_tracks * 100) if total_tracks > 0 else 0.0
        for k, v in counts.items()
    }

    tag_summary_rows = [
        ("Total Tracks Processed", str(total_tracks), None),
        (
            "MusicBrainz Matched",
            f"{counts['mb']}/{total_tracks} ({pcts['mb']:.0f}%)",
            None,
        ),
        (
            "Genre & Styles Tagged",
            f"{counts['genre']}/{total_tracks} ({pcts['genre']:.0f}%)",
            None,
        ),
        (
            "Lyrics Attached (.lrc)",
            f"{counts['lyrics']}/{total_tracks} ({pcts['lyrics']:.0f}%)",
            None,
        ),
        (
            "ISRC Registered",
            f"{counts['isrc']}/{total_tracks} ({pcts['isrc']:.0f}%)",
            None,
        ),
        (
            "Tempo / BPM Calculated",
            f"{counts['bpm']}/{total_tracks} ({pcts['bpm']:.0f}%)",
            None,
        ),
        (
            "ReplayGain Loudness",
            f"{counts['rg']}/{total_tracks} ({pcts['rg']:.0f}%)",
            None,
        ),
        (
            "Musical Key / Tonality",
            f"{counts['key']}/{total_tracks} ({pcts['key']:.0f}%)",
            None,
        ),
        (
            "Composers & Writers",
            f"{counts['composer']}/{total_tracks} ({pcts['composer']:.0f}%)",
            None,
        ),
        (
            "Producers & Studio Credits",
            f"{counts['producers']}/{total_tracks} ({pcts['producers']:.0f}%)",
            None,
        ),
        (
            "Song Stories & Annotations",
            f"{counts['genius']}/{total_tracks} ({pcts['genius']:.0f}%)",
            None,
        ),
        (
            "Parental Advisory",
            f"{counts['advisory']}/{total_tracks} ({pcts['advisory']:.0f}%)",
            None,
        ),
        (
            "Audio Quality Breakdown",
            f"{counts['lossless']} Lossless / {lossy_count} Lossy",
            None,
        ),
    ]
    LOG.summary_table("Tagging Summary", tag_summary_rows)

    if json_report:
        summary_text = (
            f"Processed {total_tracks} tracks ({len(failures)} failures). "
            f"MusicBrainz {counts['mb']}/{total_tracks} ({pcts['mb']:.0f}%), "
            f"Genre {counts['genre']}/{total_tracks} ({pcts['genre']:.0f}%), "
            f"Lyrics {counts['lyrics']}/{total_tracks} ({pcts['lyrics']:.0f}%), "
            f"BPM {counts['bpm']}/{total_tracks} ({pcts['bpm']:.0f}%), "
            f"ReplayGain {counts['rg']}/{total_tracks} ({pcts['rg']:.0f}%)."
        )
        report_data = {
            "schema": "tag_report_v1",
            "generator": "Sonora",
            "version": __version__,
            "summary_text": summary_text,
            "aborted_by_user": interrupted,
            "execution": {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "target_path": str(path.resolve()),
                "threads_used": threads,
            },
            "statistics": {
                "total_tracks": total_tracks,
                "total_failures": len(failures),
                "enrichment": {
                    "musicbrainz_matched_count": counts["mb"],
                    "musicbrainz_percentage": round(pcts["mb"], 1),
                    "genre_tagged_count": counts["genre"],
                    "genre_percentage": round(pcts["genre"], 1),
                    "lyrics_tagged_count": counts["lyrics"],
                    "lyrics_percentage": round(pcts["lyrics"], 1),
                    "isrc_tagged_count": counts["isrc"],
                    "isrc_percentage": round(pcts["isrc"], 1),
                    "bpm_calculated_count": counts["bpm"],
                    "bpm_percentage": round(pcts["bpm"], 1),
                    "replaygain_calculated_count": counts["rg"],
                    "replaygain_percentage": round(pcts["rg"], 1),
                    "initial_key_count": counts["key"],
                    "initial_key_percentage": round(pcts["key"], 1),
                    "composer_tagged_count": counts["composer"],
                    "composer_percentage": round(pcts["composer"], 1),
                    "producers_tagged_count": counts["producers"],
                    "producers_percentage": round(pcts["producers"], 1),
                    "advisory_tagged_count": counts["advisory"],
                    "advisory_percentage": round(pcts["advisory"], 1),
                    "discogs_matched_count": counts["discogs"],
                    "discogs_percentage": round(pcts["discogs"], 1),
                    "genius_matched_count": counts["genius"],
                    "genius_percentage": round(pcts["genius"], 1),
                    "theaudiodb_matched_count": counts["theaudiodb"],
                    "theaudiodb_percentage": round(pcts["theaudiodb"], 1),
                },
                "audio_formats": {
                    "lossless_tracks": counts["lossless"],
                    "lossy_tracks": lossy_count,
                },
            },
            "failures": failures,
            "tracks": [track.to_dict() for track in tagged_tracks],
        }
        _write_json_report(json_report, report_data, "tagging")


@app.command
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
    threads: ThreadsOpt = 4,
    dry_run: DryRunOpt = False,
) -> int:
    """
    Tag audio files and albums automatically with all metadata, artwork, BPM, ReplayGain & lyrics.
    """
    load_app_environment(path)
    clear_config_cache()
    init_musicbrainz()
    init_musixmatch_token()

    if force:
        set_ignore_cache(True)

    cfg = get_config()
    resolved_lastfm_key = lastfm_api_key or cfg.lastfm_api_key
    resolved_acoustid_key = acoustid_api_key or cfg.acoustid_api_key
    resolved_discogs_token = discogs_user_token or cfg.discogs_token
    resolved_genius_token = genius_api_token or cfg.genius_api_token
    resolved_fanart_key = fanart_api_key or cfg.fanart_api_key
    resolved_fanart_client_key = fanart_client_key or cfg.fanart_client_key
    has_musixmatch = bool(cfg.musixmatch_token)

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
    except KeyboardInterrupt as exc:
        interrupted = True
        if isinstance(exc, InterruptedOperationError) and isinstance(
            exc.partial_result, list
        ):
            tagged_tracks = exc.partial_result
        LOG.warning(
            "\n⏹️  [bold yellow]INTERRUPTED[/] - Tagging stopped by user (Ctrl+C). Generating summary for processed tracks..."
        )

    if not interrupted:
        LOG.success(f"Successfully tagged {len(tagged_tracks)} tracks.")
    else:
        LOG.warning(
            f"Partially processed {len(tagged_tracks)} tracks before interruption."
        )

    _emit_tag_summary_and_report(
        tagged_tracks, failures, interrupted, path, threads, json_report
    )
    if interrupted:
        return 130
    return 0


@app.command
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
    LOG.info(f"Checking music library: [bold]{path}[/bold]")
    interrupted = False
    check_report = CheckReport(
        total_files=0, corrupt_files=0, missing_metadata=0, missing_lrc=0
    )
    try:
        check_report = check_library(
            path,
            output_json=json_report,
            check_spectral=spectral_analysis,
            max_threads=threads,
            report=check_report,
        )
    except KeyboardInterrupt as exc:
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
            f"Check completed: {check_report.total_files} files scanned, {len(check_report.issues)} issues identified."
        )
    else:
        LOG.warning(
            f"Partially scanned {check_report.total_files} files before interruption."
        )

    total = check_report.total_files
    issue_count = len(check_report.issues)
    perfect_files = max(0, total - issue_count)
    perfect_pct = (perfect_files / total * 100) if total > 0 else 0.0
    issues_pct = (issue_count / total * 100) if total > 0 else 0.0
    corrupt_pct = (check_report.corrupt_files / total * 100) if total > 0 else 0.0
    missing_meta_pct = (
        (check_report.missing_metadata / total * 100) if total > 0 else 0.0
    )
    missing_lrc_pct = (check_report.missing_lrc / total * 100) if total > 0 else 0.0

    validation_summary_rows = [
        ("Total Files Scanned", str(total), None),
        (
            "✅ Perfect Audio Files",
            f"{perfect_files}/{total} ({perfect_pct:.1f}%)",
            "green" if perfect_files == total else None,
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
            f"Saved validation JSON report with all {issue_count} issue(s) to [bold]{json_report}[/bold]"
        )

    if interrupted:
        return 130
    return 0


@app.command
def rename(
    path: PathArg,
    threads: ThreadsOpt = 4,
    dry_run: DryRunOpt = False,
    json_report: JsonReportOpt = None,
) -> int:
    """
    Rename audio files and sync .lrc metadata headers.
    """
    LOG.info(f"Renaming files in directory: [bold]{path}[/bold]")
    interrupted = False
    report = RenameReport()
    try:
        rename_directory_files(
            path, dry_run=dry_run, max_threads=threads, report=report
        )
    except KeyboardInterrupt as exc:
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
                },
            },
            "renaming",
        )
    if interrupted:
        return 130
    return 0


@app.command
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
    try:
        organized_count = organize_library_singles(
            path, destination_directory, dry_run=dry_run, max_threads=threads
        )
    except KeyboardInterrupt as exc:
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
                },
            },
            "organization",
        )

    if interrupted:
        return 130
    return 0


@app.command
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
    except KeyboardInterrupt:
        LOG.warning(
            "\n⏹️  [bold yellow]INTERRUPTED[/] - Backup stopped by user (Ctrl+C)."
        )
        return 130

    LOG.success(f"Backup created at: [bold]{backup_path}[/bold]")
    backup_summary_rows = [
        ("Source Directory", str(path.resolve()), None),
        ("Backup Archive", str(backup_path), "green"),
    ]
    LOG.summary_table("Backup Summary", backup_summary_rows)
    return 0


@app.command
def restore(
    backup_file: Annotated[
        Path,
        Parameter(help="Path to JSON backup file"),
    ],
    threads: ThreadsOpt = 4,
    json_report: JsonReportOpt = None,
) -> int:
    """
    Restore audio tags from JSON backup file.
    """
    interrupted = False
    restored_count = 0
    try:
        restored_count = restore_library_tags(backup_file, max_threads=threads)
    except KeyboardInterrupt as exc:
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
                },
            },
            "restoration",
        )

    if interrupted:
        return 130
    return 0


@app.command
def normalize(
    path: PathArg,
    fetch_bpm: BpmOpt = True,
    fetch_key: KeyOpt = True,
    fetch_replaygain: ReplayGainOpt = True,
    force: ForceOpt = False,
    threads: ThreadsOpt = 4,
    dry_run: DryRunOpt = False,
) -> int:
    """
    Locally clean tags, remove bracket noise, and calculate BPM/Key/ReplayGain (100% offline).
    """
    LOG.info(f"Normalizing audio tags in [bold]{path}[/bold] (offline mode)...")
    interrupted = False
    count = 0
    try:
        results = normalize_library(
            path,
            fetch_bpm=fetch_bpm,
            fetch_key=fetch_key,
            fetch_replaygain=fetch_replaygain,
            force=force,
            dry_run=dry_run,
            max_threads=threads,
        )
        count = len(results)
    except KeyboardInterrupt as exc:
        interrupted = True
        if isinstance(exc, InterruptedOperationError) and isinstance(
            exc.partial_result, list
        ):
            count = len(exc.partial_result)
        LOG.warning(
            "\n⏹️  [bold yellow]INTERRUPTED[/] - Normalization stopped by user (Ctrl+C)."
        )

    if not interrupted:
        LOG.success(f"Normalization completed for {count} files.")
    else:
        LOG.warning(f"Partially normalized {count} files before interruption.")

    summary_rows = [
        ("Target Directory", str(path.resolve()), None),
        ("Tracks Normalized", str(count), "green" if count else "white"),
        ("BPM Included", "Yes" if fetch_bpm else "No", None),
        ("Musical Key Included", "Yes" if fetch_key else "No", None),
        ("ReplayGain Included", "Yes" if fetch_replaygain else "No", None),
    ]
    LOG.summary_table("Normalization Summary", summary_rows)
    if interrupted:
        return 130
    return 0


def _run_parallel_audio_task(
    path: Path,
    description: str,
    worker: Callable[[Path], T],
    threads: int,
) -> tuple[list[T], bool]:
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    audio_files = find_audio_files(path, recursive=True)
    if not audio_files:
        LOG.warning("No audio files found.")
        return [], False

    LOG.info(f"{description} for {len(audio_files)} files in [bold]{path}[/bold]...")
    results: list[T] = []
    interrupted = False

    with create_progress() as progress:
        task = progress.add_task(f"[cyan]{description}...", total=len(audio_files))
        with (
            interactive_pause_listener(progress, task),
            ThreadPoolExecutor(max_workers=threads) as executor,
        ):
            try:
                futures = [executor.submit(worker, f) for f in audio_files]
                for future in as_completed(futures):
                    wait_if_paused()
                    results.append(future.result())
                    progress.advance(task)
            except KeyboardInterrupt:
                executor.shutdown(wait=True, cancel_futures=True)
                interrupted = True
                LOG.warning(
                    f"\n⏹️  [bold yellow]INTERRUPTED[/] - {description} stopped by user (Ctrl+C)."
                )

    return results, interrupted


def _process_bpm_file(
    audio_path: Path, force: bool, dry_run: bool
) -> tuple[Path, float | None, bool]:
    wait_if_paused()
    try:
        info = read_track_metadata(audio_path)
        if not force and info.bpm is not None:
            return audio_path, info.bpm, False
        val = calculate_bpm(audio_path)
        if val is not None and not dry_run:
            updated = dataclasses.replace(info, bpm=val)
            write_track_metadata(updated)
        return audio_path, val, True
    except (OSError, ValueError, RuntimeError) as err:
        LOG.debug(f"BPM error for {audio_path}: {err}")
        return audio_path, None, False


@app.command
def bpm(
    path: PathArg,
    force: ForceOpt = False,
    threads: ThreadsOpt = 4,
    dry_run: DryRunOpt = False,
) -> int:
    """
    Calculate and embed audio tempo (BPM) tags locally.
    """
    worker = functools.partial(_process_bpm_file, force=force, dry_run=dry_run)
    results, interrupted = _run_parallel_audio_task(
        path, "Calculating BPM", worker, threads
    )
    if not results and not interrupted:
        return 0

    computed = sum(1 for _, val, mod in results if mod and val is not None)
    skipped = len(results) - computed
    summary_rows = [
        ("Total Files Scanned", str(len(results)), None),
        ("BPM Calculated & Tagged", str(computed), "green" if computed else "white"),
        ("Already Tagged / Skipped", str(skipped), None),
    ]
    LOG.summary_table("BPM Summary", summary_rows)
    return 130 if interrupted else 0


def _process_key_file(
    audio_path: Path, force: bool, dry_run: bool
) -> tuple[Path, str | None, bool]:
    wait_if_paused()
    try:
        info = read_track_metadata(audio_path)
        if not force and info.initial_key is not None:
            return audio_path, info.initial_key, False
        details = detect_key_details(audio_path)
        if details is not None:
            val, camelot, _ = details
            if not dry_run:
                updated = dataclasses.replace(info, initial_key=val)
                write_track_metadata(updated)
            return audio_path, f"{val} ({camelot})", True
        return audio_path, None, False
    except (OSError, ValueError, RuntimeError) as err:
        LOG.debug(f"Key detection error for {audio_path}: {err}")
        return audio_path, None, False


@app.command
def key(
    path: PathArg,
    force: ForceOpt = False,
    threads: ThreadsOpt = 4,
    dry_run: DryRunOpt = False,
) -> int:
    """
    Detect and embed musical key (INITIALKEY) and Camelot wheel tags locally.
    """
    worker = functools.partial(_process_key_file, force=force, dry_run=dry_run)
    results, interrupted = _run_parallel_audio_task(
        path, "Detecting musical key", worker, threads
    )
    if not results and not interrupted:
        return 0

    computed = sum(1 for _, val, mod in results if mod and val is not None)
    skipped = len(results) - computed
    summary_rows = [
        ("Total Files Scanned", str(len(results)), None),
        ("Key Detected & Tagged", str(computed), "green" if computed else "white"),
        ("Already Tagged / Skipped", str(skipped), None),
    ]
    LOG.summary_table("Musical Key Summary", summary_rows)
    return 130 if interrupted else 0


@app.command
def replaygain(
    path: PathArg,
    force: ForceOpt = False,
    threads: ThreadsOpt = 4,
    dry_run: DryRunOpt = False,
) -> int:
    """
    Calculate and embed ReplayGain loudness normalization tags (Track & Album mode).
    """
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    audio_files = find_audio_files(path, recursive=True)
    if not audio_files:
        LOG.warning("No audio files found.")
        return 0

    album_groups = group_files_by_parent(audio_files)
    LOG.info(
        f"Calculating ReplayGain for {len(audio_files)} files across {len(album_groups)} folders..."
    )
    albums_processed = 0
    interrupted = False

    with create_progress() as progress:
        task = progress.add_task(
            "[cyan]Calculating ReplayGain...", total=len(album_groups)
        )
        with interactive_pause_listener(progress, task):
            try:
                for files in album_groups.values():
                    wait_if_paused()
                    success = calculate_album_replaygain(
                        files,
                        force=force,
                        dry_run=dry_run,
                        max_threads=threads,
                    )
                    if success:
                        albums_processed += 1
                    progress.advance(task)
            except KeyboardInterrupt:
                interrupted = True
                LOG.warning(
                    "\n⏹️  [bold yellow]INTERRUPTED[/] - ReplayGain stopped by user (Ctrl+C)."
                )

    summary_rows = [
        ("Total Folders Scanned", str(len(album_groups)), None),
        ("Total Files Scanned", str(len(audio_files)), None),
        (
            "Albums Tagged with ReplayGain",
            str(albums_processed),
            "green" if albums_processed else "white",
        ),
    ]
    LOG.summary_table("ReplayGain Summary", summary_rows)
    if interrupted:
        return 130
    return 0


def _process_lyrics_file(
    audio_path: Path, force: bool, dry_run: bool
) -> tuple[Path, str | None, str | None]:
    wait_if_paused()
    try:
        info = read_track_metadata(audio_path)
        lrc_path = audio_path.with_suffix(".lrc")
        if not force and lrc_path.exists() and lrc_path.stat().st_size > 0:
            return audio_path, "existing", "existing"
        lyrics_text, tag_type = process_track_lyrics(
            audio_path,
            info.artist,
            info.title,
            force=force,
            dry_run=dry_run,
            isrc=info.isrc,
        )
        if lyrics_text and not dry_run:
            with contextlib.suppress(OSError, ValueError, RuntimeError):
                updated = dataclasses.replace(info, lyrics=lyrics_text)
                write_track_metadata(updated)
        return audio_path, lyrics_text, tag_type
    except (OSError, ValueError, RuntimeError) as err:
        LOG.debug(f"Lyrics error for {audio_path}: {err}")
        return audio_path, None, None


@app.command
def lyrics(
    path: PathArg,
    force: ForceOpt = False,
    threads: ThreadsOpt = 4,
    dry_run: DryRunOpt = False,
) -> int:
    """
    Fetch and save synchronized lyrics (.lrc) files and embedded lyrics.
    """
    init_musixmatch_token()
    worker = functools.partial(_process_lyrics_file, force=force, dry_run=dry_run)
    results, interrupted = _run_parallel_audio_task(
        path, "Fetching synchronized lyrics", worker, threads
    )
    if not results and not interrupted:
        return 0

    saved_count = sum(1 for _, content, typ in results if typ != "existing" and content)
    skipped_count = sum(1 for _, _, typ in results if typ == "existing")
    missing_count = len(results) - saved_count - skipped_count

    summary_rows = [
        ("Total Files Scanned", str(len(results)), None),
        (
            "Lyrics Saved / Updated",
            str(saved_count),
            "green" if saved_count else "white",
        ),
        ("Already Had Lyrics", str(skipped_count), None),
        ("Lyrics Unavailable", str(missing_count), "yellow" if missing_count else None),
    ]
    LOG.summary_table("Lyrics Summary", summary_rows)
    return 130 if interrupted else 0


cache_app = App(
    "cache",
    help="Inspect and manage Sonora cache and persistent state",
    result_action="return_value",
)
app.command(cache_app)


def _display_cache_stats(stats: CacheStats) -> None:
    rows = [
        ("Cache Directory", escape(str(stats.cache_dir)), "cyan"),
        (
            "API Cached Entries",
            f"{stats.api_entries:,}",
            "green" if stats.api_entries else "white",
        ),
        (
            "API Cache Size on Disk",
            format_filesize(stats.api_size_bytes),
            "green" if stats.api_size_bytes else "white",
        ),
        (
            "Library State Entries",
            f"{stats.state_entries:,}",
            "green" if stats.state_entries else "white",
        ),
        (
            "Library State Size on Disk",
            format_filesize(stats.state_size_bytes),
            "green" if stats.state_size_bytes else "white",
        ),
        (
            "Memory Metadata Cache",
            f"{stats.memory_metadata_entries:,} items",
            "cyan",
        ),
        (
            "Total Cache Disk Usage",
            format_filesize(stats.total_size_bytes),
            "bold magenta",
        ),
    ]
    LOG.summary_table("Sonora Cache Statistics", rows)


@cache_app.command(name="stats")
def cache_stats(
    all_layers: Annotated[
        bool,
        Parameter(
            name=["-a", "--all"],
            negative="",
            help="Display comprehensive statistics across all cache layers",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        Parameter(
            name=["--json"],
            negative="",
            help="Output statistics in JSON format",
        ),
    ] = False,
) -> int:
    """
    Display current cache statistics and disk usage.
    """
    stats = get_cache_stats()
    if json_output:
        print(orjson.dumps(stats.to_dict(), option=orjson.OPT_INDENT_2).decode("utf-8"))
        return 0

    _display_cache_stats(stats)
    return 0


@cache_app.command(name="clear")
def cache_clear_cmd(
    all_caches: Annotated[
        bool,
        Parameter(
            name=["-a", "--all"],
            negative="",
            help="Clear all cache layers (API metadata, library state, and in-memory caches)",
        ),
    ] = False,
    api: Annotated[
        bool,
        Parameter(
            name=["--api"],
            negative="",
            help="Clear API metadata cache",
        ),
    ] = False,
    state: Annotated[
        bool,
        Parameter(
            name=["--state"],
            negative="",
            help="Clear persistent library tracking state database",
        ),
    ] = False,
    memory: Annotated[
        bool,
        Parameter(
            name=["--memory"],
            negative="",
            help="Clear in-memory metadata and normalization caches",
        ),
    ] = False,
    purge: Annotated[
        bool,
        Parameter(
            name=["-p", "--purge"],
            negative="",
            help="Purge cache files and SQLite databases entirely from disk",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        Parameter(
            negative="",
            help="Simulate cache clearing without modifying or deleting files",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        Parameter(
            name=["--json"],
            negative="",
            help="Output results in JSON format",
        ),
    ] = False,
) -> int:
    """
    Clear Sonora API cache, library state, and memory caches.
    """
    if all_caches:
        do_api = True
        do_state = True
        do_memory = True
    elif not (api or state or memory):
        do_api = True
        do_state = False
        do_memory = True
    else:
        do_api = api
        do_state = state
        do_memory = memory

    result = perform_clear_cache(
        clear_api=do_api,
        clear_state=do_state,
        clear_memory=do_memory,
        purge=purge,
        dry_run=dry_run,
    )

    if json_output:
        print(
            orjson.dumps(result.to_dict(), option=orjson.OPT_INDENT_2).decode("utf-8")
        )
        return 0

    if dry_run:
        LOG.info(
            f"[yellow][DRY RUN][/yellow] Simulated cache clearing for [bold]{escape(str(result.cache_dir))}[/bold]:"
        )
        rows = [
            (
                "API Cache Target",
                (
                    f"{result.api_entries_cleared:,} entries ({format_filesize(result.api_bytes_freed)})"
                    if result.api_cleared
                    else "Skipped"
                ),
                "yellow" if result.api_cleared else "white",
            ),
            (
                "Library State Target",
                (
                    f"{result.state_entries_cleared:,} tracks ({format_filesize(result.state_bytes_freed)})"
                    if result.state_cleared
                    else "Preserved (use --state or --all to clear)"
                ),
                "yellow" if result.state_cleared else "white",
            ),
            (
                "In-Memory Cache Target",
                (
                    f"{result.memory_metadata_cleared:,} entries"
                    if result.memory_cleared
                    else "Skipped"
                ),
                "yellow" if result.memory_cleared else "white",
            ),
            (
                "Action Mode",
                "Purge completely from disk" if purge else "Clear & Reclaim",
                "cyan",
            ),
            (
                "Total Space Reclaimable",
                format_filesize(result.total_bytes_freed),
                "bold yellow",
            ),
        ]
        LOG.summary_table("Dry Run Cache Clearing", rows)
        return 0

    mode_label = "Purged" if purge else "Cleared"
    rows = [
        ("Cache Directory", escape(str(result.cache_dir)), "cyan"),
        (
            "API Metadata Cache",
            (
                f"{result.api_entries_cleared:,} entries cleared ({format_filesize(result.api_bytes_freed)} freed)"
                if result.api_cleared
                else "Skipped"
            ),
            "green" if result.api_cleared else "white",
        ),
        (
            "Library State Cache",
            (
                f"{result.state_entries_cleared:,} tracks cleared ({format_filesize(result.state_bytes_freed)} freed)"
                if result.state_cleared
                else "Preserved (use --state or --all to clear)"
            ),
            "green" if result.state_cleared else "white",
        ),
        (
            "In-Memory Caches",
            (
                f"{result.memory_metadata_cleared:,} entries reset"
                if result.memory_cleared
                else "Skipped"
            ),
            "cyan" if result.memory_cleared else "white",
        ),
        ("Action Type", mode_label, "magenta"),
        (
            "Total Space Freed",
            format_filesize(result.total_bytes_freed),
            "bold green",
        ),
    ]
    LOG.summary_table(f"Cache {mode_label} Summary", rows)
    LOG.success(f"Cache successfully {mode_label.lower()}.")
    return 0


@cache_app.default
def cache_default() -> int:
    """
    Inspect Sonora cache status and available actions.
    """
    stats = get_cache_stats()
    _display_cache_stats(stats)
    LOG.info(
        "Run [bold cyan]sonora cache clear[/bold cyan] to clear API caches, or [bold cyan]sonora cache clear --all[/bold cyan] for a complete reset."
    )
    return 0


app.command(cache_clear_cmd, name="clear-cache")


def main(arguments: Sequence[str] | None = None) -> int:
    """
    CLI Entrypoint. Executes cyclopts App with provided or sys arguments.
    """
    if arguments is None:
        arguments = sys.argv[1:]

    if not arguments:
        app(["--help"], exit_on_error=False)
        return 0

    try:
        result = app(arguments, exit_on_error=False)
        if isinstance(result, int):
            return result
        return 0
    except KeyboardInterrupt:
        LOG.warning("Aborted by user. Shutting down gracefully...")
        return 130
    except CycloptsError:
        return 2
    except (OSError, ValueError, RuntimeError) as error:
        LOG.error(f"Error: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
