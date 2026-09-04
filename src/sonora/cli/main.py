import contextlib
import dataclasses
import datetime
import os
import signal
import socket
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated

import orjson
from cyclopts import App, Parameter
from dotenv import load_dotenv
from rich.markup import escape

from sonora import __version__
from sonora.audio.bpm import calculate_bpm
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
from sonora.core.logger import (
    LOG,
    create_progress,
    interactive_pause_listener,
    wait_if_paused,
)
from sonora.core.models import CheckReport, TrackInfo
from sonora.core.utils import (
    find_audio_files,
    format_filesize,
    group_files_by_parent,
)
from sonora.modules.backup import (
    backup_library_tags,
    get_last_restored_count,
    restore_library_tags,
)
from sonora.modules.checker import (
    check_library,
    get_last_check_report,
    write_check_report_json,
)
from sonora.modules.organizer import (
    get_last_organized_count,
    organize_library_singles,
)
from sonora.modules.renamer import get_last_rename_report, rename_directory_files
from sonora.modules.tagger import (
    get_last_normalized_count,
    get_last_tagged_tracks,
    get_last_tagging_failures,
    normalize_library,
    tag_album_folder,
)
from sonora.services.lyrics import init_musixmatch_token, process_track_lyrics
from sonora.services.musicbrainz import init_musicbrainz

load_dotenv()
socket.setdefaulttimeout(15)

if hasattr(signal, "SIGCONT"):
    with contextlib.suppress(Exception):
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


@app.command
def tag(
    path: Annotated[
        Path,
        Parameter(help="Directory containing audio files to tag"),
    ],
    fetch_bpm: Annotated[
        bool,
        Parameter(
            name=["--bpm"],
            help="Calculate audio tempo (BPM)",
        ),
    ] = True,
    fetch_replaygain: Annotated[
        bool,
        Parameter(
            name=["--replaygain"],
            help="Calculate ReplayGain loudness normalization tags",
        ),
    ] = True,
    fetch_lyrics: Annotated[
        bool,
        Parameter(
            name=["--lyrics"],
            help="Fetch synchronized (.lrc) lyrics",
        ),
    ] = True,
    fetch_artwork: Annotated[
        bool,
        Parameter(
            name=["--art"],
            help="Download high-resolution album and artist artwork",
        ),
    ] = True,
    json_report: Annotated[
        Path | None,
        Parameter(
            name=["--json"],
            help="Output path to save tagging JSON report with statistics",
        ),
    ] = None,
    force: Annotated[
        bool,
        Parameter(
            negative="",
            help="Force retagging by ignoring cache and existing MusicBrainz IDs",
        ),
    ] = False,
    lastfm_api_key: Annotated[
        str | None,
        Parameter(
            name=["--lastfm-key"],
            help="Last.fm API key for genre and mood lookup",
        ),
    ] = None,
    acoustid_api_key: Annotated[
        str | None,
        Parameter(
            name=["--acoustid-key"],
            help="AcoustID API key for acoustic fingerprinting",
        ),
    ] = None,
    discogs_user_token: Annotated[
        str | None,
        Parameter(
            name=["--discogs-token"],
            help="Discogs personal user token",
        ),
    ] = None,
    genius_api_token: Annotated[
        str | None,
        Parameter(
            name=["--genius-token"],
            help="Genius API token for song descriptions",
        ),
    ] = None,
    threads: Annotated[
        int,
        Parameter(
            name=["-t", "--threads"],
            help="Number of parallel threads",
        ),
    ] = 4,
    dry_run: Annotated[
        bool,
        Parameter(
            negative="",
            help="Simulate actions without modifying files on disk",
        ),
    ] = False,
) -> int:
    """
    Tag audio files and albums automatically with all metadata, artwork, BPM, ReplayGain & lyrics.
    """
    init_musicbrainz()
    init_musixmatch_token()

    if force:
        set_ignore_cache(True)

    resolved_lastfm_key = lastfm_api_key or os.environ.get("LASTFM_API_KEY")
    resolved_acoustid_key = acoustid_api_key or os.environ.get("ACOUSTID_API_KEY")
    resolved_discogs_token = (
        discogs_user_token
        or os.environ.get("DISCOGS_TOKEN")
        or os.environ.get("DISCOGS_USER_TOKEN")
    )
    resolved_genius_token = genius_api_token or os.environ.get("GENIUS_API_TOKEN")
    has_musixmatch = bool(
        os.environ.get("MUSIXMATCH_TOKEN") or os.environ.get("MUSIXMATCH_USER_TOKEN")
    )

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

    if active_keys:
        LOG.info(f"🔑 [bold]Active API Keys/Tokens:[/] {', '.join(active_keys)}")
    else:
        LOG.info(
            "🔑 [dim]Active API Keys/Tokens:[/] [yellow]None (using free unauthenticated tiers)[/]"
        )

    LOG.info(f"Tagging album directory: [bold]{path}[/bold]")
    tagged_tracks: list[TrackInfo] = []
    interrupted = False
    try:
        tagged_tracks = tag_album_folder(
            path,
            max_threads=threads,
            fetch_bpm=fetch_bpm,
            fetch_replaygain=fetch_replaygain,
            fetch_lyrics=fetch_lyrics,
            fetch_itunes_art=fetch_artwork,
            lastfm_api_key=resolved_lastfm_key,
            acoustid_api_key=resolved_acoustid_key,
            discogs_user_token=resolved_discogs_token,
            genius_api_token=resolved_genius_token,
            force=force,
            dry_run=dry_run,
        )
    except KeyboardInterrupt:
        interrupted = True
        tagged_tracks = get_last_tagged_tracks()
        LOG.warning(
            "\n⏹️  [bold yellow]INTERRUPTED[/] - Tagging stopped by user (Ctrl+C). Generating summary for processed tracks..."
        )

    if not interrupted:
        LOG.success(f"Successfully tagged {len(tagged_tracks)} tracks.")
    else:
        LOG.warning(
            f"Partially processed {len(tagged_tracks)} tracks before interruption."
        )

    total_tracks = len(tagged_tracks)
    bpm_count = sum(1 for track in tagged_tracks if track.bpm is not None)
    replaygain_count = sum(
        1 for track in tagged_tracks if track.replaygain_track_gain is not None
    )
    musicbrainz_count = sum(
        1 for track in tagged_tracks if track.musicbrainz_trackid is not None
    )
    genre_count = sum(1 for track in tagged_tracks if track.genre is not None)
    lyrics_count = sum(
        1
        for track in tagged_tracks
        if track.lyrics is not None or track.synced_lyrics is not None
    )
    isrc_count = sum(1 for track in tagged_tracks if track.isrc is not None)
    discogs_count = sum(
        1 for track in tagged_tracks if track.discogs_release_id is not None
    )
    genius_count = sum(
        1
        for track in tagged_tracks
        if track.genius_song_id is not None or track.comment is not None
    )
    theaudiodb_count = sum(
        1
        for track in tagged_tracks
        if track.initial_key is not None
        or track.music_video_url is not None
        or track.mood is not None
    )
    key_count = sum(1 for track in tagged_tracks if track.initial_key is not None)
    composer_count = sum(1 for track in tagged_tracks if track.composer is not None)
    producers_count = sum(1 for track in tagged_tracks if track.producers is not None)
    advisory_count = sum(1 for track in tagged_tracks if track.advisory is not None)
    lossless_count = sum(1 for track in tagged_tracks if track.is_lossless)
    lossy_count = total_tracks - lossless_count

    bpm_percentage = (bpm_count / total_tracks * 100) if total_tracks > 0 else 0.0
    replaygain_percentage = (
        (replaygain_count / total_tracks * 100) if total_tracks > 0 else 0.0
    )
    musicbrainz_percentage = (
        (musicbrainz_count / total_tracks * 100) if total_tracks > 0 else 0.0
    )
    genre_percentage = (genre_count / total_tracks * 100) if total_tracks > 0 else 0.0
    lyrics_percentage = (lyrics_count / total_tracks * 100) if total_tracks > 0 else 0.0
    isrc_percentage = (isrc_count / total_tracks * 100) if total_tracks > 0 else 0.0
    key_percentage = (key_count / total_tracks * 100) if total_tracks > 0 else 0.0
    composer_percentage = (
        (composer_count / total_tracks * 100) if total_tracks > 0 else 0.0
    )
    producers_percentage = (
        (producers_count / total_tracks * 100) if total_tracks > 0 else 0.0
    )
    advisory_percentage = (
        (advisory_count / total_tracks * 100) if total_tracks > 0 else 0.0
    )
    discogs_percentage = (
        (discogs_count / total_tracks * 100) if total_tracks > 0 else 0.0
    )
    genius_percentage = (genius_count / total_tracks * 100) if total_tracks > 0 else 0.0
    theaudiodb_percentage = (
        (theaudiodb_count / total_tracks * 100) if total_tracks > 0 else 0.0
    )

    tag_summary_rows = [
        ("Total Tracks Processed", str(total_tracks), None),
        (
            "MusicBrainz Matched",
            f"{musicbrainz_count}/{total_tracks} ({musicbrainz_percentage:.0f}%)",
            None,
        ),
        (
            "Genre & Styles Tagged",
            f"{genre_count}/{total_tracks} ({genre_percentage:.0f}%)",
            None,
        ),
        (
            "Lyrics Attached (.lrc)",
            f"{lyrics_count}/{total_tracks} ({lyrics_percentage:.0f}%)",
            None,
        ),
        (
            "ISRC Registered",
            f"{isrc_count}/{total_tracks} ({isrc_percentage:.0f}%)",
            None,
        ),
        (
            "Tempo / BPM Calculated",
            f"{bpm_count}/{total_tracks} ({bpm_percentage:.0f}%)",
            None,
        ),
        (
            "ReplayGain Loudness",
            f"{replaygain_count}/{total_tracks} ({replaygain_percentage:.0f}%)",
            None,
        ),
        (
            "Musical Key / Tonality",
            f"{key_count}/{total_tracks} ({key_percentage:.0f}%)",
            None,
        ),
        (
            "Composers & Writers",
            f"{composer_count}/{total_tracks} ({composer_percentage:.0f}%)",
            None,
        ),
        (
            "Producers & Studio Credits",
            f"{producers_count}/{total_tracks} ({producers_percentage:.0f}%)",
            None,
        ),
        (
            "Song Stories & Annotations",
            f"{genius_count}/{total_tracks} ({genius_percentage:.0f}%)",
            None,
        ),
        (
            "Parental Advisory",
            f"{advisory_count}/{total_tracks} ({advisory_percentage:.0f}%)",
            None,
        ),
        (
            "Audio Quality Breakdown",
            f"{lossless_count} Lossless / {lossy_count} Lossy",
            None,
        ),
    ]
    LOG.summary_table("Tagging Summary", tag_summary_rows)

    if json_report:
        failures = get_last_tagging_failures()
        summary_text = (
            f"Processed {total_tracks} tracks ({len(failures)} failures). "
            f"MusicBrainz {musicbrainz_count}/{total_tracks} ({musicbrainz_percentage:.0f}%), "
            f"Genre {genre_count}/{total_tracks} ({genre_percentage:.0f}%), "
            f"Lyrics {lyrics_count}/{total_tracks} ({lyrics_percentage:.0f}%), "
            f"BPM {bpm_count}/{total_tracks} ({bpm_percentage:.0f}%), "
            f"ReplayGain {replaygain_count}/{total_tracks} ({replaygain_percentage:.0f}%)."
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
                    "musicbrainz_matched_count": musicbrainz_count,
                    "musicbrainz_percentage": round(musicbrainz_percentage, 1),
                    "genre_tagged_count": genre_count,
                    "genre_percentage": round(genre_percentage, 1),
                    "lyrics_tagged_count": lyrics_count,
                    "lyrics_percentage": round(lyrics_percentage, 1),
                    "isrc_tagged_count": isrc_count,
                    "isrc_percentage": round(isrc_percentage, 1),
                    "bpm_calculated_count": bpm_count,
                    "bpm_percentage": round(bpm_percentage, 1),
                    "replaygain_calculated_count": replaygain_count,
                    "replaygain_percentage": round(replaygain_percentage, 1),
                    "initial_key_count": key_count,
                    "initial_key_percentage": round(key_percentage, 1),
                    "composer_tagged_count": composer_count,
                    "composer_percentage": round(composer_percentage, 1),
                    "producers_tagged_count": producers_count,
                    "producers_percentage": round(producers_percentage, 1),
                    "advisory_tagged_count": advisory_count,
                    "advisory_percentage": round(advisory_percentage, 1),
                    "discogs_matched_count": discogs_count,
                    "discogs_percentage": round(discogs_percentage, 1),
                    "genius_matched_count": genius_count,
                    "genius_percentage": round(genius_percentage, 1),
                    "theaudiodb_matched_count": theaudiodb_count,
                    "theaudiodb_percentage": round(theaudiodb_percentage, 1),
                },
                "audio_formats": {
                    "lossless_tracks": lossless_count,
                    "lossy_tracks": lossy_count,
                },
            },
            "failures": failures,
            "tracks": [track.to_dict() for track in tagged_tracks],
        }
        json_report.write_bytes(
            orjson.dumps(
                report_data,
                option=orjson.OPT_INDENT_2
                | orjson.OPT_NON_STR_KEYS
                | orjson.OPT_SERIALIZE_DATACLASS,
            )
        )
        LOG.info(f"Saved tagging JSON report to [bold]{json_report}[/bold]")

    if interrupted:
        return 130
    return 0


@app.command
def check(
    path: Annotated[
        Path,
        Parameter(help="Directory containing music library to check"),
    ],
    json_report: Annotated[
        Path | None,
        Parameter(
            name=["--json"],
            help="Output path to save check JSON report",
        ),
    ] = None,
    spectral_analysis: Annotated[
        bool,
        Parameter(
            name=["--spectral"],
            negative="",
            help="Enable deep spectral cutoff analysis for fake lossless detection (slow)",
        ),
    ] = False,
    threads: Annotated[
        int,
        Parameter(
            name=["-t", "--threads"],
            help="Number of parallel threads",
        ),
    ] = 8,
) -> int:
    """
    Check music library for FLAC integrity, bracket corruption & missing LRCs.
    """
    LOG.info(f"Checking music library: [bold]{path}[/bold]")
    interrupted = False
    try:
        check_report = check_library(
            path,
            output_json=json_report,
            check_spectral=spectral_analysis,
            max_threads=threads,
        )
    except KeyboardInterrupt:
        interrupted = True
        check_report = get_last_check_report() or CheckReport(
            total_files=0, corrupt_files=0, missing_metadata=0, missing_lrc=0
        )
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
    path: Annotated[
        Path,
        Parameter(help="Directory containing audio files to rename"),
    ],
    threads: Annotated[
        int,
        Parameter(
            name=["-t", "--threads"],
            help="Number of threads for parallel processing",
        ),
    ] = 4,
    dry_run: Annotated[
        bool,
        Parameter(
            negative="",
            help="Simulate actions without modifying files on disk",
        ),
    ] = False,
    json_report: Annotated[
        Path | None,
        Parameter(
            name=["-j", "--json"],
            help="Save renaming report to JSON file",
        ),
    ] = None,
) -> int:
    """
    Rename audio files and sync .lrc metadata headers.
    """
    LOG.info(f"Renaming files in directory: [bold]{path}[/bold]")
    interrupted = False
    try:
        rename_directory_files(path, dry_run=dry_run, max_threads=threads)
        report = get_last_rename_report()
    except KeyboardInterrupt:
        interrupted = True
        report = get_last_rename_report()
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
        rename_json_data = {
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
        }
        json_report.write_bytes(
            orjson.dumps(rename_json_data, option=orjson.OPT_INDENT_2)
        )
        LOG.info(f"Saved renaming JSON report to [bold]{json_report}[/bold]")
    if interrupted:
        return 130
    return 0


@app.command
def organize(
    path: Annotated[
        Path,
        Parameter(help="Source music directory"),
    ],
    target_singles: Annotated[
        Path | None,
        Parameter(
            name=["--target-singles"],
            help="Destination directory for single tracks (default: <path>/Singles)",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        Parameter(
            negative="",
            help="Simulate actions without modifying files on disk",
        ),
    ] = False,
    threads: Annotated[
        int,
        Parameter(
            name=["-t", "--threads"],
            help="Number of worker threads (default: 4)",
        ),
    ] = 4,
    json_report: Annotated[
        Path | None,
        Parameter(
            name=["-j", "--json"],
            help="Save organization report to JSON file",
        ),
    ] = None,
) -> int:
    """
    Organize single tracks into a Singles directory structure.
    """
    destination_directory = target_singles or (path / "Singles")
    LOG.info(f"Organizing single tracks from {path} to {destination_directory}")
    interrupted = False
    try:
        organized_count = organize_library_singles(
            path, destination_directory, dry_run=dry_run, max_threads=threads
        )
    except KeyboardInterrupt:
        interrupted = True
        organized_count = get_last_organized_count()
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
        organize_json_data = {
            "schema": "organize_report_v1",
            "generator": "Sonora",
            "aborted_by_user": interrupted,
            "source_path": str(path.resolve()),
            "target_singles_path": str(destination_directory.resolve()),
            "summary": {
                "single_tracks_organized": organized_count,
            },
        }
        json_report.write_bytes(
            orjson.dumps(organize_json_data, option=orjson.OPT_INDENT_2)
        )
        LOG.info(f"Saved organization JSON report to [bold]{json_report}[/bold]")

    if interrupted:
        return 130
    return 0


@app.command
def backup(
    path: Annotated[
        Path,
        Parameter(help="Music directory to back up"),
    ],
    output_file: Annotated[
        Path | None,
        Parameter(
            name=["--out", "-j", "--json"],
            help="Output JSON backup file path",
        ),
    ] = None,
    threads: Annotated[
        int,
        Parameter(
            name=["-t", "--threads"],
            help="Number of parallel threads",
        ),
    ] = 4,
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
    threads: Annotated[
        int,
        Parameter(
            name=["-t", "--threads"],
            help="Number of parallel threads",
        ),
    ] = 4,
    json_report: Annotated[
        Path | None,
        Parameter(
            name=["-j", "--json"],
            help="Save restoration report to JSON file",
        ),
    ] = None,
) -> int:
    """
    Restore audio tags from JSON backup file.
    """
    interrupted = False
    try:
        restored_count = restore_library_tags(backup_file, max_threads=threads)
    except KeyboardInterrupt:
        interrupted = True
        restored_count = get_last_restored_count()
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
        restore_json_data = {
            "schema": "restore_report_v1",
            "generator": "Sonora",
            "aborted_by_user": interrupted,
            "backup_file": str(backup_file.resolve()),
            "summary": {
                "tracks_restored": restored_count,
            },
        }
        json_report.write_bytes(
            orjson.dumps(restore_json_data, option=orjson.OPT_INDENT_2)
        )
        LOG.info(f"Saved restoration JSON report to [bold]{json_report}[/bold]")

    if interrupted:
        return 130
    return 0


@app.command
def normalize(
    path: Annotated[
        Path,
        Parameter(help="Directory containing audio files to normalize"),
    ],
    fetch_bpm: Annotated[
        bool,
        Parameter(
            name=["--bpm"],
            help="Calculate audio tempo (BPM) locally",
        ),
    ] = True,
    fetch_replaygain: Annotated[
        bool,
        Parameter(
            name=["--replaygain"],
            help="Calculate ReplayGain loudness normalization locally",
        ),
    ] = True,
    force: Annotated[
        bool,
        Parameter(
            negative="",
            help="Force re-normalization and recalculation of all files",
        ),
    ] = False,
    threads: Annotated[
        int,
        Parameter(
            name=["-t", "--threads"],
            help="Number of parallel threads",
        ),
    ] = 4,
    dry_run: Annotated[
        bool,
        Parameter(
            negative="",
            help="Simulate actions without modifying files on disk",
        ),
    ] = False,
) -> int:
    """
    Locally clean tags, remove bracket noise, and calculate BPM/ReplayGain (100% offline).
    """
    LOG.info(f"Normalizing audio tags in [bold]{path}[/bold] (offline mode)...")
    interrupted = False
    try:
        results = normalize_library(
            path,
            fetch_bpm=fetch_bpm,
            fetch_replaygain=fetch_replaygain,
            force=force,
            dry_run=dry_run,
            max_threads=threads,
        )
        count = len(results)
    except KeyboardInterrupt:
        interrupted = True
        count = get_last_normalized_count()
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
        ("ReplayGain Included", "Yes" if fetch_replaygain else "No", None),
    ]
    LOG.summary_table("Normalization Summary", summary_rows)
    if interrupted:
        return 130
    return 0


@app.command
def bpm(
    path: Annotated[
        Path,
        Parameter(help="Directory containing audio files to calculate BPM for"),
    ],
    force: Annotated[
        bool,
        Parameter(
            negative="",
            help="Force recalculation even if BPM tag exists",
        ),
    ] = False,
    threads: Annotated[
        int,
        Parameter(
            name=["-t", "--threads"],
            help="Number of parallel threads",
        ),
    ] = 4,
    dry_run: Annotated[
        bool,
        Parameter(
            negative="",
            help="Simulate actions without modifying files on disk",
        ),
    ] = False,
) -> int:
    """
    Calculate and embed audio tempo (BPM) tags locally.
    """
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    audio_files = find_audio_files(path, recursive=True)
    if not audio_files:
        LOG.warning("No audio files found.")
        return 0

    LOG.info(f"Calculating BPM for {len(audio_files)} files in [bold]{path}[/bold]...")
    computed = 0
    skipped = 0
    interrupted = False

    def _process_bpm(audio_path: Path) -> tuple[Path, float | None, bool]:
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

    with create_progress() as progress:
        task = progress.add_task("[cyan]Calculating BPM...", total=len(audio_files))
        with interactive_pause_listener(progress, task):
            executor = ThreadPoolExecutor(max_workers=threads)
            try:
                futures = [executor.submit(_process_bpm, f) for f in audio_files]
                for future in as_completed(futures):
                    wait_if_paused()
                    _, bpm_val, modified = future.result()
                    if modified and bpm_val is not None:
                        computed += 1
                    else:
                        skipped += 1
                    progress.advance(task)
            except KeyboardInterrupt:
                executor.shutdown(wait=False, cancel_futures=True)
                interrupted = True
                LOG.warning(
                    "\n⏹️  [bold yellow]INTERRUPTED[/] - BPM calculation stopped by user (Ctrl+C)."
                )
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

    summary_rows = [
        ("Total Files Scanned", str(len(audio_files)), None),
        ("BPM Calculated & Tagged", str(computed), "green" if computed else "white"),
        ("Already Tagged / Skipped", str(skipped), None),
    ]
    LOG.summary_table("BPM Summary", summary_rows)
    if interrupted:
        return 130
    return 0


@app.command
def replaygain(
    path: Annotated[
        Path,
        Parameter(help="Directory containing audio files to calculate ReplayGain for"),
    ],
    force: Annotated[
        bool,
        Parameter(
            negative="",
            help="Force recalculation even if ReplayGain tags exist",
        ),
    ] = False,
    threads: Annotated[
        int,
        Parameter(
            name=["-t", "--threads"],
            help="Number of parallel threads",
        ),
    ] = 4,
    dry_run: Annotated[
        bool,
        Parameter(
            negative="",
            help="Simulate actions without modifying files on disk",
        ),
    ] = False,
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


@app.command
def lyrics(
    path: Annotated[
        Path,
        Parameter(help="Directory containing audio files to fetch lyrics for"),
    ],
    force: Annotated[
        bool,
        Parameter(
            negative="",
            help="Force re-fetching even if lyrics exist",
        ),
    ] = False,
    threads: Annotated[
        int,
        Parameter(
            name=["-t", "--threads"],
            help="Number of parallel threads",
        ),
    ] = 4,
    dry_run: Annotated[
        bool,
        Parameter(
            negative="",
            help="Simulate actions without modifying files on disk",
        ),
    ] = False,
) -> int:
    """
    Fetch and save synchronized lyrics (.lrc) files and embedded lyrics.
    """
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    init_musixmatch_token()
    audio_files = find_audio_files(path, recursive=True)
    if not audio_files:
        LOG.warning("No audio files found.")
        return 0

    LOG.info(
        f"Fetching synchronized lyrics for {len(audio_files)} files in [bold]{path}[/bold]..."
    )
    saved_count = 0
    skipped_count = 0
    missing_count = 0
    interrupted = False

    def _process_lyrics(audio_path: Path) -> tuple[Path, str | None, str | None]:
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
                try:
                    updated = dataclasses.replace(info, lyrics=lyrics_text)
                    write_track_metadata(updated)
                except (OSError, ValueError, RuntimeError):
                    pass
            return audio_path, lyrics_text, tag_type
        except (OSError, ValueError, RuntimeError) as err:
            LOG.debug(f"Lyrics error for {audio_path}: {err}")
            return audio_path, None, None

    with create_progress() as progress:
        task = progress.add_task("[cyan]Fetching lyrics...", total=len(audio_files))
        with interactive_pause_listener(progress, task):
            executor = ThreadPoolExecutor(max_workers=threads)
            try:
                futures = [executor.submit(_process_lyrics, f) for f in audio_files]
                for future in as_completed(futures):
                    wait_if_paused()
                    _, lyr_content, lyr_type = future.result()
                    if lyr_type == "existing":
                        skipped_count += 1
                    elif lyr_content:
                        saved_count += 1
                    else:
                        missing_count += 1
                    progress.advance(task)
            except KeyboardInterrupt:
                executor.shutdown(wait=False, cancel_futures=True)
                interrupted = True
                LOG.warning(
                    "\n⏹️  [bold yellow]INTERRUPTED[/] - Lyrics fetch stopped by user (Ctrl+C)."
                )
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

    summary_rows = [
        ("Total Files Scanned", str(len(audio_files)), None),
        (
            "Lyrics Saved / Updated",
            str(saved_count),
            "green" if saved_count else "white",
        ),
        ("Already Had Lyrics", str(skipped_count), None),
        ("Lyrics Unavailable", str(missing_count), "yellow" if missing_count else None),
    ]
    LOG.summary_table("Lyrics Summary", summary_rows)
    if interrupted:
        return 130
    return 0


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
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        LOG.error(f"Error: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
