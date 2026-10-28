import dataclasses
import functools
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal, TypeVar

from rich.markup import escape

from sonora.audio.bpm import calculate_bpm
from sonora.audio.key import detect_key_details, key_to_camelot
from sonora.audio.metadata import (
    get_audio_duration,
    read_track_metadata,
    write_track_metadata,
)
from sonora.audio.replaygain import calculate_album_replaygain
from sonora.cli.options import (
    DryRunOpt,
    ForceOpt,
    JsonReportOpt,
    MusixmatchOpt,
    PathArg,
    ThreadsOpt,
)
from sonora.cli.reports import _write_json_report
from sonora.core.audio_files import find_audio_files
from sonora.core.cache import api_cache_session
from sonora.core.cancellation import is_interruption
from sonora.core.config import get_config
from sonora.core.library_paths import group_files_by_album_root
from sonora.core.logger import (
    LOG,
    create_progress,
    interactive_pause_listener,
    resume_workers,
    wait_if_paused,
)
from sonora.modules.track_processing import process_track_lyrics
from sonora.services.lyrics import parse_musixmatch_token

T = TypeVar("T")


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

    LOG.info(
        f"{description} for {len(audio_files)} files in [bold]{escape(str(path))}[/bold]..."
    )
    results: list[T] = []
    interrupted = False

    with create_progress() as progress:
        task = progress.add_task(f"[cyan]{description}...", total=len(audio_files))
        with (
            interactive_pause_listener(progress, task) as pause_stop_event,
            ThreadPoolExecutor(max_workers=threads) as executor,
        ):
            try:
                futures = [
                    executor.submit(worker, audio_path) for audio_path in audio_files
                ]
                for future in as_completed(futures):
                    wait_if_paused()
                    results.append(future.result())
                    progress.advance(task)
            except (KeyboardInterrupt, RuntimeError) as exc:
                if not is_interruption(exc):
                    raise
                resume_workers(pause_stop_event)
                executor.shutdown(wait=True, cancel_futures=True)
                interrupted = True
                LOG.warning(
                    f"\n⏹️  [bold yellow]INTERRUPTED[/] - {description} stopped by user (Ctrl+C)."
                )

    return results, interrupted


def _report_audio_measurements(
    path: Path,
    results: Sequence[tuple[Path, float | str | None, bool]],
    metric: Literal["bpm", "key"],
    dry_run: bool,
    json_report: Path | None,
) -> None:
    title, action, report_label = (
        ("BPM", "calculated", "bpm")
        if metric == "bpm"
        else ("Musical Key", "detected", "musical key")
    )
    computed = sum(
        1 for _, value, measured in results if measured and value is not None
    )
    skipped = len(results) - computed
    label = "BPM Calculated" if metric == "bpm" else "Key Detected"
    LOG.summary_table(
        f"{title} Summary",
        [
            ("Total Files Scanned", str(len(results)), None),
            (
                f"{label} (preview)" if dry_run else f"{label} & Tagged",
                str(computed),
                "green" if computed else "white",
            ),
            ("Already Tagged / Skipped", str(skipped), None),
        ],
    )
    if json_report:
        report_data = {
            "summary": {
                "target_directory": str(path.resolve()),
                "total_scanned": len(results),
                "dry_run": dry_run,
                f"{metric}_{action}": computed,
                f"{metric}_tagged": 0 if dry_run else computed,
                "skipped": skipped,
            },
            "details": [
                {
                    "file": str(audio_path),
                    metric: value,
                    action: measured,
                    "modified": measured and value is not None and not dry_run,
                }
                for audio_path, value, measured in results
            ],
        }
        _write_json_report(json_report, report_data, report_label)


def _process_bpm_file(
    audio_path: Path, force: bool, dry_run: bool
) -> tuple[Path, float | None, bool]:
    wait_if_paused()
    try:
        track_info = read_track_metadata(audio_path)
        if not force and track_info.bpm is not None:
            return audio_path, track_info.bpm, False
        bpm_value = calculate_bpm(audio_path)
        if bpm_value is not None and not dry_run:
            updated = dataclasses.replace(track_info, bpm=bpm_value)
            write_track_metadata(updated)
        return audio_path, bpm_value, True
    except OSError as error:
        LOG.debug(f"BPM error for {audio_path}: {error}")
        return audio_path, None, False


def bpm(
    path: PathArg,
    force: ForceOpt = False,
    threads: ThreadsOpt = 4,
    dry_run: DryRunOpt = False,
    json_report: JsonReportOpt = None,
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

    _report_audio_measurements(path, results, "bpm", dry_run, json_report)
    return 130 if interrupted else 0


def _process_key_file(
    audio_path: Path, force: bool, dry_run: bool
) -> tuple[Path, str | None, bool]:
    wait_if_paused()
    try:
        track_info = read_track_metadata(audio_path)
        is_invalid_key = bool(
            track_info.initial_key and key_to_camelot(track_info.initial_key) is None
        )
        if not force and track_info.initial_key is not None and not is_invalid_key:
            return audio_path, track_info.initial_key, False
        details = detect_key_details(audio_path)
        if details is not None:
            musical_key, camelot, _ = details
            if not dry_run:
                updated = dataclasses.replace(track_info, initial_key=musical_key)
                write_track_metadata(updated)
            return audio_path, f"{musical_key} ({camelot})", True
        return audio_path, None, False
    except OSError as error:
        LOG.debug(f"Key detection error for {audio_path}: {error}")
        return audio_path, None, False


def key(
    path: PathArg,
    force: ForceOpt = False,
    threads: ThreadsOpt = 4,
    dry_run: DryRunOpt = False,
    json_report: JsonReportOpt = None,
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

    _report_audio_measurements(path, results, "key", dry_run, json_report)
    return 130 if interrupted else 0


def replaygain(
    path: PathArg,
    force: ForceOpt = False,
    threads: ThreadsOpt = 4,
    dry_run: DryRunOpt = False,
    json_report: JsonReportOpt = None,
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

    album_groups = group_files_by_album_root(audio_files)
    LOG.info(
        f"Calculating ReplayGain for {len(audio_files)} files across {len(album_groups)} folders..."
    )
    albums_processed = 0
    interrupted = False

    with create_progress() as progress:
        task = progress.add_task(
            "[cyan]Calculating ReplayGain...", total=len(album_groups)
        )
        with interactive_pause_listener(progress, task) as pause_stop_event:
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
            except (KeyboardInterrupt, RuntimeError) as exc:
                if not is_interruption(exc):
                    raise
                resume_workers(pause_stop_event)
                interrupted = True
                LOG.warning(
                    "\n⏹️  [bold yellow]INTERRUPTED[/] - ReplayGain stopped by user (Ctrl+C)."
                )

    summary_rows = [
        ("Total Folders Scanned", str(len(album_groups)), None),
        ("Total Files Scanned", str(len(audio_files)), None),
        (
            "Albums Analyzed (preview)" if dry_run else "Albums Tagged with ReplayGain",
            str(albums_processed),
            "green" if albums_processed else "white",
        ),
    ]
    LOG.summary_table("ReplayGain Summary", summary_rows)
    if json_report:
        report_data = {
            "summary": {
                "target_directory": str(path.resolve()),
                "total_folders": len(album_groups),
                "total_files": len(audio_files),
                "dry_run": dry_run,
                "albums_processed": albums_processed,
            }
        }
        _write_json_report(json_report, report_data, "replaygain")
    if interrupted:
        return 130
    return 0


@api_cache_session()
def _process_lyrics_file(
    audio_path: Path, force: bool, dry_run: bool, musixmatch_token: str | None = None
) -> tuple[Path, str | None, str | None]:
    wait_if_paused()
    try:
        track_info = read_track_metadata(audio_path)
        audio_duration = get_audio_duration(audio_path)

        lyrics_text, tag_type = process_track_lyrics(
            audio_path,
            track_info.artist,
            track_info.title,
            force=force,
            dry_run=dry_run,
            isrc=track_info.isrc,
            album_name=track_info.album,
            duration=audio_duration,
            existing_lyrics=track_info.lyrics,
            musixmatch_token=musixmatch_token,
        )
        if lyrics_text and not dry_run:
            try:
                updated = dataclasses.replace(track_info, lyrics=lyrics_text)
                write_track_metadata(updated)
            except OSError as write_err:
                LOG.warning(
                    f"Failed to embed lyrics for {escape(str(audio_path))}: {escape(str(write_err))}"
                )
        return audio_path, lyrics_text, tag_type
    except OSError as error:
        LOG.debug(f"Lyrics error for {audio_path}: {error}")
        return audio_path, None, None


def lyrics(
    path: PathArg,
    force: ForceOpt = False,
    musixmatch_token: MusixmatchOpt = None,
    threads: ThreadsOpt = 4,
    dry_run: DryRunOpt = False,
    json_report: JsonReportOpt = None,
) -> int:
    """
    Fetch and save synchronized lyrics (.lrc) files and embedded lyrics.
    """
    token = parse_musixmatch_token(musixmatch_token or get_config().musixmatch_token)
    worker = functools.partial(
        _process_lyrics_file, force=force, dry_run=dry_run, musixmatch_token=token
    )
    results, interrupted = _run_parallel_audio_task(
        path, "Fetching synchronized lyrics", worker, threads
    )
    if not results and not interrupted:
        return 0

    saved_count = sum(
        1
        for _, content, lyrics_status in results
        if lyrics_status != "existing" and content
    )
    skipped_count = sum(
        1 for _, _, lyrics_status in results if lyrics_status == "existing"
    )
    missing_count = len(results) - saved_count - skipped_count

    summary_rows = [
        ("Total Files Scanned", str(len(results)), None),
        (
            "Lyrics Would Save / Update" if dry_run else "Lyrics Saved / Updated",
            str(saved_count),
            "green" if saved_count else "white",
        ),
        ("Already Had Lyrics", str(skipped_count), None),
        ("Lyrics Unavailable", str(missing_count), "yellow" if missing_count else None),
    ]
    LOG.summary_table("Lyrics Summary", summary_rows)

    if json_report:
        report_data = {
            "total_files": len(results),
            "dry_run": dry_run,
            "saved": 0 if dry_run else saved_count,
            "would_update": saved_count if dry_run else 0,
            "already_had": skipped_count,
            "unavailable": missing_count,
            "details": [
                {"file": str(audio_path), "status": lyrics_status}
                for audio_path, _, lyrics_status in results
            ],
        }
        _write_json_report(json_report, report_data, "lyrics")

    return 130 if interrupted else 0
