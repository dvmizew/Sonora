import datetime
import os
import socket
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import orjson
from cyclopts import App, Parameter
from dotenv import load_dotenv

load_dotenv()
socket.setdefaulttimeout(15)

from sonora import __version__
from sonora.core.cache import set_ignore_cache
from sonora.core.logger import LOG
from sonora.modules.backup import backup_library_tags, restore_library_tags
from sonora.modules.checker import check_library
from sonora.modules.organizer import organize_library_singles
from sonora.modules.renamer import rename_directory_files
from sonora.modules.tagger import get_last_tagging_failures, tag_album_folder
from sonora.services.musicbrainz import init_musicbrainz

app = App(
    name="sonora",
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

    LOG.info(f"Tagging album directory: [bold]{path}[/bold]")
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
    LOG.success(f"Successfully tagged {len(tagged_tracks)} tracks.")

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
            f"Successfully processed {total_tracks} tracks ({len(failures)} failures). "
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
    check_report = check_library(
        path,
        output_json=json_report,
        check_spectral=spectral_analysis,
        max_threads=threads,
    )
    LOG.success(
        f"Check completed: {check_report.total_files} files scanned, {len(check_report.issues)} issues identified."
    )

    validation_summary_rows = [
        ("Total Files Scanned", str(check_report.total_files), None),
        (
            "Corrupt Files",
            str(check_report.corrupt_files),
            "red" if check_report.corrupt_files else "green",
        ),
        (
            "Missing Metadata",
            str(check_report.missing_metadata),
            "yellow" if check_report.missing_metadata else "green",
        ),
        (
            "Missing LRC Lyrics",
            str(check_report.missing_lrc),
            "yellow" if check_report.missing_lrc else "green",
        ),
        (
            "Files with Issues",
            str(len(check_report.issues)),
            "red" if check_report.issues else "green",
        ),
    ]
    LOG.summary_table("Validation Summary", validation_summary_rows)
    return 0


@app.command
def rename(
    path: Annotated[
        Path,
        Parameter(help="Directory containing audio files to rename"),
    ],
    dry_run: Annotated[
        bool,
        Parameter(
            negative="",
            help="Simulate actions without modifying files on disk",
        ),
    ] = False,
) -> int:
    """
    Rename audio files and sync .lrc metadata headers.
    """
    LOG.info(f"Renaming files in directory: [bold]{path}[/bold]")
    renamed_files = rename_directory_files(path, dry_run=dry_run)
    LOG.success(
        f"Renamed {len(renamed_files)} audio files and synchronized .lrc headers."
    )

    renaming_summary_rows = [
        ("Audio Files Processed", str(len(renamed_files)), None),
        (
            "Files Renamed & LRC Synced",
            str(len(renamed_files)),
            "green" if renamed_files else "white",
        ),
    ]
    LOG.summary_table("Renaming Summary", renaming_summary_rows)
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
) -> int:
    """
    Organize single tracks into a Singles directory structure.
    """
    destination_directory = target_singles or (path / "Singles")
    LOG.info(f"Organizing single tracks from {path} to {destination_directory}")
    organized_count = organize_library_singles(
        path, destination_directory, dry_run=dry_run
    )
    LOG.success(f"Organized and moved {organized_count} single tracks.")

    organization_summary_rows = [
        (
            "Single Tracks Organized",
            str(organized_count),
            "green" if organized_count else "white",
        ),
    ]
    LOG.summary_table("Organization Summary", organization_summary_rows)
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
            name=["--out"],
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
    backup_path = backup_library_tags(
        path, output_file=output_file, max_threads=threads
    )
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
) -> int:
    """
    Restore audio tags from JSON backup file.
    """
    restored_count = restore_library_tags(backup_file, max_threads=threads)
    LOG.success(f"Restored metadata for {restored_count} tracks.")
    restore_summary_rows = [
        ("Backup File", str(backup_file.resolve()), None),
        (
            "Tracks Restored",
            str(restored_count),
            "green" if restored_count else "white",
        ),
    ]
    LOG.summary_table("Restoration Summary", restore_summary_rows)
    return 0


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
