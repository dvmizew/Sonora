import argparse
import datetime
import os
import socket
import sys
from collections.abc import Sequence
from pathlib import Path

import orjson
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
from sonora.modules.tagger import tag_album_folder
from sonora.services.musicbrainz import init_musicbrainz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sonora",
        description="Sonora - Music tagging, library checking, and file organization",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate actions without modifying files",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    tag_parser = subparsers.add_parser(
        "tag",
        help="Tag audio files and albums automatically with all metadata, artwork, BPM, ReplayGain & lyrics",
    )
    tag_parser.add_argument(
        "path", type=Path, help="Directory containing audio files to tag"
    )
    tag_parser.add_argument(
        "--no-bpm",
        action="store_false",
        dest="fetch_bpm",
        help="Disable BPM calculation",
    )
    tag_parser.add_argument(
        "--no-replaygain",
        action="store_false",
        dest="fetch_replaygain",
        help="Disable ReplayGain calculation",
    )
    tag_parser.add_argument(
        "--no-lyrics",
        action="store_false",
        dest="fetch_lyrics",
        help="Disable LRC lyrics fetching",
    )
    tag_parser.add_argument(
        "--no-art",
        action="store_false",
        dest="fetch_itunes_art",
        help="Disable cover art downloading",
    )
    tag_parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Output path to save tagging JSON report with statistics",
    )
    tag_parser.add_argument(
        "--force",
        action="store_true",
        help="Force retagging by ignoring cache and existing MBIDs",
    )
    tag_parser.add_argument(
        "--lastfm-key",
        type=str,
        default=None,
        help="Last.fm API key for genre/mood lookup",
    )
    tag_parser.add_argument(
        "--acoustid-key",
        type=str,
        default=None,
        help="AcoustID API key for acoustic fingerprinting",
    )
    tag_parser.add_argument(
        "--discogs-token", type=str, default=None, help="Discogs personal user token"
    )
    tag_parser.add_argument(
        "--genius-token",
        type=str,
        default=None,
        help="Genius API token for song descriptions",
    )
    tag_parser.add_argument(
        "-t",
        "--threads",
        "-w",
        "--workers",
        type=int,
        default=4,
        dest="workers",
        help="Number of parallel worker threads (default: 4)",
    )
    check_parser = subparsers.add_parser(
        "check",
        help="Check music library for FLAC integrity, bracket corruption & missing LRCs",
    )
    check_parser.add_argument(
        "path", type=Path, help="Directory containing music library to check"
    )
    check_parser.add_argument(
        "--json", type=Path, default=None, help="Output path to save check JSON report"
    )
    check_parser.add_argument(
        "--spectral",
        action="store_true",
        help="Enable deep spectral cutoff analysis for fake lossless detection (slow)",
    )
    check_parser.add_argument(
        "-t",
        "--threads",
        "-w",
        "--workers",
        type=int,
        default=8,
        dest="workers",
        help="Number of parallel worker threads (default: 8)",
    )
    rename_parser = subparsers.add_parser(
        "rename", help="Rename audio files and sync .lrc metadata headers"
    )
    rename_parser.add_argument(
        "path", type=Path, help="Directory containing audio files to rename"
    )
    organize_parser = subparsers.add_parser(
        "organize", help="Organize single tracks into a Singles directory structure"
    )
    organize_parser.add_argument("path", type=Path, help="Source music directory")
    organize_parser.add_argument(
        "--target-singles",
        type=Path,
        default=None,
        help="Destination directory for single tracks (default: <path>/Singles)",
    )

    backup_parser = subparsers.add_parser(
        "backup", help="Create streaming JSON backup of audio tags"
    )
    backup_parser.add_argument("path", type=Path, help="Music directory to back up")
    backup_parser.add_argument(
        "--out", type=Path, default=None, help="Output JSON backup file path"
    )

    restore_parser = subparsers.add_parser(
        "restore", help="Restore audio tags from JSON backup file"
    )
    restore_parser.add_argument(
        "backup_file", type=Path, help="Path to JSON backup file"
    )

    return parser


def handle_tag(args: argparse.Namespace) -> int:
    init_musicbrainz()

    lastfm_key = args.lastfm_key or os.environ.get("LASTFM_API_KEY")
    acoustid_key = args.acoustid_key or os.environ.get("ACOUSTID_API_KEY")
    discogs_token = args.discogs_token or os.environ.get("DISCOGS_TOKEN")
    genius_token = args.genius_token or os.environ.get("GENIUS_API_TOKEN")

    LOG.info(f"Tagging album directory: [bold]{args.path}[/bold]")
    results = tag_album_folder(
        args.path,
        max_workers=args.workers,
        fetch_bpm=args.fetch_bpm,
        fetch_replaygain=args.fetch_replaygain,
        fetch_lyrics=args.fetch_lyrics,
        fetch_itunes_art=args.fetch_itunes_art,
        lastfm_api_key=lastfm_key,
        acoustid_api_key=acoustid_key,
        discogs_user_token=discogs_token,
        genius_api_token=genius_token,
        force=args.force,
        dry_run=args.dry_run,
    )
    LOG.success(f"Successfully tagged {len(results)} tracks.")

    total = len(results)
    bpm_count = sum(1 for track in results if track.bpm is not None)
    replaygain_count = sum(
        1 for track in results if track.replaygain_track_gain is not None
    )
    musicbrainz_count = sum(
        1 for track in results if track.musicbrainz_trackid is not None
    )
    genre_count = sum(1 for track in results if track.genre is not None)

    bpm_percentage = (bpm_count / total * 100) if total > 0 else 0.0
    replaygain_percentage = (replaygain_count / total * 100) if total > 0 else 0.0
    musicbrainz_percentage = (musicbrainz_count / total * 100) if total > 0 else 0.0
    genre_percentage = (genre_count / total * 100) if total > 0 else 0.0

    tag_rows = [
        ("Total Tracks Processed", str(total), None),
        ("BPM Calculated", f"{bpm_count}/{total} ({bpm_percentage:.0f}%)", None),
        (
            "ReplayGain Calculated",
            f"{replaygain_count}/{total} ({replaygain_percentage:.0f}%)",
            None,
        ),
        (
            "MusicBrainz Matched",
            f"{musicbrainz_count}/{total} ({musicbrainz_percentage:.0f}%)",
            None,
        ),
        (
            "Genre Tagged",
            f"{genre_count}/{total} ({genre_percentage:.0f}%)",
            None,
        ),
    ]
    LOG.summary_table("Tagging Summary", tag_rows)

    if args.json:
        summary_text = (
            f"Successfully processed {total} tracks. "
            f"BPM {bpm_count}/{total} ({bpm_percentage:.0f}%), "
            f"ReplayGain {replaygain_count}/{total} ({replaygain_percentage:.0f}%), "
            f"MusicBrainz MBID {musicbrainz_count}/{total} ({musicbrainz_percentage:.0f}%), "
            f"Genre {genre_count}/{total} ({genre_percentage:.0f}%)."
        )

        report_data = {
            "schema": "tag_report_v1",
            "generator": "Sonora",
            "version": __version__,
            "summary_text": summary_text,
            "execution": {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "target_path": str(args.path.resolve()),
                "workers_used": args.workers,
            },
            "statistics": {
                "total_tracks": total,
                "enrichment": {
                    "bpm_calculated_count": bpm_count,
                    "bpm_percentage": round(bpm_percentage, 1),
                    "replaygain_calculated_count": replaygain_count,
                    "replaygain_percentage": round(replaygain_percentage, 1),
                    "musicbrainz_matched_count": musicbrainz_count,
                    "musicbrainz_percentage": round(musicbrainz_percentage, 1),
                    "genre_tagged_count": genre_count,
                    "genre_percentage": round(genre_percentage, 1),
                },
            },
            "tracks": [track.to_dict() for track in results],
        }
        args.json.write_bytes(
            orjson.dumps(
                report_data,
                option=orjson.OPT_INDENT_2
                | orjson.OPT_NON_STR_KEYS
                | orjson.OPT_SERIALIZE_DATACLASS,
            )
        )
        LOG.info(f"Saved tagging JSON report to [bold]{args.json}[/bold]")

    return 0


def handle_check(args: argparse.Namespace) -> int:
    LOG.info(f"Checking music library: [bold]{args.path}[/bold]")
    report = check_library(
        args.path,
        output_json=args.json,
        check_spectral=args.spectral,
        max_workers=args.workers,
    )
    LOG.success(
        f"Check completed: {report.total_files} files scanned, {len(report.issues)} issues identified."
    )

    check_rows = [
        ("Total Files Scanned", str(report.total_files), None),
        (
            "Corrupt Files",
            str(report.corrupt_files),
            "red" if report.corrupt_files else "green",
        ),
        (
            "Missing Metadata",
            str(report.missing_metadata),
            "yellow" if report.missing_metadata else "green",
        ),
        (
            "Missing LRC Lyrics",
            str(report.missing_lrc),
            "yellow" if report.missing_lrc else "green",
        ),
        (
            "Files with Issues",
            str(len(report.issues)),
            "red" if report.issues else "green",
        ),
    ]
    LOG.summary_table("Validation Summary", check_rows)
    return 0


def handle_rename(args: argparse.Namespace) -> int:
    LOG.info(f"Renaming files in directory: [bold]{args.path}[/bold]")
    renamed = rename_directory_files(args.path, dry_run=args.dry_run)
    LOG.success(f"Renamed {len(renamed)} audio files and synchronized .lrc headers.")

    rename_rows = [
        ("Audio Files Processed", str(len(renamed)), None),
        (
            "Files Renamed & LRC Synced",
            str(len(renamed)),
            "green" if renamed else "white",
        ),
    ]
    LOG.summary_table("Renaming Summary", rename_rows)
    return 0


def handle_organize(args: argparse.Namespace) -> int:
    target_singles = args.target_singles or (args.path / "Singles")
    LOG.info(f"Organizing single tracks from {args.path} to {target_singles}")
    count = organize_library_singles(args.path, target_singles, dry_run=args.dry_run)
    LOG.success(f"Organized and moved {count} single tracks.")

    organize_rows = [
        ("Single Tracks Organized", str(count), "green" if count else "white"),
    ]
    LOG.summary_table("Organization Summary", organize_rows)
    return 0


def handle_backup(args: argparse.Namespace) -> int:
    backup_path = backup_library_tags(args.path, output_file=args.out)
    LOG.success(f"Backup created at: [bold]{backup_path}[/bold]")
    return 0


def handle_restore(args: argparse.Namespace) -> int:
    restored_count = restore_library_tags(args.backup_file)
    LOG.success(f"Restored metadata for {restored_count} tracks.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "verbose", False):
        LOG.verbose = True

    force = getattr(args, "force", False)
    if force:
        set_ignore_cache(True)

    if not args.command:
        parser.print_help()
        return 0

    try:
        if args.command == "tag":
            return handle_tag(args)
        elif args.command == "check":
            return handle_check(args)
        elif args.command == "rename":
            return handle_rename(args)
        elif args.command == "organize":
            return handle_organize(args)
        elif args.command == "backup":
            return handle_backup(args)
        elif args.command == "restore":
            return handle_restore(args)
        return 0
    except KeyboardInterrupt:
        LOG.warning("Aborted by user. Shutting down gracefully...")
        return 130
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        LOG.error(f"Error: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
