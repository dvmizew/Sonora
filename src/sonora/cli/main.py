"""
Command-line interface (CLI) main entrypoint for Sonora.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from sonora import __version__
from sonora.core.exceptions import SonoraError
from sonora.core.logger import LOG
from sonora.modules.auditor import audit_library
from sonora.modules.organizer import organize_library_singles
from sonora.modules.renamer import rename_directory_files
from sonora.modules.tagger import tag_album_folder


def build_parser() -> argparse.ArgumentParser:
    """Build and return the main CLI argument parser with intuitive defaults."""
    parser = argparse.ArgumentParser(
        prog="sonora",
        description="Sonora — Smart FLAC/MP3 music tagging, library auditing, and file organization",
    )
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Subcommand: tag
    tag_parser = subparsers.add_parser("tag", help="Tag audio files and albums automatically with all metadata, artwork, BPM, ReplayGain & lyrics")
    tag_parser.add_argument("path", type=Path, help="Directory containing audio files to tag")
    
    # Audio processing flags (enabled by default, use --no-* to disable)
    tag_parser.add_argument("--no-bpm", action="store_false", dest="fetch_bpm", help="Disable BPM calculation")
    tag_parser.add_argument("--no-replaygain", action="store_false", dest="fetch_replaygain", help="Disable ReplayGain calculation")
    tag_parser.add_argument("--no-lyrics", action="store_false", dest="fetch_lyrics", help="Disable LRC lyrics fetching")
    tag_parser.add_argument("--no-art", action="store_false", dest="fetch_itunes_art", help="Disable cover art downloading")
    
    # Optional API keys
    tag_parser.add_argument("--lastfm-key", type=str, default=None, help="Last.fm API key for genre/mood lookup")
    tag_parser.add_argument("--acoustid-key", type=str, default=None, help="AcoustID API key for acoustic fingerprinting")
    tag_parser.add_argument("--discogs-token", type=str, default=None, help="Discogs personal user token")
    tag_parser.add_argument("-w", "--workers", type=int, default=4, help="Number of parallel worker threads (default: 4)")

    # Subcommand: audit
    audit_parser = subparsers.add_parser("audit", help="Audit music library for FLAC integrity, bracket corruption & missing LRCs")
    audit_parser.add_argument("path", type=Path, help="Directory containing music library to audit")
    audit_parser.add_argument("--json", type=Path, default=None, help="Output path to save audit JSON report")
    audit_parser.add_argument("--spectral", action="store_true", help="Enable deep spectral cutoff analysis for fake lossless detection (slow)")

    # Subcommand: rename
    rename_parser = subparsers.add_parser("rename", help="Rename audio files and sync .lrc metadata headers")
    rename_parser.add_argument("path", type=Path, help="Directory containing audio files to rename")

    # Subcommand: organize
    organize_parser = subparsers.add_parser("organize", help="Organize single tracks into a Singles directory structure")
    organize_parser.add_argument("path", type=Path, help="Source music directory")
    organize_parser.add_argument("--target-singles", type=Path, required=True, help="Destination directory for single tracks")

    return parser


def handle_tag(args: argparse.Namespace) -> int:
    """Handle 'tag' subcommand execution."""
    from sonora.services.musicbrainz import init_musicbrainz

    init_musicbrainz()
    LOG.info(f"Tagging album directory: [bold]{args.path}[/bold]")
    results = tag_album_folder(
        args.path,
        max_workers=args.workers,
        fetch_bpm=args.fetch_bpm,
        fetch_replaygain=args.fetch_replaygain,
        fetch_lyrics=args.fetch_lyrics,
        fetch_itunes_art=args.fetch_itunes_art,
        lastfm_api_key=args.lastfm_key,
        acoustid_api_key=args.acoustid_key,
        discogs_user_token=args.discogs_token,
    )
    LOG.success(f"Successfully tagged {len(results)} tracks.")
    return 0


def handle_audit(args: argparse.Namespace) -> int:
    """Handle 'audit' subcommand execution."""
    LOG.info(f"Auditing music library: [bold]{args.path}[/bold]")
    report = audit_library(args.path, output_json=args.json, check_spectral=args.spectral)
    LOG.success(f"Audit completed: {report.total_files} files scanned, {len(report.issues)} issues identified.")
    return 0


def handle_rename(args: argparse.Namespace) -> int:
    """Handle 'rename' subcommand execution."""
    LOG.info(f"Renaming files in directory: [bold]{args.path}[/bold]")
    renamed = rename_directory_files(args.path)
    LOG.success(f"Renamed {len(renamed)} audio files and synchronized .lrc headers.")
    return 0


def handle_organize(args: argparse.Namespace) -> int:
    """Handle 'organize' subcommand execution."""
    LOG.info(f"Organizing single tracks from {args.path} to {args.target_singles}")
    count = organize_library_singles(args.path, args.target_singles)
    LOG.success(f"Organized and moved {count} single tracks.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    try:
        if args.command == "tag":
            return handle_tag(args)
        elif args.command == "audit":
            return handle_audit(args)
        elif args.command == "rename":
            return handle_rename(args)
        elif args.command == "organize":
            return handle_organize(args)
        return 0
    except SonoraError as e:
        LOG.error(f"Error: {e}")
        return 1
    except Exception as e:  # noqa: BLE001
        LOG.error(f"Unexpected failure: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
