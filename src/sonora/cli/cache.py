from typing import Annotated

import orjson
from cyclopts import App, Parameter
from rich.markup import escape

from sonora.core.cache import CacheStats, get_cache_stats
from sonora.core.cache import clear_cache as perform_clear_cache
from sonora.core.logger import LOG

cache_app = App(
    "cache",
    help="Inspect and manage Sonora cache and persistent state",
    result_action="return_value",
)


def _format_file_size(size_bytes: float) -> str:
    """Format a byte count into a human-readable string (e.g. 4.25 MB, 1.20 GB)."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size) < 1024.0:
            return f"{int(size)} B" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


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
            _format_file_size(stats.api_size_bytes),
            "green" if stats.api_size_bytes else "white",
        ),
        (
            "Library State Entries",
            f"{stats.state_entries:,}",
            "green" if stats.state_entries else "white",
        ),
        (
            "Library State Size on Disk",
            _format_file_size(stats.state_size_bytes),
            "green" if stats.state_size_bytes else "white",
        ),
        (
            "Memory Metadata Cache",
            f"{stats.memory_metadata_entries:,} items",
            "cyan",
        ),
        (
            "Total Cache Disk Usage",
            _format_file_size(stats.total_size_bytes),
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
        do_state = purge
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

    mode_label = "Purged" if purge else "Cleared"
    active_style = "yellow" if dry_run else "green"
    if dry_run:
        LOG.info(
            f"[yellow][DRY RUN][/yellow] Simulated cache clearing for [bold]{escape(str(result.cache_dir))}[/bold]:"
        )
    rows = [("Cache Directory", escape(str(result.cache_dir)), "cyan")]
    for label, selected, count, size_bytes, unit, skipped in (
        (
            "API Metadata Cache",
            result.api_cleared,
            result.api_entries_cleared,
            result.api_bytes_freed,
            "entries",
            "Skipped",
        ),
        (
            "Library State Cache",
            result.state_cleared,
            result.state_entries_cleared,
            result.state_bytes_freed,
            "tracks",
            "Preserved (use --state or --all to clear)",
        ),
        (
            "In-Memory Caches",
            result.memory_cleared,
            result.memory_metadata_cleared,
            None,
            "entries",
            "Skipped",
        ),
    ):
        description = f"{count:,} {unit}" if selected else skipped
        if selected and size_bytes is not None:
            description += f" ({_format_file_size(size_bytes)})"
        rows.append((label, description, active_style if selected else "white"))
    rows.extend(
        [
            (
                "Action Mode",
                ("Purge completely from disk" if purge else "Clear & Reclaim")
                if dry_run
                else mode_label,
                "cyan",
            ),
            (
                "Total Space Reclaimable" if dry_run else "Total Space Freed",
                _format_file_size(result.total_bytes_freed),
                f"bold {active_style}",
            ),
        ]
    )
    LOG.summary_table(
        "Dry Run Cache Clearing" if dry_run else f"Cache {mode_label} Summary", rows
    )
    if not dry_run:
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
