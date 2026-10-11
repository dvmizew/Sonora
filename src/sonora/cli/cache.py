from typing import Annotated

import orjson
from cyclopts import App, Parameter
from rich.markup import escape

from sonora.core.cache import CacheStats, get_cache_stats
from sonora.core.cache import clear_cache as perform_clear_cache
from sonora.core.logger import LOG
from sonora.core.utils import format_filesize

cache_app = App(
    "cache",
    help="Inspect and manage Sonora cache and persistent state",
    result_action="return_value",
)


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
