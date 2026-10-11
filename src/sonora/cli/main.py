import contextlib
import signal
import socket
import sqlite3
import sys
from collections.abc import Sequence

import diskcache
from cyclopts import App
from cyclopts.exceptions import CycloptsError
from rich.markup import escape

from sonora import __version__
from sonora.cli.audio import bpm, key, lyrics, replaygain
from sonora.cli.cache import cache_app, cache_clear_cmd
from sonora.cli.library import backup, check, normalize, organize, rename, restore, tag
from sonora.core.config import load_app_environment
from sonora.core.logger import LOG
from sonora.core.utils import is_interruption

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
app.command(tag)
app.command(check)
app.command(rename)
app.command(organize)
app.command(backup)
app.command(restore)
app.command(normalize)
app.command(bpm)
app.command(key)
app.command(replaygain)
app.command(lyrics)
app.command(cache_app)
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
    except (KeyboardInterrupt, RuntimeError) as exc:
        if not is_interruption(exc):
            raise
        LOG.warning("Aborted by user. Shutting down gracefully...")
        return 130
    except CycloptsError:
        return 2
    except (OSError, ValueError, TypeError, sqlite3.Error, diskcache.Timeout) as error:
        LOG.error(f"Error: {escape(str(error))}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
