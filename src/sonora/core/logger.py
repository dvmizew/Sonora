import logging
import threading
from collections.abc import Sequence

from rich.console import Console
from rich.table import Table
from rich.theme import Theme

# Silence 3rd party loggers completely as in initial/script.py
for logger_name in ["syncedlyrics", "Musixmatch", "Lrclib", "NetEase", "Megalobiz", "Genius", "urllib3", "librosa", "audioread", "resampy"]:
    _l = logging.getLogger(logger_name)
    _l.setLevel(logging.CRITICAL)
    _l.propagate = False

_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red",
    "success": "green bold"
})
CONSOLE = Console(theme=_THEME, force_terminal=True)
_LOG_LOCK = threading.Lock()


class SonoraLogger:
    def __init__(self) -> None:
        self.local = threading.local()

    def start_buffering(self) -> None:
        self.local.buf = []

    def stop_buffering(self) -> None:
        if hasattr(self.local, "buf"):
            buf = self.local.buf
            del self.local.buf
            with _LOG_LOCK:
                for msg in buf:
                    CONSOLE.print(msg, highlight=False)

    def force_info(self, message: str) -> None:
        with _LOG_LOCK:
            CONSOLE.print(message, highlight=False)

    def _log_msg(self, message: str) -> None:
        if hasattr(self.local, "buf"):
            self.local.buf.append(message)
        else:
            with _LOG_LOCK:
                CONSOLE.print(message, highlight=False)

    def info(self, message: str) -> None:
        if message.startswith(("   ∟", "✨", "📁", "🎧")):
            self._log_msg(message)
        else:
            self._log_msg(f"[info]INFO:[/info] {message}")

    def success(self, message: str) -> None:
        self._log_msg(f"[success]SUCCESS:[/success] {message}")

    def warning(self, message: str) -> None:
        self._log_msg(f"[warning]WARNING:[/warning] {message}")

    def debug(self, message: str) -> None:
        pass

    def error(self, message: str) -> None:
        self._log_msg(f"[error]ERROR:[/error] {message}")

    def summary_table(self, title: str, rows: Sequence[tuple[str, str, str | None]]) -> None:
        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        for metric, val, style in rows:
            if style:
                table.add_row(metric, val, style=style)
            else:
                table.add_row(metric, val)
        with _LOG_LOCK:
            CONSOLE.print("\n", table)

    def heartbeat(self, message: str, min_interval: float = 0.5) -> None:
        pass


LOG = SonoraLogger()
