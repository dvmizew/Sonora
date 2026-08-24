import logging
import threading
from collections.abc import Sequence

from rich.console import Console
from rich.table import Table
from rich.theme import Theme

for logger_name in [
    "syncedlyrics",
    "Musixmatch",
    "Lrclib",
    "NetEase",
    "Megalobiz",
    "Genius",
    "urllib3",
    "httpx",
]:
    external_logger = logging.getLogger(logger_name)
    external_logger.setLevel(logging.CRITICAL)
    external_logger.propagate = False

_THEME = Theme(
    {"info": "cyan", "warning": "yellow", "error": "red", "success": "green bold"}
)
CONSOLE = Console(theme=_THEME, force_terminal=True)
_LOG_LOCK = threading.Lock()


class SonoraLogger:
    def __init__(self) -> None:
        self.local = threading.local()
        self.verbose: bool = False

    def start_buffering(self) -> None:
        self.local.buffer = []

    def stop_buffering(self) -> None:
        if hasattr(self.local, "buffer"):
            messages = self.local.buffer
            del self.local.buffer
            with _LOG_LOCK:
                for message in messages:
                    CONSOLE.print(message, highlight=False)

    def force_info(self, message: str) -> None:
        with _LOG_LOCK:
            CONSOLE.print(message, highlight=False)

    def _log_message(self, message: str) -> None:
        if hasattr(self.local, "buffer"):
            self.local.buffer.append(message)
        else:
            with _LOG_LOCK:
                CONSOLE.print(message, highlight=False)

    def info(self, message: str) -> None:
        if message.startswith(("   ∟", "✨", "📁", "🎧")):
            self._log_message(message)
        else:
            self._log_message(f"[info]INFO:[/info] {message}")

    def success(self, message: str) -> None:
        self._log_message(f"[success]SUCCESS:[/success] {message}")

    def warning(self, message: str) -> None:
        self._log_message(f"[warning]WARNING:[/warning] {message}")

    def debug(self, message: str) -> None:
        if self.verbose:
            self._log_message(f"[dim]DEBUG:[/dim] {message}")

    def error(self, message: str) -> None:
        self._log_message(f"[error]ERROR:[/error] {message}")

    def summary_table(
        self, title: str, rows: Sequence[tuple[str, str, str | None]]
    ) -> None:
        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        for metric, value, style in rows:
            if style:
                table.add_row(metric, value, style=style)
            else:
                table.add_row(metric, value)
        with _LOG_LOCK:
            CONSOLE.print("\n", table)


LOG = SonoraLogger()
