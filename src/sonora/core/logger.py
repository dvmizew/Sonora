import contextlib
import logging
import select
import sys
import threading
from collections.abc import Callable, Generator, Sequence

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
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
_LOG_LOCK = threading.RLock()


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


LOG = SonoraLogger()


def create_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("[dim]/[/dim]"),
        TimeRemainingColumn(),
        console=CONSOLE,
    )


_PAUSE_EVENT = threading.Event()
_PAUSE_EVENT.set()


def wait_if_paused(poll_interval: float = 0.2) -> None:
    """Block calling thread safely until resumed."""
    while not _PAUSE_EVENT.wait(timeout=poll_interval):
        pass


@contextlib.contextmanager
def interactive_pause_listener(
    progress: Progress | None = None,
    task_id: TaskID | None = None,
    on_pause: Callable[[], None] | None = None,
    on_resume: Callable[[], None] | None = None,
) -> Generator[None, None, None]:
    """
    Listen for Space or 'p' / 'P' in background cbreak mode to pause/resume.
    Safely adjusts Rich Progress timers and displays pause state.
    """
    if not sys.stdin.isatty():
        yield
        return

    _PAUSE_EVENT.set()
    stop_event = threading.Event()
    pause_start_time: float | None = None
    original_description: str | None = None

    def _default_pause() -> None:
        nonlocal pause_start_time, original_description
        if progress is not None and task_id is not None:
            pause_start_time = progress.get_time()
            task_obj = progress._tasks.get(task_id)
            if task_obj:
                original_description = task_obj.description
            progress.stop_task(task_id)
            progress.update(
                task_id,
                description="[bold yellow]⏸️  PAUSED (Press [Space] or 'p' to resume)[/]",
            )
            progress.refresh()

    def _default_resume() -> None:
        nonlocal pause_start_time, original_description
        if progress is not None and task_id is not None:
            if pause_start_time is not None:
                pause_duration = progress.get_time() - pause_start_time
                task_obj = progress._tasks.get(task_id)
                if task_obj and task_obj.start_time is not None:
                    task_obj.start_time += pause_duration
                if task_obj:
                    task_obj.stop_time = None
                pause_start_time = None
            desc = original_description or "[cyan]Processing..."
            progress.update(task_id, description=desc)
            progress.refresh()

    def _listener_loop() -> None:
        try:
            import termios
            import tty

            orig_term = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
        except (ImportError, OSError, ValueError):
            return

        try:
            while not stop_event.is_set():
                rlist, _, _ = select.select([sys.stdin], [], [], 0.2)
                if rlist and not stop_event.is_set():
                    char = sys.stdin.read(1)
                    if char in (" ", "p", "P"):
                        if _PAUSE_EVENT.is_set():
                            _PAUSE_EVENT.clear()
                            if on_pause:
                                with contextlib.suppress(Exception):
                                    on_pause()
                            else:
                                _default_pause()
                            LOG.warning(
                                "⏸️  [bold yellow]PAUSED[/] - In-flight tracks finishing cleanly. Press [bold cyan][Space][/] or [bold cyan]'p'[/] to resume..."
                            )
                        else:
                            _PAUSE_EVENT.set()
                            if on_resume:
                                with contextlib.suppress(Exception):
                                    on_resume()
                            else:
                                _default_resume()
                            LOG.info(
                                "▶️  [bold green]RESUMED[/] - Continuing execution..."
                            )
                        while True:
                            drain_list, _, _ = select.select([sys.stdin], [], [], 0.05)
                            if drain_list:
                                sys.stdin.read(1)
                            else:
                                break
        finally:
            with contextlib.suppress(Exception):
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, orig_term)

    listener = threading.Thread(
        target=_listener_loop, name="SonoraPauseListener", daemon=True
    )
    listener.start()
    try:
        yield
    finally:
        stop_event.set()
        _PAUSE_EVENT.set()
        listener.join(timeout=0.5)
