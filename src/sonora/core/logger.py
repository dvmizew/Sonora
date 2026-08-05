"""
Unified logger and Rich console wrapper for Sonora.
"""


try:
    from rich.console import Console
    from rich.theme import Theme

    _THEME = Theme({
        "info": "cyan",
        "warning": "yellow",
        "error": "red",
        "success": "green bold"
    })
    CONSOLE: Console | None = Console(theme=_THEME, force_terminal=True)
    HAS_RICH = True
except ImportError:
    CONSOLE = None
    HAS_RICH = False


class SonoraLogger:
    """Rich console wrapper for logging messages."""

    def info(self, message: str) -> None:
        if CONSOLE and HAS_RICH:
            CONSOLE.print(f"[info]INFO:[/info] {message}")
        else:
            print(f"[INFO] {message}")

    def success(self, message: str) -> None:
        if CONSOLE and HAS_RICH:
            CONSOLE.print(f"[success]SUCCESS:[/success] {message}")
        else:
            print(f"[SUCCESS] {message}")

    def warning(self, message: str) -> None:
        if CONSOLE and HAS_RICH:
            CONSOLE.print(f"[warning]WARNING:[/warning] {message}")
        else:
            print(f"[WARNING] {message}")

    def debug(self, message: str) -> None:
        if CONSOLE and HAS_RICH:
            CONSOLE.print(f"[dim]DEBUG:[/dim] {message}")
        else:
            print(f"[DEBUG] {message}")

    def error(self, message: str) -> None:
        if CONSOLE and HAS_RICH:
            CONSOLE.print(f"[error]ERROR:[/error] {message}")
        else:
            print(f"[ERROR] {message}")


LOG = SonoraLogger()


def log_debug(message: str) -> None:
    LOG.debug(message)


def log_info(message: str) -> None:
    LOG.info(message)


def log_success(message: str) -> None:
    LOG.success(message)


def log_warning(message: str) -> None:
    LOG.warning(message)


def log_error(message: str) -> None:
    LOG.error(message)
