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
except ImportError:
    CONSOLE = None


class SonoraLogger:
    """Rich console wrapper for logging messages."""

    def info(self, message: str) -> None:
        if CONSOLE:
            CONSOLE.print(f"[info]INFO:[/info] {message}")
        else:
            print(f"[INFO] {message}")

    def success(self, message: str) -> None:
        if CONSOLE:
            CONSOLE.print(f"[success]SUCCESS:[/success] {message}")
        else:
            print(f"[SUCCESS] {message}")

    def warning(self, message: str) -> None:
        if CONSOLE:
            CONSOLE.print(f"[warning]WARNING:[/warning] {message}")
        else:
            print(f"[WARNING] {message}")

    def debug(self, message: str) -> None:
        if CONSOLE:
            CONSOLE.print(f"[dim]DEBUG:[/dim] {message}")
        else:
            print(f"[DEBUG] {message}")

    def error(self, message: str) -> None:
        if CONSOLE:
            CONSOLE.print(f"[error]ERROR:[/error] {message}")
        else:
            print(f"[ERROR] {message}")


LOG = SonoraLogger()
