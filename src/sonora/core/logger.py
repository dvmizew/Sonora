"""
Unified logger and Rich console wrapper for Sonora.
"""

from typing import Optional

try:
    from rich.console import Console
    from rich.theme import Theme

    _THEME = Theme({
        "info": "cyan",
        "warning": "yellow",
        "error": "red",
        "success": "green bold"
    })
    CONSOLE: Optional[Console] = Console(theme=_THEME, force_terminal=True)
    HAS_RICH = True
except ImportError:
    CONSOLE = None
    HAS_RICH = False


def log_info(message: str) -> None:
    """Log informational message."""
    if CONSOLE and HAS_RICH:
        CONSOLE.print(f"[info]INFO:[/info] {message}")
    else:
        print(f"[INFO] {message}")


def log_success(message: str) -> None:
    """Log success message."""
    if CONSOLE and HAS_RICH:
        CONSOLE.print(f"[success]SUCCESS:[/success] {message}")
    else:
        print(f"[SUCCESS] {message}")


def log_warning(message: str) -> None:
    """Log warning message."""
    if CONSOLE and HAS_RICH:
        CONSOLE.print(f"[warning]WARNING:[/warning] {message}")
    else:
        print(f"[WARNING] {message}")


def log_error(message: str) -> None:
    """Log error message."""
    if CONSOLE and HAS_RICH:
        CONSOLE.print(f"[error]ERROR:[/error] {message}")
    else:
        print(f"[ERROR] {message}")
