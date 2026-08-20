from collections.abc import Sequence
from rich.console import Console
from rich.table import Table
from rich.theme import Theme

_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red",
    "success": "green bold"
})
CONSOLE = Console(theme=_THEME, force_terminal=True)

class SonoraLogger:
    def info(self, message: str) -> None:
        CONSOLE.print(f"[info]INFO:[/info] {message}")

    def success(self, message: str) -> None:
        CONSOLE.print(f"[success]SUCCESS:[/success] {message}")

    def warning(self, message: str) -> None:
        CONSOLE.print(f"[warning]WARNING:[/warning] {message}")

    def debug(self, message: str) -> None:
        CONSOLE.print(f"[dim]DEBUG:[/dim] {message}")

    def error(self, message: str) -> None:
        CONSOLE.print(f"[error]ERROR:[/error] {message}")

    def summary_table(self, title: str, rows: Sequence[tuple[str, str, str | None]]) -> None:
        """
        `rows` is a list of (metric_name, value, optional_rich_style).
        """
        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        for metric, val, style in rows:
            if style:
                table.add_row(metric, val, style=style)
            else:
                table.add_row(metric, val)
        CONSOLE.print("\n", table)


LOG = SonoraLogger()
