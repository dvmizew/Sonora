from typing import Any


class InterruptedOperationError(KeyboardInterrupt):
    """Raised when an operation is cancelled via SIGINT / KeyboardInterrupt, carrying partial progress."""

    def __init__(self, partial_result: Any = None) -> None:
        super().__init__()
        self.partial_result = partial_result


def is_interruption(exc: BaseException) -> bool:
    """Recognize cancellation by exception type and chaining, without masking lock errors."""
    visited: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in visited:
        if isinstance(current, KeyboardInterrupt):
            return True
        visited.add(id(current))
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    return False
