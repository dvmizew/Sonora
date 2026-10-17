import time
from collections.abc import Callable
from typing import TypeVar

from sonora.core.logger import LOG
from sonora.core.rate_limit import RateLimiter

_T = TypeVar("_T")


def retry_network_call(
    func: Callable[[], _T],
    *,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    rate_limiter: RateLimiter | None = None,
    transient_exceptions: tuple[type[Exception], ...] = (
        OSError,
        TimeoutError,
    ),
    retry_condition: Callable[[Exception], bool] | None = None,
) -> _T:
    """
    Execute a network-dependent callable with exponential backoff retries.
    Optionally coordinates with a RateLimiter before each attempt to ensure rate
    limits are strictly respected.
    """
    if max_retries < 1:
        raise ValueError("max_retries must be positive")
    attempt = 0
    while True:
        if rate_limiter is not None:
            rate_limiter.wait()
        try:
            return func()
        except transient_exceptions as error:
            if retry_condition is not None and not retry_condition(error):
                raise
            if attempt >= max_retries - 1:
                raise
            delay = initial_delay * (backoff_factor**attempt)
            LOG.debug(
                f"Transient network error ({type(error).__name__}: {error}). "
                f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})..."
            )
            time.sleep(delay)
            attempt += 1
