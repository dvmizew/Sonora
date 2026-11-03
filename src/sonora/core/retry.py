import socket
import threading
import time
from collections.abc import Callable
from typing import TypeVar

from sonora.core.logger import LOG
from sonora.core.rate_limit import RateLimiter

_T = TypeVar("_T")
_DNS_COOLDOWN_SECONDS = 30.0
_DNS_UNAVAILABLE_UNTIL = 0.0
_DNS_LOCK = threading.Lock()


def _dns_requests_suspended() -> bool:
    with _DNS_LOCK:
        return time.monotonic() < _DNS_UNAVAILABLE_UNTIL


def _suspend_dns_requests() -> None:
    global _DNS_UNAVAILABLE_UNTIL
    with _DNS_LOCK:
        _DNS_UNAVAILABLE_UNTIL = time.monotonic() + _DNS_COOLDOWN_SECONDS


def _dns_error(error: Exception) -> socket.gaierror | None:
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, socket.gaierror):
            return current
        pending.extend(
            cause
            for cause in (
                current.__cause__,
                current.__context__,
                getattr(current, "cause", None),
                getattr(current, "reason", None),
            )
            if isinstance(cause, BaseException)
        )
    return None


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
        if _dns_requests_suspended():
            raise socket.gaierror(socket.EAI_AGAIN, "DNS lookup temporarily suspended")
        if rate_limiter is not None:
            rate_limiter.wait()
            if _dns_requests_suspended():
                raise socket.gaierror(
                    socket.EAI_AGAIN, "DNS lookup temporarily suspended"
                )
        try:
            return func()
        except transient_exceptions as error:
            dns_error = _dns_error(error)
            if dns_error is not None:
                if dns_error.errno == socket.EAI_AGAIN:
                    _suspend_dns_requests()
                raise
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
