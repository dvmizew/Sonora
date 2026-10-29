import atexit
import random
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from sonora.core.constants import (
    RATE_LIMIT_DEEZER,
    RATE_LIMIT_DISCOGS_AUTHENTICATED,
    RATE_LIMIT_DISCOGS_UNAUTHENTICATED,
    RATE_LIMIT_FANART,
    RATE_LIMIT_GENIUS,
    RATE_LIMIT_ITUNES,
    RATE_LIMIT_LASTFM,
    RATE_LIMIT_LRCLIB,
    RATE_LIMIT_THEAUDIODB,
    USER_AGENT,
)
from sonora.core.coordination import wait_for_provider
from sonora.core.logger import LOG
from sonora.core.rate_limit import RateLimiter

_HOST_LIMITERS = {
    "itunes.apple.com": RateLimiter(RATE_LIMIT_ITUNES, provider="itunes.apple.com"),
    "api.deezer.com": RateLimiter(RATE_LIMIT_DEEZER, provider="api.deezer.com"),
    "api.discogs.com": RateLimiter(
        RATE_LIMIT_DISCOGS_AUTHENTICATED, provider="api.discogs.com"
    ),
    "www.theaudiodb.com": RateLimiter(
        RATE_LIMIT_THEAUDIODB, provider="www.theaudiodb.com"
    ),
    "lrclib.net": RateLimiter(RATE_LIMIT_LRCLIB, provider="lrclib.net"),
    "webservice.fanart.tv": RateLimiter(
        RATE_LIMIT_FANART, provider="webservice.fanart.tv"
    ),
    "api.genius.com": RateLimiter(RATE_LIMIT_GENIUS, provider="api.genius.com"),
    "ws.audioscrobbler.com": RateLimiter(
        RATE_LIMIT_LASTFM, provider="ws.audioscrobbler.com"
    ),
}


def _retry_delay(retry_after: str | None, fallback: float) -> float:
    if not retry_after:
        return fallback
    try:
        return max(0.0, float(retry_after))
    except ValueError:
        try:
            deadline = parsedate_to_datetime(retry_after)
        except ValueError:
            return fallback
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        return max(0.0, (deadline - datetime.now(timezone.utc)).total_seconds())


class RetryTransport(httpx.HTTPTransport):
    """
    Transparent HTTP transport providing connection pooling and exponential backoff retries for:
    - Transient network drops and socket timeouts (httpx.TransportError, OSError)
    - Rate limit responses (HTTP 429), respecting Retry-After header
    - Server-side transient failures (HTTP 502, 503, 504)
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if max_retries < 1:
            raise ValueError("max_retries must be positive")
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        limiter = _HOST_LIMITERS.get(request.url.host)
        with (
            limiter.lock
            if limiter is not None and request.url.host == "lrclib.net"
            else nullcontext()
        ):
            attempt = 0
            while True:
                try:
                    if limiter is not None:
                        interval = (
                            RATE_LIMIT_DISCOGS_UNAUTHENTICATED
                            if request.url.host == "api.discogs.com"
                            and "Authorization" not in request.headers
                            else None
                        )
                        limiter.wait(interval)
                    response = super().handle_request(request)
                    if response.status_code in (429, 502, 503, 504):
                        delay = _retry_delay(
                            response.headers.get("Retry-After"),
                            self.backoff_factor * (2**attempt),
                        )
                        if request.url.host == "lrclib.net":
                            delay += random.uniform(0.0, delay * 0.3)
                        if limiter is not None and not limiter._disabled:
                            wait_for_provider(request.url.host, 0, defer_seconds=delay)
                        if delay > 60.0 or attempt >= self.max_retries - 1:
                            return response
                        response.close()
                        LOG.debug(
                            f"HTTP {response.status_code} for {request.url.host}. "
                            f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})..."
                        )
                        time.sleep(delay)
                        attempt += 1
                        continue
                    return response
                except (httpx.TransportError, OSError) as exc:
                    if attempt >= self.max_retries - 1:
                        raise
                    delay = self.backoff_factor * (2**attempt)
                    LOG.debug(
                        f"HTTP transport error ({type(exc).__name__}) for {request.url.host}. "
                        f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})..."
                    )
                    time.sleep(delay)
                    attempt += 1


SESSION = httpx.Client(
    transport=RetryTransport(
        max_retries=3,
        http2=True,
        retries=0,
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=30.0,
        ),
    ),
    timeout=httpx.Timeout(timeout=10.0),
    follow_redirects=True,
    headers={"User-Agent": USER_AGENT},
)

atexit.register(SESSION.close)
