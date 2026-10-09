import atexit
import time
from typing import Any

import httpx

from sonora.core.constants import USER_AGENT
from sonora.core.logger import LOG


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
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = super().handle_request(request)
                if (
                    response.status_code in (429, 502, 503, 504)
                    and attempt < self.max_retries - 1
                ):
                    retry_after = response.headers.get("Retry-After")
                    delay: float
                    if retry_after and retry_after.strip().isdigit():
                        delay = min(float(retry_after.strip()), 10.0)
                    else:
                        delay = self.backoff_factor * (2**attempt)
                    LOG.debug(
                        f"HTTP {response.status_code} for {request.url}. "
                        f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})..."
                    )
                    time.sleep(delay)
                    continue
                return response
            except (httpx.TransportError, OSError) as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    delay = self.backoff_factor * (2**attempt)
                    LOG.debug(
                        f"HTTP transport error ({type(exc).__name__}) for {request.url}. "
                        f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})..."
                    )
                    time.sleep(delay)
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("Unreachable")


SESSION = httpx.Client(
    transport=RetryTransport(
        max_retries=3,
        http2=True,
        retries=3,
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
