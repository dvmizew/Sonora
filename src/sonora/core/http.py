import atexit

import httpx

from sonora.core.constants import USER_AGENT

SESSION = httpx.Client(
    transport=httpx.HTTPTransport(
        http2=True,
        retries=3,
        limits=httpx.Limits(
            max_connections=100, max_keepalive_connections=20, keepalive_expiry=30.0
        ),
    ),
    timeout=httpx.Timeout(timeout=10.0),
    follow_redirects=True,
    headers={"User-Agent": USER_AGENT},
)

atexit.register(SESSION.close)
