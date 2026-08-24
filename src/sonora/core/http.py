from typing import Any

import httpx

from sonora.core.constants import USER_AGENT


class SonoraHTTPClient:
    def __init__(self) -> None:
        limits = httpx.Limits(
            max_connections=100, max_keepalive_connections=20, keepalive_expiry=30.0
        )
        transport = httpx.HTTPTransport(http2=True, retries=3, limits=limits)
        timeout = httpx.Timeout(timeout=10.0)
        self._client = httpx.Client(
            transport=transport,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
        self.headers = self._client.headers

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        if "allow_redirects" in kwargs:
            kwargs["follow_redirects"] = kwargs.pop("allow_redirects")
        return self._client.get(url, **kwargs)

    def head(self, url: str, **kwargs: Any) -> httpx.Response:
        if "allow_redirects" in kwargs:
            kwargs["follow_redirects"] = kwargs.pop("allow_redirects")
        return self._client.head(url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        if "allow_redirects" in kwargs:
            kwargs["follow_redirects"] = kwargs.pop("allow_redirects")
        return self._client.post(url, **kwargs)

    def close(self) -> None:
        self._client.close()


SESSION = SonoraHTTPClient()
