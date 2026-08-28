import atexit
import threading
from pathlib import Path
from typing import Any

import diskcache

from sonora.core.logger import LOG

DEFAULT_API_TTL_SECONDS: int = 2419200  # 28 days
_CACHE_DIR = Path.home() / ".cache" / "sonora"
_CACHE_INSTANCE: Any = None
_CACHE_LOCK = threading.RLock()
_IGNORE_CACHE = False


def set_ignore_cache(ignore: bool) -> None:
    global _IGNORE_CACHE
    _IGNORE_CACHE = ignore


def get_cache() -> Any:
    global _CACHE_INSTANCE
    if _CACHE_INSTANCE is None:
        with _CACHE_LOCK:
            if _CACHE_INSTANCE is None:
                try:
                    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    _CACHE_INSTANCE = diskcache.Cache(
                        str(_CACHE_DIR),
                        timeout=10,
                    )
                except (
                    OSError,
                    ValueError,
                    KeyError,
                    RuntimeError,
                    TypeError,
                ) as error:
                    LOG.debug(f"Cache initialization failed: {error}")
                    _CACHE_INSTANCE = None
    return _CACHE_INSTANCE


def get_cached_api(key: str) -> Any | None:
    if _IGNORE_CACHE:
        return None

    cache = get_cache()
    if cache is not None:
        try:
            with _CACHE_LOCK:
                return cache.get(key)
        except (
            OSError,
            ValueError,
            KeyError,
            RuntimeError,
            TypeError,
            diskcache.Timeout,
        ) as error:
            LOG.debug(f"Cache fetch failed for key '{key}': {error}")
    return None


def set_cached_api(
    key: str, value: Any, expire_seconds: int = DEFAULT_API_TTL_SECONDS
) -> None:
    """Store value into disk cache with expiration (default 28 days)."""
    if value is None:
        return
    cache = get_cache()
    if cache is not None:
        try:
            with _CACHE_LOCK:
                cache.set(key, value, expire=expire_seconds)
        except (
            OSError,
            ValueError,
            KeyError,
            RuntimeError,
            TypeError,
            diskcache.Timeout,
        ) as error:
            LOG.debug(f"Cache store failed for key '{key}': {error}")


def clear_cache() -> None:
    cache = get_cache()
    if cache is not None:
        try:
            with _CACHE_LOCK:
                cache.clear()
        except (
            OSError,
            ValueError,
            KeyError,
            RuntimeError,
            TypeError,
            diskcache.Timeout,
        ) as error:
            LOG.debug(f"Cache clear failed: {error}")


def close_cache() -> None:
    global _CACHE_INSTANCE
    with _CACHE_LOCK:
        if _CACHE_INSTANCE is not None:
            try:
                _CACHE_INSTANCE.close()
            except (
                OSError,
                ValueError,
                KeyError,
                RuntimeError,
                TypeError,
                diskcache.Timeout,
            ) as error:
                LOG.debug(f"Cache close failed: {error}")
            _CACHE_INSTANCE = None


atexit.register(close_cache)
