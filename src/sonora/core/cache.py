import atexit
import threading
from pathlib import Path
from typing import Any

import diskcache

from sonora.core.logger import LOG

_CACHE_DIR = Path.home() / ".cache" / "sonora"
_CACHE_INSTANCE: Any = None
_CACHE_LOCK = threading.Lock()
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
                    if hasattr(diskcache, "FanoutCache"):
                        _CACHE_INSTANCE = diskcache.FanoutCache(
                            str(_CACHE_DIR),
                            shards=8,
                            sqlite_journal_mode="wal",
                            sqlite_synchronous=0,
                        )
                    else:
                        _CACHE_INSTANCE = diskcache.Cache(
                            str(_CACHE_DIR),
                            sqlite_journal_mode="wal",
                            sqlite_synchronous=0,
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
            return cache.get(key)
        except (OSError, ValueError, KeyError, RuntimeError, TypeError) as error:
            LOG.debug(f"Cache fetch failed for key '{key}': {error}")
    return None


def set_cached_api(key: str, value: Any, expire_seconds: int = 604800) -> None:
    """Store value into disk cache with expiration (default 7 days)."""
    if value is None:
        return
    cache = get_cache()
    if cache is not None:
        try:
            cache.set(key, value, expire=expire_seconds)
        except (OSError, ValueError, KeyError, RuntimeError, TypeError) as error:
            LOG.debug(f"Cache store failed for key '{key}': {error}")


def close_cache() -> None:
    global _CACHE_INSTANCE
    with _CACHE_LOCK:
        if _CACHE_INSTANCE is not None:
            try:
                _CACHE_INSTANCE.close()
            except (OSError, ValueError, KeyError, RuntimeError, TypeError) as error:
                LOG.debug(f"Cache close failed: {error}")
            _CACHE_INSTANCE = None


atexit.register(close_cache)
