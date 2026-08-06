"""
Disk caching layer using diskcache for API response caching.
"""

import atexit
import threading
from pathlib import Path
from typing import Any

from sonora.core.logger import LOG

try:
    import diskcache
except ImportError:
    diskcache = None


_CACHE_DIR = Path.home() / ".cache" / "sonora"
_CACHE_INSTANCE: Any = None
_CACHE_LOCK = threading.Lock()

def get_cache() -> Any:
    """Lazy initialize and return sharded diskcache.FanoutCache instance for multi-threaded speed."""
    global _CACHE_INSTANCE
    if diskcache is not None and _CACHE_INSTANCE is None:
        with _CACHE_LOCK:
            if _CACHE_INSTANCE is None:
                try:
                    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    if hasattr(diskcache, "FanoutCache"):
                        _CACHE_INSTANCE = diskcache.FanoutCache(str(_CACHE_DIR), shards=8)
                    else:
                        _CACHE_INSTANCE = diskcache.Cache(str(_CACHE_DIR))
                except Exception as e:  # noqa: BLE001
                    LOG.debug(f"Cache initialization failed: {e}")
                    _CACHE_INSTANCE = None
    return _CACHE_INSTANCE


def get_cached_api(key: str) -> Any | None:
    """Retrieve value from disk cache by key."""
    cache = get_cache()
    if cache is not None:
        try:
            return cache.get(key)
        except Exception as e:  # noqa: BLE001
            LOG.debug(f"Cache fetch failed for key '{key}': {e}")
    return None


def set_cached_api(key: str, value: Any, expire_seconds: int = 604800) -> None:
    """Store value into disk cache with expiration (default 7 days)."""
    if value is None:
        return
    cache = get_cache()
    if cache is not None:
        try:
            cache.set(key, value, expire=expire_seconds)
        except Exception as e:  # noqa: BLE001
            LOG.debug(f"Cache store failed for key '{key}': {e}")

def close_cache() -> None:
    """Close the diskcache to release SQLite connections."""
    global _CACHE_INSTANCE
    if _CACHE_INSTANCE is not None:
        try:
            _CACHE_INSTANCE.close()
        except Exception as e:  # noqa: BLE001
            LOG.debug(f"Cache close failed: {e}")
        _CACHE_INSTANCE = None

atexit.register(close_cache)
