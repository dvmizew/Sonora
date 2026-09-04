import atexit
import dataclasses
import os
import shutil
import threading
from pathlib import Path
from typing import Any

import diskcache

from sonora.core.logger import LOG

DEFAULT_API_TTL_SECONDS: int = 2419200  # 28 days
_CACHE_INSTANCE: Any = None
_CACHE_LOCK = threading.RLock()
_IGNORE_CACHE = False


def get_cache_dir() -> Path:
    """
    Return Sonora cache directory following the XDG Base Directory specification.
    Prefers $XDG_CACHE_HOME/sonora, falling back to ~/.cache/sonora.
    """
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home and xdg_cache_home.strip():
        base = Path(xdg_cache_home.strip())
    else:
        base = Path.home() / ".cache"
    return base / "sonora"


def get_api_cache_dir() -> Path:
    """Return dedicated directory for diskcache API cache files."""
    return get_cache_dir() / "api"


def _migrate_legacy_cache(cache_dir: Path, api_cache_dir: Path) -> None:
    """Migrate legacy cache files from cache root to dedicated api/ subdirectory."""
    legacy_db = cache_dir / "cache.db"
    if not legacy_db.exists():
        return
    try:
        api_cache_dir.mkdir(parents=True, exist_ok=True)
        for item in list(cache_dir.iterdir()):
            if item.name.startswith("library_state.db") or item.name == "api":
                continue
            target = api_cache_dir / item.name
            if not target.exists():
                shutil.move(str(item), str(target))
    except OSError as error:
        LOG.debug(f"Legacy cache migration failed: {error}")


@dataclasses.dataclass(frozen=True)
class CacheStats:
    cache_dir: Path
    api_entries: int
    api_size_bytes: int
    state_entries: int
    state_size_bytes: int
    memory_metadata_entries: int
    total_size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_dir": str(self.cache_dir),
            "api_entries": self.api_entries,
            "api_size_bytes": self.api_size_bytes,
            "state_entries": self.state_entries,
            "state_size_bytes": self.state_size_bytes,
            "memory_metadata_entries": self.memory_metadata_entries,
            "total_size_bytes": self.total_size_bytes,
        }


@dataclasses.dataclass(frozen=True)
class ClearResult:
    cache_dir: Path
    api_cleared: bool
    api_entries_cleared: int
    api_bytes_freed: int
    state_cleared: bool
    state_entries_cleared: int
    state_bytes_freed: int
    memory_cleared: bool
    memory_metadata_cleared: int
    purged: bool
    dry_run: bool = False

    @property
    def total_bytes_freed(self) -> int:
        return self.api_bytes_freed + self.state_bytes_freed

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_dir": str(self.cache_dir),
            "dry_run": self.dry_run,
            "api_cleared": self.api_cleared,
            "api_entries_cleared": self.api_entries_cleared,
            "api_bytes_freed": self.api_bytes_freed,
            "state_cleared": self.state_cleared,
            "state_entries_cleared": self.state_entries_cleared,
            "state_bytes_freed": self.state_bytes_freed,
            "memory_cleared": self.memory_cleared,
            "memory_metadata_cleared": self.memory_metadata_cleared,
            "purged": self.purged,
            "total_bytes_freed": self.total_bytes_freed,
        }


def set_ignore_cache(ignore: bool) -> None:
    global _IGNORE_CACHE
    _IGNORE_CACHE = ignore


def get_cache() -> Any:
    global _CACHE_INSTANCE
    if _CACHE_INSTANCE is None:
        with _CACHE_LOCK:
            if _CACHE_INSTANCE is None:
                try:
                    cache_dir = get_cache_dir()
                    api_cache_dir = get_api_cache_dir()
                    _migrate_legacy_cache(cache_dir, api_cache_dir)
                    api_cache_dir.mkdir(parents=True, exist_ok=True)
                    _CACHE_INSTANCE = diskcache.Cache(
                        str(api_cache_dir),
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


def _get_api_cache_size(api_cache_dir: Path) -> int:
    """Calculate total size of API cache files on disk (in bytes)."""
    if not api_cache_dir.exists():
        return 0
    total = 0
    try:
        if api_cache_dir.is_file():
            return api_cache_dir.stat().st_size
        for entry in api_cache_dir.rglob("*"):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except OSError:
                    pass
    except OSError as error:
        LOG.debug(f"Failed to calculate API cache size: {error}")
    return total


def get_cache_stats() -> CacheStats:
    """Collect current cache statistics across disk and memory layers."""
    cache_dir = get_cache_dir()
    api_cache_dir = get_api_cache_dir()
    api_entries = 0
    cache = get_cache()
    if cache is not None:
        try:
            with _CACHE_LOCK:
                api_entries = len(cache)
        except (
            OSError,
            ValueError,
            KeyError,
            RuntimeError,
            TypeError,
            diskcache.Timeout,
        ) as error:
            LOG.debug(f"Failed to get cache length: {error}")

    api_size = _get_api_cache_size(api_cache_dir)

    from sonora.core.state import get_library_state

    state_mgr = get_library_state()
    state_entries = state_mgr.get_state_count()
    state_size = state_mgr.get_state_size()

    from sonora.audio.metadata import get_metadata_cache_size

    memory_entries = get_metadata_cache_size()

    return CacheStats(
        cache_dir=cache_dir,
        api_entries=api_entries,
        api_size_bytes=api_size,
        state_entries=state_entries,
        state_size_bytes=state_size,
        memory_metadata_entries=memory_entries,
        total_size_bytes=api_size + state_size,
    )


def clear_cache(
    clear_api: bool = True,
    clear_state: bool = False,
    clear_memory: bool = True,
    purge: bool = False,
    dry_run: bool = False,
) -> ClearResult:
    """
    Clear cache layers following official DiskCache, SQLite, and memory cache protocols.

    :param clear_api: Clear diskcache API metadata.
    :param clear_state: Clear SQLite library state tracking database.
    :param clear_memory: Clear in-memory metadata and utility caches.
    :param purge: If True, unlink database and shard files entirely from disk.
    :param dry_run: If True, simulate operations and calculate reclaimable space without modifying disk.
    :return: ClearResult dataclass containing execution details.
    """
    cache_dir = get_cache_dir()
    api_cache_dir = get_api_cache_dir()

    if dry_run:
        stats = get_cache_stats()
        sim_api_entries = stats.api_entries if clear_api else 0
        sim_api_bytes = stats.api_size_bytes if clear_api else 0
        sim_state_entries = stats.state_entries if clear_state else 0
        sim_state_bytes = stats.state_size_bytes if clear_state else 0
        sim_memory_entries = stats.memory_metadata_entries if clear_memory else 0

        return ClearResult(
            cache_dir=cache_dir,
            api_cleared=clear_api,
            api_entries_cleared=sim_api_entries,
            api_bytes_freed=sim_api_bytes,
            state_cleared=clear_state,
            state_entries_cleared=sim_state_entries,
            state_bytes_freed=sim_state_bytes,
            memory_cleared=clear_memory,
            memory_metadata_cleared=sim_memory_entries,
            purged=purge,
            dry_run=True,
        )

    api_entries_cleared = 0
    api_bytes_freed = 0
    state_entries_cleared = 0
    state_bytes_freed = 0
    memory_metadata_cleared = 0

    if clear_api:
        api_bytes_before = _get_api_cache_size(api_cache_dir)
        cache = get_cache()
        if cache is not None:
            try:
                with _CACHE_LOCK:
                    api_entries_cleared = len(cache)
            except (
                OSError,
                ValueError,
                KeyError,
                RuntimeError,
                TypeError,
                diskcache.Timeout,
            ) as error:
                LOG.debug(f"Failed to read cache entries before clearing: {error}")

        if purge:
            close_cache()
            if api_cache_dir.exists():
                try:
                    shutil.rmtree(api_cache_dir, ignore_errors=True)
                except OSError as error:
                    LOG.debug(f"Failed to purge API cache directory: {error}")
        else:
            if cache is not None:
                try:
                    with _CACHE_LOCK:
                        cache.clear(retry=True)
                        cache.check(fix=True, retry=True)
                except (
                    OSError,
                    ValueError,
                    KeyError,
                    RuntimeError,
                    TypeError,
                    diskcache.Timeout,
                ) as error:
                    LOG.debug(f"Cache clear/check failed: {error}")

        api_bytes_after = _get_api_cache_size(api_cache_dir)
        api_bytes_freed = max(0, api_bytes_before - api_bytes_after)

    if clear_state:
        from sonora.core.state import get_library_state, reset_library_state

        state_mgr = get_library_state()
        state_entries_cleared = state_mgr.get_state_count()
        state_bytes_before = state_mgr.get_state_size()

        state_mgr.clear_state(purge=purge)
        if purge:
            reset_library_state()
            state_bytes_after = 0
        else:
            state_bytes_after = state_mgr.get_state_size()

        state_bytes_freed = max(0, state_bytes_before - state_bytes_after)

    if clear_memory:
        from sonora.audio.metadata import clear_metadata_cache
        from sonora.core.utils import clear_utils_cache

        memory_metadata_cleared = clear_metadata_cache()
        clear_utils_cache()

    return ClearResult(
        cache_dir=cache_dir,
        api_cleared=clear_api,
        api_entries_cleared=api_entries_cleared,
        api_bytes_freed=api_bytes_freed,
        state_cleared=clear_state,
        state_entries_cleared=state_entries_cleared,
        state_bytes_freed=state_bytes_freed,
        memory_cleared=clear_memory,
        memory_metadata_cleared=memory_metadata_cleared,
        purged=purge,
    )


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
