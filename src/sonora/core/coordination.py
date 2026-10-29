import functools
import hashlib
import inspect
import sqlite3
import time
from collections.abc import Callable
from contextlib import closing
from typing import ParamSpec, TypeVar

from filelock import FileLock

from sonora.core.cache import get_cache_dir

_P = ParamSpec("_P")
_Result = TypeVar("_Result")


def coalesce_lookup(lookup: Callable[_P, _Result]) -> Callable[_P, _Result]:
    """Let duplicate requests reuse the first caller's cache entry across processes."""
    signature = inspect.signature(lookup)

    @functools.wraps(lookup)
    def coordinated(*arguments: _P.args, **keywords: _P.kwargs) -> _Result:
        bound = signature.bind(*arguments, **keywords)
        bound.apply_defaults()
        request_identity = repr(
            (
                lookup.__module__,
                lookup.__qualname__,
                {
                    name: value
                    for name, value in bound.arguments.items()
                    if name not in {"details", "lookup_state"}
                },
            )
        )
        request_digest = hashlib.sha256(request_identity.encode("utf-8")).hexdigest()
        directory = get_cache_dir() / "coordination" / "lookups"
        directory.mkdir(parents=True, exist_ok=True)
        # Native OS locks are released on process exit; no expiring lease can overlap a slow request.
        with FileLock(str(directory / f"{request_digest}.lock"), timeout=180):
            return lookup(*arguments, **keywords)

    return coordinated


def wait_for_provider(
    provider: str, interval: float, *, defer_seconds: float | None = None
) -> float:
    """Pace actual dispatch across processes, including delayed worker wakeups."""
    directory = get_cache_dir() / "coordination"
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(provider.encode("utf-8")).hexdigest()
    with (
        FileLock(str(directory / f"quota-{digest}.lock"), timeout=180),
        closing(
            sqlite3.connect(directory / "rate_limits.db", timeout=10)
        ) as connection,
    ):
        connection.execute(
            "CREATE TABLE IF NOT EXISTS provider_slots (provider TEXT PRIMARY KEY, next_request REAL NOT NULL)"
        )
        existing = connection.execute(
            "SELECT next_request FROM provider_slots WHERE provider = ?",
            (provider,),
        ).fetchone()
        if defer_seconds is not None:
            with connection:
                connection.execute(
                    "INSERT INTO provider_slots VALUES (?, ?) ON CONFLICT(provider) DO UPDATE SET next_request = MAX(next_request, excluded.next_request)",
                    (provider, time.time() + defer_seconds),
                )
            return 0.0
        delay = max(0.0, float(existing[0]) - time.time()) if existing else 0.0
        if delay > 0:
            time.sleep(delay)
        with connection:
            connection.execute(
                "INSERT INTO provider_slots VALUES (?, ?) ON CONFLICT(provider) DO UPDATE SET next_request = excluded.next_request",
                (provider, time.time() + interval),
            )
    return delay
