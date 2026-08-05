"""
syncedlyrics service client for searching and fetching synchronized LRC lyrics.
Supports Lrclib, Musixmatch, Genius, NetEase providers.
"""

import threading
import time
from pathlib import Path

from sonora.core.exceptions import APIServiceError

try:
    import syncedlyrics  # type: ignore
    HAS_SYNCEDLYRICS = True
except ImportError:
    syncedlyrics = None
    HAS_SYNCEDLYRICS = False

_LYRICS_LOCK = threading.Lock()
_LAST_LYRICS_CALL = 0.0
_LYRICS_RATE_INTERVAL = 1.0


def _wait_lyrics_turn() -> None:
    """Thread-safe rate limiter for lyrics scraping providers."""
    global _LAST_LYRICS_CALL
    with _LYRICS_LOCK:
        now = time.time()
        elapsed = now - _LAST_LYRICS_CALL
        if elapsed < _LYRICS_RATE_INTERVAL:
            time.sleep(_LYRICS_RATE_INTERVAL - elapsed)
        _LAST_LYRICS_CALL = time.time()


def fetch_synced_lyrics(
    artist: str,
    title: str,
    synced_only: bool = False,
    plain_only: bool = False,
    enhanced: bool = False,
    providers: list[str] | None = None,
    lang: str | None = None,
    save_path: Path | None = None,
) -> str | None:
    """
    Search and fetch LRC lyrics for a track using syncedlyrics.

    Parameters:
    - artist: Artist name
    - title: Track title
    - synced_only: If True, only returns lyrics with [mm:ss.xx] timestamps
    - plain_only: If True, returns plain text lyrics without timestamps
    - enhanced: If True, returns word-by-word <mm:ss.xx> enhanced LRC timestamps
    - providers: Custom list of providers e.g. ["Lrclib", "Musixmatch", "Genius", "NetEase"]
    - lang: Preferred language ISO code (e.g. "en", "ro")
    - save_path: Optional Path destination to write the .lrc file directly
    """
    if not artist or not title:
        return None

    if not HAS_SYNCEDLYRICS:
        raise APIServiceError("syncedlyrics library is not installed.")

    query = f"{artist} - {title}".strip()
    _wait_lyrics_turn()

    try:
        kwargs: dict[str, object] = {
            "plain_only": plain_only,
            "synced_only": synced_only,
            "enhanced": enhanced,
        }
        if providers:
            kwargs["providers"] = providers
        if lang:
            kwargs["lang"] = lang
        if save_path:
            kwargs["save_path"] = str(save_path)

        lrc_content = syncedlyrics.search(query, **kwargs)
        if lrc_content and len(str(lrc_content).strip()) > 0:
            return str(lrc_content).strip()
        return None
    except Exception as e:
        raise APIServiceError(f"Lyrics fetch failed for '{query}': {e}") from e
