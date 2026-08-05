"""
syncedlyrics service client for searching and fetching synchronized LRC lyrics.
Supports Lrclib, Musixmatch, Genius, NetEase providers.
"""

from pathlib import Path
from types import ModuleType

from sonora.core.exceptions import APIServiceError
from sonora.core.utils import RateLimiter

syncedlyrics: ModuleType | None = None
try:
    import syncedlyrics  # type: ignore
except ImportError:
    pass

_LYRICS_LIMITER = RateLimiter(interval_seconds=1.0)


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

    if syncedlyrics is None:
        raise APIServiceError("syncedlyrics library is not installed.")

    query = f"{artist} - {title}".strip()
    _LYRICS_LIMITER.wait()

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
