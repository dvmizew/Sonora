import threading
from typing import TYPE_CHECKING

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.exceptions import APIServiceError
from sonora.core.utils import RateLimiter, normalize_str

if TYPE_CHECKING:
    import discogs_client
else:
    try:
        import discogs_client
    except ImportError:
        discogs_client = None

_DISCOGS_LIMITER = RateLimiter(interval_seconds=1.1)

_discogs_locks: dict[str, threading.Lock] = {}
_discogs_meta_lock = threading.Lock()

def _get_discogs_lock(artist_key: str) -> threading.Lock:
    with _discogs_meta_lock:
        if len(_discogs_locks) > 1000:
            _discogs_locks.clear()
        if artist_key not in _discogs_locks:
            _discogs_locks[artist_key] = threading.Lock()
        return _discogs_locks[artist_key]

_discogs_client_instance = None
_discogs_client_token = None

def search_discogs_release(artist: str, album: str, user_token: str | None = None) -> dict[str, object] | None:
    """
    Search Discogs for release metadata using a User token.
    Returns a dict with metadata if found, otherwise None.
    """
    global _discogs_client_instance, _discogs_client_token
    if not user_token or not artist or not album:
        return None

    if discogs_client is None:
        raise APIServiceError("discogs-client library is not installed.")

    if normalize_str(album) in ["unknown album", "unknown"]:
        return None

    artist_key = normalize_str(artist)
    cache_key = f"discogs:{artist_key}:{normalize_str(album)}"
    
    with _get_discogs_lock(artist_key):
        cached = get_cached_api(cache_key)
        if isinstance(cached, dict):
            return cached

        _DISCOGS_LIMITER.wait()
        try:
            if _discogs_client_instance is None or _discogs_client_token != user_token:
                from sonora.core.constants import USER_AGENT
                _discogs_client_instance = discogs_client.Client(USER_AGENT, user_token=user_token)
                _discogs_client_token = user_token
            results = _discogs_client_instance.search(album, artist=artist, type="release")
            try:
                first = results[0]
                if first:
                    res = {
                        "id": getattr(first, "id", None),
                        "title": getattr(first, "title", None),
                        "year": getattr(first, "year", None),
                        "genres": getattr(first, "genres", []),
                    }
                    set_cached_api(cache_key, res)
                    return res
            except (IndexError, TypeError, AttributeError):
                pass
            return None
        except Exception as e:
            _discogs_client_instance = None
            if isinstance(e, APIServiceError):
                raise
            raise APIServiceError(f"Discogs search failed for {artist} - {album}: {e}") from e
