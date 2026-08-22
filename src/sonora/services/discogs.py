import threading

import discogs_client

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.utils import RateLimiter, normalize_str

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
                    labels = getattr(first, "labels", None)
                    label_name = labels[0].name if labels and len(labels) > 0 and hasattr(labels[0], "name") else None
                    cat_no = labels[0].catno if labels and len(labels) > 0 and hasattr(labels[0], "catno") else None
                    barcodes = getattr(first, "barcodes", None)
                    barcode_val = barcodes[0] if barcodes and len(barcodes) > 0 else None

                    res = {
                        "id": getattr(first, "id", None),
                        "title": getattr(first, "title", None),
                        "year": getattr(first, "year", None),
                        "genres": list(getattr(first, "genres", []) or []),
                        "country": getattr(first, "country", None),
                        "label": label_name,
                        "catalog_number": cat_no,
                        "barcode": barcode_val,
                    }
                    set_cached_api(cache_key, res)
                    return res
            except (IndexError, TypeError, AttributeError) as e:
                from sonora.core.logger import LOG
                LOG.debug(f"Discogs empty result parse: {e}")
            return None
        except Exception as e:
            _discogs_client_instance = None
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"Discogs search failed for {artist} - {album}: {e}") from e
