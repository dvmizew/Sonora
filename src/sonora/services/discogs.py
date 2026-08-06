"""
Discogs API service client.
"""

from typing import Any

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.exceptions import APIServiceError

try:
    import discogs_client
except ImportError:
    discogs_client = None


def search_discogs_release(artist: str, album: str, user_token: str | None = None) -> dict[str, Any] | None:
    """
    Search Discogs for album release metadata.
    Requires a Discogs user token.
    """
    if not user_token:
        return None

    if discogs_client is None:
        raise APIServiceError("discogs-client library is not installed.")

    cache_key = f"discogs:{artist.lower()}:{album.lower()}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return cached

    try:
        from sonora.core.constants import USER_AGENT
        client = discogs_client.Client(USER_AGENT, user_token=user_token)
        results = client.search(album, artist=artist, type="release")
        if results and len(results) > 0:
            first = results[0]
            res = {
                "id": getattr(first, "id", None),
                "title": getattr(first, "title", None),
                "year": getattr(first, "year", None),
                "genres": getattr(first, "genres", []),
            }
            set_cached_api(cache_key, res)
            return res
        return None
    except Exception as e:
        raise APIServiceError(f"Discogs search failed for {artist} - {album}: {e}") from e
