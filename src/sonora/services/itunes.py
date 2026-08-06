"""
iTunes Search API service client for metadata and high-res cover art.
"""

from typing import Any

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.exceptions import APIServiceError
from sonora.core.http import SESSION

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"


def search_itunes(artist: str, term: str, entity: str = "album", country: str = "US") -> list[dict[str, Any]]:
    """
    Search iTunes Search API for album or track metadata.
    """
    from sonora.core.utils import normalize_str
    query_term = f"{artist} {term}".strip()
    cache_key = f"itunes:{normalize_str(artist)}:{normalize_str(term)}:{entity}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, list):
        return cached

    params = {
        "term": query_term,
        "entity": entity,
        "country": country,
        "limit": "5",
    }
    try:
        resp = SESSION.get(ITUNES_SEARCH_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results: list[dict[str, Any]] = data.get("results", [])
        set_cached_api(cache_key, results)
        return results
    except Exception as e:
        raise APIServiceError(f"iTunes Search API request failed for {query_term}: {e}") from e


def fetch_itunes_cover_art_url(artist: str, album: str, resolution: int = 1400) -> str | None:
    """
    Fetch high-resolution album cover art URL from iTunes.
    Resolutions can be 600, 1400, or 3000.
    """
    results = search_itunes(artist=artist, term=album, entity="album")
    if not results:
        return None

    artwork_url = results[0].get("artworkUrl100")
    if artwork_url:
        # Upgrade low-res 100x100 URL to requested high resolution (e.g. 1400x1400 or 3000x3000)
        return artwork_url.replace("100x100bb", f"{resolution}x{resolution}bb")
    return None
