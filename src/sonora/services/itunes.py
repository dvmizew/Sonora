"""
iTunes Search API service client for metadata and high-res cover art.
"""

import json
import urllib.parse
import urllib.request
from typing import Any

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.exceptions import APIServiceError

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"


def search_itunes(artist: str, term: str, entity: str = "album", country: str = "US") -> list[dict[str, Any]]:
    """
    Search iTunes Search API for album or track metadata.
    """
    query_term = f"{artist} {term}".strip()
    cache_key = f"itunes:{artist.lower()}:{term.lower()}:{entity}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, list):
        return cached

    params = {
        "term": query_term,
        "entity": entity,
        "country": country,
        "limit": "5",
    }
    url = f"{ITUNES_SEARCH_URL}?{urllib.parse.urlencode(params)}"

    try:
        from sonora.core.constants import USER_AGENT
        req = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
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
