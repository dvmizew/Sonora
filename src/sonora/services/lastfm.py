"""
Last.fm API service client for tag and mood/style metadata lookup.
"""

import json
import urllib.parse
import urllib.request
from typing import Any

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.exceptions import APIServiceError
from sonora.core.utils import RateLimiter

_LASTFM_LIMITER = RateLimiter(interval_seconds=0.25)


def fetch_lastfm_tags(artist: str, title: str, api_key: str | None = None, mbid: str | None = None) -> list[str]:
    """
    Fetch top tags from Last.fm API to use as MOOD/STYLE tags.
    Returns list of top 5 tags (title-cased).
    """
    if not api_key:
        return []

    cache_key = f"lastfm:{artist.lower()}:{title.lower()}:{mbid or ''}"
    cached = get_cached_api(cache_key)
    if cached is not None:
        return cached  # type: ignore[no-any-return]

    _LASTFM_LIMITER.wait()
    params: dict[str, Any] = {
        "method": "track.getTopTags",
        "api_key": api_key,
        "format": "json"
    }

    if mbid:
        params["mbid"] = mbid
    elif artist and title:
        params["artist"] = artist
        params["track"] = title
    else:
        return []

    try:
        url = f"https://ws.audioscrobbler.com/2.0/?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Sonora/0.1.0 (+https://github.com/dvmizew/Sonora)"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            tags = data.get("toptags", {}).get("tag", [])
            res = [
                str(t["name"]).title()
                for t in tags
                if isinstance(t, dict) and t.get("name")
            ]
            if not res and mbid and artist and title:
                # Fallback to artist+title if MBID returned 0 tags
                return fetch_lastfm_tags(artist, title, api_key=api_key, mbid=None)
            final_res = res[:5]
            set_cached_api(cache_key, final_res)
            return final_res
    except Exception as e:
        if mbid and artist and title:
            return fetch_lastfm_tags(artist, title, api_key=api_key, mbid=None)
        raise APIServiceError(f"Last.fm tag fetch failed for {artist} - {title}: {e}") from e
