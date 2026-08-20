from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.exceptions import APIServiceError
from sonora.core.http import SESSION
from sonora.core.utils import RateLimiter, normalize_str

_LASTFM_LIMITER = RateLimiter(interval_seconds=0.25)


def fetch_lastfm_tags(artist: str, title: str, api_key: str | None = None, mbid: str | None = None, _retried: bool = False) -> list[str]:
    """
    Fetch top tags from Last.fm API to use as MOOD/STYLE tags.
    Returns list of top 5 tags (title-cased).
    """
    if not api_key:
        return []

    cache_key = f"lastfm:{normalize_str(artist)}:{normalize_str(title)}:{mbid or ''}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, list):
        return cached

    _LASTFM_LIMITER.wait()
    params: dict[str, str] = {
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
        url = "https://ws.audioscrobbler.com/2.0/"
        resp = SESSION.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        tags = data.get("toptags", {}).get("tag", [])
        res = [
            t["name"].title()
            for t in tags
            if isinstance(t, dict) and t.get("name") and isinstance(t["name"], str)
        ]
        if not res and mbid and artist and title and not _retried:
            # Fallback to artist+title if MBID returned 0 tags
            return fetch_lastfm_tags(artist, title, api_key=api_key, mbid=None, _retried=True)
        final_res = res[:5]
        set_cached_api(cache_key, final_res)
        return final_res
    except Exception as e:
        if mbid and artist and title and not _retried:
            return fetch_lastfm_tags(artist, title, api_key=api_key, mbid=None, _retried=True)
        raise APIServiceError(f"Last.fm tag fetch failed for {artist} - {title}: {e}") from e
