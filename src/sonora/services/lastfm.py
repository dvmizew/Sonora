import httpx

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.constants import RATE_LIMIT_LASTFM
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.utils import RateLimiter, is_valid_uuid, normalize_str

_LASTFM_LIMITER = RateLimiter(interval_seconds=RATE_LIMIT_LASTFM)


def fetch_lastfm_tags(
    artist: str,
    title: str,
    api_key: str | None = None,
    mbid: str | None = None,
    _retried: bool = False,
) -> list[str]:
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
        "format": "json",
    }

    if mbid and is_valid_uuid(mbid):
        params["mbid"] = mbid
    elif artist and title:
        params["artist"] = artist
        params["track"] = title
    else:
        return []

    try:
        url = "https://ws.audioscrobbler.com/2.0/"
        response = SESSION.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        tags = data.get("toptags", {}).get("tag", [])
        tag_names = [
            tag["name"].title()
            for tag in tags
            if isinstance(tag, dict)
            and tag.get("name")
            and isinstance(tag["name"], str)
        ]
        if not tag_names and mbid and artist and title and not _retried:
            # Fallback to artist+title if MBID returned 0 tags
            return fetch_lastfm_tags(
                artist, title, api_key=api_key, mbid=None, _retried=True
            )
        final_tags = tag_names[:5]
        set_cached_api(cache_key, final_tags)
        return final_tags
    except (httpx.HTTPError, OSError, ValueError, KeyError, RuntimeError) as error:
        if mbid and artist and title and not _retried:
            return fetch_lastfm_tags(
                artist, title, api_key=api_key, mbid=None, _retried=True
            )
        LOG.debug(f"Last.fm tag fetch failed for {artist} - {title}: {error}")
        return []


def fetch_lastfm_track_stats(
    artist: str, title: str, api_key: str | None = None
) -> dict[str, int] | None:
    """
    Fetch track popularity metrics (listeners, playcount) from Last.fm API.
    """
    if not api_key or not artist or not title:
        return None

    cache_key = f"lastfm_stats:{normalize_str(artist)}:{normalize_str(title)}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return cached

    _LASTFM_LIMITER.wait()
    params = {
        "method": "track.getInfo",
        "api_key": api_key,
        "artist": artist,
        "track": title,
        "format": "json",
    }
    try:
        url = "https://ws.audioscrobbler.com/2.0/"
        response = SESSION.get(url, params=params, timeout=5)
        response.raise_for_status()
        track = response.json().get("track", {})
        listeners_str = track.get("listeners")
        playcount_str = track.get("playcount")
        stats_result = {
            "listeners": int(listeners_str)
            if listeners_str and str(listeners_str).isdigit()
            else 0,
            "playcount": int(playcount_str)
            if playcount_str and str(playcount_str).isdigit()
            else 0,
        }
        set_cached_api(cache_key, stats_result)
        return stats_result
    except (httpx.HTTPError, OSError, ValueError, KeyError, RuntimeError) as error:
        LOG.debug(f"Last.fm stats fetch failed for {artist} - {title}: {error}")
        return None
