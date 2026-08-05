"""
Last.fm API service client for tag and mood/style metadata lookup.
"""

import json
import threading
import time
import urllib.parse
import urllib.request
from typing import Any

from sonora.core.exceptions import APIServiceError

_LASTFM_LOCK = threading.Lock()
_LAST_LASTFM_CALL = 0.0
_LASTFM_RATE_INTERVAL = 0.25


def _wait_lastfm_turn() -> None:
    """Thread-safe rate limiter for Last.fm API requests."""
    global _LAST_LASTFM_CALL
    with _LASTFM_LOCK:
        now = time.time()
        elapsed = now - _LAST_LASTFM_CALL
        if elapsed < _LASTFM_RATE_INTERVAL:
            time.sleep(_LASTFM_RATE_INTERVAL - elapsed)
        _LAST_LASTFM_CALL = time.time()


def fetch_lastfm_tags(artist: str, title: str, api_key: str | None = None, mbid: str | None = None) -> list[str]:
    """
    Fetch top tags from Last.fm API to use as MOOD/STYLE tags.
    Returns list of top 5 tags (title-cased).
    """
    if not api_key:
        return []

    _wait_lastfm_turn()
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
        url = f"http://ws.audioscrobbler.com/2.0/?{urllib.parse.urlencode(params)}"
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
            return res[:5]
    except Exception as e:
        if mbid and artist and title:
            return fetch_lastfm_tags(artist, title, api_key=api_key, mbid=None)
        raise APIServiceError(f"Last.fm tag fetch failed for {artist} - {title}: {e}") from e
