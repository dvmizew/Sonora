import urllib.parse

import httpx

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.constants import RATE_LIMIT_THEAUDIODB
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.utils import RateLimiter, normalize_str

_THEAUDIODB_LIMITER = RateLimiter(interval_seconds=RATE_LIMIT_THEAUDIODB)


def _download_artwork_bytes(url: str | None) -> bytes | None:
    if not url:
        return None
    _THEAUDIODB_LIMITER.wait()
    try:
        response = SESSION.get(url, timeout=10)
        if response.status_code == 200:
            return response.content
    except (httpx.HTTPError, OSError, ValueError, RuntimeError) as error:
        LOG.debug(f"Failed to fetch artwork from {url}: {error}")
    return None


def fetch_artist_images(artist_name: str) -> tuple[bytes | None, bytes | None]:
    """
    Fetch artist avatar (artist.jpg) and wide banner (banner.jpg) from TheAudioDB with disk caching.
    Returns (thumbnail_bytes, banner_bytes).
    """
    if not artist_name or artist_name in ["Various Artists", "Unknown Artist"]:
        return None, None

    artist_key = normalize_str(artist_name)
    cache_key = f"theaudiodb:{artist_key}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, tuple) and len(cached) == 2:
        return cached

    _THEAUDIODB_LIMITER.wait()
    thumbnail_bytes: bytes | None = None
    banner_bytes: bytes | None = None
    try:
        url = f"https://www.theaudiodb.com/api/v1/json/2/search.php?s={urllib.parse.quote(artist_name)}"
        response = SESSION.get(url, timeout=6)
        if response.status_code == 200:
            data = response.json()
            artists = data.get("artists")
            if artists and isinstance(artists, list) and artists[0]:
                artist_data = artists[0]
                thumbnail_url = artist_data.get("strArtistThumb") or artist_data.get(
                    "strArtistFanart"
                )
                banner_url = (
                    artist_data.get("strArtistBanner")
                    or artist_data.get("strArtistWideBanner")
                    or artist_data.get("strArtistFanart")
                )

                if thumbnail_url:
                    thumbnail_bytes = _download_artwork_bytes(thumbnail_url)
                if banner_url:
                    banner_bytes = _download_artwork_bytes(banner_url)

                result = (thumbnail_bytes, banner_bytes)
                if thumbnail_bytes or banner_bytes:
                    set_cached_api(cache_key, result)
                return result
    except (httpx.HTTPError, OSError, ValueError, KeyError, RuntimeError) as error:
        LOG.debug(f"TheAudioDB fetch_artist_images failed for {artist_name}: {error}")
    return None, None


def fetch_theaudiodb_track_details(
    artist_name: str, track_title: str
) -> dict[str, object] | None:
    """
    Fetch track details (video URL, mood, style, key, rating, description) from TheAudioDB.
    """
    if not artist_name or not track_title:
        return None
    cache_key = (
        f"theaudiodb_track:{normalize_str(artist_name)}:{normalize_str(track_title)}"
    )
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return cached

    _THEAUDIODB_LIMITER.wait()
    try:
        url = f"https://www.theaudiodb.com/api/v1/json/2/searchtrack.php?s={urllib.parse.quote(artist_name)}&t={urllib.parse.quote(track_title)}"
        response = SESSION.get(url, timeout=6)
        if response.status_code == 200:
            tracks = response.json().get("track", [])
            if tracks and isinstance(tracks, list) and tracks[0]:
                raw_track = tracks[0]
                rating_raw = raw_track.get("intScore")
                rating: float | None = None
                if rating_raw is not None:
                    try:
                        rating = float(rating_raw)
                    except (ValueError, TypeError):
                        rating = None

                def _clean_str(val: object) -> str | None:
                    if val is None:
                        return None
                    s = str(val).strip()
                    return s if s and s.lower() not in ("null", "none", "") else None

                details: dict[str, object] = {
                    "music_video_url": _clean_str(raw_track.get("strMusicVid")),
                    "mood": _clean_str(raw_track.get("strMood")),
                    "style": _clean_str(raw_track.get("strStyle")),
                    "initial_key": _clean_str(raw_track.get("strKey"))
                    or _clean_str(raw_track.get("strOpenKey")),
                    "rating": rating,
                    "description": _clean_str(raw_track.get("strDescriptionEN")),
                    "genre": _clean_str(raw_track.get("strGenre")),
                }
                set_cached_api(cache_key, details)
                return details
    except (httpx.HTTPError, OSError, ValueError, KeyError, RuntimeError) as error:
        LOG.debug(
            f"TheAudioDB track lookup failed for {artist_name} - {track_title}: {error}"
        )
    return None
