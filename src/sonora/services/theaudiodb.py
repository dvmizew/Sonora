import urllib.parse

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.utils import RateLimiter, normalize_str

_THEAUDIODB_LIMITER = RateLimiter(interval_seconds=1.0)


def fetch_artist_images(artist_name: str) -> tuple[bytes | None, bytes | None]:
    """
    Fetch artist avatar (artist.jpg) and wide banner (banner.jpg) from TheAudioDB with disk caching.
    Returns (thumb_bytes, banner_bytes).
    """
    if not artist_name or artist_name in ["Various Artists", "Unknown Artist"]:
        return None, None

    artist_key = normalize_str(artist_name)
    cache_key = f"theaudiodb:{artist_key}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, tuple) and len(cached) == 2:
        return cached

    _THEAUDIODB_LIMITER.wait()
    thumb_bytes, banner_bytes = None, None
    try:
        url = f"https://www.theaudiodb.com/api/v1/json/2/search.php?s={urllib.parse.quote(artist_name)}"
        resp = SESSION.get(url, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            artists = data.get("artists")
            if artists and isinstance(artists, list) and artists[0]:
                art = artists[0]
                thumb_url = art.get("strArtistThumb") or art.get("strArtistFanart")
                banner_url = art.get("strArtistBanner") or art.get("strArtistWideBanner") or art.get("strArtistFanart")

                if thumb_url:
                    try:
                        r_t = SESSION.get(thumb_url, timeout=6)
                        if r_t.status_code == 200:
                            thumb_bytes = r_t.content
                    except (OSError, ValueError) as e:
                        LOG.debug(f"Failed to fetch thumb image: {e}")

                if banner_url:
                    try:
                        r_b = SESSION.get(banner_url, timeout=6)
                        if r_b.status_code == 200:
                            banner_bytes = r_b.content
                    except (OSError, ValueError) as e:
                        LOG.debug(f"Failed to fetch banner image: {e}")

                res = (thumb_bytes, banner_bytes)
                if thumb_bytes or banner_bytes:
                    set_cached_api(cache_key, res, expire_seconds=2592000)  # 30 days
                return res
    except (OSError, ValueError, KeyError) as e:
        LOG.debug(f"TheAudioDB fetch_artist_images failed for {artist_name}: {e}")
    return None, None
