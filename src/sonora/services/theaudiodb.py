import urllib.parse

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.utils import RateLimiter, normalize_str

_THEAUDIODB_LIMITER = RateLimiter(interval_seconds=1.0)


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
    thumbnail_bytes, banner_bytes = None, None
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
                    try:
                        thumb_response = SESSION.get(thumbnail_url, timeout=6)
                        if thumb_response.status_code == 200:
                            thumbnail_bytes = thumb_response.content
                    except (OSError, ValueError) as error:
                        LOG.debug(f"Failed to fetch thumb image: {error}")

                if banner_url:
                    try:
                        banner_response = SESSION.get(banner_url, timeout=6)
                        if banner_response.status_code == 200:
                            banner_bytes = banner_response.content
                    except (OSError, ValueError) as error:
                        LOG.debug(f"Failed to fetch banner image: {error}")

                result = (thumbnail_bytes, banner_bytes)
                if thumbnail_bytes or banner_bytes:
                    set_cached_api(cache_key, result, expire_seconds=2592000)  # 30 days
                return result
    except (OSError, ValueError, KeyError) as error:
        LOG.debug(f"TheAudioDB fetch_artist_images failed for {artist_name}: {error}")
    return None, None


def fetch_track_video_url(artist_name: str, track_title: str) -> str | None:
    """
    Fetch official music video URL (strMusicVid) from TheAudioDB track search.
    """
    if not artist_name or not track_title:
        return None
    cache_key = (
        f"theaudiodb_vid:{normalize_str(artist_name)}:{normalize_str(track_title)}"
    )
    cached = get_cached_api(cache_key)
    if isinstance(cached, str):
        return cached

    _THEAUDIODB_LIMITER.wait()
    try:
        url = f"https://www.theaudiodb.com/api/v1/json/2/searchtrack.php?s={urllib.parse.quote(artist_name)}&t={urllib.parse.quote(track_title)}"
        response = SESSION.get(url, timeout=6)
        if response.status_code == 200:
            tracks = response.json().get("track", [])
            if tracks and isinstance(tracks, list) and tracks[0]:
                video_url = tracks[0].get("strMusicVid")
                if video_url and str(video_url).strip():
                    clean_url = str(video_url).strip()
                    set_cached_api(cache_key, clean_url)
                    return clean_url
    except (OSError, ValueError, KeyError) as error:
        LOG.debug(
            f"TheAudioDB video lookup failed for {artist_name} - {track_title}: {error}"
        )
    return None
