from typing import Any

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.exceptions import APIServiceError
from sonora.core.utils import RateLimiter, normalize_str

try:
    import musicbrainzngs
except ImportError:
    musicbrainzngs = None

_MB_LIMITER = RateLimiter(interval_seconds=1.0)


def init_musicbrainz(app_name: str = "Sonora", version: str = "0.1.0", contact: str = "danielradu02@users.noreply.github.com") -> None:
    """Initialize MusicBrainz User-Agent header and built-in rate limiter."""
    if musicbrainzngs is not None:
        musicbrainzngs.set_useragent(app_name, version, contact)
        musicbrainzngs.set_rate_limit(limit_or_interval=1.0, new_requests=1)

import threading

_discography_locks: dict[str, threading.Lock] = {}
_discography_meta_lock = threading.Lock()

def _get_discography_lock(artist_key: str) -> threading.Lock:
    with _discography_meta_lock:
        if len(_discography_locks) > 1000:
            _discography_locks.clear()
        if artist_key not in _discography_locks:
            _discography_locks[artist_key] = threading.Lock()
        return _discography_locks[artist_key]


def fetch_artist_discography(artist: str) -> list[dict[str, Any]]:
    """
    Fetch and cache the entire discography (releases) of an artist from MusicBrainz in a single API call.
    Returns list of release dicts.
    """
    if musicbrainzngs is None or not artist:
        return []

    artist_key = normalize_str(artist)
    cache_key = f"mb_discography:{artist_key}"

    with _get_discography_lock(artist_key):
        cached = get_cached_api(cache_key)
        if isinstance(cached, list):
            return cached

        _MB_LIMITER.wait()
        try:
            result = musicbrainzngs.search_releases(artist=artist, limit=100)
            releases: list[dict[str, Any]] = result.get("release-list", [])
            set_cached_api(cache_key, releases, expire_seconds=2419200)  # 30 days
            return releases
        except Exception as e:
            raise APIServiceError(f"MusicBrainz discography fetch failed for {artist}: {e}") from e


def search_musicbrainz_release(artist: str, album: str) -> dict[str, Any] | None:
    """Search MusicBrainz for an album release matching artist and album name."""
    if musicbrainzngs is None:
        raise APIServiceError("musicbrainzngs library is not installed.")
        
    if not album or normalize_str(album) in ["unknown album", "unknown"]:
        return None

    cache_key = f"mb_release:{normalize_str(artist)}:{normalize_str(album)}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return cached

    # Batch strategy: Check artist discography cache first
    discography = fetch_artist_discography(artist)
    album_lower = normalize_str(album)
    for rel in discography:
        rel_title = normalize_str(rel.get("title", ""))
        if rel_title == album_lower or album_lower in rel_title:
            set_cached_api(cache_key, rel)
            return rel

    _MB_LIMITER.wait()
    try:
        result = musicbrainzngs.search_releases(artist=artist, release=album, limit=5)
        releases: list[dict[str, Any]] = result.get("release-list", [])
        target_rel: dict[str, Any] | None = releases[0] if releases else None
        set_cached_api(cache_key, target_rel)
        return target_rel
    except Exception as e:
        raise APIServiceError(f"MusicBrainz search failed for {artist} - {album}: {e}") from e



def fetch_track_mbid(artist: str, title: str) -> str | None:
    """Search MusicBrainz for a track Recording ID (MBID)."""
    if musicbrainzngs is None:
        raise APIServiceError("musicbrainzngs library is not installed.")

    cache_key = f"mb_mbid:{normalize_str(artist)}:{normalize_str(title)}"
    cached = get_cached_api(cache_key)
    if cached is not None:
        return str(cached)

    _MB_LIMITER.wait()
    try:
        result = musicbrainzngs.search_recordings(artist=artist, recording=title, limit=5)
        recordings = result.get("recording-list", [])
        mbid = str(recordings[0].get("id")) if recordings else None
        set_cached_api(cache_key, mbid)
        return mbid
    except Exception as e:
        raise APIServiceError(f"MusicBrainz track lookup failed for {artist} - {title}: {e}") from e
