import threading

import musicbrainzngs

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.utils import RateLimiter, normalize_str

_MB_LIMITER = RateLimiter(interval_seconds=1.1)


def init_musicbrainz(app_name: str = "Sonora", version: str = "0.1.0", contact: str = "danielradu02@users.noreply.github.com") -> None:
    try:
        musicbrainzngs.set_useragent(app_name, version, contact)
    except (ValueError, AttributeError, RuntimeError) as e:
        from sonora.core.logger import LOG
        LOG.debug(f"MusicBrainz User-Agent initialization failed: {e}")

_discography_locks: dict[str, threading.Lock] = {}
_discography_meta_lock = threading.Lock()

def _get_discography_lock(artist_key: str) -> threading.Lock:
    with _discography_meta_lock:
        if len(_discography_locks) > 1000:
            _discography_locks.clear()
        if artist_key not in _discography_locks:
            _discography_locks[artist_key] = threading.Lock()
        return _discography_locks[artist_key]


def fetch_artist_discography(artist: str) -> list[dict[str, object]]:
    """
    Fetch and cache the entire discography (releases) of an artist from MusicBrainz in a single API call.
    Returns list of release dicts.
    """
    if musicbrainzngs is not None:
        init_musicbrainz()
    artist_key = normalize_str(artist)
    cache_key = f"mb_discography:{artist_key}"

    with _get_discography_lock(artist_key):
        cached = get_cached_api(cache_key)
        if isinstance(cached, list):
            return cached

        _MB_LIMITER.wait()
        try:
            result = musicbrainzngs.search_releases(artist=artist, limit=100)
            releases: list[dict[str, object]] = result.get("release-list", []) if isinstance(result, dict) else []
            set_cached_api(cache_key, releases, expire_seconds=2419200)  # 30 days
            return releases
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"MusicBrainz discography fetch failed for {artist}: {e}") from e


def search_musicbrainz_release(artist: str, album: str) -> dict[str, object] | None:
    """Search MusicBrainz for an album release matching artist and album name."""
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
        rel_title_val = rel.get("title", "")
        rel_title = normalize_str(str(rel_title_val))
        if rel_title == album_lower or album_lower in rel_title:
            set_cached_api(cache_key, rel)
            return rel

    _MB_LIMITER.wait()
    try:
        result = musicbrainzngs.search_releases(artist=artist, release=album, limit=5)
        raw_releases = result.get("release-list", []) if isinstance(result, dict) else []
        releases: list[dict[str, object]] = [r for r in raw_releases if isinstance(r, dict)]
        target_rel: dict[str, object] | None = releases[0] if releases else None
        set_cached_api(cache_key, target_rel)
        return target_rel
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError(f"MusicBrainz search failed for {artist} - {album}: {e}") from e



def fetch_track_mbid(artist: str, title: str) -> str | None:
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
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError(f"MusicBrainz track lookup failed for {artist} - {title}: {e}") from e


def fetch_cover_art_archive_url(release_mbid: str) -> str | None:
    """
    Check if Cover Art Archive has front cover art for the given MusicBrainz release MBID.
    Returns front cover image URL or None.
    """
    if not release_mbid:
        return None
    url = f"https://coverartarchive.org/release/{release_mbid}/front"
    try:
        from sonora.core.http import SESSION
        resp = SESSION.head(url, allow_redirects=True, timeout=5)
        if resp.status_code == 200:
            return resp.url or url
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        from sonora.core.logger import LOG
        LOG.debug(f"Cover Art Archive lookup failed: {e}")
    return None

