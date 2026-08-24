import threading
from typing import Any

import httpx
import musicbrainzngs
from musicbrainzngs import MusicBrainzError

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.utils import (
    RateLimiter,
    clean_title,
    is_valid_uuid,
    match_score,
    normalize_str,
)

_MB_LIMITER = RateLimiter(interval_seconds=1.1)


def init_musicbrainz(
    app_name: str = "Sonora",
    version: str = "0.1.0",
    contact: str = "danielradu02@users.noreply.github.com",
) -> None:
    try:
        musicbrainzngs.set_useragent(app_name, version, contact)
    except (ValueError, AttributeError, RuntimeError) as e:
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
            releases: list[dict[str, object]] = (
                result.get("release-list", []) if isinstance(result, dict) else []
            )
            set_cached_api(cache_key, releases, expire_seconds=2419200)  # 30 days
            return releases
        except (
            MusicBrainzError,
            OSError,
            ValueError,
            KeyError,
            RuntimeError,
        ) as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(
                f"MusicBrainz discography fetch failed for {artist}: {e}"
            ) from e


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
        raw_releases = (
            result.get("release-list", []) if isinstance(result, dict) else []
        )
        releases: list[dict[str, object]] = [
            r for r in raw_releases if isinstance(r, dict)
        ]
        target_rel: dict[str, object] | None = releases[0] if releases else None
        set_cached_api(cache_key, target_rel)
        return target_rel
    except (
        MusicBrainzError,
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
    ) as e:
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError(
            f"MusicBrainz search failed for {artist} - {album}: {e}"
        ) from e


def fetch_track_mbid(artist: str, title: str) -> str | None:
    c_title = clean_title(title)
    cache_key = f"mb_mbid:{normalize_str(artist)}:{normalize_str(c_title)}"
    cached = get_cached_api(cache_key)
    if cached is not None:
        return str(cached) if cached else None

    _MB_LIMITER.wait()
    try:
        result = musicbrainzngs.search_recordings(
            artist=artist, recording=c_title, limit=5
        )
        recordings = result.get("recording-list", [])
        if not recordings:
            set_cached_api(cache_key, None)
            return None

        best_mbid = None
        best_score = 0.0

        for rec in recordings:
            rec_title = str(rec.get("title", ""))
            rec_artist = ""
            artist_credit: Any = rec.get("artist-credit", [])
            if (
                isinstance(artist_credit, list)
                and artist_credit
                and isinstance(artist_credit[0], dict)
            ):
                rec_artist = str(artist_credit[0].get("artist", {}).get("name", ""))

            score = match_score(artist, c_title, rec_artist, rec_title)
            if score > best_score:
                best_score = score
                best_mbid = str(rec.get("id"))

        # Threshold check: require score >= 65 to avoid wrong MBIDs
        if best_score < 65.0 or not is_valid_uuid(best_mbid):
            best_mbid = None

        set_cached_api(cache_key, best_mbid)
        return best_mbid
    except (
        MusicBrainzError,
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
    ) as e:
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError(
            f"MusicBrainz track lookup failed for {artist} - {title}: {e}"
        ) from e


def fetch_cover_art_archive_url(release_mbid: str) -> str | None:
    """
    Check if Cover Art Archive has front cover art for the given MusicBrainz release MBID.
    Returns front cover image URL or None.
    """
    if not release_mbid or not is_valid_uuid(release_mbid):
        return None
    cache_key = f"caa_url:{release_mbid}"
    cached = get_cached_api(cache_key)
    if cached is not None:
        return str(cached) if cached else None

    url = f"https://coverartarchive.org/release/{release_mbid}/front"
    try:
        resp = SESSION.head(url, allow_redirects=True, timeout=5)
        if resp.status_code == 200:
            res_url = str(resp.url) or url
            set_cached_api(cache_key, res_url)
            return res_url
        set_cached_api(cache_key, None)
    except (httpx.HTTPError, OSError, ValueError, KeyError, RuntimeError) as e:
        LOG.debug(f"Cover Art Archive lookup failed: {e}")
    return None


def fetch_album_track_mbids(release_mbid: str) -> dict[int, str]:
    """
    Fetch all track recording MBIDs for an entire album release in ONE single API call.
    Returns mapping of track_position (1-indexed) -> recording_mbid.
    """
    if not release_mbid or not is_valid_uuid(release_mbid):
        return {}

    cache_key = f"mb_album_tracks:{release_mbid}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return {int(k): str(v) for k, v in cached.items()}

    _MB_LIMITER.wait()
    try:
        rel = musicbrainzngs.get_release_by_id(
            release_mbid, includes=["recordings", "media", "artist-credits"]
        )
        mediums = (
            rel.get("release", {}).get("medium-list", [])
            if isinstance(rel, dict)
            else []
        )
        mapping: dict[int, str] = {}
        for m in mediums:
            if isinstance(m, dict):
                for t in m.get("track-list", []):
                    if isinstance(t, dict):
                        pos = t.get("position")
                        rec_id = t.get("recording", {}).get("id")
                        if pos and rec_id:
                            mapping[int(pos)] = str(rec_id)
        set_cached_api(cache_key, mapping)
        return mapping
    except (
        MusicBrainzError,
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
    ) as e:
        if isinstance(e, RuntimeError):
            raise
        LOG.debug(f"MusicBrainz album track fetch failed for {release_mbid}: {e}")
        return {}
