"""
MusicBrainz API service client with rate limiting.
"""

import threading
import time
from typing import Any

from sonora.core.exceptions import APIServiceError

try:
    import musicbrainzngs  # type: ignore
    HAS_MUSICBRAINZ = True
except ImportError:
    HAS_MUSICBRAINZ = False

_MB_LOCK = threading.Lock()
_LAST_MB_CALL = 0.0
_MB_RATE_INTERVAL = 1.0  # 1 request per second compliant with MB policy


def _wait_musicbrainz_turn() -> None:
    """Thread-safe rate limiter for MusicBrainz API requests."""
    global _LAST_MB_CALL
    with _MB_LOCK:
        now = time.time()
        elapsed = now - _LAST_MB_CALL
        if elapsed < _MB_RATE_INTERVAL:
            time.sleep(_MB_RATE_INTERVAL - elapsed)
        _LAST_MB_CALL = time.time()


def init_musicbrainz(app_name: str = "Sonora", version: str = "0.1.0", contact: str = "danielradu02@users.noreply.github.com") -> None:
    """Initialize MusicBrainz User-Agent header."""
    if HAS_MUSICBRAINZ and musicbrainzngs:
        musicbrainzngs.set_useragent(app_name, version, contact)


def search_musicbrainz_release(artist: str, album: str) -> dict[str, Any] | None:
    """Search MusicBrainz for an album release matching artist and album name."""
    if not HAS_MUSICBRAINZ or not musicbrainzngs:
        raise APIServiceError("musicbrainzngs library is not installed.")

    _wait_musicbrainz_turn()
    try:
        result = musicbrainzngs.search_releases(artist=artist, release=album, limit=5)
        releases = result.get("release-list", [])
        if releases:
            return releases[0]
        return None
    except Exception as e:
        raise APIServiceError(f"MusicBrainz search failed for {artist} - {album}: {e}") from e


def fetch_track_mbid(artist: str, title: str) -> str | None:
    """Search MusicBrainz for a track Recording ID (MBID)."""
    if not HAS_MUSICBRAINZ or not musicbrainzngs:
        raise APIServiceError("musicbrainzngs library is not installed.")

    _wait_musicbrainz_turn()
    try:
        result = musicbrainzngs.search_recordings(artist=artist, recording=title, limit=5)
        recordings = result.get("recording-list", [])
        if recordings:
            return str(recordings[0].get("id"))
        return None
    except Exception as e:
        raise APIServiceError(f"MusicBrainz track lookup failed for {artist} - {title}: {e}") from e
