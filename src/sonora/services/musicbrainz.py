from types import ModuleType
from typing import Any

from sonora.core.exceptions import APIServiceError
from sonora.core.utils import RateLimiter

musicbrainzngs: ModuleType | None = None
try:
    import musicbrainzngs  # type: ignore
except ImportError:
    pass

_MB_LIMITER = RateLimiter(interval_seconds=1.0)


def init_musicbrainz(app_name: str = "Sonora", version: str = "0.1.0", contact: str = "danielradu02@users.noreply.github.com") -> None:
    """Initialize MusicBrainz User-Agent header."""
    if musicbrainzngs is not None:
        musicbrainzngs.set_useragent(app_name, version, contact)


def search_musicbrainz_release(artist: str, album: str) -> dict[str, Any] | None:
    """Search MusicBrainz for an album release matching artist and album name."""
    if musicbrainzngs is None:
        raise APIServiceError("musicbrainzngs library is not installed.")

    _MB_LIMITER.wait()
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
    if musicbrainzngs is None:
        raise APIServiceError("musicbrainzngs library is not installed.")

    _MB_LIMITER.wait()
    try:
        result = musicbrainzngs.search_recordings(artist=artist, recording=title, limit=5)
        recordings = result.get("recording-list", [])
        if recordings:
            return str(recordings[0].get("id"))
        return None
    except Exception as e:
        raise APIServiceError(f"MusicBrainz track lookup failed for {artist} - {title}: {e}") from e
