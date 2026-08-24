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
    except (ValueError, AttributeError, RuntimeError) as error:
        LOG.debug(f"MusicBrainz User-Agent initialization failed: {error}")


def fetch_artist_discography(artist: str) -> list[dict[str, object]]:
    """
    Fetch and cache the entire discography (releases) of an artist from MusicBrainz in a single API call.
    Returns list of release dicts.
    """
    if musicbrainzngs is not None:
        init_musicbrainz()
    artist_key = normalize_str(artist)
    cache_key = f"mb_discography:{artist_key}"

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
    ) as error:
        if isinstance(error, RuntimeError):
            raise
        raise RuntimeError(
            f"MusicBrainz discography fetch failed for {artist}: {error}"
        ) from error


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
    for release in discography:
        release_title_value = release.get("title", "")
        release_title = normalize_str(str(release_title_value))
        if release_title == album_lower or album_lower in release_title:
            set_cached_api(cache_key, release)
            return release

    _MB_LIMITER.wait()
    try:
        result = musicbrainzngs.search_releases(artist=artist, release=album, limit=5)
        raw_releases = (
            result.get("release-list", []) if isinstance(result, dict) else []
        )
        releases: list[dict[str, object]] = [
            release for release in raw_releases if isinstance(release, dict)
        ]
        target_release: dict[str, object] | None = releases[0] if releases else None
        set_cached_api(cache_key, target_release)
        return target_release
    except (
        MusicBrainzError,
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
    ) as error:
        if isinstance(error, RuntimeError):
            raise
        raise RuntimeError(
            f"MusicBrainz search failed for {artist} - {album}: {error}"
        ) from error


def fetch_track_mbid(artist: str, title: str) -> str | None:
    cleaned_title = clean_title(title)
    cache_key = f"mb_mbid:{normalize_str(artist)}:{normalize_str(cleaned_title)}"
    cached = get_cached_api(cache_key)
    if cached is not None:
        return str(cached) if cached else None

    _MB_LIMITER.wait()
    try:
        result = musicbrainzngs.search_recordings(
            artist=artist, recording=cleaned_title, limit=5
        )
        recordings = result.get("recording-list", [])
        if not recordings:
            set_cached_api(cache_key, None)
            return None

        best_mbid = None
        best_score = 0.0

        for recording in recordings:
            recording_title = str(recording.get("title", ""))
            recording_artist = ""
            artist_credit: Any = recording.get("artist-credit", [])
            if (
                isinstance(artist_credit, list)
                and artist_credit
                and isinstance(artist_credit[0], dict)
            ):
                recording_artist = str(
                    artist_credit[0].get("artist", {}).get("name", "")
                )

            score = match_score(
                artist, cleaned_title, recording_artist, recording_title
            )
            if score > best_score:
                best_score = score
                best_mbid = str(recording.get("id"))

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
    ) as error:
        if isinstance(error, RuntimeError):
            raise
        raise RuntimeError(
            f"MusicBrainz track lookup failed for {artist} - {title}: {error}"
        ) from error


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
        response = SESSION.head(url, timeout=5)
        if response.status_code == 200:
            result_url = str(response.url) or url
            set_cached_api(cache_key, result_url)
            return result_url
        set_cached_api(cache_key, None)
    except (httpx.HTTPError, OSError, ValueError, KeyError, RuntimeError) as error:
        LOG.debug(f"Cover Art Archive lookup failed: {error}")
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
        return {
            int(track_position_key): str(recording_mbid_value)
            for track_position_key, recording_mbid_value in cached.items()
        }

    _MB_LIMITER.wait()
    try:
        release_data = musicbrainzngs.get_release_by_id(
            release_mbid, includes=["recordings", "media", "artist-credits"]
        )
        mediums = (
            release_data.get("release", {}).get("medium-list", [])
            if isinstance(release_data, dict)
            else []
        )
        mapping: dict[int, str] = {}
        for medium in mediums:
            if isinstance(medium, dict):
                for track in medium.get("track-list", []):
                    if isinstance(track, dict):
                        position = track.get("position")
                        recording_id = track.get("recording", {}).get("id")
                        if position and recording_id:
                            mapping[int(position)] = str(recording_id)
        set_cached_api(cache_key, mapping)
        return mapping
    except (
        MusicBrainzError,
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
    ) as error:
        if isinstance(error, RuntimeError):
            raise
        LOG.debug(f"MusicBrainz album track fetch failed for {release_mbid}: {error}")
        return {}
