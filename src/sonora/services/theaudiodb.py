import urllib.parse

import httpx

from sonora.core.api_payloads import require_api_object, require_api_records
from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.text import normalize_str, safe_float


def _download_artwork_bytes(url: str | None) -> bytes | None:
    if not url:
        return None

    try:
        response = SESSION.get(url, timeout=10)
        if response.status_code == 200:
            return response.content
    except (httpx.HTTPError, OSError) as error:
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

    thumbnail_bytes: bytes | None = None
    banner_bytes: bytes | None = None
    try:
        url = f"https://www.theaudiodb.com/api/v1/json/123/search.php?s={urllib.parse.quote(artist_name)}"
        response = SESSION.get(url, timeout=6)
        if response.status_code == 200:
            theaudiodb_payload = require_api_object(
                response.json(), "TheAudioDB artist response"
            )
            raw_artists = theaudiodb_payload["artists"]
            artists = (
                []
                if raw_artists is None
                else require_api_records(raw_artists, "TheAudioDB artists")
            )
            if artists:
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
    except (httpx.HTTPError, OSError) as error:
        LOG.debug(f"TheAudioDB fetch_artist_images failed for {artist_name}: {error}")
    return None, None


def _clean_theaudiodb_str(metadata_value: object) -> str | None:
    if metadata_value is None:
        return None
    cleaned_value = str(metadata_value).strip()
    return (
        cleaned_value
        if cleaned_value and cleaned_value.lower() not in ("null", "none", "")
        else None
    )


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
        return cached if cached else None

    try:
        url = f"https://www.theaudiodb.com/api/v1/json/123/searchtrack.php?s={urllib.parse.quote(artist_name)}&t={urllib.parse.quote(track_title)}"
        response = SESSION.get(url, timeout=6)
        if response.status_code == 200:
            theaudiodb_payload = require_api_object(
                response.json(), "TheAudioDB track response"
            )
            raw_tracks = theaudiodb_payload["track"]
            tracks = (
                []
                if raw_tracks is None
                else require_api_records(raw_tracks, "TheAudioDB tracks")
            )
            if tracks:
                raw_track = tracks[0]
                rating = safe_float(raw_track.get("intScore"))

                details: dict[str, object] = {
                    "music_video_url": _clean_theaudiodb_str(
                        raw_track.get("strMusicVid")
                    ),
                    "mood": _clean_theaudiodb_str(raw_track.get("strMood")),
                    "style": _clean_theaudiodb_str(raw_track.get("strStyle")),
                    "initial_key": _clean_theaudiodb_str(raw_track.get("strKey"))
                    or _clean_theaudiodb_str(raw_track.get("strOpenKey")),
                    "rating": rating,
                    "description": _clean_theaudiodb_str(
                        raw_track.get("strDescriptionEN")
                    ),
                    "genre": _clean_theaudiodb_str(raw_track.get("strGenre")),
                }
                set_cached_api(cache_key, details)
                return details
            set_cached_api(cache_key, {})
    except (httpx.HTTPError, OSError) as error:
        LOG.debug(
            f"TheAudioDB track lookup failed for {artist_name} - {track_title}: {error}"
        )
    return None
