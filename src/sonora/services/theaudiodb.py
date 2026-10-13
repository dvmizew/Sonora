import urllib.parse

import httpx
from rich.markup import escape

from sonora.core.api_payloads import require_api_object, require_api_records
from sonora.core.artists import get_primary_artist
from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.coordination import coalesce_lookup
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.text import is_valid_uuid, normalize_str, safe_float
from sonora.core.titles import recording_identity_matches


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


@coalesce_lookup
def fetch_artist_images(
    artist_name: str, *, artist_mbid: str | None = None
) -> tuple[bytes | None, bytes | None]:
    """
    Fetch artist avatar (artist.jpg) and wide banner (banner.jpg) from TheAudioDB with disk caching.
    Returns (thumbnail_bytes, banner_bytes).
    """
    if not artist_name or artist_name in ["Various Artists", "Unknown Artist"]:
        return None, None

    artist_key = normalize_str(artist_name)
    if not artist_key:
        return None, None
    cache_key = f"theaudiodb_artist:v2:{artist_key}:{artist_mbid or ''}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, tuple) and len(cached) == 2:
        return cached

    thumbnail_bytes: bytes | None = None
    banner_bytes: bytes | None = None
    try:
        url = (
            f"https://www.theaudiodb.com/api/v1/json/123/artist-mb.php?i={artist_mbid}"
            if is_valid_uuid(artist_mbid)
            else f"https://www.theaudiodb.com/api/v1/json/123/search.php?s={urllib.parse.quote(artist_name)}"
        )
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
            matching_artists = [
                artist_record
                for artist_record in artists
                if normalize_str(
                    get_primary_artist(str(artist_record.get("strArtist") or ""))
                )
                == normalize_str(get_primary_artist(artist_name))
                and (
                    not is_valid_uuid(artist_mbid)
                    or artist_record.get("strMusicBrainzID") == artist_mbid
                )
            ]
            if len(matching_artists) == 1:
                artist_metadata = matching_artists[0]
                thumbnail_url = artist_metadata.get(
                    "strArtistThumb"
                ) or artist_metadata.get("strArtistFanart")
                banner_url = (
                    artist_metadata.get("strArtistBanner")
                    or artist_metadata.get("strArtistWideBanner")
                    or artist_metadata.get("strArtistFanart")
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


@coalesce_lookup
def fetch_theaudiodb_track_details(
    artist_name: str,
    track_title: str,
    *,
    recording_mbid: str | None = None,
    duration: float | None = None,
) -> dict[str, object] | None:
    """
    Fetch track details (video URL, mood, style, key, rating, description) from TheAudioDB.
    """
    if not artist_name or not track_title:
        return None
    cache_key = f"theaudiodb_track:v3:{normalize_str(artist_name)}:{normalize_str(track_title)}:{recording_mbid or ''}:{duration}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return cached or None
    if cached == "":
        return None

    try:
        url = f"https://www.theaudiodb.com/api/v1/json/123/searchtrack.php?s={urllib.parse.quote(artist_name)}&t={urllib.parse.quote(track_title)}"
        lookup_urls = (
            [
                f"https://www.theaudiodb.com/api/v1/json/123/track-mb.php?i={recording_mbid}"
            ]
            if is_valid_uuid(recording_mbid)
            else []
        ) + [url]
        all_responses_valid = True
        for lookup_url in lookup_urls:
            response = SESSION.get(lookup_url, timeout=6)
            if response.status_code != 200:
                all_responses_valid = False
                continue
            theaudiodb_payload = require_api_object(
                response.json(), "TheAudioDB track response"
            )
            raw_tracks = theaudiodb_payload["track"]
            tracks = (
                []
                if raw_tracks is None
                else require_api_records(raw_tracks, "TheAudioDB tracks")
            )
            candidates = [
                track
                for track in tracks
                if recording_identity_matches(
                    artist_name,
                    track_title,
                    str(track.get("strArtist") or ""),
                    str(track.get("strTrack") or ""),
                    expected_duration=duration,
                    candidate_duration=(safe_float(track.get("intDuration")) or 0)
                    / 1000
                    or None,
                )
                and (
                    not is_valid_uuid(recording_mbid)
                    or track.get("strMusicBrainzID") == recording_mbid
                    or (
                        "track-mb.php" not in lookup_url
                        and not track.get("strMusicBrainzID")
                    )
                )
            ]
            if len(candidates) > 1:
                LOG.warning(
                    f"Ambiguous TheAudioDB match for {escape(artist_name)} - {escape(track_title)}"
                )
                return None
            if candidates:
                raw_track = candidates[0]
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
        if all_responses_valid:
            set_cached_api(cache_key, {})
    except (httpx.HTTPError, OSError) as error:
        LOG.debug(
            f"TheAudioDB track lookup failed for {artist_name} - {track_title}: {error}"
        )
    return None
