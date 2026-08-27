import urllib.parse

import httpx

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.constants import RATE_LIMIT_DEEZER
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.utils import RateLimiter, clean_title, match_score, normalize_str

_DEEZER_LIMITER = RateLimiter(interval_seconds=RATE_LIMIT_DEEZER)


def fetch_deezer_cover_art_url(artist: str, album: str) -> str | None:
    if not (artist and album):
        return None

    clean_album = clean_title(album)
    cache_key = f"deezer_art:{normalize_str(artist)}:{normalize_str(clean_album)}"
    cached = get_cached_api(cache_key)
    if cached is not None:
        return str(cached) if cached else None

    _DEEZER_LIMITER.wait()
    query = f'artist:"{artist}" album:"{clean_album}"'
    url = f"https://api.deezer.com/search/album?q={urllib.parse.quote(query)}"

    try:
        response = SESSION.get(url, timeout=6)
        if response.status_code != 200:
            set_cached_api(cache_key, None)
            return None

        data = response.json()
        items = data.get("data", []) if isinstance(data, dict) else []
        if not items:
            query2 = f"{artist} {clean_album}"
            url2 = f"https://api.deezer.com/search/album?q={urllib.parse.quote(query2)}"
            fallback_response = SESSION.get(url2, timeout=6)
            if fallback_response.status_code == 200:
                data2 = fallback_response.json()
                items = data2.get("data", []) if isinstance(data2, dict) else []

        best_cover_url = None
        best_score = 0.0

        for item in items:
            if not isinstance(item, dict):
                continue
            item_artist = str(item.get("artist", {}).get("name", ""))
            item_album = str(item.get("title", ""))
            cover_xl = str(item.get("cover_xl", "")) or str(item.get("cover_big", ""))

            if not cover_xl:
                continue

            score = match_score(artist, clean_album, item_artist, item_album)
            if score > best_score and score >= 70.0:
                best_score = score
                best_cover_url = cover_xl

            set_cached_api(cache_key, best_cover_url)
        return best_cover_url
    except (httpx.HTTPError, OSError, ValueError, KeyError, RuntimeError) as error:
        LOG.debug(f"Deezer cover art lookup failed for {artist} - {album}: {error}")
        return None


def fetch_deezer_album_details(
    artist: str, album: str
) -> dict[str, str | int | bool | None] | None:
    if not (artist and album):
        return None

    clean_album = clean_title(album)
    cache_key = f"deezer_meta:{normalize_str(artist)}:{normalize_str(clean_album)}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return cached

    _DEEZER_LIMITER.wait()
    query = f'artist:"{artist}" album:"{clean_album}"'
    url = f"https://api.deezer.com/search/album?q={urllib.parse.quote(query)}"

    try:
        response = SESSION.get(url, timeout=6)
        if response.status_code != 200:
            set_cached_api(cache_key, None)
            return None

        items = (
            response.json().get("data", []) if isinstance(response.json(), dict) else []
        )
        if not items:
            return None

        album_id = items[0].get("id")
        if not album_id:
            return None

        _DEEZER_LIMITER.wait()
        detail_response = SESSION.get(
            f"https://api.deezer.com/album/{album_id}", timeout=6
        )
        if detail_response.status_code != 200:
            return None

        album_data = detail_response.json()
        genres = [
            genre_item["name"]
            for genre_item in album_data.get("genres", {}).get("data", [])
            if isinstance(genre_item, dict) and "name" in genre_item
        ]

        result = {
            "label": album_data.get("label"),
            "barcode": album_data.get("upc"),
            "release_date": album_data.get("release_date"),
            "explicit_lyrics": album_data.get("explicit_lyrics"),
            "cover_url": album_data.get("cover_xl") or album_data.get("cover_big"),
            "genre": genres[0] if genres else None,
        }
        set_cached_api(cache_key, result)
        return result
    except (httpx.HTTPError, OSError, ValueError, KeyError, RuntimeError) as error:
        LOG.debug(f"Deezer album details lookup failed for {artist} - {album}: {error}")
        return None


def fetch_deezer_track_details(
    artist: str, title: str
) -> dict[str, str | int | float | bool | None] | None:
    if not (artist and title):
        return None

    clean_track_title = clean_title(title)
    cache_key = (
        f"deezer_track:{normalize_str(artist)}:{normalize_str(clean_track_title)}"
    )
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return cached

    _DEEZER_LIMITER.wait()
    query = f"{artist} {clean_track_title}"
    url = f"https://api.deezer.com/search/track?q={urllib.parse.quote(query)}"

    try:
        response = SESSION.get(url, timeout=6)
        if response.status_code != 200:
            set_cached_api(cache_key, None)
            return None

        items = (
            response.json().get("data", []) if isinstance(response.json(), dict) else []
        )
        if not items:
            return None

        best_item = None
        best_score = 0.0
        for item in items:
            if not isinstance(item, dict):
                continue
            item_artist = str(item.get("artist", {}).get("name", ""))
            item_title = str(item.get("title", ""))
            score = match_score(artist, clean_track_title, item_artist, item_title)
            if score > best_score and score >= 65.0:
                best_score = score
                best_item = item

        if not best_item:
            set_cached_api(cache_key, None)
            return None

        track_id = best_item.get("id")
        if not track_id:
            return None

        _DEEZER_LIMITER.wait()
        detail_response = SESSION.get(
            f"https://api.deezer.com/track/{track_id}", timeout=6
        )
        if detail_response.status_code != 200:
            return None

        track_data = detail_response.json()
        contributors = track_data.get("contributors", [])
        featured: list[str] = []
        producers: list[str] = []
        if isinstance(contributors, list):
            for c in contributors:
                if isinstance(c, dict):
                    c_name = c.get("name")
                    c_role = str(c.get("role", "")).lower()
                    if not c_name:
                        continue
                    if "featured" in c_role:
                        featured.append(c_name)
                    elif "producer" in c_role:
                        producers.append(c_name)

        result = {
            "isrc": track_data.get("isrc"),
            "bpm": track_data.get("bpm"),
            "gain": track_data.get("gain"),
            "explicit_lyrics": track_data.get("explicit_lyrics"),
            "featured_artists": ", ".join(dict.fromkeys(featured))
            if featured
            else None,
            "producers": ", ".join(dict.fromkeys(producers)) if producers else None,
            "track_position": track_data.get("track_position"),
            "disk_number": track_data.get("disk_number"),
            "release_date": track_data.get("release_date"),
        }
        set_cached_api(cache_key, result)
        return result
    except (httpx.HTTPError, OSError, ValueError, KeyError, RuntimeError) as error:
        LOG.debug(f"Deezer track details lookup failed for {artist} - {title}: {error}")
        return None
