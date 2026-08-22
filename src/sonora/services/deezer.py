import urllib.parse

import httpx

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.utils import clean_title, match_score, normalize_str


def fetch_deezer_cover_art_url(artist: str, album: str) -> str | None:
    if not (artist and album):
        return None

    clean_alb = clean_title(album)
    cache_key = f"deezer_art:{normalize_str(artist)}:{normalize_str(clean_alb)}"
    cached = get_cached_api(cache_key)
    if cached is not None:
        return str(cached) if cached else None

    query = f'artist:"{artist}" album:"{clean_alb}"'
    url = f"https://api.deezer.com/search/album?q={urllib.parse.quote(query)}"

    try:
        resp = SESSION.get(url, timeout=6)
        if resp.status_code != 200:
            set_cached_api(cache_key, None)
            return None

        data = resp.json()
        items = data.get("data", []) if isinstance(data, dict) else []
        if not items:
            query2 = f"{artist} {clean_alb}"
            url2 = f"https://api.deezer.com/search/album?q={urllib.parse.quote(query2)}"
            resp2 = SESSION.get(url2, timeout=6)
            if resp2.status_code == 200:
                data2 = resp2.json()
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

            score = match_score(artist, clean_alb, item_artist, item_album)
            if score > best_score and score >= 70.0:
                best_score = score
                best_cover_url = cover_xl

        set_cached_api(cache_key, best_cover_url)
        return best_cover_url
    except (httpx.HTTPError, OSError, ValueError, KeyError, RuntimeError) as e:
        LOG.debug(f"Deezer cover art lookup failed for {artist} - {album}: {e}")
        return None


def fetch_deezer_album_details(artist: str, album: str) -> dict[str, str | int | bool | None] | None:
    if not (artist and album):
        return None

    clean_alb = clean_title(album)
    cache_key = f"deezer_meta:{normalize_str(artist)}:{normalize_str(clean_alb)}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return cached

    query = f'artist:"{artist}" album:"{clean_alb}"'
    url = f"https://api.deezer.com/search/album?q={urllib.parse.quote(query)}"

    try:
        resp = SESSION.get(url, timeout=6)
        if resp.status_code != 200:
            set_cached_api(cache_key, None)
            return None

        items = resp.json().get("data", []) if isinstance(resp.json(), dict) else []
        if not items:
            return None

        album_id = items[0].get("id")
        if not album_id:
            return None

        detail_resp = SESSION.get(f"https://api.deezer.com/album/{album_id}", timeout=6)
        if detail_resp.status_code != 200:
            return None

        alb_data = detail_resp.json()
        genres = [g["name"] for g in alb_data.get("genres", {}).get("data", []) if isinstance(g, dict) and "name" in g]

        result = {
            "label": alb_data.get("label"),
            "barcode": alb_data.get("upc"),
            "release_date": alb_data.get("release_date"),
            "explicit_lyrics": alb_data.get("explicit_lyrics"),
            "cover_url": alb_data.get("cover_xl") or alb_data.get("cover_big"),
            "genre": genres[0] if genres else None,
        }
        set_cached_api(cache_key, result)
        return result
    except (httpx.HTTPError, OSError, ValueError, KeyError, RuntimeError) as e:
        LOG.debug(f"Deezer album details lookup failed for {artist} - {album}: {e}")
        return None


def fetch_deezer_track_details(artist: str, title: str) -> dict[str, str | int | float | bool | None] | None:
    if not (artist and title):
        return None

    clean_t = clean_title(title)
    cache_key = f"deezer_track:{normalize_str(artist)}:{normalize_str(clean_t)}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return cached

    query = f"{artist} {clean_t}"
    url = f"https://api.deezer.com/search/track?q={urllib.parse.quote(query)}"

    try:
        resp = SESSION.get(url, timeout=6)
        if resp.status_code != 200:
            set_cached_api(cache_key, None)
            return None

        items = resp.json().get("data", []) if isinstance(resp.json(), dict) else []
        if not items:
            return None

        best_item = None
        best_score = 0.0
        for item in items:
            if not isinstance(item, dict):
                continue
            item_artist = str(item.get("artist", {}).get("name", ""))
            item_title = str(item.get("title", ""))
            score = match_score(artist, clean_t, item_artist, item_title)
            if score > best_score and score >= 65.0:
                best_score = score
                best_item = item

        if not best_item:
            set_cached_api(cache_key, None)
            return None

        track_id = best_item.get("id")
        if not track_id:
            return None

        detail_resp = SESSION.get(f"https://api.deezer.com/track/{track_id}", timeout=6)
        if detail_resp.status_code != 200:
            return None

        t_data = detail_resp.json()
        result = {
            "isrc": t_data.get("isrc"),
            "bpm": t_data.get("bpm"),
            "gain": t_data.get("gain"),
            "explicit_lyrics": t_data.get("explicit_lyrics"),
        }
        set_cached_api(cache_key, result)
        return result
    except (httpx.HTTPError, OSError, ValueError, KeyError, RuntimeError) as e:
        LOG.debug(f"Deezer track details lookup failed for {artist} - {title}: {e}")
        return None
