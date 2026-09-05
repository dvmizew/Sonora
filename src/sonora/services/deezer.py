import urllib.parse

import httpx

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.constants import RATE_LIMIT_DEEZER
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.utils import (
    RateLimiter,
    clean_title,
    match_score,
    normalize_str,
    safe_int,
)

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
            set_cached_api(cache_key, None)
            return None

        best_item = None
        best_score = 0.0
        for item in items:
            if not isinstance(item, dict):
                continue
            item_title = str(item.get("title", ""))
            item_artist = str(item.get("artist", {}).get("name", ""))
            score = match_score(artist, clean_album, item_artist, item_title)
            if normalize_str(item_title) == normalize_str(clean_album):
                score += 50.0
            if score > best_score and score >= 60.0:
                best_score = score
                best_item = item

        if not best_item and items and isinstance(items[0], dict):
            best_item = items[0]

        if not best_item:
            set_cached_api(cache_key, None)
            return None

        album_id = best_item.get("id")
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

        tracks_by_position: dict[int, dict[str, object]] = {}
        tracks_by_title: dict[str, dict[str, object]] = {}
        tracks_data = (
            album_data.get("tracks", {}).get("data", [])
            if isinstance(album_data.get("tracks"), dict)
            else []
        )

        for idx, item in enumerate(tracks_data, start=1):
            if not isinstance(item, dict):
                continue
            raw_pos = item.get("track_position")
            pos = raw_pos if isinstance(raw_pos, int) else idx
            t_name = str(item.get("title", ""))
            artist_name = (
                item.get("artist", {}).get("name")
                if isinstance(item.get("artist"), dict)
                else None
            )
            explicit = bool(item.get("explicit_lyrics"))

            track_dict: dict[str, object] = {
                "id": item.get("id"),
                "title": t_name,
                "artist": artist_name,
                "track_position": pos,
                "disk_number": item.get("disk_number", 1),
                "isrc": item.get("isrc"),
                "bpm": item.get("bpm"),
                "gain": item.get("gain"),
                "explicit_lyrics": explicit,
                "release_date": album_data.get("release_date"),
                "genre": genres[0] if genres else None,
            }
            tracks_by_position[pos] = track_dict
            if t_name:
                tracks_by_title[normalize_str(clean_title(t_name))] = track_dict
                tracks_by_title[normalize_str(t_name)] = track_dict

        result = {
            "title": album_data.get("title"),
            "artist": (
                album_data.get("artist", {}).get("name")
                if isinstance(album_data.get("artist"), dict)
                else None
            ),
            "nb_tracks": album_data.get("nb_tracks"),
            "label": album_data.get("label"),
            "barcode": album_data.get("upc"),
            "release_date": album_data.get("release_date"),
            "explicit_lyrics": album_data.get("explicit_lyrics"),
            "cover_url": album_data.get("cover_xl") or album_data.get("cover_big"),
            "genre": genres[0] if genres else None,
            "tracks_by_position": tracks_by_position,
            "tracks_by_title": tracks_by_title,
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
        composers: list[str] = []
        lyricists: list[str] = []
        if isinstance(contributors, list):
            for contributor in contributors:
                if isinstance(contributor, dict):
                    contributor_name = contributor.get("name")
                    contributor_role = str(contributor.get("role", "")).lower()
                    if not contributor_name:
                        continue
                    if "featured" in contributor_role:
                        featured.append(contributor_name)
                    elif "producer" in contributor_role:
                        producers.append(contributor_name)
                    elif "composer" in contributor_role:
                        composers.append(contributor_name)
                    elif (
                        "author" in contributor_role
                        or "lyricist" in contributor_role
                        or "writer" in contributor_role
                    ):
                        lyricists.append(contributor_name)

        track_pos = safe_int(track_data.get("track_position"))
        disk_num = safe_int(track_data.get("disk_number"))

        result = {
            "isrc": track_data.get("isrc"),
            "explicit_lyrics": track_data.get("explicit_lyrics"),
            "featured_artists": ", ".join(dict.fromkeys(featured))
            if featured
            else None,
            "producers": ", ".join(dict.fromkeys(producers)) if producers else None,
            "composer": ", ".join(dict.fromkeys(composers)) if composers else None,
            "lyricist": ", ".join(dict.fromkeys(lyricists)) if lyricists else None,
            "track_position": track_pos,
            "disk_number": disk_num,
            "release_date": track_data.get("release_date"),
        }
        set_cached_api(cache_key, result)
        return result
    except (httpx.HTTPError, OSError, ValueError, KeyError, RuntimeError) as error:
        LOG.debug(f"Deezer track details lookup failed for {artist} - {title}: {error}")
        return None
