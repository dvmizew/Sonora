import urllib.parse
from typing import Any

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

        deezer_payload = response.json()
        items = (
            deezer_payload.get("data", []) if isinstance(deezer_payload, dict) else []
        )
        if not items:
            query2 = f"{artist} {clean_album}"
            url2 = f"https://api.deezer.com/search/album?q={urllib.parse.quote(query2)}"
            fallback_response = SESSION.get(url2, timeout=6)
            if fallback_response.status_code == 200:
                fallback_payload = fallback_response.json()
                items = (
                    fallback_payload.get("data", [])
                    if isinstance(fallback_payload, dict)
                    else []
                )

        best_cover_url = None
        best_score = 0.0

        for track_candidate in items:
            if not isinstance(track_candidate, dict):
                continue
            item_artist = str(track_candidate.get("artist", {}).get("name", ""))
            item_album = str(track_candidate.get("title", ""))
            cover_xl = str(track_candidate.get("cover_xl", "")) or str(
                track_candidate.get("cover_big", "")
            )

            if not cover_xl:
                continue

            score = match_score(artist, clean_album, item_artist, item_album)
            if score > best_score and score >= 70.0:
                best_score = score
                best_cover_url = cover_xl

        set_cached_api(cache_key, best_cover_url)
        return best_cover_url
    except (httpx.HTTPError, OSError) as error:
        LOG.debug(f"Deezer cover art lookup failed for {artist} - {album}: {error}")
        return None


def fetch_deezer_album_details(
    artist: str, album: str, expected_track_count: int | None = None
) -> dict[str, Any] | None:
    if not (artist and album):
        return None

    clean_album = clean_title(album)
    cache_suffix = f":{expected_track_count}" if expected_track_count else ""
    cache_key = f"deezer_meta:{normalize_str(artist)}:{normalize_str(clean_album)}{cache_suffix}"
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
        for track_candidate in items:
            if not isinstance(track_candidate, dict):
                continue
            item_title = str(track_candidate.get("title", ""))
            item_artist = str(track_candidate.get("artist", {}).get("name", ""))
            score = match_score(artist, clean_album, item_artist, item_title)
            if normalize_str(item_title) == normalize_str(clean_album):
                score += 50.0

            nb_tracks = safe_int(track_candidate.get("nb_tracks"))
            record_type = str(track_candidate.get("record_type", "")).lower()
            if expected_track_count is not None and nb_tracks is not None:
                diff = abs(nb_tracks - expected_track_count)
                if diff == 0:
                    score += 40.0
                elif diff <= 2:
                    score += 20.0
                elif expected_track_count >= 3 and nb_tracks == 1:
                    score -= 60.0
            if (
                expected_track_count is not None
                and expected_track_count >= 3
                and record_type == "single"
            ):
                score -= 40.0

            if score > best_score and score >= 60.0:
                best_score = score
                best_item = track_candidate

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

        album_deezer_payload = detail_response.json()
        if not isinstance(album_deezer_payload, dict) or album_deezer_payload.get(
            "error"
        ):
            return None
        genres = [
            genre_item["name"]
            for genre_item in album_deezer_payload.get("genres", {}).get("data", [])
            if isinstance(genre_item, dict) and "name" in genre_item
        ]

        tracks_by_position: dict[int, dict[str, object]] = {}
        tracks_by_title: dict[str, dict[str, object]] = {}
        tracks_deezer_payload = (
            album_deezer_payload.get("tracks", {}).get("data", [])
            if isinstance(album_deezer_payload.get("tracks"), dict)
            else []
        )

        for idx, track_candidate in enumerate(tracks_deezer_payload, start=1):
            if not isinstance(track_candidate, dict):
                continue
            raw_pos = track_candidate.get("track_position")
            pos = raw_pos if isinstance(raw_pos, int) else idx
            t_name = str(track_candidate.get("title", ""))
            artist_name = (
                track_candidate.get("artist", {}).get("name")
                if isinstance(track_candidate.get("artist"), dict)
                else None
            )
            explicit = bool(track_candidate.get("explicit_lyrics"))

            track_dict: dict[str, object] = {
                "id": track_candidate.get("id"),
                "title": t_name,
                "artist": artist_name,
                "track_position": pos,
                "disk_number": track_candidate.get("disk_number", 1),
                "isrc": track_candidate.get("isrc"),
                "bpm": track_candidate.get("bpm"),
                "gain": track_candidate.get("gain"),
                "explicit_lyrics": explicit,
                "release_date": album_deezer_payload.get("release_date"),
                "genre": genres[0] if genres else None,
            }
            tracks_by_position[pos] = track_dict
            if t_name:
                tracks_by_title[normalize_str(clean_title(t_name))] = track_dict
                tracks_by_title[normalize_str(t_name)] = track_dict

        result = {
            "title": album_deezer_payload.get("title"),
            "artist": (
                album_deezer_payload.get("artist", {}).get("name")
                if isinstance(album_deezer_payload.get("artist"), dict)
                else None
            ),
            "nb_tracks": album_deezer_payload.get("nb_tracks"),
            "label": album_deezer_payload.get("label"),
            "barcode": album_deezer_payload.get("upc"),
            "release_date": album_deezer_payload.get("release_date"),
            "explicit_lyrics": album_deezer_payload.get("explicit_lyrics"),
            "cover_url": album_deezer_payload.get("cover_xl")
            or album_deezer_payload.get("cover_big"),
            "genre": genres[0] if genres else None,
            "tracks_by_position": tracks_by_position,
            "tracks_by_title": tracks_by_title,
        }
        set_cached_api(cache_key, result)
        return result
    except (httpx.HTTPError, OSError) as error:
        LOG.debug(f"Deezer album details lookup failed for {artist} - {album}: {error}")
        return None


def _parse_deezer_track_payload(
    track_deezer_payload: dict[str, Any],
) -> dict[str, str | int | float | bool | None]:
    contributors = track_deezer_payload.get("contributors", [])
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
                    featured.append(str(contributor_name))
                elif "producer" in contributor_role:
                    producers.append(str(contributor_name))
                elif "composer" in contributor_role:
                    composers.append(str(contributor_name))
                elif (
                    "author" in contributor_role
                    or "lyricist" in contributor_role
                    or "writer" in contributor_role
                ):
                    lyricists.append(str(contributor_name))

    track_pos = safe_int(track_deezer_payload.get("track_position"))
    disk_num = safe_int(track_deezer_payload.get("disk_number"))
    artist_deezer_payload = track_deezer_payload.get("artist")
    artist_name = (
        str(artist_deezer_payload.get("name"))
        if isinstance(artist_deezer_payload, dict) and artist_deezer_payload.get("name")
        else None
    )
    album_deezer_payload = track_deezer_payload.get("album")
    album_title = (
        str(album_deezer_payload.get("title"))
        if isinstance(album_deezer_payload, dict) and album_deezer_payload.get("title")
        else None
    )

    return {
        "title": track_deezer_payload.get("title"),
        "artist": artist_name,
        "album": album_title,
        "isrc": track_deezer_payload.get("isrc"),
        "explicit_lyrics": track_deezer_payload.get("explicit_lyrics"),
        "featured_artists": ", ".join(dict.fromkeys(featured)) if featured else None,
        "producers": ", ".join(dict.fromkeys(producers)) if producers else None,
        "composer": ", ".join(dict.fromkeys(composers)) if composers else None,
        "lyricist": ", ".join(dict.fromkeys(lyricists)) if lyricists else None,
        "track_position": track_pos,
        "disk_number": disk_num,
        "release_date": track_deezer_payload.get("release_date"),
    }


def fetch_deezer_track_by_isrc(
    isrc: str,
) -> dict[str, str | int | float | bool | None] | None:
    clean_isrc = isrc.strip().upper()
    if not clean_isrc:
        return None

    cache_key = f"deezer_isrc:{clean_isrc}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return cached

    _DEEZER_LIMITER.wait()
    url = f"https://api.deezer.com/track/isrc:{urllib.parse.quote(clean_isrc)}"

    try:
        response = SESSION.get(url, timeout=6)
        if response.status_code != 200:
            set_cached_api(cache_key, None)
            return None

        track_deezer_payload = response.json()
        if (
            not isinstance(track_deezer_payload, dict)
            or track_deezer_payload.get("error")
            or not track_deezer_payload.get("id")
        ):
            set_cached_api(cache_key, None)
            return None

        result = _parse_deezer_track_payload(track_deezer_payload)
        set_cached_api(cache_key, result)
        return result
    except (httpx.HTTPError, OSError) as error:
        LOG.debug(f"Deezer track lookup by ISRC failed for {isrc}: {error}")
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
        for track_candidate in items:
            if not isinstance(track_candidate, dict):
                continue
            item_artist = str(track_candidate.get("artist", {}).get("name", ""))
            item_title = str(track_candidate.get("title", ""))
            score = match_score(artist, clean_track_title, item_artist, item_title)
            if score > best_score and score >= 65.0:
                best_score = score
                best_item = track_candidate

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

        track_deezer_payload = detail_response.json()
        if not isinstance(track_deezer_payload, dict) or track_deezer_payload.get(
            "error"
        ):
            return None

        result = _parse_deezer_track_payload(track_deezer_payload)
        set_cached_api(cache_key, result)
        return result
    except (httpx.HTTPError, OSError) as error:
        LOG.debug(f"Deezer track details lookup failed for {artist} - {title}: {error}")
        return None
