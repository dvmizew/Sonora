import urllib.parse
from typing import Any

import httpx
from rich.markup import escape

from sonora.core.api_payloads import require_api_object, require_api_records
from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.config import get_config
from sonora.core.coordination import coalesce_lookup
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.text import normalize_str, safe_float, safe_int
from sonora.core.titles import (
    clean_title,
    recording_identity_matches,
    score_artist_title_match,
)


def _read_explicit_lyrics(explicit_lyrics: object) -> bool | None:
    if explicit_lyrics is None or isinstance(explicit_lyrics, bool):
        return explicit_lyrics
    raise TypeError("Deezer explicit_lyrics must be a boolean or null")


@coalesce_lookup
def fetch_deezer_cover_art_url(artist: str, album: str) -> str | None:
    if not (artist and album):
        return None

    clean_album = clean_title(album)
    cache_key = f"deezer_art:{normalize_str(artist)}:{normalize_str(clean_album)}"
    cached = get_cached_api(cache_key)
    if cached is not None:
        return str(cached) if cached else None

    query = f'artist:"{artist}" album:"{clean_album}"'
    url = f"https://api.deezer.com/search/album?q={urllib.parse.quote(query)}"

    try:
        response = SESSION.get(url, timeout=6)
        if response.status_code != 200:
            if response.status_code == 404:
                set_cached_api(cache_key, None)
            return None

        deezer_payload = require_api_object(response.json(), "Deezer search response")
        if deezer_payload.get("error"):
            return None
        catalog_candidates = require_api_records(
            deezer_payload.get("data"), "Deezer search results"
        )
        if not catalog_candidates:
            fallback_query = f"{artist} {clean_album}"
            fallback_url = f"https://api.deezer.com/search/album?q={urllib.parse.quote(fallback_query)}"

            fallback_response = SESSION.get(fallback_url, timeout=6)
            if fallback_response.status_code == 200:
                fallback_payload = require_api_object(
                    fallback_response.json(), "Deezer fallback response"
                )
                if fallback_payload.get("error"):
                    return None
                catalog_candidates = require_api_records(
                    fallback_payload.get("data"), "Deezer fallback results"
                )

        best_cover_url = None
        best_score = 0.0

        for track_candidate in catalog_candidates:
            candidate_artist = str(track_candidate.get("artist", {}).get("name", ""))
            candidate_album = str(track_candidate.get("title", ""))
            cover_xl = str(track_candidate.get("cover_xl", "")) or str(
                track_candidate.get("cover_big", "")
            )

            if not cover_xl:
                continue

            score = score_artist_title_match(
                artist, clean_album, candidate_artist, candidate_album
            )
            if score > best_score:
                best_score = score
                best_cover_url = cover_xl

        set_cached_api(cache_key, best_cover_url)
        return best_cover_url
    except (httpx.HTTPError, OSError) as error:
        LOG.debug(f"Deezer cover art lookup failed for {artist} - {album}: {error}")
        return None


@coalesce_lookup
def fetch_deezer_album_details(
    artist: str, album: str, expected_track_count: int | None = None
) -> dict[str, Any] | None:
    if not (artist and album):
        return None

    clean_album = clean_title(album)
    cache_suffix = f":{expected_track_count}" if expected_track_count else ""
    cache_key = f"deezer_meta:v7:{normalize_str(artist)}:{normalize_str(clean_album)}{cache_suffix}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return cached
    if cached == "":
        return None

    query = f'artist:"{artist}" album:"{clean_album}"'
    url = f"https://api.deezer.com/search/album?q={urllib.parse.quote(query)}"

    try:
        response = SESSION.get(url, timeout=6)
        if response.status_code != 200:
            if response.status_code == 404:
                set_cached_api(cache_key, None)
            return None

        deezer_payload = require_api_object(response.json(), "Deezer search response")
        if deezer_payload.get("error"):
            return None
        catalog_candidates = require_api_records(
            deezer_payload.get("data"), "Deezer search results"
        )
        if not catalog_candidates:
            set_cached_api(cache_key, None)
            return None

        best_candidate = None
        best_score = 0.0
        tied_candidates = False
        for track_candidate in catalog_candidates:
            candidate_title = str(track_candidate.get("title", ""))
            candidate_artist = str(track_candidate.get("artist", {}).get("name", ""))
            score = score_artist_title_match(
                artist, clean_album, candidate_artist, candidate_title
            )
            if score <= 0.0:
                continue
            if normalize_str(candidate_title) == normalize_str(clean_album):
                score += 50.0

            candidate_track_count = safe_int(track_candidate.get("nb_tracks"))
            record_type = str(track_candidate.get("record_type", "")).lower()
            if expected_track_count is not None and candidate_track_count is not None:
                track_count_difference = abs(
                    candidate_track_count - expected_track_count
                )
                if track_count_difference == 0:
                    score += 40.0
                elif track_count_difference <= 2:
                    score += 20.0
                elif expected_track_count >= 3 and candidate_track_count == 1:
                    score -= 60.0
            if (
                expected_track_count is not None
                and expected_track_count >= 3
                and record_type == "single"
            ):
                score -= 40.0

            if (
                score > best_score
                and score >= get_config().album_candidate_rank_threshold
            ):
                best_score = score
                best_candidate = track_candidate
                tied_candidates = False
            elif (
                best_candidate is not None
                and score == best_score
                and track_candidate.get("id") != best_candidate.get("id")
            ):
                tied_candidates = True

        if tied_candidates:
            LOG.detail(f"Ambiguous Deezer album for {escape(artist)} - {escape(album)}")
            return None
        if not best_candidate:
            set_cached_api(cache_key, None)
            return None

        album_id = best_candidate.get("id")
        if not album_id:
            return None

        detail_response = SESSION.get(
            f"https://api.deezer.com/album/{album_id}", timeout=6
        )
        if detail_response.status_code != 200:
            return None

        album_deezer_payload = require_api_object(
            detail_response.json(), "Deezer album response"
        )
        if album_deezer_payload.get("error"):
            return None
        genres = [
            genre_item["name"]
            for genre_item in album_deezer_payload.get("genres", {}).get("data", [])
            if isinstance(genre_item, dict) and "name" in genre_item
        ]

        tracks_by_disc_and_position: dict[tuple[int, int], dict[str, object]] = {}
        tracks_by_title: dict[str, dict[str, object]] = {}
        repeated_titles: set[str] = set()
        tracks_page = require_api_object(
            album_deezer_payload.get("tracks", {}), "Deezer album tracks"
        )
        tracks_deezer_payload = list(
            require_api_records(
                tracks_page.get("data", []), "Deezer album track records"
            )
        )
        visited_pages: set[str] = set()
        while tracks_page.get("next"):
            page_url = urllib.parse.urlsplit(str(tracks_page["next"]))
            if page_url.hostname != "api.deezer.com" or page_url.scheme not in (
                "http",
                "https",
            ):
                raise ValueError("Unexpected Deezer pagination host")
            next_page = page_url._replace(scheme="https").geturl()
            if next_page in visited_pages:
                raise ValueError("Repeated Deezer pagination link")
            visited_pages.add(next_page)
            page_response = SESSION.get(next_page, timeout=6)
            page_response.raise_for_status()
            tracks_page = require_api_object(page_response.json(), "Deezer tracks page")
            tracks_deezer_payload.extend(
                require_api_records(
                    tracks_page.get("data"), "Deezer tracks page records"
                )
            )

        for track_index, track_candidate in enumerate(tracks_deezer_payload, start=1):
            raw_pos = track_candidate.get("track_position")
            track_position = raw_pos if isinstance(raw_pos, int) else track_index
            disc_number = safe_int(track_candidate.get("disk_number")) or 1
            track_title = str(track_candidate.get("title", ""))
            track_dict: dict[str, object] = dict(
                _parse_deezer_track_payload(track_candidate)
            )
            track_dict.update(
                {
                    "id": track_candidate.get("id"),
                    "track_position": track_position,
                    "disk_number": disc_number,
                    "bpm": track_candidate.get("bpm"),
                    "gain": track_candidate.get("gain"),
                    "release_date": album_deezer_payload.get("release_date"),
                    "genre": genres[0] if genres else None,
                }
            )
            tracks_by_disc_and_position[(disc_number, track_position)] = track_dict
            if track_title:
                for title_key in {
                    normalize_str(clean_title(track_title)),
                    normalize_str(track_title),
                }:
                    if title_key in tracks_by_title or title_key in repeated_titles:
                        repeated_titles.add(title_key)
                        tracks_by_title.pop(title_key, None)
                    else:
                        tracks_by_title[title_key] = track_dict

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
            "tracks_by_disc_and_position": tracks_by_disc_and_position,
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
    disc_number = safe_int(track_deezer_payload.get("disk_number"))
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
        "duration": safe_float(track_deezer_payload.get("duration")),
        "explicit_lyrics": _read_explicit_lyrics(
            track_deezer_payload.get("explicit_lyrics")
        ),
        "featured_artists": ", ".join(dict.fromkeys(featured)) if featured else None,
        "producers": ", ".join(dict.fromkeys(producers)) if producers else None,
        "composer": ", ".join(dict.fromkeys(composers)) if composers else None,
        "lyricist": ", ".join(dict.fromkeys(lyricists)) if lyricists else None,
        "track_position": track_pos,
        "disk_number": disc_number,
        "release_date": track_deezer_payload.get("release_date"),
    }


@coalesce_lookup
def fetch_deezer_track_by_isrc(
    isrc: str,
) -> dict[str, str | int | float | bool | None] | None:
    clean_isrc = isrc.strip().upper()
    if not clean_isrc:
        return None

    cache_key = f"deezer_isrc:v2:{clean_isrc}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return cached
    if cached == "":
        return None

    url = f"https://api.deezer.com/track/isrc:{urllib.parse.quote(clean_isrc)}"

    try:
        response = SESSION.get(url, timeout=6)
        if response.status_code != 200:
            if response.status_code == 404:
                set_cached_api(cache_key, None)
            return None

        track_deezer_payload = require_api_object(
            response.json(), "Deezer ISRC response"
        )
        if track_deezer_payload.get("error") or not track_deezer_payload.get("id"):
            error = track_deezer_payload.get("error")
            if isinstance(error, dict) and error.get("code") == 800:
                set_cached_api(cache_key, None)
            return None

        result = _parse_deezer_track_payload(track_deezer_payload)
        set_cached_api(cache_key, result)
        return result
    except (httpx.HTTPError, OSError) as error:
        LOG.debug(f"Deezer track lookup by ISRC failed for {isrc}: {error}")
        return None


@coalesce_lookup
def fetch_deezer_track_details(
    artist: str,
    title: str,
    *,
    duration: float | None = None,
) -> dict[str, str | int | float | bool | None] | None:
    if not (artist and title):
        return None

    clean_track_title = clean_title(title)
    cache_key = (
        f"deezer_track:v5:{normalize_str(artist)}:{normalize_str(clean_track_title)}"
        f":{duration}:{get_config().single_duration_tolerance_seconds}"
    )
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return cached
    if cached == "":
        return None

    query = f"{artist} {clean_track_title}"
    url = f"https://api.deezer.com/search/track?q={urllib.parse.quote(query)}"

    try:
        response = SESSION.get(url, timeout=6)
        if response.status_code != 200:
            if response.status_code == 404:
                set_cached_api(cache_key, None)
            return None

        deezer_payload = require_api_object(response.json(), "Deezer search response")
        if deezer_payload.get("error"):
            return None
        catalog_candidates = require_api_records(
            deezer_payload.get("data"), "Deezer search results"
        )
        if not catalog_candidates:
            set_cached_api(cache_key, None)
            return None

        best_candidate = None
        best_score = 0.0
        candidate_scores: dict[str, float] = {}
        for track_candidate in catalog_candidates:
            candidate_artist = str(track_candidate.get("artist", {}).get("name", ""))
            candidate_title = str(track_candidate.get("title", ""))
            score = score_artist_title_match(
                artist, clean_track_title, candidate_artist, candidate_title
            )
            if not recording_identity_matches(
                artist,
                title,
                candidate_artist,
                candidate_title,
                expected_duration=duration,
                candidate_duration=safe_float(track_candidate.get("duration")),
            ):
                continue
            candidate_scores[str(track_candidate.get("id") or "")] = score
            if score > best_score:
                best_score = score
                best_candidate = track_candidate

        if sum(score == best_score for score in candidate_scores.values()) > 1:
            LOG.detail(
                f"Ambiguous Deezer recording for {escape(artist)} - {escape(title)}"
            )
            return None
        if not best_candidate:
            set_cached_api(cache_key, None)
            return None

        track_id = best_candidate.get("id")
        if not track_id:
            return None

        detail_response = SESSION.get(
            f"https://api.deezer.com/track/{track_id}", timeout=6
        )
        if detail_response.status_code != 200:
            return None

        track_deezer_payload = require_api_object(
            detail_response.json(), "Deezer track response"
        )
        if track_deezer_payload.get("error"):
            return None

        result = _parse_deezer_track_payload(track_deezer_payload)
        if not recording_identity_matches(
            artist,
            title,
            str(result.get("artist") or ""),
            str(result.get("title") or ""),
            expected_duration=duration,
            candidate_duration=safe_float(result.get("duration")),
        ):
            return None
        set_cached_api(cache_key, result)
        return result
    except (httpx.HTTPError, OSError) as error:
        LOG.debug(f"Deezer track details lookup failed for {artist} - {title}: {error}")
        return None
