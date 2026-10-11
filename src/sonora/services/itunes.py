import re
from typing import Any

import httpx
from rapidfuzz import fuzz

from sonora.core.api_payloads import require_api_object, require_api_records
from sonora.core.artists import get_primary_artist
from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.config import get_config
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.text import normalize_country_name, normalize_str, safe_int
from sonora.core.titles import extract_series_number, score_artist_title_match

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"


def search_itunes(
    artist: str, term: str, entity: str = "album", country: str = "US"
) -> list[dict[str, object]]:
    """
    Search iTunes Search API for album or track metadata.
    """
    query_term = f"{artist} {term}".strip()
    cache_key = f"itunes:{normalize_str(artist)}:{normalize_str(term)}:{entity}:{country.upper()}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, list):
        return cached

    params = {
        "term": query_term,
        "entity": entity,
        "country": country,
        "limit": "5",
    }

    try:
        response = SESSION.get(ITUNES_SEARCH_URL, params=params, timeout=10)
        response.raise_for_status()
        itunes_payload = require_api_object(response.json(), "iTunes search response")
        results = require_api_records(
            itunes_payload.get("results"), "iTunes search results"
        )
        set_cached_api(cache_key, results)
        return results
    except (httpx.HTTPError, OSError) as error:
        LOG.debug(f"iTunes Search API request failed for {query_term}: {error}")
        return []


def fetch_itunes_cover_art_url(
    artist: str, album: str, resolution: int = 1400
) -> str | None:
    """
    Match album artwork and request a best-effort larger CDN variant.

    Apple's documented API supplies 60/100-pixel URLs. Larger size substitutions
    are a CDN convention; downloaded dimensions are verified by the artwork pipeline.
    """
    results = search_itunes(artist=artist, term=album, entity="album")
    if not results:
        primary_artist = get_primary_artist(artist)
        if primary_artist and primary_artist.lower() != artist.lower():
            results = search_itunes(artist=primary_artist, term=album, entity="album")
    if not results:
        return None

    normalized_target = normalize_str(album)
    best_result: dict[str, object] | None = None
    primary_orig = get_primary_artist(artist)
    eligible_results: list[tuple[dict[str, object], str]] = []

    for result in results:
        collection_name = str(result.get("collectionName", ""))
        candidate_artist = str(result.get("artistName", ""))
        primary_cand = get_primary_artist(candidate_artist)
        if score_artist_title_match(
            artist, album, candidate_artist, collection_name
        ) <= 0.0 and (
            not primary_orig
            or score_artist_title_match(
                primary_orig, album, primary_cand, collection_name
            )
            <= 0.0
        ):
            continue
        normalized_collection = normalize_str(collection_name)
        eligible_results.append((result, normalized_collection))
        if normalized_collection == normalized_target:
            best_result = result
            break

    if best_result is None:
        target_series = extract_series_number(normalized_target)
        for result, normalized_collection in eligible_results:
            if (
                fuzz.token_set_ratio(normalized_target, normalized_collection)
                < get_config().album_match_threshold
            ):
                continue

            collection_series = extract_series_number(normalized_collection)

            # Reject mismatch between album series (e.g. Savage Mode vs Savage Mode II, or Pt 1 vs Pt 2)
            if collection_series != target_series:
                continue

            best_result = result
            break

    if best_result is not None:
        artwork_url = best_result.get("artworkUrl100")
        if isinstance(artwork_url, str):
            if "100x100bb" in artwork_url:
                return artwork_url.replace("100x100bb", f"{resolution}x{resolution}bb")
            upgraded_url = re.sub(
                r"\b\d+x\d+(?:bb)?(?:-\d+)?\b",
                f"{resolution}x{resolution}bb",
                artwork_url,
            )
            return upgraded_url

    return None


def fetch_itunes_track_metadata(artist: str, title: str) -> dict[str, object] | None:
    """
    Fetch comprehensive track metadata from iTunes Search API.
    """
    results = search_itunes(artist=artist, term=title, entity="song")
    if not results:
        return None

    normalized_target = normalize_str(title)
    best_result: dict[str, object] | None = None

    for result in results:
        track_name = str(result.get("trackName", ""))
        result_artist = str(result.get("artistName", ""))
        if score_artist_title_match(artist, title, result_artist, track_name) <= 0.0:
            continue

        normalized_name = normalize_str(track_name)
        if (
            normalized_name == normalized_target
            or fuzz.token_set_ratio(normalized_target, normalized_name)
            >= get_config().album_token_match_threshold
        ):
            best_result = result
            break

    if best_result is None:
        return None

    explicitness = str(best_result.get("trackExplicitness", "")).lower()
    advisory = "Explicit" if explicitness == "explicit" else None

    release_date_raw = best_result.get("releaseDate")
    release_date_str = str(release_date_raw)[:10] if release_date_raw else None

    return {
        "genre": best_result.get("primaryGenreName"),
        "advisory": advisory,
        "copyright": best_result.get("copyright"),
        "itunes_trackid": str(best_result["trackId"])
        if best_result.get("trackId")
        else None,
        "itunes_collectionid": str(best_result["collectionId"])
        if best_result.get("collectionId")
        else None,
        "itunes_artistid": str(best_result["artistId"])
        if best_result.get("artistId")
        else None,
        "release_country": normalize_country_name(str(best_result["country"]))
        if best_result.get("country")
        else None,
        "track_number": best_result.get("trackNumber"),
        "total_tracks": best_result.get("trackCount"),
        "disc_number": best_result.get("discNumber"),
        "total_discs": best_result.get("discCount"),
        "date": release_date_str,
    }


def _score_itunes_album(
    album_dict: dict[str, object],
    artist: str,
    album: str,
    normalized_target: str,
    expected_track_count: int | None = None,
) -> float:
    score = 0.0
    coll_name = str(album_dict.get("collectionName", ""))
    cand_art = str(album_dict.get("artistName", ""))
    score += score_artist_title_match(artist, album, cand_art, coll_name)
    if normalize_str(coll_name) == normalized_target:
        score += 30.0
    if str(album_dict.get("collectionExplicitness", "")).lower() == "explicit":
        score += 15.0
    tc = safe_int(album_dict.get("trackCount"))
    if expected_track_count is not None and tc is not None:
        diff = abs(tc - expected_track_count)
        if diff == 0:
            score += 50.0
        elif diff <= 2:
            score += 25.0
        elif expected_track_count >= 3 and tc == 1:
            score -= 60.0
    return score


def fetch_itunes_album_details(
    artist: str, album: str, expected_track_count: int | None = None
) -> dict[str, object] | None:
    """
    Fetch album metadata and all track details in ONE single lookup from iTunes Search API.
    Returns mapping of album details and track items indexed by track number and title.
    """
    results = search_itunes(artist=artist, term=album, entity="album")
    if not results:
        return None

    normalized_target = normalize_str(album)
    matching_albums: list[dict[str, object]] = []

    for result in results:
        collection_name = str(result.get("collectionName", ""))
        candidate_artist = str(result.get("artistName", ""))
        if (
            score_artist_title_match(artist, album, candidate_artist, collection_name)
            <= 0.0
        ):
            continue
        normalized_name = normalize_str(collection_name)
        if (
            normalized_name == normalized_target
            or fuzz.token_set_ratio(normalized_target, normalized_name)
            >= get_config().album_token_match_threshold
        ):
            matching_albums.append(result)

    if not matching_albums:
        return None

    best_album = max(
        matching_albums,
        key=lambda cand: _score_itunes_album(
            cand, artist, album, normalized_target, expected_track_count
        ),
    )

    collection_id = best_album.get("collectionId")
    if not collection_id:
        return None

    cache_key = f"itunes_album_lookup:{collection_id}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return cached

    try:
        url = "https://itunes.apple.com/lookup"
        params: dict[str, str | int] = {
            "id": str(collection_id),
            "entity": "song",
            "limit": 200,
        }
        response = SESSION.get(url, params=params, timeout=10)
        response.raise_for_status()
        itunes_payload = require_api_object(response.json(), "iTunes lookup response")
        raw_items = require_api_records(
            itunes_payload.get("results"), "iTunes lookup results"
        )

        tracks_by_number: dict[Any, dict[str, object]] = {}
        tracks_by_disc_and_position: dict[tuple[int, int], dict[str, object]] = {}
        tracks_by_title: dict[str, dict[str, object]] = {}
        album_meta: dict[str, object] = {
            "itunes_collectionid": str(collection_id),
            "itunes_artistid": str(best_album.get("artistId"))
            if best_album.get("artistId")
            else None,
            "genre": best_album.get("primaryGenreName"),
            "copyright": best_album.get("copyright"),
            "release_country": normalize_country_name(str(best_album["country"]))
            if best_album.get("country")
            else None,
            "total_tracks": best_album.get("trackCount"),
            "tracks_by_number": tracks_by_number,
            "tracks_by_disc_and_position": tracks_by_disc_and_position,
            "tracks_by_title": tracks_by_title,
        }

        for track_entry in raw_items:
            if track_entry.get("wrapperType") != "track":
                continue
            track_number = safe_int(track_entry.get("trackNumber"))
            disc_num = safe_int(track_entry.get("discNumber")) or 1
            track_title = str(track_entry.get("trackName", ""))
            explicitness = str(track_entry.get("trackExplicitness", "")).lower()
            advisory = "Explicit" if explicitness == "explicit" else None
            r_date = (
                str(track_entry.get("releaseDate"))[:10]
                if track_entry.get("releaseDate")
                else None
            )

            t_info: dict[str, object] = {
                "title": track_title,
                "artist": track_entry.get("artistName"),
                "genre": track_entry.get("primaryGenreName"),
                "advisory": advisory,
                "itunes_trackid": str(track_entry["trackId"])
                if track_entry.get("trackId")
                else None,
                "itunes_collectionid": str(collection_id),
                "itunes_artistid": str(track_entry.get("artistId"))
                if track_entry.get("artistId")
                else None,
                "release_country": normalize_country_name(str(track_entry["country"]))
                if track_entry.get("country")
                else None,
                "track_number": track_number,
                "total_tracks": track_entry.get("trackCount"),
                "disc_number": disc_num,
                "date": r_date,
            }
            if track_number is not None:
                tracks_by_disc_and_position[(disc_num, track_number)] = t_info
                tracks_by_number[(disc_num, track_number)] = t_info
                tracks_by_number[f"{disc_num}-{track_number}"] = t_info
                if disc_num == 1 or track_number not in tracks_by_number:
                    tracks_by_number[track_number] = t_info
            if track_title:
                tracks_by_title[normalize_str(track_title)] = t_info

        set_cached_api(cache_key, album_meta)
        return album_meta
    except (httpx.HTTPError, OSError) as error:
        LOG.debug(f"iTunes album lookup failed for ID {collection_id}: {error}")
        return None
