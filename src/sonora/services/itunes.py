import re
from collections.abc import Mapping

import httpx
from rapidfuzz import fuzz
from rich.markup import escape

from sonora.core.api_payloads import require_api_object, require_api_records
from sonora.core.artists import get_primary_artist
from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.config import get_config
from sonora.core.coordination import coalesce_lookup
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.text import normalize_str, safe_float, safe_int
from sonora.core.titles import (
    extract_series_number,
    recording_identity_matches,
    score_artist_title_match,
)

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"


@coalesce_lookup
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
    target_primary_artist = get_primary_artist(artist)
    eligible_results: list[tuple[dict[str, object], str]] = []

    for result in results:
        collection_name = str(result.get("collectionName", ""))
        candidate_artist = str(result.get("artistName", ""))
        candidate_primary_artist = get_primary_artist(candidate_artist)
        if score_artist_title_match(
            artist, album, candidate_artist, collection_name
        ) <= 0.0 and (
            not target_primary_artist
            or score_artist_title_match(
                target_primary_artist, album, candidate_primary_artist, collection_name
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

            # Numbered releases in a series may share the same base title.
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


@coalesce_lookup
def lookup_itunes_track(track_id: str) -> dict[str, object] | None:
    """Retrieve the requested recording without treating its stored ID as validated."""
    if not track_id.isdigit():
        return None
    lookup_key = f"itunes_lookup:v1:{track_id}"
    payload = get_cached_api(lookup_key)
    if payload is None:
        response = SESSION.get(
            "https://itunes.apple.com/lookup",
            params={"id": track_id, "entity": "song"},
            timeout=10,
        )
        response.raise_for_status()
        payload = require_api_object(response.json(), "iTunes recording lookup")
        set_cached_api(lookup_key, payload)
    payload = require_api_object(payload, "Cached iTunes recording lookup")
    candidates = [
        candidate
        for candidate in require_api_records(
            payload.get("results"), "iTunes recording lookup results"
        )
        if str(candidate.get("trackId")) == track_id
    ]
    return _parse_itunes_track(candidates[0]) if len(candidates) == 1 else None


def fetch_itunes_track_metadata(
    artist: str,
    title: str,
    *,
    track_id: str | None = None,
    duration: float | None = None,
) -> dict[str, object] | None:
    """
    Fetch comprehensive track metadata from iTunes Search API.
    """
    if track_id:
        try:
            candidate = lookup_itunes_track(track_id)
        except (httpx.HTTPError, OSError) as error:
            LOG.warning(f"iTunes identifier lookup unavailable: {escape(str(error))}")
            return None
        if candidate and recording_identity_matches(
            artist,
            title,
            str(candidate.get("artist") or ""),
            str(candidate.get("title") or ""),
            expected_duration=duration,
            candidate_duration=safe_float(candidate.get("duration")),
        ):
            return candidate
    results = search_itunes(artist=artist, term=title, entity="song")

    candidates: dict[str, tuple[float, dict[str, object]]] = {}
    for result in results:
        track_name = str(result.get("trackName") or "")
        result_artist = str(result.get("artistName") or "")
        if not recording_identity_matches(
            artist,
            title,
            result_artist,
            track_name,
            expected_duration=duration,
            candidate_duration=(safe_float(result.get("trackTimeMillis")) or 0) / 1000
            or None,
        ):
            continue
        score = score_artist_title_match(artist, title, result_artist, track_name)
        candidate_id = str(result.get("trackId") or "")
        if candidate_id:
            candidates[candidate_id] = (score, result)
    ranked = sorted(
        candidates.values(), key=lambda candidate: candidate[0], reverse=True
    )
    if not ranked or (len(ranked) > 1 and ranked[0][0] == ranked[1][0]):
        if ranked:
            LOG.detail(
                f"Ambiguous iTunes recording for {escape(artist)} - {escape(title)}; metadata unchanged"
            )
        return None
    best_result = ranked[0][1]

    return _parse_itunes_track(best_result)


def _parse_itunes_track(track_entry: Mapping[str, object]) -> dict[str, object]:
    explicitness = str(track_entry.get("trackExplicitness", "")).lower()
    advisory = "Explicit" if explicitness == "explicit" else None

    release_date_raw = track_entry.get("releaseDate")
    release_date_str = str(release_date_raw)[:10] if release_date_raw else None

    return {
        "title": track_entry.get("trackName"),
        "artist": track_entry.get("artistName"),
        "duration": (safe_float(track_entry.get("trackTimeMillis")) or 0) / 1000
        or None,
        "genre": track_entry.get("primaryGenreName"),
        "advisory": advisory,
        "explicitness": explicitness,
        "copyright": track_entry.get("copyright"),
        "itunes_trackid": str(track_entry["trackId"])
        if track_entry.get("trackId")
        else None,
        "itunes_collectionid": str(track_entry["collectionId"])
        if track_entry.get("collectionId")
        else None,
        "itunes_artistid": str(track_entry["artistId"])
        if track_entry.get("artistId")
        else None,
        "track_number": track_entry.get("trackNumber"),
        "total_tracks": track_entry.get("trackCount"),
        "disc_number": track_entry.get("discNumber"),
        "total_discs": track_entry.get("discCount"),
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
    collection_title = str(album_dict.get("collectionName", ""))
    candidate_artist = str(album_dict.get("artistName", ""))
    score += score_artist_title_match(artist, album, candidate_artist, collection_title)
    if normalize_str(collection_title) == normalized_target:
        score += 30.0
    if str(album_dict.get("collectionExplicitness", "")).lower() == "explicit":
        score += 15.0
    candidate_track_count = safe_int(album_dict.get("trackCount"))
    if expected_track_count is not None and candidate_track_count is not None:
        track_count_difference = abs(candidate_track_count - expected_track_count)
        if track_count_difference == 0:
            score += 50.0
        elif track_count_difference <= 2:
            score += 25.0
        elif expected_track_count >= 3 and candidate_track_count == 1:
            score -= 60.0
    return score


@coalesce_lookup
def fetch_itunes_album_details(
    artist: str, album: str, expected_track_count: int | None = None
) -> dict[str, object] | None:
    """
    Fetch album details and index its recordings by title and disc position.
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

    ranked_albums = sorted(
        (
            (
                _score_itunes_album(
                    candidate, artist, album, normalized_target, expected_track_count
                ),
                candidate,
            )
            for candidate in {
                str(candidate.get("collectionId")): candidate
                for candidate in matching_albums
            }.values()
        ),
        key=lambda ranked: ranked[0],
        reverse=True,
    )
    if len(ranked_albums) > 1 and ranked_albums[0][0] == ranked_albums[1][0]:
        LOG.detail(f"Ambiguous iTunes album for {escape(artist)} - {escape(album)}")
        return None
    best_album = ranked_albums[0][1]

    collection_id = best_album.get("collectionId")
    if not collection_id:
        return None

    cache_key = f"itunes_album_lookup:v6:{collection_id}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return cached
    if cached == "":
        return None

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

        tracks_by_disc_and_position: dict[tuple[int, int], dict[str, object]] = {}
        tracks_by_title: dict[str, dict[str, object]] = {}
        repeated_titles: set[str] = set()
        album_meta: dict[str, object] = {
            "itunes_collectionid": str(collection_id),
            "itunes_artistid": str(best_album.get("artistId"))
            if best_album.get("artistId")
            else None,
            "genre": best_album.get("primaryGenreName"),
            "copyright": best_album.get("copyright"),
            "total_tracks": best_album.get("trackCount"),
            "tracks_by_disc_and_position": tracks_by_disc_and_position,
            "tracks_by_title": tracks_by_title,
        }

        for track_entry in raw_items:
            if track_entry.get("wrapperType") != "track":
                continue
            track_number = safe_int(track_entry.get("trackNumber"))
            disc_number = safe_int(track_entry.get("discNumber")) or 1
            track_title = str(track_entry.get("trackName", ""))
            track_metadata = _parse_itunes_track(track_entry)
            track_metadata["itunes_collectionid"] = str(collection_id)
            track_metadata["track_number"] = track_number
            track_metadata["disc_number"] = disc_number
            if track_number is not None:
                tracks_by_disc_and_position[(disc_number, track_number)] = (
                    track_metadata
                )
            if track_title:
                title_key = normalize_str(track_title)
                if title_key in tracks_by_title or title_key in repeated_titles:
                    repeated_titles.add(title_key)
                    tracks_by_title.pop(title_key, None)
                else:
                    tracks_by_title[title_key] = track_metadata

        set_cached_api(cache_key, album_meta)
        return album_meta
    except (httpx.HTTPError, OSError) as error:
        LOG.debug(f"iTunes album lookup failed for ID {collection_id}: {error}")
        return None
