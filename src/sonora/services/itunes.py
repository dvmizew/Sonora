from rapidfuzz import fuzz

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.constants import ALBUM_MATCH_THRESHOLD, RATE_LIMIT_ITUNES
from sonora.core.http import SESSION
from sonora.core.utils import RateLimiter, extract_series_number, normalize_str

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
_ITUNES_LIMITER = RateLimiter(interval_seconds=RATE_LIMIT_ITUNES)


def search_itunes(
    artist: str, term: str, entity: str = "album", country: str = "US"
) -> list[dict[str, object]]:
    """
    Search iTunes Search API for album or track metadata.
    """
    query_term = f"{artist} {term}".strip()
    cache_key = f"itunes:{normalize_str(artist)}:{normalize_str(term)}:{entity}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, list):
        return cached

    params = {
        "term": query_term,
        "entity": entity,
        "country": country,
        "limit": "5",
    }
    _ITUNES_LIMITER.wait()
    try:
        response = SESSION.get(ITUNES_SEARCH_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        raw_results = data.get("results", []) if isinstance(data, dict) else []
        results: list[dict[str, object]] = [
            item for item in raw_results if isinstance(item, dict)
        ]
        set_cached_api(cache_key, results)
        return results
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        raise RuntimeError(
            f"iTunes Search API request failed for {query_term}: {error}"
        ) from error


def fetch_itunes_cover_art_url(
    artist: str, album: str, resolution: int = 1400
) -> str | None:
    """
    Fetch high-resolution album cover art URL from iTunes.
    Resolutions can be 600, 1400, or 3000.
    Ensures album name matches target to avoid wrong album series artwork.
    """
    results = search_itunes(artist=artist, term=album, entity="album")
    if not results:
        return None

    normalized_target = normalize_str(album)
    best_result: dict[str, object] | None = None

    # Step 1: Look for exact normalized title match
    for result in results:
        collection_name = str(result.get("collectionName", ""))
        if normalize_str(collection_name) == normalized_target:
            best_result = result
            break

    # Step 2: Fallback matching if no exact match found
    if best_result is None:
        target_series = extract_series_number(normalized_target)
        for result in results:
            collection_name = str(result.get("collectionName", ""))
            normalized_collection = normalize_str(collection_name)

            # Strict similarity requirement: collection_name must be fuzzy similar to album title!
            if (
                fuzz.token_set_ratio(normalized_target, normalized_collection)
                < ALBUM_MATCH_THRESHOLD
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
            return artwork_url.replace("100x100bb", f"{resolution}x{resolution}bb")

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
        normalized_name = normalize_str(track_name)
        if (
            normalized_name == normalized_target
            or fuzz.token_set_ratio(normalized_target, normalized_name) >= 80.0
        ):
            best_result = result
            break

    if best_result is None:
        return None

    explicitness = str(best_result.get("trackExplicitness", "")).lower()
    advisory = (
        "Explicit"
        if explicitness == "explicit"
        else "Clean"
        if explicitness == "cleaned"
        else None
    )

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
        "release_country": best_result.get("country"),
        "track_number": best_result.get("trackNumber"),
        "total_tracks": best_result.get("trackCount"),
        "disc_number": best_result.get("discNumber"),
        "total_discs": best_result.get("discCount"),
        "date": release_date_str,
    }
