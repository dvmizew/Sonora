from rapidfuzz import fuzz

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.http import SESSION
from sonora.core.utils import RateLimiter, normalize_str

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
_ITUNES_LIMITER = RateLimiter(interval_seconds=3.2)


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
        target_has_numeric_part = any(
            word in normalized_target.split()
            for word in [
                "ii",
                "2",
                "two",
                "part 2",
                "pt 2",
                "pt. 2",
                "vol 2",
                "vol. 2",
                "iii",
                "3",
                "iv",
                "4",
            ]
        )
        for result in results:
            collection_name = str(result.get("collectionName", ""))
            normalized_collection = normalize_str(collection_name)

            # Strict similarity requirement: collection_name must be fuzzy similar to album title!
            if fuzz.token_set_ratio(normalized_target, normalized_collection) < 75.0:
                continue

            collection_has_numeric_part = any(
                word in normalized_collection.split()
                for word in [
                    "ii",
                    "2",
                    "two",
                    "part 2",
                    "pt 2",
                    "pt. 2",
                    "vol 2",
                    "vol. 2",
                    "iii",
                    "3",
                    "iv",
                    "4",
                ]
            )

            # Reject mismatch between album series (e.g. Savage Mode vs Savage Mode II)
            if collection_has_numeric_part != target_has_numeric_part:
                continue

            best_result = result
            break

    if best_result is not None:
        artwork_url = best_result.get("artworkUrl100")
        if isinstance(artwork_url, str):
            return artwork_url.replace("100x100bb", f"{resolution}x{resolution}bb")

    return None
