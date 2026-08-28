import httpx

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.constants import GENIUS_MATCH_THRESHOLD, RATE_LIMIT_GENIUS
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.utils import RateLimiter, clean_title, match_score, normalize_str

_GENIUS_LIMITER = RateLimiter(interval_seconds=RATE_LIMIT_GENIUS)


def fetch_genius_description(
    artist: str, title: str, api_token: str | None = None
) -> str | None:
    details = fetch_genius_song_details(artist, title, api_token)
    return (
        str(details["description"]) if details and details.get("description") else None
    )


def fetch_genius_song_details(
    artist: str, title: str, api_token: str | None = None
) -> dict[str, object] | None:
    if not api_token or not artist or not title:
        return None

    cleaned_title = clean_title(title)
    cache_key = f"genius_song:{normalize_str(artist)}:{normalize_str(cleaned_title)}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return cached

    _GENIUS_LIMITER.wait()
    try:
        query = f"{artist} {cleaned_title}"
        search_url = "https://api.genius.com/search"
        response = SESSION.get(
            search_url,
            params={"q": query},
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        hits = data.get("response", {}).get("hits", [])
        if not hits:
            return None

        best_hit = None
        best_score = 0.0

        for hit in hits:
            result_item = hit.get("result", {})
            hit_artist = str(result_item.get("primary_artist", {}).get("name", ""))
            hit_title = str(result_item.get("title", ""))

            score = match_score(artist, cleaned_title, hit_artist, hit_title)
            if score > best_score:
                best_score = score
                best_hit = result_item

        if not best_hit or best_score < GENIUS_MATCH_THRESHOLD:
            return None

        api_path = best_hit.get("api_path")
        if not api_path:
            return None

        _GENIUS_LIMITER.wait()
        song_url = f"https://api.genius.com{api_path}"
        song_response = SESSION.get(
            song_url,
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=5,
        )
        song_response.raise_for_status()
        song_data = song_response.json().get("response", {}).get("song", {})

        plain_description = song_data.get("description", {}).get("plain")
        if plain_description and "Lyrics for this song are unavailable" in str(
            plain_description
        ):
            plain_description = None

        genius_song_id = song_data.get("id")

        # Parse featured artists
        featured_list = song_data.get("featured_artists", [])
        featured_names = [
            str(featured["name"])
            for featured in featured_list
            if isinstance(featured, dict) and featured.get("name")
        ]

        # Parse producers
        producer_list = song_data.get("producer_artists", [])
        producer_names = [
            str(producer["name"])
            for producer in producer_list
            if isinstance(producer, dict) and producer.get("name")
        ]

        # Parse writers / composers
        writer_list = song_data.get("writer_artists", [])
        writer_names = [
            str(writer["name"])
            for writer in writer_list
            if isinstance(writer, dict) and writer.get("name")
        ]

        release_date = song_data.get("release_date")

        result = {
            "genius_song_id": genius_song_id,
            "description": plain_description,
            "featured_artists": ", ".join(featured_names) if featured_names else None,
            "producers": ", ".join(producer_names) if producer_names else None,
            "writers": ", ".join(writer_names) if writer_names else None,
            "release_date": release_date,
        }
        set_cached_api(cache_key, result)
        return result

    except (httpx.HTTPError, OSError, ValueError, KeyError) as error:
        LOG.debug(f"Genius song details fetch failed for {artist} - {title}: {error}")
        return None
