import httpx

from sonora.core.api_payloads import require_api_object, require_api_records
from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.config import get_config
from sonora.core.coordination import coalesce_lookup
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.text import clean_disambiguation, normalize_str
from sonora.core.titles import clean_title, score_artist_title_match


def _normalize_genius_description(plain_description: object) -> str | None:
    if plain_description is None:
        return None
    if not isinstance(plain_description, str):
        raise TypeError("Unexpected Genius plain-text description payload")
    song_description = plain_description.strip()
    if (
        song_description in ("", "?")
        or "Lyrics for this song are unavailable" in song_description
    ):
        return None
    return song_description


@coalesce_lookup
def fetch_genius_song_details(
    artist: str, title: str, api_token: str | None = None
) -> dict[str, object] | None:
    if not api_token or not artist or not title:
        return None

    cleaned_title = clean_title(title)
    cache_key = f"genius_song:v2:{normalize_str(artist)}:{normalize_str(title)}"
    cached = get_cached_api(cache_key)
    if cached == "":
        return None
    if isinstance(cached, dict):
        cached["description"] = _normalize_genius_description(cached.get("description"))
        return cached

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
        genius_payload = require_api_object(response.json(), "Genius search response")
        search_results = require_api_object(
            genius_payload.get("response"), "Genius search envelope"
        )
        hits = require_api_records(search_results.get("hits"), "Genius search hits")
        if not hits:
            set_cached_api(cache_key, None)
            return None

        best_hit = None
        best_score = 0.0

        for hit in hits:
            song_candidate = require_api_object(hit.get("result"), "Genius search hit")
            hit_artist = str(song_candidate.get("primary_artist", {}).get("name", ""))
            hit_title = str(song_candidate.get("title", ""))

            score = score_artist_title_match(artist, title, hit_artist, hit_title)
            if score > best_score:
                best_score = score
                best_hit = song_candidate

        if not best_hit or best_score < get_config().genius_match_threshold:
            return None

        api_path = best_hit.get("api_path")
        if not api_path:
            return None

        song_url = f"https://api.genius.com{api_path}"
        song_response = SESSION.get(
            song_url,
            params={"text_format": "plain"},
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=5,
        )
        song_response.raise_for_status()
        genius_payload = require_api_object(
            song_response.json(), "Genius song response"
        )
        song_envelope = require_api_object(
            genius_payload.get("response"), "Genius song envelope"
        )
        song_genius_payload = require_api_object(
            song_envelope.get("song"), "Genius song"
        )
        primary_artist = require_api_object(
            song_genius_payload.get("primary_artist"), "Genius song primary artist"
        )
        if (
            score_artist_title_match(
                artist,
                title,
                str(primary_artist.get("name") or ""),
                str(song_genius_payload.get("title") or ""),
            )
            <= 0
        ):
            return None

        plain_description = _normalize_genius_description(
            require_api_object(
                song_genius_payload.get("description", {}), "Genius description"
            ).get("plain")
        )

        genius_song_id = song_genius_payload.get("id")

        featured_list = require_api_records(
            song_genius_payload.get("featured_artists", []), "Genius featured_artists"
        )
        featured_names = [
            clean_disambiguation(str(featured["name"]))
            for featured in featured_list
            if isinstance(featured, dict) and featured.get("name")
        ]

        producer_list = require_api_records(
            song_genius_payload.get("producer_artists", []), "Genius producer_artists"
        )
        producer_names = [
            clean_disambiguation(str(producer["name"]))
            for producer in producer_list
            if isinstance(producer, dict) and producer.get("name")
        ]

        writer_list = require_api_records(
            song_genius_payload.get("writer_artists", []), "Genius writer_artists"
        )
        writer_names = [
            clean_disambiguation(str(writer["name"]))
            for writer in writer_list
            if isinstance(writer, dict) and writer.get("name")
        ]

        release_date = song_genius_payload.get("release_date")

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

    except (httpx.HTTPError, OSError) as error:
        LOG.debug(f"Genius song details fetch failed for {artist} - {title}: {error}")
        return None
