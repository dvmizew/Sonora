import httpx

from sonora.core.api_payloads import require_api_object, require_api_records
from sonora.core.artists import extract_featured_artist_tokens
from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.config import get_config
from sonora.core.coordination import coalesce_lookup
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.text import clean_disambiguation, normalize_str
from sonora.core.titles import clean_title, score_artist_title_match


def score_genius_song_match(artist: str, title: str, song: dict[str, object]) -> float:
    """Validate title and complete collaboration credits regardless of billing order."""
    candidate_title = str(song.get("title") or "")
    if normalize_str(clean_title(title)) != normalize_str(clean_title(candidate_title)):
        return 0.0
    expected_artists = {
        normalize_str(name) for name in extract_featured_artist_tokens(artist)
    }
    candidate_artist = str(song.get("artist") or "")
    if len(expected_artists) > 1:
        credited_artists = {
            normalize_str(name)
            for name in extract_featured_artist_tokens(
                [candidate_artist, str(song.get("featured_artists") or "")]
            )
        }
        if expected_artists != credited_artists:
            return 0.0
        candidate_artist = artist
    return score_artist_title_match(artist, title, candidate_artist, candidate_title)


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


def _parse_genius_song(song_genius_payload: dict[str, object]) -> dict[str, object]:
    primary_artist = require_api_object(
        song_genius_payload.get("primary_artist"), "Genius song primary artist"
    )
    primary_artists = require_api_records(
        song_genius_payload.get("primary_artists", []), "Genius primary artists"
    ) or [primary_artist]
    credited_names = [
        clean_disambiguation(str(artist["name"]))
        for artist in primary_artists
        if artist.get("name")
    ]
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
        "artist": " & ".join(credited_names),
        "title": str(song_genius_payload.get("title") or ""),
        "genius_song_id": genius_song_id,
        "description": plain_description,
        "featured_artists": ", ".join(featured_names) if featured_names else None,
        "producers": ", ".join(producer_names) if producer_names else None,
        "writers": ", ".join(writer_names) if writer_names else None,
        "release_date": release_date,
    }
    return result


@coalesce_lookup
def lookup_genius_song_details(
    song_id: str, api_token: str | None = None
) -> dict[str, object] | None:
    """Fetch a specific Genius song without treating its stored ID as identity proof."""
    if (
        not api_token
        or not song_id.isascii()
        or not song_id.isdigit()
        or int(song_id) <= 0
    ):
        return None
    identifier = int(song_id)
    cache_key = f"genius_recording:v1:{identifier}"
    cached = get_cached_api(cache_key)
    if cached == "":
        return None
    if cached is not None:
        song_payload = require_api_object(cached, "Cached Genius song")
    else:
        try:
            response = SESSION.get(
                f"https://api.genius.com/songs/{identifier}",
                params={"text_format": "plain"},
                headers={"Authorization": f"Bearer {api_token}"},
                timeout=5,
            )
            response.raise_for_status()
            envelope = require_api_object(response.json(), "Genius song response")
            song_payload = require_api_object(
                require_api_object(
                    envelope.get("response"), "Genius song envelope"
                ).get("song"),
                "Genius song",
            )
        except (httpx.HTTPError, OSError) as error:
            LOG.debug(f"Genius song lookup failed for {identifier}: {error}")
            return None
    returned_id = song_payload.get("id")
    if (
        isinstance(returned_id, bool)
        or not isinstance(returned_id, int)
        or returned_id <= 0
    ):
        raise TypeError("Genius song id must be a positive integer")
    if returned_id != identifier:
        LOG.warning(
            f"Genius returned song {returned_id} for requested ID {identifier}; result withheld"
        )
        return None
    details = _parse_genius_song(song_payload)
    if cached is None:
        set_cached_api(cache_key, song_payload)
    return details


@coalesce_lookup
def fetch_genius_song_details(
    artist: str, title: str, api_token: str | None = None
) -> dict[str, object] | None:
    if not api_token or not artist or not title:
        return None

    cleaned_title = clean_title(title)
    normalized_title = normalize_str(cleaned_title)
    cache_key = f"genius_song:v6:{normalize_str(artist)}:{normalize_str(title)}"
    cached = get_cached_api(cache_key)
    if cached == "":
        return None
    if isinstance(cached, dict):
        cached_score = score_genius_song_match(artist, title, cached)
        if (
            cached_score <= 0
            or cached_score < get_config().genius_match_threshold
            or normalize_str(clean_title(str(cached.get("title") or "")))
            != normalized_title
        ):
            return None
        return {
            **cached,
            "description": _normalize_genius_description(cached.get("description")),
        }

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

        scored_hits: dict[str, tuple[float, dict[str, object]]] = {}

        for hit in hits:
            song_candidate = require_api_object(hit.get("result"), "Genius search hit")
            candidate_artist = require_api_object(
                song_candidate.get("primary_artist"), "Genius search primary artist"
            )
            hit_artist = str(candidate_artist.get("name") or "")
            hit_title = str(song_candidate.get("title", ""))

            expected_artists = {
                normalize_str(name) for name in extract_featured_artist_tokens(artist)
            }
            search_artist = (
                artist
                if len(expected_artists) > 1
                and normalize_str(hit_artist) in expected_artists
                else hit_artist
            )
            score = score_artist_title_match(artist, title, search_artist, hit_title)
            candidate_path = str(song_candidate.get("api_path") or "")
            if (
                candidate_path
                and score > 0
                and score >= get_config().genius_match_threshold
                and normalize_str(clean_title(hit_title)) == normalized_title
            ):
                previous_score = scored_hits.get(candidate_path, (0.0, {}))[0]
                if score > previous_score:
                    scored_hits[candidate_path] = (score, song_candidate)

        if not scored_hits:
            set_cached_api(cache_key, None)
            return None
        ranked_hits = sorted(
            scored_hits.values(), key=lambda ranked_hit: ranked_hit[0], reverse=True
        )
        if len(ranked_hits) > 1 and ranked_hits[0][0] == ranked_hits[1][0]:
            set_cached_api(cache_key, None)
            return None
        _, best_hit = ranked_hits[0]

        api_path = best_hit.get("api_path")
        if not isinstance(api_path, str) or not api_path.startswith("/songs/"):
            return None

        song_id = str(api_path).removeprefix("/songs/")
        result = lookup_genius_song_details(song_id, api_token)
        if result is None:
            return None
        detail_score = score_genius_song_match(artist, title, result)
        if (
            detail_score <= 0
            or detail_score < get_config().genius_match_threshold
            or normalize_str(clean_title(str(result["title"]))) != normalized_title
        ):
            return None

        set_cached_api(cache_key, result)
        return result

    except (httpx.HTTPError, OSError) as error:
        LOG.debug(f"Genius song details fetch failed for {artist} - {title}: {error}")
        return None
