"""Online artist identity resolution, isolated from deterministic metadata normalization."""

import re
from typing import Any

import httpx

from sonora.core.api_payloads import require_api_object, require_api_records
from sonora.core.artists import (
    _load_user_overrides,
    normalize_artist_name,
)
from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.coordination import coalesce_lookup
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.text import (
    clean_disambiguation,
    clean_unicode_punct,
    normalize_str,
)
from sonora.services import musicbrainz


@coalesce_lookup
def resolve_artist_name(raw_name: str | None) -> str:
    """Resolve canonical artist branding through validated provider names and aliases."""
    clean_name = normalize_artist_name(raw_name)
    if not raw_name or not str(raw_name).strip():
        return clean_name
    normalized = normalize_str(clean_name)
    if (
        not normalized
        or normalize_str(clean_disambiguation(raw_name)) in _load_user_overrides()
    ):
        return clean_name

    cache_key = f"canonical_artist:v2:{normalized}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, str) and cached:
        cached_words = cached.split()
        normalized_words = normalized.split()
        is_corrupt_expansion = (
            len(normalized_words) == 1
            and len(cached_words) > 1
            and normalized in [word.lower() for word in cached_words]
        )
        is_cached_acronym = (
            len(cached.replace(".", "")) <= 3
            or "." in cached
            or bool(re.search(r"\d", cached))
        )
        if is_corrupt_expansion or (
            clean_name.istitle() and cached.isupper() and not is_cached_acronym
        ):
            set_cached_api(cache_key, None)
        elif not (cached.isupper() and not is_cached_acronym):
            return clean_unicode_punct(cached)

    artists: list[dict[str, Any]] = []
    try:
        artists = musicbrainz.search_musicbrainz_artists(
            query=f'artist:"{clean_name}" OR alias:"{clean_name}"', limit=5
        )
        for artist in artists:
            candidate_artist = str(artist.get("name", "")).strip()
            if candidate_artist == clean_name:
                resolved_artist = clean_unicode_punct(candidate_artist)
                set_cached_api(cache_key, resolved_artist)
                return resolved_artist

        clean_has_punct = bool(re.search(r"[^\w\s]", clean_name))
        for artist in artists:
            candidate_artist = str(artist.get("name", "")).strip()
            if not candidate_artist:
                continue
            is_exact = candidate_artist.lower() == clean_name.lower()
            is_norm = normalize_str(candidate_artist) == normalized
            if is_exact or is_norm:
                cand_has_punct = bool(re.search(r"[^\w\s]", candidate_artist))
                if not is_exact and not clean_has_punct and cand_has_punct:
                    continue
                is_short_acronym = (
                    len(clean_name.replace(".", "")) <= 3
                    or "." in clean_name
                    or bool(re.search(r"\d", clean_name))
                )
                if (
                    clean_name.istitle()
                    and candidate_artist.isupper()
                    and len(clean_name.replace(".", "")) > 3
                ) or (
                    clean_name.isupper()
                    and is_short_acronym
                    and not candidate_artist.isupper()
                ):
                    resolved_name = clean_name
                else:
                    resolved_name = candidate_artist
                resolved_artist = clean_unicode_punct(resolved_name)
                set_cached_api(cache_key, resolved_artist)
                return resolved_artist
    except (httpx.HTTPError, OSError) as error:
        LOG.debug(f"MusicBrainz API error during artist resolution: {error}")

    try:
        response = SESSION.get(
            "https://api.deezer.com/search/artist",
            params={"q": clean_name},
            timeout=5,
        )
        if response.status_code == 200:
            deezer_payload = require_api_object(
                response.json(), "Deezer artist response"
            )
            deezer_results = (
                []
                if deezer_payload.get("error")
                else require_api_records(
                    deezer_payload.get("data"), "Deezer artist results"
                )
            )
            for deezer_entry in deezer_results[:10]:
                candidate_artist = str(deezer_entry.get("name", "")).strip()
                if candidate_artist == clean_name:
                    resolved_artist = clean_unicode_punct(candidate_artist)
                    set_cached_api(cache_key, resolved_artist)
                    return resolved_artist

            clean_has_punct = bool(re.search(r"[^\w\s]", clean_name))
            for deezer_entry in deezer_results[:10]:
                candidate_artist = str(deezer_entry.get("name", "")).strip()
                if not candidate_artist:
                    continue
                is_exact = candidate_artist.lower() == clean_name.lower()
                is_norm = normalize_str(candidate_artist) == normalized
                if is_exact or is_norm:
                    if (
                        not is_exact
                        and not clean_has_punct
                        and bool(re.search(r"[^\w\s]", candidate_artist))
                    ):
                        continue
                    if (clean_name.istitle() and candidate_artist.isupper()) or (
                        clean_name.isupper() and len(clean_name.replace(".", "")) <= 5
                    ):
                        resolved_name = clean_name
                    else:
                        resolved_name = candidate_artist
                    resolved_artist = clean_unicode_punct(resolved_name)
                    set_cached_api(cache_key, resolved_artist)
                    return resolved_artist
    except (httpx.HTTPError, OSError) as error:
        LOG.debug(f"Deezer API error during artist resolution: {error}")

    clean_words = clean_name.split()
    alias_candidates: dict[str, str] = {}
    for artist in artists:
        candidate_artist = str(artist.get("name", "")).strip()
        if not candidate_artist:
            continue
        candidate_words = candidate_artist.split()

        # Short stage names can also be aliases of unrelated, longer artist names.
        if len(clean_words) == 1 and len(candidate_words) > 1:
            continue
        for alias_item in artist.get("alias-list", []):
            if isinstance(alias_item, dict):
                alias_name = str(alias_item.get("alias") or "")
                alias_type = str(alias_item.get("type") or "").strip().lower()
            else:
                alias_name = str(alias_item)
                alias_type = ""

            if alias_type == "search hint" or (
                alias_type == "legal name" and len(clean_words) < 2
            ):
                continue

            if alias_name and (
                alias_name.lower() == clean_name.lower()
                or normalize_str(alias_name) == normalized
            ):
                canonical_artist = clean_unicode_punct(candidate_artist)
                alias_candidates.setdefault(
                    normalize_str(canonical_artist), canonical_artist
                )

    if len(alias_candidates) == 1:
        canonical_artist = next(iter(alias_candidates.values()))
        set_cached_api(cache_key, canonical_artist)
        return canonical_artist

    normalized_artist = clean_unicode_punct(clean_name)
    set_cached_api(cache_key, normalized_artist)
    return normalized_artist
