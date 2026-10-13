import json
import re
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import ftfy
import httpx
from rapidfuzz import fuzz
from rich.markup import escape

from sonora.core.api_payloads import require_api_object, require_api_records
from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.config import (
    get_artist_split_pattern,
    get_balanced_feat_pattern,
    get_feat_tokens_pattern,
)
from sonora.core.constants import DIRS, FEAT_KEYWORDS
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.text import (
    _COLLAPSE_SPACES_PATTERN,
    clean_disambiguation,
    clean_unicode_punct,
    normalize_str,
    preserve_unicode_repertoire,
    safe_int,
    sanitize_name,
)


@lru_cache(maxsize=1)
def _load_user_overrides() -> dict[str, str]:
    candidate_paths = [
        DIRS.user_config_path / "aliases.json",
        Path.home() / ".config" / "sonora" / "aliases.json",
        Path("sonora_aliases.json"),
    ]
    seen_paths: set[Path] = set()
    overrides: dict[str, str] = {}
    for path in candidate_paths:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        try:
            if path.exists() and path.is_file() and path.stat().st_size > 0:
                alias_dict = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(alias_dict, dict):
                    for src, dest in alias_dict.items():
                        overrides[normalize_str(src)] = str(dest).strip()
        except (OSError, ValueError) as error:
            LOG.warning(
                f"Failed to load user aliases from {escape(str(path))}: {escape(str(error))}"
            )
    return overrides


@lru_cache(maxsize=4096)
def resolve_artist_name(raw_name: str | None, allow_network: bool = True) -> str:
    """
    Resolve legal names, aliases, or variations to canonical stage names.
    Returns the resolved canonical name or the cleaned input if not an alias.
    """
    if not raw_name or not str(raw_name).strip():
        return "Unknown Artist"

    clean_name = clean_unicode_punct(clean_disambiguation(str(raw_name).strip()))
    normalized = normalize_str(clean_name)
    if not normalized:
        return clean_unicode_punct(clean_name)

    user_overrides = _load_user_overrides()
    if normalized in user_overrides:
        return clean_unicode_punct(user_overrides[normalized])

    if not allow_network:
        return clean_unicode_punct(clean_name)

    cache_key = f"canonical_artist:{normalized}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, str) and cached:
        cached_words = cached.split()
        norm_words = normalized.split()
        is_corrupt_expansion = (
            len(norm_words) == 1
            and len(cached_words) > 1
            and normalized in [w.lower() for w in cached_words]
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
        from sonora.services.musicbrainz import search_musicbrainz_artists

        artists = search_musicbrainz_artists(
            query=f'artist:"{clean_name}" OR alias:"{clean_name}"', limit=5
        )
        for artist in artists:
            art_name = str(artist.get("name", "")).strip()
            if art_name == clean_name:
                clean_res = clean_unicode_punct(art_name)
                set_cached_api(cache_key, clean_res)
                return clean_res

        clean_has_punct = bool(re.search(r"[^\w\s]", clean_name))
        for artist in artists:
            art_name = str(artist.get("name", "")).strip()
            if not art_name:
                continue
            is_exact = art_name.lower() == clean_name.lower()
            is_norm = normalize_str(art_name) == normalized
            if is_exact or is_norm:
                cand_has_punct = bool(re.search(r"[^\w\s]", art_name))
                if not is_exact and not clean_has_punct and cand_has_punct:
                    continue
                is_short_acronym = (
                    len(clean_name.replace(".", "")) <= 3
                    or "." in clean_name
                    or bool(re.search(r"\d", clean_name))
                )
                if (
                    clean_name.istitle()
                    and art_name.isupper()
                    and len(clean_name.replace(".", "")) > 3
                ) or (
                    clean_name.isupper() and is_short_acronym and not art_name.isupper()
                ):
                    res_name = clean_name
                else:
                    res_name = art_name
                clean_res = clean_unicode_punct(res_name)
                set_cached_api(cache_key, clean_res)
                return clean_res
    except (httpx.HTTPError, OSError) as e:
        LOG.debug(f"MusicBrainz API error during artist resolution: {e}")

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
                d_name = str(deezer_entry.get("name", "")).strip()
                if d_name == clean_name:
                    clean_deezer = clean_unicode_punct(d_name)
                    set_cached_api(cache_key, clean_deezer)
                    return clean_deezer

            clean_has_punct = bool(re.search(r"[^\w\s]", clean_name))
            for deezer_entry in deezer_results[:10]:
                d_name = str(deezer_entry.get("name", "")).strip()
                if not d_name:
                    continue
                is_exact = d_name.lower() == clean_name.lower()
                is_norm = normalize_str(d_name) == normalized
                if is_exact or is_norm:
                    if (
                        not is_exact
                        and not clean_has_punct
                        and bool(re.search(r"[^\w\s]", d_name))
                    ):
                        continue
                    if (clean_name.istitle() and d_name.isupper()) or (
                        clean_name.isupper() and len(clean_name.replace(".", "")) <= 5
                    ):
                        res_name = clean_name
                    else:
                        res_name = d_name
                    clean_deezer = clean_unicode_punct(res_name)
                    set_cached_api(cache_key, clean_deezer)
                    return clean_deezer
    except (httpx.HTTPError, OSError) as e:
        LOG.debug(f"Deezer API error during artist resolution: {e}")

    clean_words = clean_name.split()
    for artist in artists:
        art_name = str(artist.get("name", "")).strip()
        if not art_name:
            continue
        art_words = art_name.split()

        # Guardrail: Single-word stage names must never expand into multi-word entities via aliases
        if len(clean_words) == 1 and len(art_words) > 1:
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
                clean_art = clean_unicode_punct(art_name)
                set_cached_api(cache_key, clean_art)
                return clean_art

    clean_final = clean_unicode_punct(clean_name)
    set_cached_api(cache_key, clean_final)
    return clean_final


@lru_cache(maxsize=4096)
def is_single_group_artist(raw_name: str | None, allow_network: bool = True) -> bool:
    """
    Determine if an artist name containing delimiters ('&', '+', ',') is a registered
    single band/group entity (e.g. 'Simon & Garfunkel', 'Earth, Wind & Fire', 'Play & Win')
    or a temporary collaboration (e.g. 'Drake & 21 Savage').
    """
    if not raw_name or not str(raw_name).strip():
        return False

    clean_name = str(raw_name).strip()
    normalized = normalize_str(clean_name)
    if not normalized:
        return False

    delimiters = ("&", "+", ",", "/")
    has_delimiter = any(delim in clean_name for delim in delimiters)
    has_conjunction = False
    if not has_delimiter:
        from sonora.core.config import get_config

        name_lower = f" {clean_name.lower()} "
        has_conjunction = any(
            f" {conj} " in name_lower for conj in get_config().featuring_conjunctions
        )
    if not (has_delimiter or has_conjunction):
        return False

    user_overrides = _load_user_overrides()
    if normalized in user_overrides:
        return True

    from sonora.core.config import get_config

    if any(normalize_str(g) == normalized for g in get_config().known_group_artists):
        return True

    if not allow_network:
        return False

    cache_key = f"is_group_entity:{normalized}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, bool):
        return cached

    query_name = re.sub(r"\s*([&+,/])\s*", r" \1 ", clean_name).strip()

    try:
        from sonora.services.musicbrainz import search_musicbrainz_artists

        artist_list = search_musicbrainz_artists(
            query=f'artist:"{query_name}"', limit=5
        )
        for artist in artist_list:
            name_match = normalize_str(artist.get("name")) == normalized
            score = safe_int(artist.get("ext:score")) or 0
            artist_type = artist.get("type")
            if name_match and (score >= 90 or artist_type == "Group"):
                canonical_name = clean_unicode_punct(
                    str(artist.get("name", "")).strip()
                )
                set_cached_api(f"canonical_artist:{normalized}", canonical_name)
                set_cached_api(cache_key, True)
                return True
        set_cached_api(cache_key, False)
        return False
    except (
        httpx.HTTPError,
        OSError,
    ):
        return False


def get_primary_artist(artist_name: str | None, allow_network: bool = False) -> str:
    """
    Extract primary artist from raw artist string by resolving aliases and stripping
    transient featured artists/delimiters, while preserving single group/band entities
    (e.g., 'Simon & Garfunkel', 'Play & Win', 'Earth, Wind & Fire').
    """
    if not artist_name:
        return "Unknown"

    raw_artist_name = str(artist_name).strip()
    formatted_artist_name = re.sub(r"\s*([&+,/])\s*", r" \1 ", raw_artist_name).strip()
    if is_single_group_artist(raw_artist_name, allow_network=allow_network):
        return sanitize_name(
            resolve_artist_name(formatted_artist_name, allow_network=allow_network)
        )

    parts = get_artist_split_pattern().split(raw_artist_name, maxsplit=1)
    primary = parts[0].strip()
    return sanitize_name(resolve_artist_name(primary, allow_network=allow_network))


def extract_balanced_features(
    text: str,
) -> tuple[str, list[str], str, str]:
    """
    Extract featuring artists and clean base title using balanced bracket scanning.
    Avoids regex truncation on nested parenthesized disambiguations (e.g. '(ROU)', '(Rapper)').
    Returns (cleaned_base_title, list_of_raw_feature_strings, open_char, close_char).
    """
    bracket_pairs = {"(": ")", "[": "]", "{": "}"}
    matches: list[str] = []
    i = 0
    n = len(text)
    base_parts: list[str] = []
    last_end = 0
    open_char = "("
    close_char = ")"

    while i < n:
        c = text[i]
        if c in bracket_pairs:
            start = i
            target_close = bracket_pairs[c]
            depth = 1
            i += 1
            while i < n and depth > 0:
                if text[i] == c:
                    depth += 1
                elif text[i] == target_close:
                    depth -= 1
                i += 1
            if depth == 0:
                end = i
                inner = text[start + 1 : end - 1].strip()
                feat_m = get_balanced_feat_pattern().match(inner)
                if feat_m:
                    base_parts.append(text[last_end:start])
                    matches.append(feat_m.group(1).strip())
                    if c == "[":
                        open_char, close_char = "[", "]"
                    last_end = end
        else:
            i += 1

    base_parts.append(text[last_end:])
    base_title = _COLLAPSE_SPACES_PATTERN.sub(" ", "".join(base_parts)).strip()
    return base_title, matches, open_char, close_char


def extract_featured_artist_tokens(
    featured_artists_input: str | Sequence[str | None] | None,
    primary_artist: str | None = None,
    allow_network: bool = False,
) -> list[str]:
    """
    Extract, normalize and deduplicate featured artists into a canonical list of individual artist names.
    - Strips composite conjunction strings (e.g. 'A & B' alongside 'A, B').
    - Handles multilingual conjunctions ('feat', 'ft', 'with', 'w/', 'w.', 'and', 'si', 'și', '&').
    - Preserves single registered group/band entities containing delimiters (e.g. 'Vargas & Lagola', 'Play & Win').
    - Reconciles Unicode diacritics / canonical repertoire (e.g. 'DJ Flamă' over 'Dj Flama').
    - Strips self-featured occurrences matching the primary track artist.
    """
    if not featured_artists_input:
        return []

    raw_items: list[str] = (
        [featured_artists_input]
        if isinstance(featured_artists_input, str)
        else [str(raw_entry) for raw_entry in featured_artists_input if raw_entry]
    )
    if not raw_items:
        return []

    tokens: list[str] = []
    feat_pattern = get_feat_tokens_pattern()
    for raw_chunk in raw_items:
        cleaned_chunk = clean_unicode_punct(raw_chunk).strip()
        if not cleaned_chunk:
            continue
        if is_single_group_artist(cleaned_chunk, allow_network=allow_network):
            tokens.append(cleaned_chunk)
            continue

        comma_parts = re.split(r"[,;\n]+|\s+/\s+", cleaned_chunk)
        for part in comma_parts:
            part_strip = part.strip()
            if not part_strip:
                continue
            if is_single_group_artist(part_strip, allow_network=allow_network):
                tokens.append(part_strip)
            else:
                for subpart in feat_pattern.split(part_strip):
                    subpart_clean = subpart.strip()
                    if subpart_clean:
                        tokens.append(subpart_clean)

    unique_artists: list[str] = []
    norm_to_index: dict[str, int] = {}
    primary_norm = normalize_str(primary_artist) if primary_artist else None
    user_overrides = _load_user_overrides()

    for tok in tokens:
        clean_tok = clean_disambiguation(tok)
        clean_tok = re.sub(r"[\(\)\[\]\{\}]", "", clean_tok).strip()
        clean_tok = re.sub(
            r"^(?:fea?t(?:uring)?|ft|with|w/(?!\s*[oO](?:ut)?\b)|w\.|and|si|și|cu)\.?\s+",
            "",
            clean_tok,
            flags=re.IGNORECASE,
        ).strip()
        if not clean_tok:
            continue

        norm = normalize_str(clean_tok)
        if norm in user_overrides:
            clean_tok = user_overrides[norm]
            norm = normalize_str(clean_tok)
        if (
            not norm
            or norm
            in (
                "unknown",
                "unknown artist",
                "various",
                "various artists",
                "untitled",
                "none",
                "null",
            )
            or clean_tok.lower()
            in (
                "unknown",
                "unknown artist",
                "various",
                "various artists",
                "untitled",
                "none",
                "null",
            )
        ):
            continue

        if primary_norm and (
            norm == primary_norm or fuzz.ratio(norm, primary_norm) >= 88
        ):
            continue

        if norm in norm_to_index:
            existing_idx = norm_to_index[norm]
            unique_artists[existing_idx] = preserve_unicode_repertoire(
                unique_artists[existing_idx], clean_tok
            )
            continue

        fuzzy_matched = False
        for existing_norm, idx in norm_to_index.items():
            if fuzz.ratio(norm, existing_norm) >= 88:
                unique_artists[idx] = preserve_unicode_repertoire(
                    unique_artists[idx], clean_tok
                )
                fuzzy_matched = True
                break

        if not fuzzy_matched:
            norm_to_index[norm] = len(unique_artists)
            unique_artists.append(clean_tok)

    registered_groups = [
        artist_token
        for artist_token in unique_artists
        if is_single_group_artist(artist_token, allow_network=allow_network)
    ]
    if registered_groups:
        feat_token_pattern = get_feat_tokens_pattern()
        filtered_unique_artists: list[str] = []
        for artist_token in unique_artists:
            is_subtoken = False
            token_norm = normalize_str(artist_token)
            for registered_group in registered_groups:
                if artist_token == registered_group:
                    continue
                group_subtokens = [
                    normalize_str(subtok)
                    for subtok in feat_token_pattern.split(registered_group)
                    if subtok.strip()
                ]
                if any(
                    token_norm == subtok_norm
                    or fuzz.ratio(token_norm, subtok_norm) >= 90
                    for subtok_norm in group_subtokens
                ):
                    is_subtoken = True
                    break
            if not is_subtoken:
                filtered_unique_artists.append(artist_token)
        unique_artists = filtered_unique_artists

    return unique_artists


def normalize_featured_artists(
    featured_artists_input: str | Sequence[str | None] | None,
    primary_artist: str | None = None,
    allow_network: bool = False,
) -> str | None:
    """
    Format normalized featured artists as a canonical comma-separated string,
    or None if no featured artists are present.
    """
    tokens = extract_featured_artist_tokens(
        featured_artists_input,
        primary_artist=primary_artist,
        allow_network=allow_network,
    )
    return ", ".join(tokens) if tokens else None


_FEAT_ARTIST_PATTERN = re.compile(rf"\s+(?:{FEAT_KEYWORDS})\.?\s*(.+)$", re.IGNORECASE)


def extract_artist_features(
    artist_name: str | None, allow_network: bool = False
) -> tuple[str, list[str]]:
    """
    Extract featuring artists embedded within an artist tag, returning (base_artist, featured_artists_list).
    (e.g., '21 Savage feat. Metro Boomin' -> ('21 Savage', ['Metro Boomin'])).
    Preserves registered collaborative and multi-artist bands (e.g. 'Vargas & Lagola').
    """
    if not artist_name:
        return ("", [])
    raw_artist = ftfy.fix_text(str(artist_name)).strip()
    if is_single_group_artist(raw_artist, allow_network=allow_network):
        return (raw_artist, [])

    feat_match = _FEAT_ARTIST_PATTERN.search(raw_artist)
    if not feat_match:
        return (raw_artist, [])

    base_artist = raw_artist[: feat_match.start()].strip()
    raw_featured = feat_match.group(1).strip()
    featured_tokens = extract_featured_artist_tokens(
        raw_featured, primary_artist=base_artist, allow_network=allow_network
    )
    return (base_artist, featured_tokens)


def is_artist_acronym(artist_name: str | None) -> bool:
    """
    Check if an artist string represents an acronym, initials, or alphanumeric band code
    (e.g., 'ABBA', 'M.G.L.', 'U2', 'DMX', '3OH!3') that should preserve uppercase branding.
    """
    if not artist_name:
        return False
    clean = artist_name.strip()
    return (
        len(clean.replace(".", "")) <= 3
        or "." in clean
        or any(char.isdigit() for char in clean)
    )


def harmonize_artist_casing(candidate_artist: str, reference_artist: str) -> str:
    """
    Reconcile two case variations of the same artist (whose normalized strings match).
    Demotes ALL-CAPS screaming to mixed/title case unless the artist is an acronym/initials.
    """
    if candidate_artist == reference_artist:
        return candidate_artist
    if normalize_str(candidate_artist) != normalize_str(reference_artist):
        return candidate_artist

    is_acronym = is_artist_acronym(candidate_artist)
    if candidate_artist.isupper() and not is_acronym and not reference_artist.isupper():
        return reference_artist
    if reference_artist.isupper() and not is_acronym and not candidate_artist.isupper():
        return candidate_artist
    return reference_artist
