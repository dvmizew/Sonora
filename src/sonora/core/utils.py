import json
import os
import re
import threading
import time
import unicodedata
import uuid
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeGuard

import anyascii
import ftfy
import httpx
import pycountry
from music_metadata_filter.filter import MetadataFilter
from music_metadata_filter.functions import (
    fix_track_suffix,
    remove_clean_explicit,
    remove_feature,
    remove_parody,
    remove_reissue,
    remove_remastered,
    remove_zero_width,
    replace_nbsp,
)
from pathvalidate import sanitize_filename
from rapidfuzz import fuzz

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.config import (
    get_artist_split_pattern,
    get_balanced_feat_pattern,
    get_bracket_feat_pattern,
    get_disambiguation_pattern,
    get_duplicate_feat_pattern,
    get_feat_tokens_pattern,
)
from sonora.core.constants import (
    COMPANION_LYRICS_EXTS,
    DIRS,
    SUPPORTED_EXTS,
)
from sonora.core.http import SESSION
from sonora.core.logger import LOG

_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50}


def _parse_roman_numeral(token: str) -> int | None:
    total, prev = 0, 0
    for char in reversed(token):
        val = _ROMAN_VALUES.get(char)
        if val is None:
            return None
        total += val if val >= prev else -val
        prev = val
    return total if 1 <= total <= 50 else None


_WORD_NUMBERS = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
]
_WORD_NUMBER_MAP: dict[str, int] = {w: i for i, w in enumerate(_WORD_NUMBERS, 1)}


class InterruptedOperationError(KeyboardInterrupt):
    """Raised when an operation is cancelled via SIGINT / KeyboardInterrupt, carrying partial progress."""

    def __init__(self, partial_result: Any = None) -> None:
        super().__init__()
        self.partial_result = partial_result


def extract_series_number(text: str | None) -> int | None:
    """
    Extract album or track series/volume number (e.g. 'Savage Mode II' -> 2, 'Pt. 2' -> 2, 'Vol. 3' -> 3).
    Returns integer series number or None if not part of a numbered series.
    """
    if not text:
        return None

    clean_text = ftfy.fix_text(str(text)).strip().lower()

    match = re.search(
        r"\b(?:vol(?:ume)?|pt|part|chapter|act|book)\.?\s*(\d{1,2}|[a-z]+)\b",
        clean_text,
        re.IGNORECASE,
    ) or re.search(r"\b(\d{1,2}|[a-z]+)\s*$", clean_text, re.IGNORECASE)

    if match:
        token = match.group(1).lower()
        if token.isdigit():
            return int(token)
        return _parse_roman_numeral(token) or _WORD_NUMBER_MAP.get(token)

    return None


def safe_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    val_str = str(value).split("/")[0].strip()
    return int(val_str) if val_str.isdigit() else None


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    val_str = str(value).replace(" dB", "").strip()
    try:
        return float(val_str)
    except ValueError:
        return None


_UNICODE_HYPHENS_PATTERN = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015]")
_ZERO_WIDTH_PATTERN = re.compile(r"[\u200B\u200C\u200D\uFEFF]")
_SPACES_BEFORE_COMMA_PATTERN = re.compile(r"\s+,")


def clean_unicode_punct(text: str | None) -> str:
    if not text:
        return ""
    cleaned = _ZERO_WIDTH_PATTERN.sub("", str(text))
    cleaned = _UNICODE_HYPHENS_PATTERN.sub("-", cleaned)
    return _SPACES_BEFORE_COMMA_PATTERN.sub(",", cleaned)


_COLLAPSE_SPACES_PATTERN = re.compile(r"\s+")


@lru_cache(maxsize=8192)
def clean_disambiguation(name: str | None) -> str:
    """
    Strips disambiguation country codes or numeric suffixes anywhere in an artist string
    (e.g., 'Armin (ROU)' -> 'Armin', 'Rafoo, Armin (ROU), ASSAF (ROU)' -> 'Rafoo, Armin, ASSAF', 'Jony (10)' -> 'Jony').
    """
    if not name:
        return ""
    cleaned = get_disambiguation_pattern().sub("", str(name)).strip()
    return _COLLAPSE_SPACES_PATTERN.sub(" ", cleaned)


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
            LOG.warning(f"Failed to load user aliases from {path}: {error}")
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

    # Tier 1: User custom config overrides (~/.config/sonora/aliases.json)
    user_overrides = _load_user_overrides()
    if normalized in user_overrides:
        return clean_unicode_punct(user_overrides[normalized])

    # Tier 2: Persistent DiskCache
    cache_key = f"canonical_artist:{normalized}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, str):
        return clean_unicode_punct(cached)

    if not allow_network:
        return clean_unicode_punct(clean_name)

    # Tier 3: MusicBrainz Alias / Legal Name lookup
    try:
        from sonora.services.musicbrainz import search_musicbrainz_artists

        artists = search_musicbrainz_artists(
            query=f'artist:"{clean_name}" OR alias:"{clean_name}"', limit=5
        )

        # Priority 1: Exact case-insensitive name match
        for artist in artists:
            art_name = str(artist.get("name", "")).strip()
            if not art_name:
                continue
            if art_name.lower() == clean_name.lower():
                if (
                    clean_name.isupper()
                    and len(clean_name.replace(".", "")) <= 5
                    and not art_name.isupper()
                ):
                    res_name = clean_name
                else:
                    res_name = art_name
                clean_res = clean_unicode_punct(res_name)
                set_cached_api(cache_key, clean_res)
                return clean_res

        # Priority 2: Normalized exact match (ignoring punctuation/diacritics)
        clean_has_punct = bool(re.search(r"[^\w\s]", clean_name))
        for artist in artists:
            art_name = str(artist.get("name", "")).strip()
            if not art_name:
                continue
            if normalize_str(art_name) == normalized:
                cand_has_punct = bool(re.search(r"[^\w\s]", art_name))
                if not clean_has_punct and cand_has_punct:
                    continue
                clean_art = clean_unicode_punct(art_name)
                set_cached_api(cache_key, clean_art)
                return clean_art

        # Priority 3: Exact alias match
        for artist in artists:
            art_name = str(artist.get("name", "")).strip()
            if not art_name:
                continue
            for alias_item in artist.get("alias-list", []):
                alias_name = (
                    alias_item.get("alias")
                    if isinstance(alias_item, dict)
                    else str(alias_item)
                )
                if alias_name and alias_name.lower() == clean_name.lower():
                    clean_art = clean_unicode_punct(art_name)
                    set_cached_api(cache_key, clean_art)
                    return clean_art
                if alias_name and normalize_str(alias_name) == normalized:
                    clean_art = clean_unicode_punct(art_name)
                    set_cached_api(cache_key, clean_art)
                    return clean_art
    except (httpx.HTTPError, OSError) as e:
        LOG.debug(f"MusicBrainz API error during artist resolution: {e}")

    # Tier 4: Deezer Artist lookup
    try:
        response = SESSION.get(
            "https://api.deezer.com/search/artist",
            params={"q": clean_name},
            timeout=5,
        )
        if response.status_code == 200:
            deezer_results = response.json().get("data", [])
            if deezer_results and isinstance(deezer_results, list):
                deezer_name = str(deezer_results[0].get("name", "")).strip()
                if deezer_name and normalize_str(deezer_name) == normalized:
                    # Do not override all-caps acronyms (e.g. M.G.L) with lowercased titles
                    if (
                        clean_name.isupper()
                        and not deezer_name.isupper()
                        and len(clean_name.replace(".", "")) <= 5
                    ):
                        deezer_name = clean_name
                    clean_deezer = clean_unicode_punct(deezer_name)
                    set_cached_api(cache_key, clean_deezer)
                    return clean_deezer
    except (httpx.HTTPError, OSError) as e:
        LOG.debug(f"Deezer API error during artist resolution: {e}")

    clean_final = clean_unicode_punct(clean_name)
    set_cached_api(cache_key, clean_final)
    return clean_final


@lru_cache(maxsize=4096)
def is_single_group_artist(raw_name: str | None) -> bool:
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

    cache_key = f"is_group_entity:{normalized}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, bool):
        return cached

    # Format query ensuring spacing around delimiters like '&'
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
                if canonical_name:
                    set_cached_api(f"canonical_artist:{normalized}", canonical_name)
                set_cached_api(cache_key, True)
                return True
        set_cached_api(cache_key, False)
        return False
    except (
        httpx.HTTPError,
        OSError,
        ValueError,
        RuntimeError,
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
    if is_single_group_artist(raw_artist_name):
        return sanitize_name(
            resolve_artist_name(formatted_artist_name, allow_network=allow_network)
        )

    parts = get_artist_split_pattern().split(raw_artist_name, maxsplit=1)
    primary = parts[0].strip() if parts else raw_artist_name
    return sanitize_name(
        resolve_artist_name(primary, allow_network=allow_network) or "Unknown"
    )


_METADATA_FILTER = MetadataFilter(
    {
        "track": (
            remove_zero_width,
            replace_nbsp,
            remove_clean_explicit,
            remove_reissue,
            remove_remastered,
            remove_parody,
            remove_feature,
            fix_track_suffix,
        ),
        "album": (
            remove_zero_width,
            replace_nbsp,
            remove_clean_explicit,
            remove_reissue,
            remove_remastered,
            fix_track_suffix,
        ),
        "artist": (
            remove_zero_width,
            replace_nbsp,
        ),
    }
)

_TITLE_EDITION_PATTERN = re.compile(
    r"\s*[\(\[\{](?:\d{4}\s+)?(?:deluxe|bonus\s+track|mono|stereo|hq|hd|album\s+version|clean\s+version|explicit\s+version|parody|official\s+(?:music\s+)?video|official\s+audio).*?[\)\]\}]",
    re.IGNORECASE,
)


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


@lru_cache(maxsize=8192)
def deduplicate_title_features(
    title: str | None, primary_artist: str | None = None
) -> str:
    if not title:
        return ""
    fixed_title = clean_unicode_punct(ftfy.fix_text(str(title)))
    cleaned = get_duplicate_feat_pattern().sub("", fixed_title)

    base_title, matches, open_char, close_char = extract_balanced_features(cleaned)
    if not matches:
        return _COLLAPSE_SPACES_PATTERN.sub(" ", cleaned).strip()

    unique_artists: list[str] = []
    seen_normalized: set[str] = set()
    primary_norm = normalize_str(primary_artist) if primary_artist else None

    for raw_feats in matches:
        tokens = get_feat_tokens_pattern().split(raw_feats)
        for tok in tokens:
            if not tok.strip():
                continue
            clean_tok = clean_disambiguation(tok.strip())
            clean_tok = re.sub(r"[\(\)\[\]\{\}]", "", clean_tok).strip()
            if not clean_tok:
                continue
            user_overrides = _load_user_overrides()
            norm = normalize_str(clean_tok)
            if norm in user_overrides:
                clean_tok = user_overrides[norm]
                norm = normalize_str(clean_tok)
            if not norm or norm in seen_normalized:
                continue
            if primary_norm and (
                norm == primary_norm or fuzz.ratio(norm, primary_norm) >= 88
            ):
                continue
            if any(fuzz.ratio(norm, s) >= 88 for s in seen_normalized):
                continue
            seen_normalized.add(norm)
            unique_artists.append(clean_tok)

    if not unique_artists:
        formatted_title = base_title
    elif len(unique_artists) == 1:
        formatted_title = (
            f"{base_title} {open_char}feat. {unique_artists[0]}{close_char}"
        )
    else:
        feat_str = ", ".join(unique_artists[:-1]) + f" & {unique_artists[-1]}"
        formatted_title = f"{base_title} {open_char}feat. {feat_str}{close_char}"

    return _COLLAPSE_SPACES_PATTERN.sub(" ", formatted_title).strip()


@lru_cache(maxsize=8192)
def clean_title(title: str) -> str:
    """Clean track title by removing feat./ft./with brackets, remaster suffixes, and mojibake text."""
    if not title:
        return ""
    fixed_title = clean_unicode_punct(ftfy.fix_text(str(title)))
    deduped = deduplicate_title_features(fixed_title)
    cleaned = _METADATA_FILTER.filter_field("track", deduped)
    cleaned = get_bracket_feat_pattern().sub("", cleaned)
    cleaned = _TITLE_EDITION_PATTERN.sub("", cleaned)
    return cleaned.strip()


_VERSION_OR_REMIX_KEYWORDS = frozenset(
    {
        "remix",
        "rework",
        "edit",
        "mix",
        "live",
        "acoustic",
        "instrumental",
        "version",
        "demo",
        "sped up",
        "slowed",
        "freestyle",
    }
)


_REMIX_MODIFIER_PATTERN = re.compile(
    r"[\(\[\{]([^\)\]\}]+?(?:remix|rework|edit|mix|version|dub|flip|vip|bootleg|demo|live|acoustic|instrumental))[\]\)\}]",
    re.IGNORECASE,
)


def extract_version_modifier(title: str) -> str | None:
    match = _REMIX_MODIFIER_PATTERN.search(title)
    if match:
        return match.group(1).lower().strip()
    return None


def is_version_or_remix(text: str) -> bool:
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in _VERSION_OR_REMIX_KEYWORDS)


def count_unicode_accents(text: str) -> int:
    """
    Count combining diacritical marks and accented characters universally
    across all Unicode scripts (Latin, Cyrillic, Greek, Vietnamese, etc.)
    using standard Unicode canonical decomposition (NFD).
    """
    decomposed = unicodedata.normalize("NFD", text)
    return sum(1 for char in decomposed if unicodedata.category(char).startswith("M"))


@lru_cache(maxsize=8192)
def preserve_unicode_repertoire(current: str | None, candidate: str | None) -> str:
    """
    Reconcile two representations of the same text, preserving authentic Unicode characters
    when remote metadata databases return lossy ASCII-7 or unaccented Romanized transliterations
    (e.g., 'Bombe în rai' vs 'Bombe in rai', 'Andreea Bănică' vs 'Andreea Banica', 'Sigur Rós' vs 'Sigur Ros').
    Operates universally across all languages and scripts without language-specific hardcoding.
    """
    if not current:
        return candidate or ""
    if not candidate:
        return current
    if current == candidate:
        return current

    if normalize_str(current) == normalize_str(candidate):
        current_accents = count_unicode_accents(current)
        candidate_accents = count_unicode_accents(candidate)

        if current_accents > candidate_accents:
            return current
        if candidate_accents > current_accents:
            return candidate
        return current if current_accents > 0 else candidate

    return candidate


# Alias for backward compatibility
prefer_diacritics = preserve_unicode_repertoire


_NON_WORD_SPACES_PATTERN = re.compile(r"[^\w\s]")


def match_score(
    query_artist: str,
    query_title: str,
    candidate_artist: str,
    candidate_title: str,
) -> float:
    """
    Calculate a combined 0-100 similarity score between query (artist, title)
    and candidate (artist, title) using RapidFuzz ratio with version, remix modifier, and series penalties.
    """
    if not query_title or not candidate_title:
        return 0.0

    query_artist_clean = clean_title(query_artist).lower()
    candidate_artist_clean = clean_title(candidate_artist).lower()
    query_title_clean = clean_title(query_title).lower()
    candidate_title_clean = clean_title(candidate_title).lower()

    # Reject series mismatches (e.g. Vol. 1 vs Vol. 2 or Part 1 vs Part 3)
    q_series = extract_series_number(query_title)
    c_series = extract_series_number(candidate_title)
    if q_series is not None and c_series is not None and q_series != c_series:
        return 0.0
    if (q_series is not None) != (c_series is not None) and max(
        q_series or 0, c_series or 0
    ) > 1:
        return 0.0

    if query_title_clean == candidate_title_clean:
        title_score = 100.0
    else:
        title_ratio = float(fuzz.ratio(query_title_clean, candidate_title_clean))
        title_token_sort = float(
            fuzz.token_sort_ratio(query_title_clean, candidate_title_clean)
        )
        title_score = max(title_ratio, title_token_sort)

        query_word = _NON_WORD_SPACES_PATTERN.sub(" ", query_title_clean).strip()
        candidate_word = _NON_WORD_SPACES_PATTERN.sub(
            " ", candidate_title_clean
        ).strip()
        if (
            query_word
            and candidate_word
            and (
                query_word != query_title_clean
                or candidate_word != candidate_title_clean
            )
        ):
            title_score = max(
                title_score,
                float(fuzz.ratio(query_word, candidate_word)),
                float(fuzz.token_sort_ratio(query_word, candidate_word)),
            )
            query_tokens = query_word.split()
            candidate_tokens = candidate_word.split()
            if (
                min(len(query_tokens), len(candidate_tokens)) >= 3
                and min(len(query_word), len(candidate_word))
                / max(len(query_word), len(candidate_word))
                >= 0.5
            ):
                title_score = max(
                    title_score, float(fuzz.token_set_ratio(query_word, candidate_word))
                )

        query_no_articles = re.sub(r"^(?:the|a|an)\s+", "", query_title_clean).strip()
        candidate_no_articles = re.sub(
            r"^(?:the|a|an)\s+", "", candidate_title_clean
        ).strip()
        if (
            query_no_articles
            and candidate_no_articles
            and (
                query_no_articles != query_title_clean
                or candidate_no_articles != candidate_title_clean
            )
        ):
            art_ratio = float(fuzz.ratio(query_no_articles, candidate_no_articles))
            art_sort = float(
                fuzz.token_sort_ratio(query_no_articles, candidate_no_articles)
            )
            title_score = max(title_score, art_ratio, art_sort)

    query_version = is_version_or_remix(query_title) or is_version_or_remix(
        query_title_clean
    )
    candidate_version = is_version_or_remix(candidate_title) or is_version_or_remix(
        candidate_title_clean
    )
    if query_version != candidate_version:
        if query_title_clean == candidate_title_clean:
            title_score -= 15.0
        else:
            title_score -= 35.0
    elif query_version and candidate_version:
        q_mod = extract_version_modifier(query_title)
        c_mod = extract_version_modifier(candidate_title)
        if q_mod and c_mod:
            mod_ratio = float(fuzz.ratio(q_mod, c_mod))
            mod_sort = float(fuzz.token_sort_ratio(q_mod, c_mod))
            if max(mod_ratio, mod_sort) < 70.0:
                title_score -= 60.0
        else:
            for kw in _VERSION_OR_REMIX_KEYWORDS:
                if (kw in query_title_clean) != (kw in candidate_title_clean):
                    title_score -= 35.0
                    break

    title_score = max(0.0, min(100.0, title_score))

    if query_artist_clean:
        if not candidate_artist_clean:
            artist_score = 50.0
        elif query_artist_clean == candidate_artist_clean:
            artist_score = 100.0
        else:
            query_primary = clean_title(get_primary_artist(query_artist_clean)).lower()
            candidate_primary = clean_title(
                get_primary_artist(candidate_artist_clean)
            ).lower()
            if query_primary == candidate_primary:
                artist_score = 100.0
            else:
                q_core = re.sub(r"^(?:the|a|an)\s+", "", query_primary).strip()
                c_core = re.sub(r"^(?:the|a|an)\s+", "", candidate_primary).strip()
                if q_core == c_core:
                    artist_score = 100.0
                else:
                    min_len = min(len(q_core), len(c_core))
                    if min_len <= 3:
                        artist_score = 100.0 if q_core == c_core else 0.0
                    else:
                        core_ratio = float(fuzz.ratio(q_core, c_core))
                        core_sort = float(fuzz.token_sort_ratio(q_core, c_core))
                        artist_score = max(core_ratio, core_sort)

        if title_score < 70.0 or artist_score < 70.0:
            return 0.0

        return (title_score * 0.6) + (artist_score * 0.4)

    return float(title_score)


_DATE_ISO_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")
_DATE_YEAR_PATTERN = re.compile(r"(\d{4})")


@lru_cache(maxsize=8192)
def normalize_str(text: str | None) -> str:
    """
    Converts text to clean normalized lowercase ASCII form:
    - Repairs mojibake/UTF-8 artifacts via ftfy
    - Substitutes stylistic artist symbols ('$' -> 's', '_' -> ' ')
    - Transliterates international Unicode (Scandinavia, Germany, Cyrillic, CJK, etc.) via anyascii
    """
    if not text:
        return ""
    fixed_text = ftfy.fix_text(str(text)).replace("$", "s").replace("_", " ")
    ascii_text = anyascii.anyascii(fixed_text)
    cleaned_text = "".join(
        char
        for char in unicodedata.normalize("NFD", ascii_text.lower())
        if unicodedata.category(char) != "Mn"
    )
    cleaned_text = _NON_WORD_SPACES_PATTERN.sub(" ", cleaned_text)
    return _COLLAPSE_SPACES_PATTERN.sub(" ", cleaned_text).strip()


@lru_cache(maxsize=8192)
def normalize_date(date_value: str | None) -> str | None:
    """Ensure date is in YYYY-MM-DD or YYYY format, rejecting invalid/zero dates."""
    if not date_value:
        return None
    date_str = str(date_value).strip()
    if date_str in ("0", "0000", "None", "null", ""):
        return None
    match = _DATE_ISO_PATTERN.search(date_str)
    if match:
        year = int(match.group(1)[:4])
        if 1900 <= year <= 2030:
            return match.group(1)
        return None
    match = _DATE_YEAR_PATTERN.search(date_str)
    if match:
        year = int(match.group(1))
        if 1900 <= year <= 2030:
            return match.group(1)
        return None
    return None


_GENRE_GROUPS: dict[str, str] = {
    "Hip-Hop/Rap": (
        "hip hop, hip-hop, hip hop/rap, hip-hop/rap, rap/hip hop, rap/hip-hop, rap, trap, "
        "trap music, trap/hip-hop, pop rap, conscious hip hop, hardcore hip hop, christian hip hop, "
        "gangsta rap, east coast hip hop, west coast hip hop, southern hip hop, drill, uk drill, "
        "cloud rap, boom bap, emo rap, trap latino, urbano latino"
    ),
    "R&B/Soul": "rnb, r&b, r&b/soul, soul, contemporary r&b, rhythm and blues, neo-soul, neo soul",
    "Pop": (
        "pop, dance-pop, dance pop, synth-pop, synthpop, electropop, electro-pop, french pop, "
        "afro-pop, afropop, k-pop, j-pop, pop/rock, teen pop"
    ),
    "Synth-pop": "synth-pop, synthpop",
    "Electronic": "electronic, electronica, electro, edm",
    "Dance": "dance, club / dance, club/dance",
    "House": "house, euro house, deep house, tech house, progressive house, electro house",
    "Trance": "trance",
    "Techno": "techno",
    "Dubstep": "dubstep",
    "Drum & Bass": "drum and bass, drum & bass, dnb, jungle/drum'n'bass",
    "Alternative": "alternative, alternativă, alt rock, alt-rock, alternative rock, indie, indie rock, indie pop",
    "Rock": "rock, hard rock, classic rock, punk, pop punk, pop-punk",
    "Metal": "metal, heavy metal",
    "Soundtrack": "soundtrack",
    "Reggae": "reggae",
    "Reggaeton": "reggaeton",
    "Latin": "latin",
    "Country": "country",
    "Classical": "classical",
    "Jazz": "jazz",
    "Blues": "blues",
    "Folk": "folk",
    "Singer/Songwriter": "singer/songwriter",
}

_CANONICAL_GENRE_MAP: dict[str, str] = {
    alias.strip(): canonical
    for canonical, aliases in _GENRE_GROUPS.items()
    for alias in aliases.split(",")
}

_NOISE_GENRES: frozenset[str] = frozenset(
    {
        "billboard",
        "hot 100",
        "top 40",
        "amazon",
        "itunes",
        "unknown",
        "release",
        "music",
        "digital",
        "various",
        "produced by",
        "written by",
        "mixed by",
        "mastered by",
        "engineer",
        "composer",
        "fitness",
        "workout",
        "miscellaneous",
        "instrumental",
        "karaoke",
        "other",
        "audio",
        "sound",
    }
)


def is_noise_genre(genre_value: str | None) -> bool:
    """Return True if a genre string represents noise or non-musical placeholder tags."""
    if not genre_value or not str(genre_value).strip():
        return True
    genre_lower = str(genre_value).strip().lower()
    return any(noise in genre_lower for noise in _NOISE_GENRES)


@lru_cache(maxsize=8192)
def normalize_genre(genre_value: str | None) -> str | None:
    """Clean and standardize genre strings with noise filtering and canonical mapping."""
    if not genre_value or not str(genre_value).strip():
        return None

    raw_genre = str(genre_value).strip()
    genre_lower = raw_genre.lower()

    if (
        raw_genre.isdigit()
        or raw_genre.replace(".", "", 1).isdigit()
        or raw_genre.replace(",", "", 1).isdigit()
    ):
        return None

    if any(noise in genre_lower for noise in _NOISE_GENRES):
        return None

    if genre_lower in _CANONICAL_GENRE_MAP:
        return _CANONICAL_GENRE_MAP[genre_lower]

    return raw_genre.title()


@lru_cache(maxsize=2048)
def normalize_country_name(country_input: str | None) -> str | None:
    """Standardize a country identifier (ISO 3166-1 alpha-2, alpha-3, historic, or common name) into official name."""
    if not country_input:
        return None
    cleaned = country_input.strip()
    if not cleaned:
        return None

    upper = cleaned.upper()
    if upper in ("XW", "WORLDWIDE", "GLOBAL", "WORLD"):
        return "Worldwide"
    if upper in ("XE", "EUROPE"):
        return "Europe"

    try:
        match = pycountry.countries.lookup(cleaned)
        if match is not None and hasattr(match, "name"):
            return str(match.name)
    except LookupError:
        pass

    try:
        historic = pycountry.historic_countries.lookup(cleaned)
        if historic is not None and hasattr(historic, "name"):
            return str(historic.name)
    except LookupError:
        pass

    try:
        fuzzy_matches = pycountry.countries.search_fuzzy(cleaned)
        if fuzzy_matches and hasattr(fuzzy_matches[0], "name"):
            return str(fuzzy_matches[0].name)
    except LookupError:
        pass

    return cleaned


@lru_cache(maxsize=1024)
def normalize_language_name(language_input: str | None) -> str | None:
    """Standardize an ISO 639-1 / 639-2 language code or localized name into official English name."""
    if not language_input:
        return None
    cleaned = language_input.strip()
    if not cleaned:
        return None

    lower = cleaned.lower()
    try:
        if len(cleaned) == 2:
            lang = pycountry.languages.get(alpha_2=lower)
            if lang is not None and hasattr(lang, "name"):
                return str(lang.name)

        if len(cleaned) == 3:
            lang = pycountry.languages.get(alpha_3=lower) or pycountry.languages.get(
                bibliographic=lower
            )
            if lang is not None and hasattr(lang, "name"):
                return str(lang.name)

        lang = pycountry.languages.get(name=cleaned)
        if lang is not None and hasattr(lang, "name"):
            return str(lang.name)

        match = pycountry.languages.lookup(cleaned)
        if match is not None and hasattr(match, "name"):
            return str(match.name)
    except LookupError:
        pass

    return cleaned


@lru_cache(maxsize=512)
def normalize_script_name(script_input: str | None) -> str | None:
    """Standardize an ISO 15924 4-letter script code or script name into official English name."""
    if not script_input:
        return None
    cleaned = script_input.strip()
    if not cleaned:
        return None

    try:
        match = pycountry.scripts.lookup(cleaned)
        if match is not None and hasattr(match, "name"):
            return str(match.name)
    except LookupError:
        pass

    return cleaned


def sanitize_name(name: str | None) -> str:
    """
    Clean string for safe cross-platform filesystem paths.
    Replaces / and \\ with _, strips invalid OS characters, handles Windows reserved device names,
    and strips trailing dots/whitespace.
    """
    if not name:
        return "Unknown"
    fixed_text = ftfy.fix_text(str(name)).replace("/", "_").replace("\\", "_")
    sanitized = sanitize_filename(fixed_text, replacement_text="")
    sanitized = re.sub(r"\s+", " ", sanitized).strip().rstrip(".")
    return sanitized or "Unknown"


class RateLimiter:
    """Thread-safe rate limiter with precise target_time scheduling."""

    _disabled: bool = False

    def __init__(self, interval_seconds: float) -> None:
        self.interval = interval_seconds
        self.lock = threading.Lock()
        self.last_call = 0.0

    @classmethod
    def set_disabled(cls, disabled: bool) -> None:
        cls._disabled = disabled

    def wait(self, interval_seconds: float | None = None) -> float:
        if self._disabled:
            return 0.0
        interval = self.interval if interval_seconds is None else interval_seconds
        if interval <= 0:
            return 0.0
        with self.lock:
            now = time.monotonic()
            target_time = max(now, self.last_call + interval)
            sleep_time = target_time - now
            self.last_call = target_time

        if sleep_time > 0:
            time.sleep(sleep_time)
        return sleep_time


def is_valid_uuid(
    uuid_candidate: object, allow_multivalue: bool = False
) -> TypeGuard[str]:
    """
    Validate that uuid_candidate is a 36-character canonical RFC 4122 UUID (e.g. MusicBrainz MBID).
    If allow_multivalue is True, also validates multiple UUIDs delimited by ';', '/', or ','.
    """
    if not uuid_candidate or not isinstance(uuid_candidate, str):
        return False
    cleaned_uuid = uuid_candidate.strip()
    if not cleaned_uuid:
        return False
    if allow_multivalue and any(delim in cleaned_uuid for delim in (";", "/", ",")):
        tokens = [t.strip() for t in re.split(r"[;/,\s]+", cleaned_uuid) if t.strip()]
        return bool(tokens) and all(
            is_valid_uuid(t, allow_multivalue=False) for t in tokens
        )
    if len(cleaned_uuid) != 36:
        return False
    try:
        parsed = uuid.UUID(cleaned_uuid)
        return str(parsed).lower() == cleaned_uuid.lower()
    except ValueError:
        return False


def find_audio_files(
    directory: Path, recursive: bool = True, include_hidden: bool = False
) -> list[Path]:
    """Find all supported audio files in a directory, ignoring hidden directories/files by default."""
    if not directory.exists() or not directory.is_dir():
        return []
    glob_iter = directory.rglob("*") if recursive else directory.glob("*")
    files: list[Path] = []
    for candidate in glob_iter:
        if not candidate.is_file() or candidate.suffix.lower() not in SUPPORTED_EXTS:
            continue
        if not include_hidden:
            try:
                rel_parts = candidate.relative_to(directory).parts
                if any(part.startswith(".") for part in rel_parts):
                    continue
            except ValueError:
                if any(part.startswith(".") for part in candidate.parts):
                    continue
        files.append(candidate)
    return sorted(files)


def find_companion_lyrics(audio_file: Path) -> list[Path]:
    """Find all existing companion lyric files (.lrc) for a given audio file."""
    parent = audio_file.parent
    stem = audio_file.stem
    results: list[Path] = []
    for ext in COMPANION_LYRICS_EXTS:
        candidate = parent / f"{stem}{ext}"
        if candidate.exists() and candidate.is_file():
            results.append(candidate)
    return results


def group_files_by_parent(files: Sequence[Path]) -> dict[Path, list[Path]]:
    grouped: dict[Path, list[Path]] = {}
    for file_path in files:
        grouped.setdefault(file_path.parent, []).append(file_path)
    return grouped


def safe_case_rename(src: Path, dst: Path) -> Path:
    """
    Safely rename a file or directory across all platforms, including case-only renames
    on case-insensitive filesystems (NTFS, FAT32, exFAT, APFS).
    """
    if src.resolve() == dst.resolve() and src.name == dst.name:
        return src

    if not os.access(src, os.W_OK) or not os.access(src.parent, os.W_OK):
        raise PermissionError(f"Cannot rename {src}: write permission denied")

    if (
        src.parent == dst.parent
        and src.name.lower() == dst.name.lower()
        and src.name != dst.name
    ):
        tmp_name = src.parent / f".tmp_{src.name}"
        src.rename(tmp_name)
        tmp_name.rename(dst)
    else:
        src.rename(dst)
    return dst


def relocate_companion_lyrics(
    src_audio: Path, dst_audio: Path, dry_run: bool = False
) -> list[Path]:
    """
    Move or rename all companion lyric files (.lrc) alongside an audio file to match the new audio location/stem.
    """
    moved_lyrics: list[Path] = []
    for companion in find_companion_lyrics(src_audio):
        if not companion.exists():
            continue
        suffix = companion.name[len(src_audio.stem) :]
        target_companion = dst_audio.parent / f"{dst_audio.stem}{suffix}"
        if target_companion.exists() and target_companion != companion:
            if not dry_run:
                companion.unlink(missing_ok=True)
            continue

        if not dry_run:
            safe_case_rename(companion, target_companion)
        moved_lyrics.append(target_companion)
    return moved_lyrics


def format_filesize(size_bytes: float) -> str:
    """Format a byte count into a human-readable string (e.g. 4.25 MB, 1.20 GB)."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024.0 or unit == "TB":
            return f"{int(size)} B" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def clear_utils_cache() -> None:
    """Clear all in-memory LRU caches in Sonora utils."""
    clean_disambiguation.cache_clear()
    _load_user_overrides.cache_clear()
    resolve_artist_name.cache_clear()
    is_single_group_artist.cache_clear()
    deduplicate_title_features.cache_clear()
    clean_title.cache_clear()
    normalize_str.cache_clear()
    normalize_date.cache_clear()
    normalize_genre.cache_clear()
    normalize_country_name.cache_clear()
    normalize_language_name.cache_clear()
    normalize_script_name.cache_clear()
    preserve_unicode_repertoire.cache_clear()
