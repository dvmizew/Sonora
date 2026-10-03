import json
import os
import re
import shutil
import threading
import time
import unicodedata
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
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
    youtube,
)
from pathvalidate import sanitize_filename
from rapidfuzz import fuzz

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.config import (
    get_artist_split_pattern,
    get_balanced_feat_pattern,
    get_bracket_feat_pattern,
    get_config,
    get_disambiguation_pattern,
    get_duplicate_feat_pattern,
    get_feat_tokens_pattern,
)
from sonora.core.constants import (
    ALBUM_COVER_NAMES,
    COMPANION_LYRICS_EXTS,
    DIRS,
    FEAT_KEYWORDS,
    SUPPORTED_EXTS,
)
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.models import TrackInfo

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


def is_interruption(exc: BaseException) -> bool:
    """Return True if an exception represents a SIGINT/Ctrl+C user cancellation,
    including threading Condition lock release errors triggered during signal interrupts."""
    if isinstance(exc, (KeyboardInterrupt, InterruptedOperationError)):
        return True
    if isinstance(exc, RuntimeError) and "release unlocked lock" in str(exc):
        return True
    context = getattr(exc, "__context__", None)
    return bool(context and isinstance(context, KeyboardInterrupt))


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


def safe_int_pair(value: object) -> tuple[int | None, int | None]:
    """Safely parse paired numeric values (e.g. '1/2', '01/12', 1) into (current, total)."""
    if value is None:
        return None, None
    if isinstance(value, int):
        return value, None
    val_str = str(value).strip()
    if not val_str:
        return None, None
    if "/" in val_str:
        parts = val_str.split("/", 1)
        curr = safe_int(parts[0])
        tot = safe_int(parts[1])
        return curr, tot
    return safe_int(val_str), None


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
_SPACES_BEFORE_COMMA_PATTERN = re.compile(r"\s+,")
_ROMANIAN_CHARS_PATTERN = re.compile(r"[ăĂîÎâÂțȚţŢ]")
_ROMANIAN_WORDS_OR_SUFFIXES = re.compile(
    r"\b(?:[şŞ]i|[şŞ]ofer|[şŞ]i-|[şŞ]osea|[Ll]e[şŞ]|[Şş]tii?|[Mm]a[şŞ]in[aăe]?|[Pp]e[şŞ]te|[Ss]f[âa]r[şŞ]it|[Oo]ra[şŞ])\b|"
    r"[A-Za-zĂăÎîÂâȚțȘș]*(?:e[şŞ]ti|e[şŞ]te|[şŞ]oi|[şŞ]el|[şŞ]or)\b",
    re.IGNORECASE,
)
_TURKISH_CHARS_PATTERN = re.compile(r"[ğĞıİ]")
_T_CEDILLA_TRANSLATION = str.maketrans({"ţ": "ț", "Ţ": "Ț"})
_S_CEDILLA_TRANSLATION = str.maketrans({"ş": "ș", "Ş": "Ș"})


@lru_cache(maxsize=8192)
def normalize_legacy_diacritics(text: str | None) -> str:
    """
    Standardize obsolete legacy diacritics into canonical Unicode characters.
    - Unconditionally converts legacy T-cedilla ('Ţ'/'ţ') to standard Romanian comma-below ('Ț'/'ț')
      as T-cedilla exists exclusively as an ISO-8859-2 encoding artifact.
    - Converts legacy S-cedilla ('Ş'/'ş') to Romanian comma-below ('Ș'/'ș') when Romanian
      orthographic markers, words, or suffixes are present, preserving Turkish S-cedilla.
    """
    if not text:
        return ""

    if "ţ" not in text and "Ţ" not in text and "ş" not in text and "Ş" not in text:
        return text

    normalized = text.translate(_T_CEDILLA_TRANSLATION)

    if "ş" in normalized or "Ş" in normalized:
        has_ro = bool(
            _ROMANIAN_CHARS_PATTERN.search(normalized)
            or _ROMANIAN_WORDS_OR_SUFFIXES.search(normalized)
        )
        has_tr = bool(_TURKISH_CHARS_PATTERN.search(normalized))
        if has_ro and not has_tr:
            normalized = normalized.translate(_S_CEDILLA_TRANSLATION)

    return normalized


def clean_unicode_punct(text: str | None) -> str:
    if not text:
        return ""
    cleaned = ftfy.fix_text(str(text))
    cleaned = remove_zero_width(cleaned)
    cleaned = _UNICODE_HYPHENS_PATTERN.sub("-", cleaned)
    cleaned = _SPACES_BEFORE_COMMA_PATTERN.sub(",", cleaned)
    return normalize_legacy_diacritics(cleaned)


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
        is_cached_acronym = (
            len(cached.replace(".", "")) <= 3
            or "." in cached
            or bool(re.search(r"\d", cached))
        )
        if not (cached.isupper() and not is_cached_acronym):
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
                is_acronym_or_initialism = (
                    len(clean_name.replace(".", "")) <= 3
                    or "." in clean_name
                    or bool(re.search(r"\d", clean_name))
                )
                if (
                    clean_name.isupper()
                    and is_acronym_or_initialism
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

    cache_key = f"is_group_entity:{normalized}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, bool):
        return cached

    if not allow_network:
        return False

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
    if is_single_group_artist(raw_artist_name, allow_network=allow_network):
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


_UNBRACKETED_FEAT_PATTERN = re.compile(
    r"(?:[\(\[\{\s]+|\s+)(?:fea?t(?:uring)?|ft|w/(?!\s*[oO](?:ut)?\b)|w\.)\.?\s+([^\)\]\}\n]+)[\)\]\}\s]*",
    re.IGNORECASE,
)


def extract_title_features(
    title: str | None, primary_artist: str | None = None
) -> tuple[str, list[str]]:
    """
    Extract featuring artists from a track title and return (clean_title, list_of_featured_artists).
    Removes bracketed and unbracketed featuring phrases, keeping title clean.
    """
    if not title:
        return ("", [])
    fixed_title = clean_unicode_punct(ftfy.fix_text(str(title)))
    deduped_title = get_duplicate_feat_pattern().sub("", fixed_title)
    clean_base, bracket_feats, _, _ = extract_balanced_features(deduped_title)

    unbracketed_feats: list[str] = []
    feat_m = _UNBRACKETED_FEAT_PATTERN.search(clean_base)
    if feat_m:
        unbracketed_feats.append(feat_m.group(1).strip())
        clean_base = clean_base[: feat_m.start()].strip()

    cleaned = _METADATA_FILTER.filter_field("track", clean_base)
    cleaned = get_bracket_feat_pattern().sub("", cleaned)
    cleaned = _TITLE_EDITION_PATTERN.sub("", cleaned)
    cleaned_title = _COLLAPSE_SPACES_PATTERN.sub(" ", cleaned).strip()

    all_raw_feats = bracket_feats + unbracketed_feats
    featured_tokens = extract_featured_artist_tokens(
        all_raw_feats, primary_artist=primary_artist, allow_network=False
    )
    return cleaned_title, featured_tokens


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


@lru_cache(maxsize=8192)
def clean_title(title: str | None) -> str:
    """Clean track title by removing feat./ft./with markers, remaster suffixes, and mojibake text."""
    if not title:
        return ""
    clean_t, _ = extract_title_features(title)
    return clean_t


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


_BRACKET_PATTERN = re.compile(r"[\(\[\{][^\(\)\[\]\{\}]+[\)\]\}]")


def extract_bracket_tokens(text: str) -> list[tuple[str, set[str]]]:
    """Extract all bracketed substrings and their constituent alphanumeric word tokens."""
    results: list[tuple[str, set[str]]] = []
    for match in _BRACKET_PATTERN.finditer(text):
        full_bracket = match.group(0)
        inner = full_bracket[1:-1].lower()
        tokens = set("".join(c if c.isalnum() else " " for c in inner).split())
        results.append((full_bracket, tokens))
    return results


def _is_corrupt_bracket(full_bracket: str, tokens: set[str]) -> bool:
    if is_version_or_remix(full_bracket) or re.search(
        FEAT_KEYWORDS, full_bracket, re.IGNORECASE
    ):
        return False

    inner_content = full_bracket.strip("()[]{}").strip()
    if inner_content.isdigit() and len(inner_content) == 4:
        return False

    dummy_title = f"Track {full_bracket}"
    if (
        youtube(dummy_title) == "Track"
        or remove_remastered(dummy_title) == "Track"
        or remove_clean_explicit(dummy_title) == "Track"
    ):
        return True

    return bool(tokens & get_config().codec_rip_keywords)


def strip_corrupt_brackets(text: str) -> str:
    """
    Remove unwanted or corrupt bracket metadata (e.g., [FLAC], (Official Video), [HQ], [Prod: ...])
    from a title or artist string, preserving legitimate features and version brackets.
    """
    if not text:
        return ""
    cleaned = text
    for full_bracket, tokens in extract_bracket_tokens(text):
        if _is_corrupt_bracket(full_bracket, tokens):
            cleaned = cleaned.replace(full_bracket, "")
    return re.sub(r"\s{2,}", " ", cleaned).strip()


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
        return normalize_legacy_diacritics(candidate) if candidate else ""
    if not candidate:
        return normalize_legacy_diacritics(current)

    clean_cur = normalize_legacy_diacritics(current)
    clean_cand = normalize_legacy_diacritics(candidate)
    if clean_cur == clean_cand:
        return clean_cur

    if normalize_str(clean_cur) == normalize_str(clean_cand):
        current_accents = count_unicode_accents(clean_cur)
        candidate_accents = count_unicode_accents(clean_cand)

        if current_accents > candidate_accents:
            return clean_cur
        if candidate_accents > current_accents:
            return clean_cand
        return clean_cur if current_accents > 0 else clean_cand

    return clean_cand


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

        min_title_len = min(len(query_title_clean), len(candidate_title_clean))
        title_thresh = 95.0 if min_title_len < 8 else 85.0

        if title_score < title_thresh or artist_score < 85.0:
            return 0.0

        return (title_score * 0.6) + (artist_score * 0.4)

    min_title_len = min(len(query_title_clean), len(candidate_title_clean))
    title_thresh = 95.0 if min_title_len < 8 else 85.0
    if title_score < title_thresh:
        return 0.0
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
    max_year = datetime.now(tz=timezone.utc).year + 1
    match = _DATE_ISO_PATTERN.search(date_str)
    if match:
        year = int(match.group(1)[:4])
        if 1900 <= year <= max_year:
            return match.group(1)
        return None
    match = _DATE_YEAR_PATTERN.search(date_str)
    if match:
        year = int(match.group(1))
        if 1900 <= year <= max_year:
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
        "mpb",
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


def find_companion_lyrics(
    audio_file: Path,
    track_number: int | str | None = None,
    in_singles: bool = False,
) -> list[Path]:
    """Find all existing companion lyric files (.lrc) for a given audio file."""
    parent = audio_file.parent
    stem = audio_file.stem
    results: list[Path] = []
    for ext in COMPANION_LYRICS_EXTS:
        candidate = parent / f"{stem}{ext}"
        if candidate.exists() and candidate.is_file():
            results.append(candidate)

    if not results and track_number is not None and parent.is_dir():
        parsed_track = safe_int(track_number)
        if parsed_track is not None:
            prefix = f"{parsed_track:02d}"
            prefix_unpadded = str(parsed_track)
            try:
                for candidate in parent.iterdir():
                    if candidate.suffix.lower() in COMPANION_LYRICS_EXTS and (
                        candidate.name.startswith(prefix)
                        or candidate.name.startswith(prefix_unpadded)
                    ):
                        results.append(candidate)
                        break
            except OSError:
                pass

    if not results and in_singles and parent.is_dir():
        try:
            lrc_candidates = [
                c
                for c in parent.iterdir()
                if c.suffix.lower() in COMPANION_LYRICS_EXTS and c.is_file()
            ]
            if len(lrc_candidates) == 1:
                results.append(lrc_candidates[0])
        except OSError:
            pass

    return results


def group_files_by_parent(files: Sequence[Path]) -> dict[Path, list[Path]]:
    grouped: dict[Path, list[Path]] = {}
    for file_path in files:
        grouped.setdefault(file_path.parent, []).append(file_path)
    return grouped


_CD_PREFIXED_FILENAME_PATTERN = re.compile(
    r"^(?:cd|disc)\s*(\d{1,2})\s*[-_.]\s*(\d{1,3})\s*[-._\s]\s*(.*)$",
    re.IGNORECASE,
)
_MULTI_DISC_FILENAME_PATTERN = re.compile(r"^(\d{1,2})[-.](\d{1,3})\s*[-._\s]\s*(.*)$")
_SINGLE_TRACK_FILENAME_PATTERN = re.compile(r"^(\d{1,3})\s*[-._\s]\s*(.*)$")


def parse_track_filename(filename: str) -> tuple[int | None, int | None, str]:
    """
    Parse filename for disc number, track number, and clean title stem.
    Returns (disc_number, track_number, clean_title).
    """
    stem = Path(filename).stem
    cd_match = _CD_PREFIXED_FILENAME_PATTERN.match(stem)
    if cd_match:
        disc_num = safe_int(cd_match.group(1))
        track_num = safe_int(cd_match.group(2))
        title_part = cd_match.group(3).strip()
        return disc_num, track_num, title_part

    multi_match = _MULTI_DISC_FILENAME_PATTERN.match(stem)
    if multi_match:
        disc_num = safe_int(multi_match.group(1))
        track_num = safe_int(multi_match.group(2))
        title_part = multi_match.group(3).strip()
        return disc_num, track_num, title_part

    single_match = _SINGLE_TRACK_FILENAME_PATTERN.match(stem)
    if single_match:
        track_num = safe_int(single_match.group(1))
        title_part = single_match.group(2).strip()
        return None, track_num, title_part

    return None, None, stem.strip()


def extract_disc_number_from_folder(folder_name: str) -> int | None:
    """
    Extract disc number from disc folder name if it matches SonoraConfig.disc_folder_patterns.
    (e.g., 'CD 1' -> 1, 'Disc 02' -> 2, 'Side A' -> 1).
    """
    from sonora.core.config import get_config

    clean = folder_name.strip()
    if not get_config().is_disc_folder(clean):
        return None
    num_match = re.search(r"\d+", clean)
    if num_match:
        return safe_int(num_match.group(0))
    side_match = re.search(r"side\s*([a-z])", clean, re.IGNORECASE)
    if side_match:
        char_code = ord(side_match.group(1).lower()) - ord("a") + 1
        return char_code if char_code > 0 else None
    return None


def get_album_root_directory(folder: Path) -> Path:
    """Resolve disc subdirectory (e.g., 'CD 1', 'Disc 2') to its canonical album root directory."""
    from sonora.core.config import get_config

    if get_config().is_disc_folder(folder.name) and folder.parent != folder:
        return folder.parent
    return folder


def group_files_by_album_root(files: Sequence[Path]) -> dict[Path, list[Path]]:
    """
    Group audio files by their canonical album root directory.
    If files reside inside disc subdirectories (e.g., 'Album/CD 1/', 'Album/Disc 2/'),
    they are unified under 'Album/'.
    """
    grouped: dict[Path, list[Path]] = {}
    for file_path in files:
        album_root = get_album_root_directory(file_path.parent)
        grouped.setdefault(album_root, []).append(file_path)
    return grouped


def resolve_unique_path(target_path: Path, current_path: Path | None = None) -> Path:
    """
    Resolve a unique, non-colliding destination path by incrementing a counter
    'Stem (2).ext', 'Stem (3).ext' if target_path already exists on disk.
    """
    if not target_path.exists() or (
        current_path and target_path.resolve() == current_path.resolve()
    ):
        return target_path
    counter = 2
    parent_dir = target_path.parent
    base_stem = target_path.stem
    extension = target_path.suffix
    while True:
        candidate = parent_dir / f"{base_stem} ({counter}){extension}"
        if not candidate.exists() or (
            current_path and candidate.resolve() == current_path.resolve()
        ):
            return candidate
        counter += 1


def relocate_companion_artwork(
    source_dir: Path, target_dir: Path, dry_run: bool = False
) -> list[Path]:
    """
    Move companion album artwork ('cover.jpg', 'cover.png', etc.) from source_dir to target_dir.
    Returns list of relocated artwork files.
    """
    relocated: list[Path] = []
    if source_dir.resolve() == target_dir.resolve():
        return relocated

    for art_name in ALBUM_COVER_NAMES:
        source_art = source_dir / art_name
        target_art = target_dir / art_name
        if source_art.exists() and source_art.is_file() and not target_art.exists():
            if not dry_run:
                try:
                    shutil.move(str(source_art), str(target_art))
                    relocated.append(target_art)
                except OSError as error:
                    LOG.debug(
                        f"Failed to relocate companion artwork {source_art}: {error}"
                    )
            else:
                relocated.append(target_art)
    return relocated


def is_in_singles_hierarchy(path: Path, root_dir: Path | None = None) -> bool:
    """
    Check if a path is located inside a Singles container or directory.
    Universally matches 'singles' folder components and configured container names
    without misinterpreting root library folders (e.g. 'FLAC', 'Music').
    """
    if any(p.lower() in ("singles", "single") for p in path.parts):
        return True
    if root_dir is not None:
        try:
            rel_parts = path.relative_to(root_dir).parts
            dir_parts = rel_parts[:-1] if not path.is_dir() else rel_parts
            from sonora.core.config import get_config

            config = get_config()
            return any(config.is_generic_container(p) for p in dir_parts)
        except ValueError:
            pass
    return False


def get_single_release_title(track_info: TrackInfo) -> str:
    """
    Return the release title for a single folder, preserving version descriptors
    (e.g. 'Deluxe Edition', 'Monoir Remix', 'Extended Mix', 'Live')
    from the album tag if absent from the track title.
    """
    title = (track_info.title or "").strip() or "Untitled"
    album = (track_info.album or "").strip()
    from sonora.core.config import get_config

    if not album or get_config().is_generic_container(album):
        return title

    matches = re.findall(r"(\((?:[^()]+)\)|\[(?:[^\[\]]+)\])", album)
    descriptors_to_add: list[str] = []
    for d in matches:
        if re.search(r"\b(?:feat|ft|featuring)\b", d, re.IGNORECASE):
            continue
        if normalize_str(d) not in normalize_str(title):
            descriptors_to_add.append(d)

    if descriptors_to_add:
        return f"{title} " + " ".join(descriptors_to_add)
    return title


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
    clean_title.cache_clear()
    normalize_str.cache_clear()
    normalize_date.cache_clear()
    normalize_genre.cache_clear()
    normalize_country_name.cache_clear()
    normalize_language_name.cache_clear()
    normalize_script_name.cache_clear()
    preserve_unicode_repertoire.cache_clear()
    normalize_legacy_diacritics.cache_clear()
