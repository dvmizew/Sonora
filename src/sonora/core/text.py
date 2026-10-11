import math
import re
import unicodedata
import uuid
from datetime import date, datetime, timezone
from functools import lru_cache
from typing import TypeGuard

import anyascii
import ftfy
import pycountry
from music_metadata_filter.functions import remove_zero_width
from pathvalidate import sanitize_filename

from sonora.core.config import get_config, get_disambiguation_pattern


def safe_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    numeric_text = str(value).split("/")[0].strip()
    return int(numeric_text) if numeric_text.isdigit() else None


def safe_int_pair(value: object) -> tuple[int | None, int | None]:
    """Safely parse paired numeric values (e.g. '1/2', '01/12', 1) into (current, total)."""
    if value is None:
        return None, None
    if isinstance(value, int):
        return value, None
    numeric_text = str(value).strip()
    if not numeric_text:
        return None, None
    if "/" in numeric_text:
        parts = numeric_text.split("/", 1)
        curr = safe_int(parts[0])
        tot = safe_int(parts[1])
        return curr, tot
    return safe_int(numeric_text), None


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    numeric_text = str(value).replace(" dB", "").strip()
    try:
        parsed_float = float(numeric_text)
        return parsed_float if math.isfinite(parsed_float) else None
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
            try:
                date.fromisoformat(match.group(1))
            except ValueError:
                return None
            return match.group(1)
        return None
    match = _DATE_YEAR_PATTERN.search(date_str)
    if match:
        year = int(match.group(1))
        if 1900 <= year <= max_year:
            return match.group(1)
        return None
    return None


def _matches_genre_policy(genre: str, policy: frozenset[str]) -> bool:
    genre_tokens = set(re.findall(r"\w+", genre))
    policy_tokens = {token for name in policy for token in re.findall(r"\w+", name)}
    phrase_match = any(
        " " in name and re.search(rf"(?<!\w){re.escape(name)}(?!\w)", genre)
        for name in policy
    )
    return (
        genre in policy
        or phrase_match
        or bool(genre_tokens and genre_tokens <= policy_tokens)
    )


def is_noise_genre(genre_value: str | None) -> bool:
    """Return True if a genre string represents noise or non-musical placeholder tags."""
    if not genre_value or not str(genre_value).strip():
        return True
    genre_lower = str(genre_value).strip().lower()
    return _matches_genre_policy(genre_lower, get_config().outlier_genres)


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

    if _matches_genre_policy(genre_lower, get_config().noise_genres):
        return None

    genre_aliases = dict(get_config().genre_aliases)
    if genre_lower in genre_aliases:
        return genre_aliases[genre_lower]

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
        return str(match.name)
    except LookupError:
        pass

    try:
        historic = pycountry.historic_countries.lookup(cleaned)
        return str(historic.name)
    except LookupError:
        pass

    try:
        fuzzy_matches = pycountry.countries.search_fuzzy(cleaned)
        if fuzzy_matches:
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

    # ISO codes take precedence: "En" is also an unrelated language's name.
    for code_field in ("alpha_2", "alpha_3", "bibliographic"):
        language = pycountry.languages.get(**{code_field: cleaned.lower()})
        if language is not None:
            return str(language.name)
    try:
        match = pycountry.languages.lookup(cleaned)
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
