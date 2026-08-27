import json
import re
import threading
import time
import unicodedata
import uuid
from functools import lru_cache
from pathlib import Path
from typing import TypeGuard

import ftfy
import httpx
import musicbrainzngs
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
from rapidfuzz import fuzz

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.constants import (
    COMPANION_LYRICS_EXTS,
    SUPPORTED_EXTS,
)
from sonora.core.http import SESSION

_ARTIST_SEPARATORS = [
    r"\s+fea?t\.?\s+",
    r"\s+featuring\s+",
    r"\s+and\s+",
    r"\s+și\s+",
    r"\s+si\s+",
    r"\s+cu\s+",
    r"\s+vs\.?\s+",
    r"\s+[xX×]\s+",
    r"\s*&\s*",
    r"\s*,\s*",
    r"\s*;\s*",
    r"\s*/\s*",
]
_ARTIST_SPLIT_PATTERN = re.compile("|".join(_ARTIST_SEPARATORS), re.IGNORECASE)

_ROMAN_VALUES: dict[str, int] = dict(zip("ivxlcdm", (1, 5, 10, 50, 100, 500, 1000)))
_NUMBER_WORDS: tuple[str, ...] = (
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
)


def _parse_roman(token: str) -> int | None:
    """Parse Roman numeral (I to MMMCMXCIX) to integer."""
    s = token.lower().strip()
    if not s or not all(c in _ROMAN_VALUES for c in s) or len(s) > 10:
        return None
    total, prev_val = 0, 0
    for c in reversed(s):
        val = _ROMAN_VALUES[c]
        total += -val if val < prev_val else val
        prev_val = val
    return total if total > 0 else None


def extract_series_number(text: str | None) -> int | None:
    """
    Extract album or track series/volume number (e.g. 'Savage Mode II' -> 2, 'Pt. 2' -> 2, 'Vol. 3' -> 3).
    Returns integer series number or None if not part of a numbered series.
    """
    if not text:
        return None

    clean_text = ftfy.fix_text(str(text)).strip().lower()

    # Match explicit volume/part patterns (e.g. 'Vol 2', 'Pt. 2', 'Part II', 'Act 1')
    prefix_match = re.search(
        r"\b(?:vol(?:ume)?|pt|part|chapter|act)\.?\s*(\d+|[ivxlcdm]+|[a-z]+)\b",
        clean_text,
        re.IGNORECASE,
    )
    if prefix_match:
        token = prefix_match.group(1).lower()
        if token.isdigit() and len(token) <= 2:
            return int(token)
        roman_val = _parse_roman(token)
        if roman_val is not None:
            return roman_val
        if token in _NUMBER_WORDS:
            return _NUMBER_WORDS.index(token) + 1

    # Match trailing Roman numerals (II, III, IV, etc.) or digits at the end of title
    trailing_match = re.search(
        r"\b(\d{1,2}|[ivxlcdm]+)\s*$",
        clean_text,
        re.IGNORECASE,
    )
    if trailing_match:
        token = trailing_match.group(1).lower()
        if token.isdigit():
            return int(token)
        roman_val = _parse_roman(token)
        if roman_val is not None:
            return roman_val

    return None


def _load_user_overrides() -> dict[str, str]:
    candidate_paths = [
        Path.home() / ".config" / "sonora" / "aliases.json",
        Path("sonora_aliases.json"),
    ]
    overrides: dict[str, str] = {}
    for path in candidate_paths:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    for key, value in data.items():
                        overrides[normalize_str(key)] = str(value).strip()
            except (OSError, ValueError):
                pass
    return overrides


@lru_cache(maxsize=4096)
def resolve_artist_name(raw_name: str | None) -> str:
    """
    Resolve legal names, aliases, or variations to canonical stage names.
    Returns the resolved canonical name or the cleaned input if not an alias.
    """
    if not raw_name or not str(raw_name).strip():
        return "Unknown Artist"

    clean_name = str(raw_name).strip()
    normalized = normalize_str(clean_name)
    if not normalized:
        return clean_name

    # Tier 1: User custom config overrides (~/.config/sonora/aliases.json)
    user_overrides = _load_user_overrides()
    if normalized in user_overrides:
        return user_overrides[normalized]

    # Tier 2: Persistent DiskCache
    cache_key = f"canonical_artist:{normalized}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, str):
        return cached

    # Tier 3: MusicBrainz Alias / Legal Name lookup
    try:
        res = musicbrainzngs.search_artists(
            query=f'artist:"{clean_name}" OR alias:"{clean_name}"', limit=5
        )
        artist_list = res.get("artist-list", [])
        # Pass 1: Exact case-insensitive ASCII match (prioritize "Nane" over accented "Nané")
        for artist in artist_list:
            art_name = str(artist.get("name", "")).strip()
            if (
                art_name.lower() == clean_name.lower()
                and art_name == clean_name.title()
            ):
                set_cached_api(cache_key, art_name, expire_seconds=86400 * 30)
                return art_name
        # Pass 2: High score official alias / entity match (e.g. "mgl" -> "M.G.L.", legal name -> stage name)
        for artist in artist_list:
            score = int(artist.get("ext:score", 0))
            if score >= 90:
                canonical_name = artist.get("name")
                if canonical_name:
                    set_cached_api(cache_key, canonical_name, expire_seconds=86400 * 30)
                    return canonical_name
        # Pass 3: Case-insensitive fallback
        for artist in artist_list:
            art_name = str(artist.get("name", "")).strip()
            if art_name.lower() == clean_name.lower():
                set_cached_api(cache_key, art_name, expire_seconds=86400 * 30)
                return art_name
    except (
        httpx.HTTPError,
        OSError,
        ValueError,
        RuntimeError,
        musicbrainzngs.MusicBrainzError,
    ):
        pass

    # Tier 4: Deezer Artist lookup
    try:
        response = SESSION.get(
            "https://api.deezer.com/search/artist",
            params={"q": clean_name},
            timeout=5,
        )
        if response.status_code == 200:
            data = response.json().get("data", [])
            if data and isinstance(data, list):
                deezer_name = str(data[0].get("name", "")).strip()
                if deezer_name and normalize_str(deezer_name) == normalized:
                    set_cached_api(cache_key, deezer_name, expire_seconds=86400 * 30)
                    return deezer_name
    except (httpx.HTTPError, OSError, ValueError, RuntimeError):
        pass

    set_cached_api(cache_key, clean_name, expire_seconds=86400 * 7)
    return clean_name


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

    if not any(
        char in clean_name.lower()
        for char in ("&", "+", ",", " și ", " si ", " with ", " / ")
    ):
        return False

    user_overrides = _load_user_overrides()
    if normalized in user_overrides:
        return True

    cache_key = f"is_group_entity:{normalized}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, bool):
        return cached

    try:
        res = musicbrainzngs.search_artists(query=f'artist:"{clean_name}"', limit=3)
        artist_list = res.get("artist-list", [])
        for artist in artist_list:
            name_match = normalize_str(artist.get("name")) == normalized
            score = int(artist.get("ext:score", 0))
            artist_type = artist.get("type")
            if name_match and (score >= 95 or artist_type == "Group"):
                set_cached_api(cache_key, True, expire_seconds=86400 * 30)
                return True
    except (
        httpx.HTTPError,
        OSError,
        ValueError,
        RuntimeError,
        musicbrainzngs.MusicBrainzError,
    ):
        pass

    set_cached_api(cache_key, False, expire_seconds=86400 * 30)
    return False


def get_primary_artist(artist_name: str | None) -> str:
    """
    Extract primary artist from raw artist string by resolving aliases and stripping
    transient featured artists/delimiters, while preserving single group/band entities
    (e.g., 'Simon & Garfunkel', 'Play & Win', 'Earth, Wind & Fire').
    """
    if not artist_name:
        return "Unknown"

    raw_artist_name = str(artist_name).strip()
    if is_single_group_artist(raw_artist_name):
        return sanitize_name(resolve_artist_name(raw_artist_name))

    parts = _ARTIST_SPLIT_PATTERN.split(raw_artist_name, maxsplit=1)
    primary = parts[0].strip() if parts else raw_artist_name
    return sanitize_name(resolve_artist_name(primary) or "Unknown")


_METADATA_FILTER = MetadataFilter(
    {
        "track": (
            remove_zero_width,
            replace_nbsp,
            youtube,
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


def clean_title(title: str) -> str:
    """Clean track title by removing feat./ft./with brackets, remaster suffixes, and mojibake text."""
    if not title:
        return ""
    fixed_title = ftfy.fix_text(str(title))
    cleaned = _METADATA_FILTER.filter_field("track", fixed_title)
    cleaned = re.sub(
        r"\s*[\(\[\{](?:\d{4}\s+)?(?:deluxe|bonus\s+track|mono|stereo|hq|hd).*?[\)\]\}]",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
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


def is_version_or_remix(text: str) -> bool:
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in _VERSION_OR_REMIX_KEYWORDS)


def match_score(
    query_artist: str,
    query_title: str,
    candidate_artist: str,
    candidate_title: str,
) -> float:
    """
    Calculate a combined 0-100 similarity score between query (artist, title)
    and candidate (artist, title) using RapidFuzz WRatio and ratio with version penalties.
    """
    if not query_title or not candidate_title:
        return 0.0

    query_artist_clean = clean_title(query_artist).lower()
    candidate_artist_clean = clean_title(candidate_artist).lower()
    query_title_clean = clean_title(query_title).lower()
    candidate_title_clean = clean_title(candidate_title).lower()

    if query_title_clean == candidate_title_clean:
        title_score = 100.0
    else:
        title_wratio = fuzz.WRatio(query_title_clean, candidate_title_clean)
        title_ratio = fuzz.ratio(query_title_clean, candidate_title_clean)
        if len(query_title_clean) <= 3:
            title_score = float(title_ratio)
        else:
            title_score = max(title_wratio, title_ratio)

    if not is_version_or_remix(query_title_clean) and is_version_or_remix(
        candidate_title_clean
    ):
        title_score -= 35.0

    title_score = max(0.0, min(100.0, title_score))

    if query_artist_clean and candidate_artist_clean:
        artist_w = fuzz.WRatio(query_artist_clean, candidate_artist_clean)
        artist_token = fuzz.token_set_ratio(query_artist_clean, candidate_artist_clean)
        artist_score = max(artist_w, artist_token)
        return (title_score * 0.6) + (artist_score * 0.4)

    return float(title_score)


def normalize_str(text: str | None) -> str:
    """
    Converts to lowercase, fixes mojibake via ftfy, normalizes NFD diacritics,
    replaces $, replaces non-alphanumeric characters with space, and collapses spaces.
    """
    if not text:
        return ""
    cleaned_text = ftfy.fix_text(str(text))
    cleaned_text = cleaned_text.replace("$", "s")
    cleaned_text = cleaned_text.replace("_", " ")
    cleaned_text = "".join(
        char
        for char in unicodedata.normalize("NFD", cleaned_text.lower())
        if unicodedata.category(char) != "Mn"
    )
    cleaned_text = re.sub(r"[^\w\s]", " ", cleaned_text)
    return re.sub(r"\s+", " ", cleaned_text).strip()


def normalize_date(date_value: str | None) -> str | None:
    """Ensure date is in YYYY-MM-DD format."""
    if not date_value:
        return None
    date_str = str(date_value).strip()
    match = re.search(r"(\d{4}-\d{2}-\d{2})", date_str)
    if match:
        return match.group(1)
    match = re.search(r"(\d{4})", date_str)
    if match:
        return match.group(1)
    return date_str if date_str else None


_CANONICAL_GENRE_MAP: dict[str, str] = {
    "hip hop": "Hip-Hop/Rap",
    "hip-hop": "Hip-Hop/Rap",
    "rap": "Hip-Hop/Rap",
    "trap": "Hip-Hop/Rap",
    "rnb": "R&B/Soul",
    "r&b": "R&B/Soul",
    "soul": "R&B/Soul",
    "pop/rock": "Pop",
    "drum and bass": "Drum & Bass",
    "drum & bass": "Drum & Bass",
    "dnb": "Drum & Bass",
    "synthpop": "Synth-pop",
    "synth-pop": "Synth-pop",
    "alternative rock": "Alternative",
    "indie rock": "Indie",
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
    }
)


def normalize_genre(genre_value: str | None) -> str | None:
    """Clean and standardize genre strings with noise filtering and canonical mapping."""
    if not genre_value or not str(genre_value).strip():
        return None

    raw_genre = str(genre_value).strip()
    genre_lower = raw_genre.lower()

    # Reject numeric or pure decimal tags
    if (
        raw_genre.isdigit()
        or raw_genre.replace(".", "", 1).isdigit()
        or raw_genre.replace(",", "", 1).isdigit()
    ):
        return None

    # Reject spam / noise tags
    if any(noise in genre_lower for noise in _NOISE_GENRES):
        return None

    # Direct canonical map
    if genre_lower in _CANONICAL_GENRE_MAP:
        return _CANONICAL_GENRE_MAP[genre_lower]

    # Standard title-case formatting
    return raw_genre.title()


def sanitize_name(name: str | None) -> str:
    """
    Clean string for safe filesystem paths.
    Fixes mojibake via ftfy, replaces / and \\ with _, strips invalid Windows/Linux bad chars (<>:"|?*),
    and strips trailing dots/whitespace.
    """
    if not name:
        return "Unknown"
    sanitized_text = ftfy.fix_text(str(name))
    sanitized_text = sanitized_text.replace("/", "_").replace("\\", "_")
    bad_chars = '<>:"|?*'
    for char in bad_chars:
        sanitized_text = sanitized_text.replace(char, "")
    sanitized_text = re.sub(r"\s+", " ", sanitized_text).strip().rstrip(".")
    return sanitized_text or "Unknown"


class RateLimiter:
    """Thread-safe rate limiter with precise target_time scheduling."""

    def __init__(self, interval_seconds: float) -> None:
        self.interval = interval_seconds
        self.lock = threading.Lock()
        self.last_call = 0.0

    def wait(self) -> float:
        with self.lock:
            now = time.monotonic()
            target_time = max(now, self.last_call + self.interval)
            sleep_time = target_time - now
            self.last_call = target_time

        if sleep_time > 0:
            time.sleep(sleep_time)
        return sleep_time


def is_valid_uuid(uuid_candidate: object) -> TypeGuard[str]:
    """Validate that uuid_candidate is a 36-character canonical RFC 4122 UUID (e.g. MusicBrainz MBID)."""
    if not uuid_candidate or not isinstance(uuid_candidate, str):
        return False
    cleaned_uuid = uuid_candidate.strip()
    if len(cleaned_uuid) != 36:
        return False
    try:
        parsed = uuid.UUID(cleaned_uuid)
        return str(parsed).lower() == cleaned_uuid.lower()
    except (ValueError, AttributeError, TypeError):
        return False


def find_audio_files(directory: Path, recursive: bool = True) -> list[Path]:
    """Find all supported audio files in a directory, sorted lexicographically."""
    if not directory.exists() or not directory.is_dir():
        return []
    glob_iter = directory.rglob("*") if recursive else directory.glob("*")
    return sorted(
        candidate_path
        for candidate_path in glob_iter
        if candidate_path.is_file() and candidate_path.suffix.lower() in SUPPORTED_EXTS
    )


def find_companion_lyrics(audio_file: Path) -> list[Path]:
    """Find all existing companion lyric files (.lrc, .synced.lrc, .enhanced.lrc, .txt) for a given audio file."""
    parent = audio_file.parent
    stem = audio_file.stem
    results: list[Path] = []
    for ext in COMPANION_LYRICS_EXTS:
        candidate = parent / f"{stem}{ext}"
        if candidate.exists() and candidate.is_file():
            results.append(candidate)
    return results
