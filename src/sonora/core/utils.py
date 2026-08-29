import json
import re
import threading
import time
import unicodedata
import uuid
from collections.abc import Sequence
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
from pathvalidate import sanitize_filename
from rapidfuzz import fuzz

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.constants import (
    COMPANION_LYRICS_EXTS,
    SUPPORTED_EXTS,
)
from sonora.core.http import SESSION
from sonora.core.logger import LOG

_ARTIST_SEPARATORS = [
    r"\s+fea?t\.?\s+",
    r"\s+featuring\s+",
    r"\s+and\s+",
    r"\s+și\s+",
    r"\s+si\s+",
    r"\s+cu\s+",
    r"\s+vs\.?\s+",
    r"\s+[xX\u00d7]\s+",
    r"\s*&\s*",
    r"\s*,\s*",
    r"\s*;\s*",
    r"\s*/\s*",
]
_ARTIST_SPLIT_PATTERN = re.compile("|".join(_ARTIST_SEPARATORS), re.IGNORECASE)

_ROMAN_MAP: dict[str, int] = {
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "v": 5,
    "vi": 6,
    "vii": 7,
    "viii": 8,
    "ix": 9,
    "x": 10,
    "xi": 11,
    "xii": 12,
    "xiii": 13,
    "xiv": 14,
    "xv": 15,
    "xvi": 16,
    "xvii": 17,
    "xviii": 18,
    "xix": 19,
    "xx": 20,
    "xxi": 21,
    "xxii": 22,
    "xxiii": 23,
    "xxiv": 24,
    "xxv": 25,
    "xxvi": 26,
    "xxvii": 27,
    "xxviii": 28,
    "xxix": 29,
    "xxx": 30,
    "xl": 40,
    "l": 50,
}

_WORD_NUMBER_MAP: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def extract_series_number(text: str | None) -> int | None:
    """
    Extract album or track series/volume number (e.g. 'Savage Mode II' -> 2, 'Pt. 2' -> 2, 'Vol. 3' -> 3).
    Returns integer series number or None if not part of a numbered series.
    """
    if not text:
        return None

    clean_text = ftfy.fix_text(str(text)).strip().lower()

    prefix_match = re.search(
        r"\b(?:vol(?:ume)?|pt|part|chapter|act|book)\.?\s*(\d{1,2}|[a-z]+)\b",
        clean_text,
        re.IGNORECASE,
    )
    if prefix_match:
        token = prefix_match.group(1).lower()
        if token.isdigit():
            return int(token)
        if token in _ROMAN_MAP:
            return _ROMAN_MAP[token]
        return _WORD_NUMBER_MAP.get(token)

    trailing_match = re.search(
        r"\b(\d{1,2}|[a-z]+)\s*$",
        clean_text,
        re.IGNORECASE,
    )
    if trailing_match:
        token = trailing_match.group(1).lower()
        if token.isdigit():
            return int(token)
        if token in _ROMAN_MAP:
            return _ROMAN_MAP[token]
        return _WORD_NUMBER_MAP.get(token)

    return None


_DISAMBIGUATION_PATTERN = re.compile(r"\s*\((?:[A-Za-z]{2,4}|\d+)\)(?=\s*(?:[,;/]|$))")
_COLLAPSE_SPACES_PATTERN = re.compile(r"\s+")


@lru_cache(maxsize=8192)
def clean_disambiguation(name: str | None) -> str:
    """
    Strips disambiguation country codes or numeric suffixes anywhere in an artist string
    (e.g., 'Armin (ROU)' -> 'Armin', 'Rafoo, Armin (ROU), ASSAF (ROU)' -> 'Rafoo, Armin, ASSAF', 'Jony (10)' -> 'Jony').
    """
    if not name:
        return ""
    cleaned = _DISAMBIGUATION_PATTERN.sub("", str(name)).strip()
    return _COLLAPSE_SPACES_PATTERN.sub(" ", cleaned)


@lru_cache(maxsize=1)
def _load_user_overrides() -> dict[str, str]:
    candidate_paths = [
        Path.home() / ".config" / "sonora" / "aliases.json",
        Path("sonora_aliases.json"),
    ]
    overrides: dict[str, str] = {}
    for path in candidate_paths:
        try:
            if path.exists() and path.is_file() and path.stat().st_size > 0:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    for key, value in data.items():
                        overrides[normalize_str(key)] = str(value).strip()
        except (OSError, ValueError) as error:
            LOG.warning(f"Failed to load user aliases from {path}: {error}")
    return overrides


def _ensure_musicbrainz_init() -> None:
    try:
        from sonora import __version__

        musicbrainzngs.set_useragent(
            "Sonora", __version__, "https://github.com/dvmizew/Sonora"
        )
    except (OSError, ValueError, RuntimeError, musicbrainzngs.MusicBrainzError):
        pass


@lru_cache(maxsize=4096)
def resolve_artist_name(raw_name: str | None) -> str:
    """
    Resolve legal names, aliases, or variations to canonical stage names.
    Returns the resolved canonical name or the cleaned input if not an alias.
    """
    if not raw_name or not str(raw_name).strip():
        return "Unknown Artist"

    clean_name = clean_disambiguation(str(raw_name).strip())
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

    _ensure_musicbrainz_init()

    # Tier 3: MusicBrainz Alias / Legal Name lookup
    try:
        res = musicbrainzngs.search_artists(
            query=f'artist:"{clean_name}" OR alias:"{clean_name}"', limit=5
        )
        artists = res.get("artist-list", [])
        clean_has_punct = bool(re.search(r"[^\w\s]", clean_name))
        for artist in artists:
            art_name = str(artist.get("name", "")).strip()
            if not art_name:
                continue
            if art_name.lower() == clean_name.lower():
                set_cached_api(cache_key, art_name)
                return art_name
            if normalize_str(art_name) == normalized:
                cand_has_punct = bool(re.search(r"[^\w\s]", art_name))
                if not clean_has_punct and cand_has_punct:
                    continue
                set_cached_api(cache_key, art_name)
                return art_name

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
                if alias_name and normalize_str(alias_name) == normalized:
                    set_cached_api(cache_key, art_name)
                    return art_name
            score = int(artist.get("ext:score", 0))
            artist_id = str(artist.get("id", "")).strip()
            if score >= 95 and is_valid_uuid(artist_id):
                try:
                    artist_data = musicbrainzngs.get_artist_by_id(
                        artist_id, includes=["aliases"]
                    )
                    full_aliases = artist_data.get("artist", {}).get("alias-list", [])
                    for alias_item in full_aliases:
                        alias_name = (
                            alias_item.get("alias")
                            if isinstance(alias_item, dict)
                            else str(alias_item)
                        )
                        if alias_name and normalize_str(alias_name) == normalized:
                            set_cached_api(cache_key, art_name)
                            return art_name
                except (
                    httpx.HTTPError,
                    OSError,
                    ValueError,
                    RuntimeError,
                    musicbrainzngs.MusicBrainzError,
                ):
                    pass
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
                    # Do not override all-caps acronyms (e.g. M.G.L) with lowercased titles
                    if (
                        clean_name.isupper()
                        and not deezer_name.isupper()
                        and len(clean_name.replace(".", "")) <= 5
                    ):
                        deezer_name = clean_name
                    set_cached_api(cache_key, deezer_name)
                    return deezer_name
    except (httpx.HTTPError, OSError, ValueError, RuntimeError):
        pass

    set_cached_api(cache_key, clean_name)
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

    _ensure_musicbrainz_init()

    # Format query ensuring spacing around delimiters like '&'
    query_name = re.sub(r"\s*([&+,/])\s*", r" \1 ", clean_name).strip()

    try:
        res = musicbrainzngs.search_artists(query=f'artist:"{query_name}"', limit=5)
        artist_list = res.get("artist-list", [])
        for artist in artist_list:
            name_match = normalize_str(artist.get("name")) == normalized
            score = int(artist.get("ext:score", 0))
            artist_type = artist.get("type")
            if name_match and (score >= 90 or artist_type == "Group"):
                set_cached_api(cache_key, True)
                return True
        set_cached_api(cache_key, False)
        return False
    except (
        httpx.HTTPError,
        OSError,
        ValueError,
        RuntimeError,
        musicbrainzngs.MusicBrainzError,
    ):
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


_DUPLICATE_FEAT_PATTERN = re.compile(
    r"\s+(?:cu|și|si|with|fea?t\.?|featuring)\s+([A-Za-z0-9\s\.\'\-]+?)(?=\s*[\(\[\{]\s*(?:fea?t\.?|featuring|with|cu)\s+\1[\)\]\}])",
    re.IGNORECASE,
)


_TITLE_ROMANIAN_FEAT_PATTERN = re.compile(
    r"\s*[\(\[\{]\s*(?:cu|și|si)\s+.*?[\)\]\}]", re.IGNORECASE
)
_TITLE_EDITION_PATTERN = re.compile(
    r"\s*[\(\[\{](?:\d{4}\s+)?(?:deluxe|bonus\s+track|mono|stereo|hq|hd).*?[\)\]\}]",
    re.IGNORECASE,
)


@lru_cache(maxsize=8192)
def deduplicate_title_features(title: str | None) -> str:
    if not title:
        return ""
    cleaned = _DUPLICATE_FEAT_PATTERN.sub("", str(title))
    return _COLLAPSE_SPACES_PATTERN.sub(" ", cleaned).strip()


@lru_cache(maxsize=8192)
def clean_title(title: str) -> str:
    """Clean track title by removing feat./ft./with brackets, remaster suffixes, and mojibake text."""
    if not title:
        return ""
    fixed_title = ftfy.fix_text(str(title))
    deduped = deduplicate_title_features(fixed_title)
    cleaned = _METADATA_FILTER.filter_field("track", deduped)
    cleaned = _TITLE_ROMANIAN_FEAT_PATTERN.sub("", cleaned)
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
    and candidate (artist, title) using RapidFuzz WRatio and ratio with version and series penalties.
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
        title_wratio = fuzz.WRatio(query_title_clean, candidate_title_clean)
        title_ratio = fuzz.ratio(query_title_clean, candidate_title_clean)
        if len(query_title_clean) <= 3:
            title_score = float(title_ratio)
        else:
            title_score = max(title_wratio, title_ratio)

    q_ver = is_version_or_remix(query_title) or is_version_or_remix(query_title_clean)
    c_ver = is_version_or_remix(candidate_title) or is_version_or_remix(
        candidate_title_clean
    )
    if q_ver != c_ver:
        title_score -= 35.0
    elif q_ver and c_ver:
        for kw in _VERSION_OR_REMIX_KEYWORDS:
            if (kw in query_title_clean) != (kw in candidate_title_clean):
                title_score -= 35.0
                break

    title_score = max(0.0, min(100.0, title_score))

    if query_artist_clean and candidate_artist_clean:
        if query_artist_clean == candidate_artist_clean:
            artist_score = 100.0
        else:
            artist_w = fuzz.WRatio(query_artist_clean, candidate_artist_clean)
            artist_token = fuzz.token_set_ratio(
                query_artist_clean, candidate_artist_clean
            )
            artist_score = max(artist_w, artist_token)

        if title_score < 70.0 or artist_score < 70.0:
            return 0.0

        return (title_score * 0.6) + (artist_score * 0.4)

    return float(title_score)


_NON_WORD_SPACES_PATTERN = re.compile(r"[^\w\s]")
_DATE_ISO_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")
_DATE_YEAR_PATTERN = re.compile(r"(\d{4})")


@lru_cache(maxsize=8192)
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
    cleaned_text = _NON_WORD_SPACES_PATTERN.sub(" ", cleaned_text)
    return _COLLAPSE_SPACES_PATTERN.sub(" ", cleaned_text).strip()


@lru_cache(maxsize=8192)
def normalize_date(date_value: str | None) -> str | None:
    """Ensure date is in YYYY-MM-DD format."""
    if not date_value:
        return None
    date_str = str(date_value).strip()
    match = _DATE_ISO_PATTERN.search(date_str)
    if match:
        return match.group(1)
    match = _DATE_YEAR_PATTERN.search(date_str)
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


@lru_cache(maxsize=8192)
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
    except (ValueError, AttributeError, TypeError):
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
