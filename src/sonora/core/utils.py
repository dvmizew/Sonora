import re
import threading
import time
import unicodedata
import uuid
from pathlib import Path
from typing import TypeGuard

import ftfy
from music_metadata_filter.functions import (
    remove_clean_explicit,
    remove_feature,
    remove_reissue,
    remove_remastered,
)
from rapidfuzz import fuzz

from sonora.core.constants import (
    BROAD_GENRE_KEYWORDS,
    COMPANION_LYRICS_EXTS,
    GENRE_BLACKLIST,
    GENRE_MAP,
    PROTECTED_ARTISTS,
    SUPPORTED_EXTS,
)

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


def get_primary_artist(artist_name: str | None) -> str:
    """
    Extract primary artist from raw artist string by stripping featured artists/delimiters
    (feat., ft., &, comma, etc.), respecting PROTECTED_ARTISTS.
    """
    if not artist_name:
        return "Unknown"

    raw_artist_name = str(artist_name).strip()
    is_protected = any(
        protected.lower() == raw_artist_name.lower() for protected in PROTECTED_ARTISTS
    )
    if is_protected:
        return sanitize_name(raw_artist_name)

    parts = _ARTIST_SPLIT_PATTERN.split(raw_artist_name, maxsplit=1)
    primary = parts[0].strip() if parts else raw_artist_name
    return sanitize_name(primary or "Unknown")


def clean_title(title: str) -> str:
    """Clean track title by removing feat./ft./with brackets, remaster suffixes, and mojibake text."""
    if not title:
        return ""
    title_text = ftfy.fix_text(str(title))
    # Apply official music-metadata-filter standard pipeline
    title_text = remove_clean_explicit(remove_reissue(remove_remastered(title_text)))
    title_text = remove_feature(title_text)
    title_text = re.sub(
        r"\s*[\(\[\{](?:\d{4}\s+)?(?:remaster(?:ed)?|deluxe|bonus\s+track|mono|stereo|official(?:\s+(?:video|audio))?|hq|hd).*?[\)\]\}]",
        "",
        title_text,
        flags=re.IGNORECASE,
    )
    return title_text.strip()


def is_version_or_remix(text: str) -> bool:
    keywords = [
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
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in keywords)


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


def normalize_genre(genre_value: str | None) -> str | None:
    """Clean and standardize genre strings with strict keyword filtering."""
    if not genre_value or not str(genre_value).strip():
        return None

    raw_genre = str(genre_value).strip()
    genre_title = raw_genre.title()
    genre_lower = raw_genre.lower()

    try:
        float(raw_genre.replace(",", ""))
        return None
    except ValueError:
        pass

    if (
        any(
            blacklisted_genre.lower() in genre_lower
            for blacklisted_genre in GENRE_BLACKLIST
        )
        or raw_genre.isdigit()
    ):
        return None

    if not any(keyword.lower() in genre_lower for keyword in BROAD_GENRE_KEYWORDS):
        return None

    return GENRE_MAP.get(genre_title, genre_title)


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
