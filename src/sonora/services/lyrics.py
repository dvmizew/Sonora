import logging
import re
from pathlib import Path
from typing import Any

import syncedlyrics

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.utils import (
    RateLimiter,
    clean_title,
    get_primary_artist,
    normalize_str,
)

logging.getLogger("syncedlyrics").setLevel(logging.CRITICAL)
for _provider in ["Musixmatch", "Lrclib", "NetEase", "Megalobiz", "RentAnAdviser"]:
    logging.getLogger(_provider).setLevel(logging.CRITICAL)

_LYRICS_LIMITER = RateLimiter(interval_seconds=1.0)


def clean_lyrics_text(text: str | None) -> str | None:
    """
    Strips web scraping artifacts (Genius, Musixmatch, AZLyrics, Lrclib junk)
    while preserving timestamped LRC lines ([mm:ss.xx] and <mm:ss.xx>).
    """
    if not text:
        return text

    lines = text.splitlines()
    cleaned: list[str] = []

    junk_patterns = [
        r"^\d+\s*(?:Contributor|Translation|Embed)s?$",
        r"^You might also like$",
        r"^Read More$",
        r"^See .* Live(?:Get tickets.*)?$",
        r"^\[?(?:Produced|Written|Arranged|Composed|Mastered|Mixed|Recorded|Engineered)\b.*\]?$",
        r"^\[?(?:Producer|Writer|Composer|Arranger|Engineer|Mixer|Release Date|Recording Date|Studio|Label)s?\s*:.*\]?$",
        r"^https?://\S+$",
        r"^www\.\S+$",
        r"^Synced by \S+$",
    ]

    def is_timestamped(line_str: str) -> bool:
        return bool(re.match(r"^(?:\[|<\d{1,2}:\d{2})", line_str))

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue

        if is_timestamped(stripped_line):
            cleaned.append(line)
            continue

        is_junk = any(
            re.match(pattern, stripped_line, re.IGNORECASE) for pattern in junk_patterns
        )
        if (
            not is_junk
            and stripped_line.startswith("[")
            and stripped_line.endswith("]")
            and not is_timestamped(stripped_line)
        ):
            # Check for non-timestamped brackets like [Verse 1], [Chorus] if plain text
            pass  # Preserve section headers for plain text readability

        if not is_junk:
            cleaned.append(line)

    if not cleaned:
        return ""

    # Strip trailing "Embed" or digit+Embed on the last non-empty line
    while cleaned and (
        re.search(r"\bEmbed\b\s*$", cleaned[-1], re.IGNORECASE) or cleaned[-1] == ""
    ):
        if cleaned[-1] == "":
            cleaned.pop()
            continue
        cleaned[-1] = re.sub(
            r"\d*\s*Embed\s*$", "", cleaned[-1], flags=re.IGNORECASE
        ).strip()
        if not cleaned[-1]:
            cleaned.pop()

    return "\n".join(cleaned).strip()


def get_lyrics_quality(text: str | None) -> int:
    """
    Returns quality level for lyrics text:
      0: missing or empty
      1: plain text
      2: line-synced [00:12.34]
      3: enhanced word-synced <00:12.34>
    """
    if not text or not text.strip():
        return 0
    if re.search(r"<\d{1,2}:\d{2}[\.:]\d{2,3}>", text):
        return 3
    if re.search(r"\[\d{2}:\d{2}[\.:]\d{2,3}\]", text):
        return 2
    return 1


def detect_lrc_quality(file_path: Path) -> int:
    """Detect quality level of existing lyrics file on disk for a given track path."""
    lrc_path = file_path.with_suffix(".lrc")
    txt_path = file_path.with_suffix(".txt")
    target_path = (
        lrc_path if lrc_path.exists() else (txt_path if txt_path.exists() else None)
    )
    if not target_path:
        return 0
    try:
        content = target_path.read_text(encoding="utf-8", errors="ignore")
        return get_lyrics_quality(content)
    except (OSError, ValueError):
        return 0


def _query_syncedlyrics(
    query_str: str,
    plain_only: bool = False,
    synced_only: bool = False,
    enhanced: bool = False,
    providers: list[str] | None = None,
    lang: str | None = None,
) -> str | None:
    kwargs: dict[str, Any] = {
        "plain_only": plain_only,
        "synced_only": synced_only,
        "enhanced": enhanced,
    }
    if providers is not None:
        kwargs["providers"] = providers
    if lang:
        kwargs["lang"] = lang
    result = syncedlyrics.search(query_str, **kwargs)
    if isinstance(result, str) and result.strip():
        return clean_lyrics_text(result.strip())
    return None


def fetch_synced_lyrics(
    artist: str,
    title: str,
    isrc: str | None = None,
    plain_only: bool = False,
    synced_only: bool = False,
    enhanced: bool = False,
    providers: list[str] | None = None,
    lang: str | None = None,
) -> str | None:
    """
    Search and fetch LRC lyrics for a track using syncedlyrics.
    Quality preference order: Enhanced (word-synced) -> Line-synced -> Plain text.
    Uses the 3-step surgical approach from the initial prototype.
    """
    if not artist or not title:
        return None

    cache_key = f"lyrics:{normalize_str(artist)}:{normalize_str(title)}:{isrc or ''}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, str):
        return cached

    _LYRICS_LIMITER.wait()

    search_args = (plain_only, synced_only, enhanced, providers, lang)

    last_exception: Exception | None = None
    lyrics_content = None

    # ATTEMPT 1: ISRC LOOKUP
    if isrc:
        try:
            lyrics_content = _query_syncedlyrics(isrc, *search_args)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            last_exception = error

    # ATTEMPT 2: Standard/Surgical Query
    if not lyrics_content:
        # Standard query format (matches unit tests)
        default_query = f"{artist.lower()} - {title.lower()}".strip()
        try:
            lyrics_content = _query_syncedlyrics(default_query, *search_args)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            last_exception = error

    # ATTEMPT 3: Surgical Clean Title Fallback
    if not lyrics_content and ("(" in title or "[" in title or "feat" in title.lower()):
        cleaned_track_title = clean_title(title)
        primary_artist = get_primary_artist(artist)
        query = f"{cleaned_track_title} {primary_artist}".strip()
        try:
            lyrics_content = _query_syncedlyrics(query, *search_args)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            last_exception = error

    if lyrics_content:
        set_cached_api(cache_key, lyrics_content)
        return lyrics_content

    if last_exception:
        raise RuntimeError(
            f"Lyrics fetch failed for '{title}': {last_exception}"
        ) from last_exception

    return None


def process_track_lyrics(
    file_path: Path,
    artist: str,
    title: str,
    force: bool = False,
    dry_run: bool = False,
    isrc: str | None = None,
) -> tuple[str | None, str | None]:
    """
    Fetches, cleans, and saves track lyrics according to quality hierarchy.
    Returns (lyrics_text, quality_tag) or (None, None).
    """
    current_quality = detect_lrc_quality(file_path)
    if not force and current_quality >= 3:
        return None, None

    lyrics_text = fetch_synced_lyrics(artist, title, enhanced=True, isrc=isrc)
    if not lyrics_text:
        return None, None

    new_quality = get_lyrics_quality(lyrics_text)
    if not force and current_quality > 0 and new_quality <= current_quality:
        return None, None

    lrc_path = file_path.with_suffix(".lrc")
    if not dry_run:
        lrc_path.write_text(lyrics_text, encoding="utf-8")
        if new_quality == 3:
            synced_copy = re.sub(r"<\d{1,2}:\d{2}[\.:]\d{2,3}>", "", lyrics_text)
            synced_path = file_path.with_suffix(".synced.lrc")
            synced_path.write_text(synced_copy, encoding="utf-8")
        txt_path = file_path.with_suffix(".txt")
        if txt_path.exists() and lrc_path != txt_path:
            txt_path.unlink(missing_ok=True)

    tag_type = (
        "enhanced" if new_quality == 3 else ("synced" if new_quality == 2 else "plain")
    )
    return lyrics_text, tag_type
