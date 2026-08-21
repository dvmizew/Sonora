import re
from pathlib import Path
from typing import TYPE_CHECKING

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.exceptions import APIServiceError
from sonora.core.utils import RateLimiter, normalize_str

if TYPE_CHECKING:
    import syncedlyrics
else:
    try:
        import logging

        import syncedlyrics
        logging.getLogger("syncedlyrics").setLevel(logging.CRITICAL)
        for _provider in ["Musixmatch", "Lrclib", "NetEase", "Megalobiz", "RentAnAdviser"]:
            logging.getLogger(_provider).setLevel(logging.CRITICAL)
    except ImportError:
        syncedlyrics = None

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

    is_timestamped = lambda s: bool(re.match(r"^(?:\[|<\d{1,2}:\d{2})", s))

    for line in lines:
        s_line = line.strip()
        if not s_line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue

        if is_timestamped(s_line):
            cleaned.append(line)
            continue

        is_junk = any(re.match(p, s_line, re.IGNORECASE) for p in junk_patterns)
        if not is_junk and s_line.startswith("[") and s_line.endswith("]") and not is_timestamped(s_line):
            # Check for non-timestamped brackets like [Verse 1], [Chorus] if plain text
            pass  # Preserve section headers for plain text readability

        if not is_junk:
            cleaned.append(line)

    if not cleaned:
        return ""

    # Strip trailing "Embed" or digit+Embed on the last non-empty line
    while cleaned and (re.search(r"\bEmbed\b\s*$", cleaned[-1], re.IGNORECASE) or cleaned[-1] == ""):
        if cleaned[-1] == "":
            cleaned.pop()
            continue
        cleaned[-1] = re.sub(r"\d*\s*Embed\s*$", "", cleaned[-1], flags=re.IGNORECASE).strip()
        if not cleaned[-1]:
            cleaned.pop()

    return "\n".join(cleaned).strip()


def fetch_synced_lyrics(
    artist: str,
    title: str,
    synced_only: bool = False,
    plain_only: bool = False,
    enhanced: bool = False,
    providers: list[str] | None = None,
    lang: str | None = None,
    save_path: Path | None = None,
    isrc: str | None = None,
) -> str | None:
    """
    Search and fetch LRC lyrics for a track using syncedlyrics.
    Quality preference order: Enhanced (word-synced) -> Line-synced -> Plain text.
    Uses the 3-step surgical approach from the initial prototype.
    """
    if not artist or not title:
        return None

    if syncedlyrics is None:
        raise APIServiceError("syncedlyrics library is not installed.")

    cache_key = f"lyrics:{normalize_str(artist)}:{normalize_str(title)}:{isrc or ''}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, str):
        return cached

    _LYRICS_LIMITER.wait()

    def _do_search(query_str: str) -> str | None:
        import typing
        kwargs: dict[str, typing.Any] = {
            "plain_only": plain_only,
            "synced_only": synced_only,
            "enhanced": enhanced,
        }
        if providers is not None:
            kwargs["providers"] = providers
        if lang:
            kwargs["lang"] = lang
        if syncedlyrics is None:
            return None
        res = syncedlyrics.search(query_str, **kwargs)
        if isinstance(res, str) and res.strip():
            return clean_lyrics_text(res.strip())
        return None

    last_exception: Exception | None = None
    lrc = None

    # ATTEMPT 1: ISRC LOOKUP
    if isrc:
        try:
            lrc = _do_search(isrc)
        except (APIServiceError, OSError, ValueError, KeyError, RuntimeError) as e:
            last_exception = e

    # ATTEMPT 2: Standard/Surgical Query
    if not lrc:
        # Standard query format (matches unit tests)
        default_query = f"{artist.lower()} - {title.lower()}".strip()
        try:
            lrc = _do_search(default_query)
        except (APIServiceError, OSError, ValueError, KeyError, RuntimeError) as e:
            last_exception = e

    # ATTEMPT 3: Surgical Clean Title Fallback
    if not lrc and ("(" in title or "[" in title or "feat" in title.lower()):
        clean_title = re.sub(r'[\(\[\{].*?[\)\]\}]', '', title).strip()
        clean_title = re.sub(r'\s+(?:fea?t|ft)\.?\s+.*$', '', clean_title, flags=re.IGNORECASE).strip()
        primary_artist = artist.split(',')[0].split('&')[0].split(';')[0].strip()
        query = f"{clean_title} {primary_artist}".strip()
        try:
            lrc = _do_search(query)
        except (APIServiceError, OSError, ValueError, KeyError, RuntimeError) as e:
            last_exception = e

    if lrc:
        set_cached_api(cache_key, lrc)
        return lrc

    if last_exception:
        raise APIServiceError(f"Lyrics fetch failed for '{title}': {last_exception}") from last_exception

    return None
