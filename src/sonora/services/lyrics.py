"""
syncedlyrics service client for searching and fetching synchronized LRC lyrics.
Supports Lrclib, Musixmatch, Genius, NetEase providers.
"""

import re
from pathlib import Path

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.exceptions import APIServiceError
from sonora.core.utils import RateLimiter, normalize_str

try:
    import syncedlyrics
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
            # Avoid accumulating multiple consecutive empty lines
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
) -> str | None:
    """
    Search and fetch LRC lyrics for a track using syncedlyrics.
    Preserves strict quality preference order: Enhanced (word-synced) -> Line-synced -> Plain text.
    """
    if not artist or not title:
        return None

    if syncedlyrics is None:
        raise APIServiceError("syncedlyrics library is not installed.")

    cache_key = f"lyrics:{normalize_str(artist)}:{normalize_str(title)}:{synced_only}:{enhanced}:{plain_only}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, str):
        return cached

    query = f"{normalize_str(artist)} - {normalize_str(title)}".strip()
    _LYRICS_LIMITER.wait()

    def _do_search(en: bool, syn: bool, pl: bool) -> str | None:
        if syncedlyrics is None:
            return None
        kwargs: dict[str, object] = {
            "plain_only": pl,
            "synced_only": syn,
            "enhanced": en,
        }
        if providers:
            kwargs["providers"] = providers
        if lang:
            kwargs["lang"] = lang
        if save_path:
            kwargs["save_path"] = str(save_path)

        res = syncedlyrics.search(query, **kwargs)
        if isinstance(res, str) and res.strip():
            return clean_lyrics_text(res.strip())
        return None

    try:
        result = _do_search(en=enhanced, syn=synced_only, pl=plain_only)

        if result:
            set_cached_api(cache_key, result)
            return result
        return None
    except Exception as e:
        raise APIServiceError(f"Lyrics fetch failed for '{query}': {e}") from e
