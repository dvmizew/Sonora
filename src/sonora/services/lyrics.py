import html
import json
import logging
import os
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any

import ftfy
import httpx
import syncedlyrics

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.constants import RATE_LIMIT_LRCLIB, RATE_LIMIT_LYRICS
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.utils import (
    RateLimiter,
    clean_title,
    get_primary_artist,
    normalize_str,
)

logging.getLogger("syncedlyrics").setLevel(logging.CRITICAL)
for _provider in ["Musixmatch", "Lrclib", "NetEase", "Megalobiz", "RentAnAdviser"]:
    logging.getLogger(_provider).setLevel(logging.CRITICAL)

_LRCLIB_LIMITER = RateLimiter(interval_seconds=RATE_LIMIT_LRCLIB)
_LYRICS_LIMITER = RateLimiter(interval_seconds=RATE_LIMIT_LYRICS)


def init_musixmatch_token(token_str: str | None = None) -> bool:
    """
    Initialize Musixmatch token in syncedlyrics cache from env or explicit argument.
    Supports raw user token or URL-encoded JSON cookie dump containing web-desktop-app-v1.0.
    """
    raw = (
        token_str
        if token_str is not None
        else (
            os.environ.get("MUSIXMATCH_TOKEN")
            or os.environ.get("MUSIXMATCH_USER_TOKEN")
        )
    )
    if not raw or not raw.strip():
        return False

    extracted_token: str | None = None
    try:
        decoded = urllib.parse.unquote(raw.strip())
        if decoded.startswith("{") and "tokens" in decoded:
            data = json.loads(decoded)
            tokens = data.get("tokens", {})
            extracted_token = (
                tokens.get("web-desktop-app-v1.0")
                or tokens.get("mxm-account-v1.0")
                or tokens.get("user_token")
            )
        elif decoded.startswith("{") and "token" in decoded:
            data = json.loads(decoded)
            extracted_token = data.get("token") or data.get("user_token")
        else:
            extracted_token = decoded
    except (ValueError, KeyError):
        extracted_token = raw.strip()

    if extracted_token:
        try:
            from syncedlyrics.utils import get_cache_path

            token_path = get_cache_path("syncedlyrics", False) / "musixmatch_token.json"
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_data = {
                "token": extracted_token,
                "expiration_time": int(time.time()) + 31536000,  # 1 year
            }
            token_path.write_text(json.dumps(token_data), encoding="utf-8")
            return True
        except (OSError, ValueError):
            pass
    return False


_LYRICS_TIMESTAMP_REGEX = re.compile(
    r"^(?:\[\d{1,2}:\d{2}[\.:]\d{2,3}\]|<\d{1,2}:\d{2}[\.:]\d{2,3}>)\s*(.*)$"
)
_LYRICS_JUNK_PATTERNS = [
    re.compile(r"^\d+\s*(?:Contributor|Translation|Embed)s?$", re.IGNORECASE),
    re.compile(r"^You might also like$", re.IGNORECASE),
    re.compile(r"^Read More$", re.IGNORECASE),
    re.compile(r"^See .* Live(?:Get tickets.*)?$", re.IGNORECASE),
    re.compile(
        r"^\[?(?:Produced|Written|Arranged|Composed|Mastered|Mixed|Recorded|Engineered)\b.*\]?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\[?(?:Producer|Writer|Composer|Arranger|Engineer|Mixer|Release Date|Recording Date|Studio|Label)s?\s*:.*\]?$",
        re.IGNORECASE,
    ),
    re.compile(r"^Lyrics (?:powered|licensed|provided) by.*$", re.IGNORECASE),
    re.compile(r"^Paroles de la chanson .* par .*$", re.IGNORECASE),
    re.compile(r"^Commercial use is strictly forbidden.*$", re.IGNORECASE),
    re.compile(r"^https?://\S+$", re.IGNORECASE),
    re.compile(r"^www\.\S+$", re.IGNORECASE),
    re.compile(
        r"^(?:Synced|Created|Uploaded|Encoded|Downloaded|LRC)\s*(?:by|from|using|with)?\s+.*$",
        re.IGNORECASE,
    ),
    re.compile(r"<!--.*?-->", re.IGNORECASE),
]
_LYRICS_EMBED_END_REGEX = re.compile(r"\d*\s*Embed\s*$", re.IGNORECASE)


def clean_lyrics_text(text: str | None) -> str | None:
    """
    Strips web scraping artifacts (Genius, Musixmatch, AZLyrics, LRCLIB, Megalobiz junk)
    and unescapes HTML entities while preserving timestamped LRC lines ([mm:ss.xx] and <mm:ss.xx>).
    """
    if text is None:
        return None
    if not text.strip():
        return ""

    decoded = html.unescape(ftfy.fix_text(str(text)))
    lines = decoded.splitlines()
    cleaned: list[str] = []

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue

        ts_match = _LYRICS_TIMESTAMP_REGEX.match(stripped_line)
        content_to_check = ts_match.group(1).strip() if ts_match else stripped_line

        if any(pat.search(content_to_check) for pat in _LYRICS_JUNK_PATTERNS):
            continue

        cleaned.append(stripped_line)

    while cleaned and (
        _LYRICS_EMBED_END_REGEX.search(cleaned[-1]) or cleaned[-1] == ""
    ):
        if cleaned[-1] == "":
            cleaned.pop()
            continue
        cleaned[-1] = _LYRICS_EMBED_END_REGEX.sub("", cleaned[-1]).strip()
        if not cleaned[-1]:
            cleaned.pop()

    return "\n".join(cleaned).strip()


def get_lyrics_quality(lyrics_text: str | None) -> int:
    """
    Determines quality tier of LRC lyrics string:
      3: Enhanced / Word-level synced (<00:01.23> word timestamps)
      2: Synced / Line-level synced ([00:01.23] line timestamps)
      1: Plain text (no timestamps)
      0: None or empty
    """
    if not lyrics_text or not lyrics_text.strip():
        return 0
    if re.search(r"<\d{1,2}:\d{2}[\.:]\d{2,3}>", lyrics_text):
        return 3
    if re.search(r"^\[\d{1,2}:\d{2}[\.:]\d{2,3}\]", lyrics_text, re.MULTILINE):
        return 2
    return 1


def detect_lrc_quality(audio_path: Path) -> int:
    """
    Check the quality of existing .lrc lyrics alongside an audio file.
    Returns 3 for enhanced, 2 for synced, 1 for plain, 0 for missing.
    """
    lrc_path = audio_path.with_suffix(".lrc")
    if not lrc_path.exists() or lrc_path.stat().st_size == 0:
        return 0
    try:
        content = lrc_path.read_text(encoding="utf-8", errors="ignore")
        return get_lyrics_quality(content)
    except (OSError, ValueError):
        return 0


def _query_lrclib(
    query_str: str,
    plain_only: bool = False,
    synced_only: bool = False,
) -> str | None:
    try:
        _LRCLIB_LIMITER.wait()
        if " - " in query_str:
            parts = query_str.split(" - ", 1)
            url = "https://lrclib.net/api/get"
            params = {
                "artist_name": parts[0].strip(),
                "track_name": parts[1].strip(),
            }
            response = SESSION.get(url, params=params, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict):
                    synced = data.get("syncedLyrics")
                    plain = data.get("plainLyrics")
                    if (
                        synced
                        and not plain_only
                        and isinstance(synced, str)
                        and synced.strip()
                    ):
                        return clean_lyrics_text(synced.strip())
                    if (
                        plain
                        and not synced_only
                        and isinstance(plain, str)
                        and plain.strip()
                    ):
                        return clean_lyrics_text(plain.strip())
            return None

        # Fallback to /api/search for non-standard or single-token queries
        search_url = "https://lrclib.net/api/search"
        response = SESSION.get(search_url, params={"q": query_str}, timeout=5.0)
        if response.status_code == 200:
            results = response.json()
            if isinstance(results, list) and results:
                first = results[0]
                if isinstance(first, dict):
                    synced = first.get("syncedLyrics")
                    plain = first.get("plainLyrics")
                    if (
                        synced
                        and not plain_only
                        and isinstance(synced, str)
                        and synced.strip()
                    ):
                        return clean_lyrics_text(synced.strip())
                    if (
                        plain
                        and not synced_only
                        and isinstance(plain, str)
                        and plain.strip()
                    ):
                        return clean_lyrics_text(plain.strip())
    except (httpx.HTTPError, OSError, ValueError, RuntimeError):
        pass
    return None


def _query_syncedlyrics(
    query_str: str,
    plain_only: bool,
    synced_only: bool,
    enhanced: bool,
    providers: list[str] | None,
    lang: str | None,
) -> str | None:
    # 1. High-speed direct HTTP/2 LRCLIB fast-path when not restricted to other providers
    if not enhanced and (not providers or "Lrclib" in providers) and not lang:
        lrclib_result = _query_lrclib(
            query_str, plain_only=plain_only, synced_only=synced_only
        )
        if lrclib_result:
            return lrclib_result

    # 2. Multi-provider fallback via syncedlyrics
    kwargs: dict[str, Any] = {
        "plain_only": plain_only,
        "synced_only": synced_only,
        "enhanced": enhanced,
    }
    if providers is not None:
        kwargs["providers"] = providers
    if lang:
        kwargs["lang"] = lang

    _LYRICS_LIMITER.wait()
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

    search_args = (plain_only, synced_only, enhanced, providers, lang)

    last_exception: Exception | None = None
    lyrics_content = None

    # ATTEMPT 1: ISRC LOOKUP
    if isrc:
        try:
            lyrics_content = _query_syncedlyrics(isrc, *search_args)
        except (
            httpx.HTTPError,
            OSError,
            ValueError,
            KeyError,
            RuntimeError,
            TypeError,
            AttributeError,
            TimeoutError,
        ) as error:
            last_exception = error

    # ATTEMPT 2: Standard/Surgical Query
    if not lyrics_content:
        # Standard query format (matches unit tests)
        default_query = f"{artist.lower()} - {title.lower()}".strip()
        try:
            lyrics_content = _query_syncedlyrics(default_query, *search_args)
        except (
            httpx.HTTPError,
            OSError,
            ValueError,
            KeyError,
            RuntimeError,
            TypeError,
            AttributeError,
            TimeoutError,
        ) as error:
            last_exception = error

    # ATTEMPT 3: Surgical Clean Title Fallback
    if not lyrics_content and ("(" in title or "[" in title or "feat" in title.lower()):
        cleaned_track_title = clean_title(title)
        primary_artist = get_primary_artist(artist)
        query = f"{cleaned_track_title} {primary_artist}".strip()
        try:
            lyrics_content = _query_syncedlyrics(query, *search_args)
        except (
            httpx.HTTPError,
            OSError,
            ValueError,
            KeyError,
            RuntimeError,
            TypeError,
            AttributeError,
            TimeoutError,
        ) as error:
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
    Fetches, cleans, and saves track lyrics to an accompanying .lrc file.
    Supports smart quality upgrade:
      - Enhanced (quality 3, word-synced): Maximum quality, skip network query unless force=True.
      - Line-synced (quality 2): Queries online to attempt upgrade to Enhanced (quality 3).
      - Plain text (quality 1): Queries online to attempt upgrade to Synced or Enhanced (quality 2/3).
      - Missing (quality 0): Queries online for any available lyrics.
    Returns (lyrics_text, quality_tag) or (None, None).
    """
    lrc_path = file_path.with_suffix(".lrc")
    current_quality = detect_lrc_quality(file_path)
    existing_content: str | None = None

    if lrc_path.exists() and lrc_path.stat().st_size > 0:
        try:
            existing_content = lrc_path.read_text(
                encoding="utf-8", errors="ignore"
            ).strip()
        except (OSError, ValueError):
            existing_content = None

    # Fast-path: Already at maximum quality (Enhanced word-synced)
    if not force and current_quality >= 3 and existing_content:
        return existing_content, "enhanced"

    # Attempt to fetch higher quality lyrics online
    try:
        lyrics_text = fetch_synced_lyrics(artist, title, enhanced=True, isrc=isrc)
    except (
        httpx.HTTPError,
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        TypeError,
        AttributeError,
        TimeoutError,
    ) as error:
        LOG.debug(f"Lyrics lookup error for {title}: {error}")
        lyrics_text = None

    if not lyrics_text:
        # Remote search returned nothing: preserve existing local lyrics if any
        if existing_content and current_quality > 0:
            tag_type = (
                "enhanced"
                if current_quality == 3
                else ("synced" if current_quality == 2 else "plain")
            )
            return existing_content, tag_type
        return None, None

    new_quality = get_lyrics_quality(lyrics_text)

    # If not forcing and remote lyrics are not better than existing local lyrics, keep existing
    if not force and current_quality > 0 and new_quality <= current_quality:
        if existing_content:
            tag_type = (
                "enhanced"
                if current_quality == 3
                else ("synced" if current_quality == 2 else "plain")
            )
            return existing_content, tag_type
        return None, None

    # Upgrade / Save new lyrics to .lrc
    if not dry_run:
        lrc_path.write_text(lyrics_text, encoding="utf-8")
        txt_path = file_path.with_suffix(".txt")
        if txt_path.exists() and lrc_path != txt_path:
            txt_path.unlink(missing_ok=True)

    tag_type = (
        "enhanced" if new_quality == 3 else ("synced" if new_quality == 2 else "plain")
    )
    return lyrics_text, tag_type
