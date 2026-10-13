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

from sonora.audio.metadata import get_audio_duration
from sonora.core.api_payloads import require_api_object
from sonora.core.artists import get_primary_artist
from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.config import get_config
from sonora.core.constants import RATE_LIMIT_LRCLIB, RATE_LIMIT_LYRICS, USER_AGENT
from sonora.core.coordination import coalesce_lookup
from sonora.core.files import atomic_write_bytes
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.rate_limit import RateLimiter
from sonora.core.text import normalize_str
from sonora.core.titles import clean_title, score_artist_title_match
from sonora.core.utils import retry_network_call

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
            lyrics_payload = json.loads(decoded)
            tokens = lyrics_payload.get("tokens", {})
            extracted_token = (
                tokens.get("web-desktop-app-v1.0")
                or tokens.get("mxm-account-v1.0")
                or tokens.get("user_token")
            )
        elif decoded.startswith("{") and "token" in decoded:
            lyrics_payload = json.loads(decoded)
            extracted_token = lyrics_payload.get("token") or lyrics_payload.get(
                "user_token"
            )
        else:
            extracted_token = decoded
    except ValueError:
        extracted_token = raw.strip()

    if extracted_token:
        try:
            from syncedlyrics.utils import get_cache_path

            token_path = get_cache_path("syncedlyrics", False) / "musixmatch_token.json"
            token_path.parent.mkdir(parents=True, exist_ok=True)
            lyrics_token_payload = {
                "token": extracted_token,
                "expiration_time": int(time.time()) + 31536000,  # 1 year
            }
            token_path.write_text(json.dumps(lyrics_token_payload), encoding="utf-8")
            return True
        except (OSError, ValueError):
            pass
    return False


_LYRICS_TIMESTAMP_REGEX = re.compile(
    r"^(?:\[\d{1,2}:\d{2}(?:[\.:]\d{2,3})?(?:[+-]\d+)?\]|<\d{1,2}:\d{2}(?:[\.:]\d{2,3})?>)\s*(.*)$"
)
_LYRICS_EXTRACT_TS_REGEX = re.compile(
    r"[\[<](\d{1,3}):([0-5]\d)(?:[\.:](\d{2,3}))?[\]>]"
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
    re.compile(r"^\[?(?:re|ve|by|length|offset)\s*:.*\]?$", re.IGNORECASE),
    re.compile(r".*megalobiz.*", re.IGNORECASE),
    re.compile(r"^(?:作词|作曲|编曲|制作人)\s*:.*$", re.IGNORECASE),
    re.compile(r"^This song is called\b.*$", re.IGNORECASE),
    re.compile(r"^\[?Instrumental\]?$", re.IGNORECASE),
    re.compile(r"^Lyrics (?:powered|licensed|provided) by.*$", re.IGNORECASE),
    re.compile(r"^Paroles de la chanson .* par .*$", re.IGNORECASE),
    re.compile(r"^Commercial use is strictly forbidden.*$", re.IGNORECASE),
    re.compile(r"^https?://\S+$", re.IGNORECASE),
    re.compile(r"^www\.\S+$", re.IGNORECASE),
    re.compile(
        r"^(?:Synced|Created|Uploaded|Encoded|Downloaded|LRC)\s*(?:by|from|using|with|via|:)\s+.*$",
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

        timestamp_match = _LYRICS_TIMESTAMP_REGEX.match(stripped_line)
        content_to_check = (
            timestamp_match.group(1).strip() if timestamp_match else stripped_line
        )

        if timestamp_match and not content_to_check:
            continue

        if any(
            junk_pattern.search(content_to_check)
            for junk_pattern in _LYRICS_JUNK_PATTERNS
        ):
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


def extract_max_timestamp(lyrics_text: str | None) -> float | None:
    """Extract the maximum timestamp in seconds from LRC lyrics text."""
    if not lyrics_text:
        return None
    timestamps: list[float] = []
    for match in _LYRICS_EXTRACT_TS_REGEX.finditer(lyrics_text):
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        fraction_str = match.group(3)
        total_seconds = minutes * 60.0 + seconds
        if fraction_str:
            total_seconds += float(f"0.{fraction_str}")
        timestamps.append(total_seconds)
    return max(timestamps) if timestamps else None


def is_lyrics_duration_valid(
    lyrics_text: str | None,
    audio_duration: float | None,
    tolerance_seconds: float = 15.0,
) -> bool:
    """
    Validates that lyrics timestamps do not significantly exceed audio duration.
    Prevents cross-song collisions (e.g. short tracks matching longer songs).
    """
    if not lyrics_text or audio_duration is None or audio_duration <= 0:
        return True
    max_timestamp = extract_max_timestamp(lyrics_text)
    if max_timestamp is None:
        return True
    return max_timestamp <= (audio_duration + tolerance_seconds)


def get_lyrics_quality(lyrics_text: str | None) -> int:
    """
    Determines quality tier of LRC lyrics string:
      3: Enhanced / Word-level synced (<00:01.23> word timestamps)
      2: Synced / Line-level synced ([00:01.23] line timestamps)
      1: Plain text (no timestamps)
      0: None or empty / invalid
    """
    cleaned = clean_lyrics_text(lyrics_text)
    if not cleaned:
        return 0

    non_header_lines = [
        line.strip()
        for line in cleaned.splitlines()
        if line.strip()
        and not re.match(
            r"^\[(?:ar|ti|al|by|re|ve|length|offset):",
            line.strip(),
            re.IGNORECASE,
        )
    ]
    if not non_header_lines:
        return 0

    if re.search(r"<\d{1,2}:\d{2}[\.:]\d{2,3}>", cleaned):
        return 3
    if re.search(r"^\[\d{1,3}:[0-5]\d(?:[\.:]\d{2,3})?\]", cleaned, re.MULTILINE):
        return 2
    return 1


def detect_lrc_quality(audio_path: Path, audio_duration: float | None = None) -> int:
    """
    Check the quality of existing .lrc lyrics alongside an audio file.
    Returns 3 for enhanced, 2 for synced, 1 for plain, 0 for missing or invalid.
    """
    lrc_path = audio_path.with_suffix(".lrc")
    if not lrc_path.exists() or lrc_path.stat().st_size == 0:
        return 0
    try:
        content = lrc_path.read_text(encoding="utf-8", errors="ignore")
        resolved_duration = (
            audio_duration
            if audio_duration is not None
            else get_audio_duration(audio_path)
        )
        if not is_lyrics_duration_valid(content, resolved_duration):
            return 0
        return get_lyrics_quality(content)
    except (OSError, ValueError):
        return 0


def _query_lrclib(
    query_str: str,
    plain_only: bool = False,
    synced_only: bool = False,
    album_name: str | None = None,
    duration: float | None = None,
) -> str | None:
    if " - " not in query_str:
        return None
    try:
        _LRCLIB_LIMITER.wait()
        artist_name, track_name = query_str.split(" - ", 1)
        params: dict[str, str | int | float] = {
            "artist_name": artist_name.strip(),
            "track_name": track_name.strip(),
        }
        if album_name:
            params["album_name"] = album_name
        if duration is not None:
            params["duration"] = int(duration)
        response = SESSION.get(
            "https://lrclib.net/api/get",
            params=params,
            timeout=5.0,
            headers={"User-Agent": USER_AGENT},
        )
        if response.status_code != 200:
            return None
        lyrics_payload = require_api_object(response.json(), "LRCLIB lyrics response")

        synced_lyrics = lyrics_payload.get("syncedLyrics")
        plain_lyrics = lyrics_payload.get("plainLyrics")
        for lyrics_text in (synced_lyrics, plain_lyrics):
            if lyrics_text is not None and not isinstance(lyrics_text, str):
                raise TypeError("LRCLIB lyrics must be strings or null")
        provider_artist = lyrics_payload.get("artistName")
        provider_title = lyrics_payload.get("trackName")
        if (
            not provider_artist
            or not provider_title
            or score_artist_title_match(
                artist_name, track_name, str(provider_artist), str(provider_title)
            )
            <= 0
        ):
            return None
        provider_duration = lyrics_payload.get("duration")
        if (
            duration is not None
            and isinstance(provider_duration, (int, float))
            and abs(provider_duration - duration)
            > get_config().single_duration_tolerance_seconds
        ):
            return None

        if synced_lyrics and not plain_only:
            cleaned_synced = clean_lyrics_text(synced_lyrics.strip())
            if cleaned_synced and is_lyrics_duration_valid(cleaned_synced, duration):
                return cleaned_synced
        if plain_lyrics and not synced_only:
            return clean_lyrics_text(plain_lyrics.strip()) or None
    except (httpx.HTTPError, OSError) as error:
        LOG.debug(f"LRCLIB request failed: {error}")
    return None


def _query_syncedlyrics(
    query_str: str,
    plain_only: bool,
    synced_only: bool,
    enhanced: bool,
    providers: list[str] | None,
    lang: str | None,
    album_name: str | None = None,
    duration: float | None = None,
) -> str | None:
    checked_lrclib = (not providers or "Lrclib" in providers) and not lang
    if checked_lrclib:
        lrclib_result = _query_lrclib(
            query_str,
            plain_only=plain_only,
            synced_only=synced_only,
            album_name=album_name,
            duration=duration,
        )
        if lrclib_result:
            return lrclib_result

    if not providers and not get_config().allow_unverified_lyrics:
        return None
    LOG.warning(
        "Using an explicitly enabled lyrics source without returned recording identity"
    )

    kwargs: dict[str, Any] = {
        "plain_only": plain_only,
        "synced_only": synced_only,
        "enhanced": enhanced,
    }
    if checked_lrclib:
        # The SDK's text-only LRCLIB search must not bypass the identity checks above.
        fallback_providers = [
            provider
            for provider in (
                providers or ["Musixmatch", "NetEase", "Megalobiz", "Genius"]
            )
            if provider != "Lrclib"
        ]
        if not fallback_providers:
            return None
        kwargs["providers"] = fallback_providers
    elif providers is not None:
        kwargs["providers"] = providers
    if lang:
        kwargs["lang"] = lang

    result = retry_network_call(
        lambda: syncedlyrics.search(query_str, **kwargs),
        max_retries=2,
        initial_delay=1.0,
        rate_limiter=_LYRICS_LIMITER,
        transient_exceptions=(
            httpx.HTTPError,
            OSError,
            TimeoutError,
        ),
    )
    if isinstance(result, str) and result.strip():
        cleaned_lyrics = clean_lyrics_text(result.strip())
        if cleaned_lyrics and is_lyrics_duration_valid(cleaned_lyrics, duration):
            return cleaned_lyrics
    return None


@coalesce_lookup
def fetch_synced_lyrics(
    artist: str,
    title: str,
    isrc: str | None = None,
    plain_only: bool = False,
    synced_only: bool = False,
    enhanced: bool = False,
    providers: list[str] | None = None,
    lang: str | None = None,
    album_name: str | None = None,
    duration: float | None = None,
) -> str | None:
    """
    Search and fetch LRC lyrics for a track using syncedlyrics.
    Quality preference order: Enhanced (word-synced) -> Line-synced -> Plain text.
    """
    if not artist or not title:
        return None

    constraints = json.dumps(
        [
            plain_only,
            synced_only,
            enhanced,
            sorted(providers or []),
            lang,
            album_name,
            duration,
        ],
        ensure_ascii=False,
    )
    cache_key = f"lyrics:v4:{get_config().allow_unverified_lyrics}:{normalize_str(artist)}:{normalize_str(title)}:{isrc or ''}:{constraints}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, str):
        cached_quality = get_lyrics_quality(cached)
        if not cached:
            return None
        if (
            is_lyrics_duration_valid(cached, duration)
            and (not synced_only or cached_quality >= 2)
            and (not plain_only or cached_quality == 1)
            and (not enhanced or cached_quality >= 3)
        ):
            return cached

    search_args = (plain_only, synced_only, enhanced, providers, lang)

    last_exception: Exception | None = None
    lyrics_content = None

    default_query = f"{artist.lower()} - {title.lower()}".strip()
    try:
        lyrics_content = _query_syncedlyrics(
            default_query, *search_args, album_name=album_name, duration=duration
        )
    except (
        httpx.HTTPError,
        OSError,
        ValueError,
        RuntimeError,
        TimeoutError,
    ) as error:
        last_exception = error

    if not lyrics_content and ("(" in title or "[" in title or "feat" in title.lower()):
        cleaned_track_title = clean_title(title)
        primary_artist = get_primary_artist(artist)
        if (
            score_artist_title_match(artist, title, primary_artist, cleaned_track_title)
            <= 0
        ):
            if last_exception:
                raise last_exception
            return None
        query = f"{primary_artist} - {cleaned_track_title}".strip()
        try:
            lyrics_content = _query_syncedlyrics(
                query, *search_args, album_name=album_name, duration=duration
            )
        except (
            httpx.HTTPError,
            OSError,
            ValueError,
            RuntimeError,
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

    set_cached_api(cache_key, "")
    return None


def process_track_lyrics(
    file_path: Path,
    artist: str,
    title: str,
    force: bool = False,
    dry_run: bool = False,
    isrc: str | None = None,
    album_name: str | None = None,
    duration: float | None = None,
    existing_lyrics: str | None = None,
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
    resolved_duration = (
        duration if duration is not None else get_audio_duration(file_path)
    )
    current_quality = detect_lrc_quality(file_path, audio_duration=resolved_duration)
    existing_content: str | None = None

    if lrc_path.exists() and lrc_path.stat().st_size > 0:
        try:
            existing_content = lrc_path.read_text(
                encoding="utf-8", errors="ignore"
            ).strip()
        except (OSError, ValueError):
            existing_content = None

    embedded_quality = get_lyrics_quality(existing_lyrics)
    if embedded_quality > current_quality and is_lyrics_duration_valid(
        existing_lyrics, resolved_duration
    ):
        current_quality = embedded_quality
        existing_content = existing_lyrics
        if existing_content and not dry_run:
            atomic_write_bytes(lrc_path, existing_content.encode("utf-8"))

    if not force and current_quality >= 3 and existing_content:
        return existing_content, "existing"

    try:
        lyrics_text = fetch_synced_lyrics(
            artist,
            title,
            enhanced=True,
            isrc=isrc,
            album_name=album_name,
            duration=resolved_duration,
        )
    except (
        httpx.HTTPError,
        OSError,
        ValueError,
        RuntimeError,
        TimeoutError,
    ) as error:
        LOG.debug(f"Lyrics lookup error for {title}: {error}")
        lyrics_text = None

    if (
        lyrics_text
        and resolved_duration is None
        and extract_max_timestamp(lyrics_text) is not None
    ):
        LOG.warning(
            f"Cannot validate timestamped lyrics without audio duration: {title}"
        )
        lyrics_text = None
    if lyrics_text and not is_lyrics_duration_valid(lyrics_text, resolved_duration):
        LOG.warning(
            f"   ∟ ⚠️  Rejected mismatched lyrics for {title}: "
            f"lyrics duration exceeds audio length ({int(resolved_duration or 0)}s)."
        )
        lyrics_text = None

    if not lyrics_text:
        # Remote search returned nothing: preserve valid existing local lyrics if any
        if existing_content and current_quality > 0:
            return existing_content, "existing"
        return None, None

    new_quality = get_lyrics_quality(lyrics_text)

    # Downgrade protection: NEVER overwrite synced or enhanced lyrics with plain text
    if current_quality >= 2 and new_quality < current_quality:
        if existing_content:
            return existing_content, "existing"
        return None, None

    if not force and current_quality > 0 and new_quality <= current_quality:
        if existing_content:
            return existing_content, "existing"
        return None, None

    if not dry_run:
        atomic_write_bytes(lrc_path, lyrics_text.encode("utf-8"))

    tag_type = (
        "enhanced" if new_quality == 3 else ("synced" if new_quality == 2 else "plain")
    )
    return lyrics_text, tag_type
