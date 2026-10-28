import hashlib
import json
import math
import urllib.parse
from typing import Any

import httpx
import syncedlyrics
from syncedlyrics.providers import Musixmatch

from sonora.core.api_payloads import require_api_object
from sonora.core.artists import get_primary_artist
from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.config import get_config
from sonora.core.constants import RATE_LIMIT_LRCLIB, RATE_LIMIT_LYRICS, USER_AGENT
from sonora.core.coordination import coalesce_lookup
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.rate_limit import RateLimiter
from sonora.core.retry import retry_network_call
from sonora.core.sidecars import (
    clean_lyrics_text,
    get_lyrics_quality,
    is_lyrics_duration_valid,
)
from sonora.core.text import normalize_str
from sonora.core.titles import clean_title, score_artist_title_match

_LRCLIB_LIMITER = RateLimiter(interval_seconds=RATE_LIMIT_LRCLIB)
_LYRICS_LIMITER = RateLimiter(interval_seconds=RATE_LIMIT_LYRICS)


def parse_musixmatch_token(raw_token: str | None) -> str | None:
    """Accept a user token or exported cookie without persisting SDK cache files."""
    if not raw_token or not raw_token.strip():
        return None
    decoded_token = urllib.parse.unquote(raw_token.strip())
    if not decoded_token.startswith("{"):
        return decoded_token
    try:
        token_payload = require_api_object(
            json.loads(decoded_token), "Musixmatch credentials"
        )
    except json.JSONDecodeError as error:
        raise ValueError("Musixmatch credentials contain invalid JSON") from error
    credential_fields = require_api_object(
        token_payload.get("tokens", token_payload), "Musixmatch token fields"
    )
    for field_name in (
        "web-desktop-app-v1.0",
        "mxm-account-v1.0",
        "user_token",
        "token",
    ):
        token_value = credential_fields.get(field_name)
        if isinstance(token_value, str) and token_value.strip():
            return token_value.strip()
    raise ValueError("Musixmatch credentials do not contain a user token")


def _query_authenticated_musixmatch(
    query: str,
    token: str,
    plain_only: bool,
    synced_only: bool,
    enhanced: bool,
    lang: str | None,
) -> str | None:
    # The SDK exposes the token on each client; no cache schema or expiry is owned here.
    client = Musixmatch(lang=lang, enhanced=enhanced)
    client.token = token
    try:
        _LYRICS_LIMITER.wait()
        lyrics_result = client.get_lrc(query)
        if lyrics_result is None:
            return None
        if plain_only:
            return str(lyrics_result.unsynced) if lyrics_result.unsynced else None
        if lyrics_result.synced:
            return str(lyrics_result.synced)
        return (
            str(lyrics_result.unsynced)
            if not synced_only and lyrics_result.unsynced
            else None
        )
    except (OSError, ValueError) as error:
        LOG.debug(f"Musixmatch lyrics request failed: {error}")
        return None
    finally:
        client.session.close()


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
            if response.status_code != 404:
                response.raise_for_status()
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
            or normalize_str(clean_title(str(provider_title)))
            != normalize_str(clean_title(track_name))
            or score_artist_title_match(
                artist_name, track_name, str(provider_artist), str(provider_title)
            )
            <= 0
        ):
            return None
        provider_duration = lyrics_payload.get("duration")
        if duration is not None and (
            isinstance(provider_duration, bool)
            or not isinstance(provider_duration, (int, float))
            or not math.isfinite(provider_duration)
            or provider_duration <= 0
            or abs(provider_duration - duration)
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
        raise
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
    musixmatch_token: str | None = None,
) -> str | None:
    checked_lrclib = (not providers or "Lrclib" in providers) and not lang
    lrclib_error: Exception | None = None
    if checked_lrclib:
        try:
            lrclib_result = _query_lrclib(
                query_str,
                plain_only=plain_only,
                synced_only=synced_only,
                album_name=album_name,
                duration=duration,
            )
            if lrclib_result:
                return lrclib_result
        except (httpx.HTTPError, OSError) as error:
            lrclib_error = error

    if not providers and not get_config().allow_unverified_lyrics:
        if lrclib_error:
            raise lrclib_error
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
            if lrclib_error:
                raise lrclib_error
            return None
        kwargs["providers"] = fallback_providers
    elif providers is not None:
        kwargs["providers"] = providers
    if lang:
        kwargs["lang"] = lang

    selected_providers = kwargs.get("providers") or [
        "Musixmatch",
        "Lrclib",
        "NetEase",
        "Megalobiz",
        "Genius",
    ]
    result: str | None = None
    if musixmatch_token and "Musixmatch" in selected_providers:
        result = _query_authenticated_musixmatch(
            query_str, musixmatch_token, plain_only, synced_only, enhanced, lang
        )
        kwargs["providers"] = [
            provider for provider in selected_providers if provider != "Musixmatch"
        ]
        if lang and result is None:
            return None
    if result is None and not kwargs.get("providers") and musixmatch_token:
        if lrclib_error:
            raise lrclib_error
        return None
    if result is None:
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
    if lrclib_error:
        raise lrclib_error
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
    musixmatch_token: str | None = None,
) -> str | None:
    """
    Search and fetch LRC lyrics for a track using syncedlyrics.
    Quality preference order: Enhanced (word-synced) -> Line-synced -> Plain text.
    """
    if not artist or not title or (plain_only and synced_only):
        return None

    token = parse_musixmatch_token(musixmatch_token or get_config().musixmatch_token)

    constraints = json.dumps(
        [
            plain_only,
            synced_only,
            enhanced,
            sorted(providers or []),
            lang,
            album_name,
            duration,
            get_config().single_duration_tolerance_seconds,
            hashlib.sha256(token.encode()).hexdigest() if token else None,
        ],
        ensure_ascii=False,
    )
    cache_key = f"lyrics:v8:{get_config().allow_unverified_lyrics}:{normalize_str(artist)}:{normalize_str(title)}:{isrc or ''}:{constraints}"
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
            default_query,
            *search_args,
            album_name=album_name,
            duration=duration,
            musixmatch_token=token,
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
                query,
                *search_args,
                album_name=album_name,
                duration=duration,
                musixmatch_token=token,
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
