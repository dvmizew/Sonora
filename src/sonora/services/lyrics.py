import hashlib
import json
import math
import urllib.parse
from contextlib import nullcontext
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
    lyrics_identity_matches,
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
        raise
    finally:
        client.session.close()


def _query_lrclib(
    query_str: str,
    plain_only: bool = False,
    synced_only: bool = False,
    album_name: str | None = None,
    duration: float | None = None,
    details: dict[str, Any] | None = None,
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
            params["duration"] = duration
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
        if lyrics_payload.get("instrumental") is True:
            return None

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
        if album_name:
            provider_album = lyrics_payload.get("albumName")
            if not isinstance(provider_album, str) or normalize_str(
                provider_album
            ) != normalize_str(album_name):
                return None

        if details is not None:
            details["source"] = details["lookup_source"] = "LRCLIB"
            if isinstance(lyrics_payload.get("id"), int):
                details["record_id"] = lyrics_payload["id"]

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
    details: dict[str, Any] | None = None,
) -> str | None:
    checked_lrclib = (not providers or "Lrclib" in providers) and not lang
    provider_error: Exception | None = None
    if checked_lrclib:
        try:
            lrclib_result = _query_lrclib(
                query_str,
                plain_only=plain_only,
                synced_only=synced_only,
                album_name=album_name,
                duration=duration,
                **({"details": details} if details is not None else {}),
            )
            if lrclib_result:
                return lrclib_result
        except (httpx.HTTPError, OSError) as error:
            provider_error = error

    if not providers and not get_config().allow_unverified_lyrics:
        if provider_error:
            raise provider_error
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
            if provider_error:
                raise provider_error
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
        try:
            result = _query_authenticated_musixmatch(
                query_str, musixmatch_token, plain_only, synced_only, enhanced, lang
            )
        except (httpx.HTTPError, OSError, ValueError) as error:
            provider_error = error
        if result is not None and details is not None:
            details["source"] = details["lookup_source"] = "Musixmatch"
        kwargs["providers"] = [
            provider for provider in selected_providers if provider != "Musixmatch"
        ]
        if lang and result is None:
            if provider_error:
                raise provider_error
            return None
    if result is None and not kwargs.get("providers") and musixmatch_token:
        if provider_error:
            raise provider_error
        return None
    if result is None:
        if details is not None:
            details["source"] = details["lookup_source"] = "syncedlyrics"
            details["providers"] = kwargs.get("providers", selected_providers)
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
    if provider_error:
        raise provider_error
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
    details: dict[str, Any] | None = None,
    lookup_state: dict[str, Any] | None = None,
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
    cache_key = f"lyrics:v9:{get_config().allow_unverified_lyrics}:{normalize_str(artist)}:{normalize_str(title)}:{isrc or ''}:{constraints}"
    lookup_details = details if details is not None else {}
    cached = get_cached_api(cache_key)
    cached_text = cached.get("text") if isinstance(cached, dict) else cached
    if isinstance(cached_text, str):
        if not cached_text:
            lookup_details.update(
                cache_hit=True,
                lookup_status="unavailable",
                source="cache",
                lookup_source="cache",
            )
            return None
        cached_text = clean_lyrics_text(cached_text)
        cached_quality = get_lyrics_quality(cached_text)
        if (
            cached_quality > 0
            and lyrics_identity_matches(cached_text, artist, title)
            and is_lyrics_duration_valid(cached_text, duration)
            and (not synced_only or cached_quality >= 2)
            and (not plain_only or cached_quality == 1)
        ):
            lookup_details.update(cache_hit=True, lookup_status="cache", source="cache")
            if isinstance(cached, dict):
                lookup_details.update(
                    {
                        key: cached[key]
                        for key in ("source", "record_id", "providers")
                        if key in cached
                    }
                )
            lookup_details["lookup_source"] = lookup_details["source"]
            return cached_text

    with lookup_state["lock"] if lookup_state is not None else nullcontext():
        if lookup_state is not None and lookup_state["suspended"]:
            lookup_details.update(
                cache_hit=False,
                lookup_status="deferred",
                lookup_reason="Network lookups suspended after repeated provider failures; retry in a later run",
            )
            return None
        queries = [f"{artist.lower()} - {title.lower()}".strip()]
        if "(" in title or "[" in title or "feat" in title.lower():
            cleaned_track_title = clean_title(title)
            primary_artist = get_primary_artist(artist)
            if (
                score_artist_title_match(
                    artist, title, primary_artist, cleaned_track_title
                )
                > 0
            ):
                alternative = f"{primary_artist} - {cleaned_track_title}".strip()
                if alternative != queries[0]:
                    queries.append(alternative)
        last_exception: Exception | None = None
        lyrics_content = None
        for query in queries:
            try:
                lyrics_content = _query_syncedlyrics(
                    query,
                    plain_only,
                    synced_only,
                    enhanced,
                    providers,
                    lang,
                    album_name=album_name,
                    duration=duration,
                    musixmatch_token=token,
                    details=lookup_details,
                )
            except (
                httpx.HTTPError,
                OSError,
                ValueError,
                RuntimeError,
                TimeoutError,
            ) as error:
                last_exception = error
            if lyrics_content or last_exception is not None:
                break

        if lyrics_content and not lyrics_identity_matches(
            lyrics_content, artist, title
        ):
            lyrics_content = None

        if lyrics_content:
            if lookup_state is not None:
                lookup_state["failures"] = 0
            lookup_details.update(cache_hit=False, lookup_status="found")
            set_cached_api(
                cache_key,
                {
                    "text": lyrics_content,
                    **{
                        key: lookup_details[key]
                        for key in ("source", "record_id", "providers")
                        if key in lookup_details
                    },
                },
            )
            return lyrics_content

        if last_exception:
            transient = isinstance(
                last_exception, (httpx.TransportError, OSError, TimeoutError)
            ) or (
                isinstance(last_exception, httpx.HTTPStatusError)
                and (
                    last_exception.response.status_code == 429
                    or last_exception.response.status_code >= 500
                )
            )
            lookup_details.update(
                cache_hit=False,
                lookup_status="failed",
                lookup_reason=(
                    f"Lyrics provider returned HTTP {last_exception.response.status_code}"
                    if isinstance(last_exception, httpx.HTTPStatusError)
                    else str(last_exception)
                ),
            )
            if transient and lookup_state is not None:
                lookup_state["failures"] += 1
                if lookup_state["failures"] >= 3:
                    lookup_state["suspended"] = True
                    LOG.warning(
                        "Lyrics provider keeps failing; network lookups are suspended for this run. Cached and local lyrics remain available."
                    )
            elif lookup_state is not None:
                lookup_state["failures"] = 0
            if isinstance(last_exception, httpx.HTTPStatusError):
                raise RuntimeError(
                    f"Lyrics provider returned HTTP {last_exception.response.status_code}; try again later"
                ) from last_exception
            raise RuntimeError(
                f"Lyrics fetch failed for '{title}': {last_exception}"
            ) from last_exception

        if lookup_state is not None:
            lookup_state["failures"] = 0
        lookup_details.update(cache_hit=False, lookup_status="unavailable")
        set_cached_api(cache_key, "")
        return None
