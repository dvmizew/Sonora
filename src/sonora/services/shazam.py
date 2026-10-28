from __future__ import annotations

import asyncio
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from stat import S_ISREG
from typing import Any

import aiohttp
from shazamio import Shazam
from shazamio.exceptions import (
    BadContentType,
    BadParseData,
    BadResponseStatus,
    FailedDecodeJson,
)
from shazamio_core import SignatureError

from sonora.core.api_payloads import (
    require_api_object,
    require_api_records,
    require_api_strings,
)
from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.config import get_config
from sonora.core.constants import RATE_LIMIT_SHAZAM
from sonora.core.coordination import coalesce_lookup
from sonora.core.logger import LOG
from sonora.core.rate_limit import RateLimiter

_SHAZAM_LIMITER = RateLimiter(interval_seconds=RATE_LIMIT_SHAZAM)


@dataclass(frozen=True)
class ShazamTrackInfo:
    title: str
    artist: str
    album: str | None = None
    genre: str | None = None
    apple_music_id: str | None = None
    isrc: str | None = None
    cover_art_url: str | None = None
    release_date: str | None = None
    label: str | None = None
    lyrics: str | None = None


async def _recognize_async(file_path: Path) -> dict[str, Any] | None:
    """Execute asynchronous acoustic recognition against the Shazam API."""
    try:
        async with Shazam() as shazam_client:
            raw_result = await asyncio.wait_for(
                shazam_client.recognize(file_path.resolve()), timeout=30.0
            )
        return require_api_object(raw_result, "Shazam recognition response")
    except (
        aiohttp.ClientError,
        OSError,
        SignatureError,
        BadParseData,
        BadContentType,
        BadResponseStatus,
        FailedDecodeJson,
        asyncio.TimeoutError,
    ) as error:
        LOG.debug(f"Shazam recognition error for {file_path.name}: {error}")
        return None


@coalesce_lookup
def recognize_audio_track(file_path: Path) -> ShazamTrackInfo | None:
    """
    Recognize an audio track via acoustic fingerprinting against the Shazam catalog.
    Returns ShazamTrackInfo with title, artist, album, genre, label, lyrics, and artwork URL or None if unmatched.
    """
    if not get_config().enable_shazam:
        return None

    try:
        audio_stat = file_path.stat()
        if not S_ISREG(audio_stat.st_mode) or audio_stat.st_size == 0:
            return None
        path_digest = hashlib.sha256(str(file_path.resolve()).encode()).hexdigest()
        cache_key = (
            f"shazam:v2:{path_digest}:{audio_stat.st_size}:{audio_stat.st_mtime_ns}"
        )
    except OSError as error:
        LOG.debug(f"Cannot inspect audio for recognition: {file_path.name}: {error}")
        return None

    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return ShazamTrackInfo(
            title=str(cached.get("title", "")),
            artist=str(cached.get("artist", "")),
            album=cached.get("album"),
            genre=cached.get("genre"),
            apple_music_id=cached.get("apple_music_id"),
            isrc=cached.get("isrc"),
            cover_art_url=cached.get("cover_art_url"),
            release_date=cached.get("release_date"),
            label=cached.get("label"),
            lyrics=cached.get("lyrics"),
        )

    _SHAZAM_LIMITER.wait()
    raw_payload = asyncio.run(_recognize_async(file_path))
    if not raw_payload:
        return None

    track_payload = raw_payload.get("track")
    if track_payload is None:
        return None
    track = require_api_object(track_payload, "Shazam track")

    title = track.get("title")
    artist = track.get("subtitle")
    if (
        not isinstance(title, str)
        or not title.strip()
        or not isinstance(artist, str)
        or not artist.strip()
    ):
        raise TypeError("Shazam track must contain nonempty title and subtitle strings")

    album_title: str | None = None
    release_date: str | None = None
    label_name: str | None = None
    lyrics_text: str | None = None

    sections = require_api_records(track.get("sections", []), "Shazam track sections")
    for section in sections:
        if section.get("type") == "SONG":
            credits = require_api_records(
                section.get("metadata", []), "Shazam song metadata"
            )
            for credit in credits:
                credit_title = str(credit.get("title", "")).lower()
                if credit_title == "album":
                    album_title = credit.get("text")
                elif credit_title in ("released", "release date"):
                    release_date = credit.get("text")
                elif credit_title in ("label", "record label"):
                    label_name = credit.get("text")
        elif section.get("type") == "LYRICS":
            lyric_lines = require_api_strings(section.get("text", []), "Shazam lyrics")
            lyrics_text = (
                "\n".join(line.strip() for line in lyric_lines if line.strip()) or None
            )

    genres = require_api_object(track.get("genres", {}), "Shazam genres")
    genre_name = genres.get("primary")
    images = require_api_object(track.get("images", {}), "Shazam images")
    cover_art_url = images.get("coverarthq") or images.get("coverart")

    apple_id: str | None = None
    hub = require_api_object(track.get("hub", {}), "Shazam track hub")
    actions = require_api_records(hub.get("actions", []), "Shazam hub actions")
    for action in actions:
        if action.get("type") == "applemusicplay" and action.get("id"):
            apple_id = str(action["id"])
            break
    isrc: str | None = track.get("isrc")

    result = ShazamTrackInfo(
        title=str(title).strip(),
        artist=str(artist).strip(),
        album=str(album_title).strip() if album_title else None,
        genre=str(genre_name).strip() if genre_name else None,
        apple_music_id=apple_id,
        isrc=str(isrc).strip() if isrc else None,
        cover_art_url=str(cover_art_url).strip() if cover_art_url else None,
        release_date=str(release_date).strip() if release_date else None,
        label=str(label_name).strip() if label_name else None,
        lyrics=lyrics_text,
    )

    set_cached_api(cache_key, asdict(result))
    return result
