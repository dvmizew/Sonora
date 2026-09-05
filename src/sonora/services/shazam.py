from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.config import get_config
from sonora.core.constants import RATE_LIMIT_SHAZAM
from sonora.core.logger import LOG
from sonora.core.utils import RateLimiter

_T = TypeVar("_T")
_SHAZAM_LIMITER = RateLimiter(interval_seconds=RATE_LIMIT_SHAZAM)

from shazamio_core.shazamio_core import SignatureError


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


def _run_async(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run an async coroutine synchronously, safely handling running event loops in any thread."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


async def _recognize_async(file_path: Path) -> dict[str, Any] | None:
    """Execute asynchronous acoustic recognition against the Shazam API."""
    try:
        from shazamio import Shazam

        shazam_client = Shazam()
        raw_result = await shazam_client.recognize(str(file_path.resolve()))
        return raw_result if isinstance(raw_result, dict) else None
    except (
        ImportError,
        OSError,
        ValueError,
        RuntimeError,
        TypeError,
        KeyError,
        AttributeError,
        SignatureError,
    ) as error:
        LOG.debug(f"Shazam recognition error for {file_path.name}: {error}")
        return None


async def _track_about_async(track_id: int) -> dict[str, Any] | None:
    """Fetch extended track details by track ID from Shazam."""
    try:
        from shazamio import Shazam

        shazam_client = Shazam()
        raw_result = await shazam_client.track_about(track_id)
        return raw_result if isinstance(raw_result, dict) else None
    except (
        ImportError,
        OSError,
        ValueError,
        RuntimeError,
        TypeError,
        KeyError,
        AttributeError,
    ) as error:
        LOG.debug(f"Shazam track_about error for {track_id}: {error}")
        return None


def get_shazam_track_about(track_id: int) -> dict[str, Any] | None:
    """Query extended track metadata directly from Shazam by track ID."""
    if track_id <= 0:
        return None
    cache_key = f"shazam_about:{track_id}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return cached

    _SHAZAM_LIMITER.wait()
    data = _run_async(_track_about_async(track_id))
    if data:
        set_cached_api(cache_key, data)
    return data


def recognize_audio_track(file_path: Path) -> ShazamTrackInfo | None:
    """
    Recognize an audio track via acoustic fingerprinting against the Shazam catalog.
    Returns ShazamTrackInfo with title, artist, album, genre, label, lyrics, and artwork URL or None if unmatched.
    """
    if not file_path.is_file() or file_path.stat().st_size == 0:
        return None

    if not get_config().enable_shazam:
        return None

    try:
        stat = file_path.stat()
        cache_key = f"shazam:{file_path.name}:{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        cache_key = f"shazam:{file_path.name}"

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
    raw_payload = _run_async(_recognize_async(file_path))
    if not raw_payload:
        return None

    track = raw_payload.get("track")
    if not isinstance(track, dict):
        return None

    title = track.get("title")
    artist = track.get("subtitle")
    if not title or not artist:
        return None

    album_title: str | None = None
    release_date: str | None = None
    label_name: str | None = None
    lyrics_text: str | None = None

    sections = track.get("sections")
    if isinstance(sections, list):
        for sec in sections:
            if isinstance(sec, dict):
                sec_type = sec.get("type")
                if sec_type == "SONG":
                    metadata = sec.get("metadata")
                    if isinstance(metadata, list):
                        for meta in metadata:
                            if isinstance(meta, dict):
                                meta_title = str(meta.get("title", "")).lower()
                                if meta_title == "album":
                                    album_title = meta.get("text")
                                elif meta_title in ("released", "release date"):
                                    release_date = meta.get("text")
                                elif meta_title in ("label", "record label"):
                                    label_name = meta.get("text")
                elif sec_type == "LYRICS":
                    lines = sec.get("text")
                    if isinstance(lines, list):
                        valid_lines = [
                            str(line).strip()
                            for line in lines
                            if isinstance(line, str) and line.strip()
                        ]
                        if valid_lines:
                            lyrics_text = "\n".join(valid_lines)

    genre_name: str | None = None
    genres_data = track.get("genres")
    if isinstance(genres_data, dict):
        genre_name = genres_data.get("primary")

    images_data = track.get("images")
    cover_art_url: str | None = None
    if isinstance(images_data, dict):
        cover_art_url = images_data.get("coverarthq") or images_data.get("coverart")

    apple_id: str | None = None
    hub = track.get("hub")
    if isinstance(hub, dict):
        actions = hub.get("actions")
        if isinstance(actions, list):
            for action in actions:
                if isinstance(action, dict) and action.get("id"):
                    apple_id = str(action["id"])
                    break
    if not apple_id and track.get("key"):
        apple_id = str(track["key"])

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
        lyrics=result_lyrics if (result_lyrics := lyrics_text) else None,
    )

    set_cached_api(
        cache_key,
        {
            "title": result.title,
            "artist": result.artist,
            "album": result.album,
            "genre": result.genre,
            "apple_music_id": result.apple_music_id,
            "isrc": result.isrc,
            "cover_art_url": result.cover_art_url,
            "release_date": result.release_date,
            "label": result.label,
            "lyrics": result.lyrics,
        },
    )
    return result
