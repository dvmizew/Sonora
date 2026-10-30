from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from sonora.core.api_payloads import require_api_object, require_api_records
from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.config import get_config
from sonora.core.coordination import coalesce_lookup
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.text import is_valid_uuid, safe_int

_BASE_URL = "https://webservice.fanart.tv/v3.2/music"


@dataclass(frozen=True)
class FanartArtistArtwork:
    logo_urls: tuple[str, ...] = ()
    background_urls: tuple[str, ...] = ()
    banner_urls: tuple[str, ...] = ()
    thumb_urls: tuple[str, ...] = ()


@dataclass(frozen=True)
class FanartLabelArtwork:
    label_urls: tuple[str, ...] = ()


@dataclass(frozen=True)
class FanartAlbumArtwork:
    cdart_urls: tuple[str, ...] = ()
    cover_urls: tuple[str, ...] = ()


def _extract_image_urls(raw_items: Any) -> list[str]:
    """Extract valid image URLs from a Fanart.tv array of image descriptors sorted by likes."""
    if raw_items is None:
        return []
    image_records = require_api_records(raw_items, "Fanart image descriptors")

    sorted_items = sorted(
        image_records,
        key=lambda image_entry: safe_int(image_entry.get("likes")) or 0,
        reverse=True,
    )
    urls: list[str] = []
    for image_entry in sorted_items:
        url = image_entry.get("url")
        if isinstance(url, str) and url.startswith("http"):
            urls.append(url)
    return urls


def download_fanart_image_bytes(url: str | None) -> bytes | None:
    """Download image payload from Fanart.tv with rate limiting and robust error handling."""
    if not url:
        return None

    try:
        response = SESSION.get(url, timeout=12)
        if response.status_code == 200:
            return response.content
    except (httpx.HTTPError, OSError) as error:
        LOG.debug(f"Failed to download fanart image from {url}: {error}")
    return None


def _fetch_artwork_urls(
    identifier: str,
    kind: str,
    fields: dict[str, tuple[str, ...]],
    api_key: str | None,
    client_key: str | None,
) -> dict[str, tuple[str, ...]] | None:
    if not is_valid_uuid(identifier):
        return None
    config = get_config()
    effective_api_key = api_key or config.fanart_api_key
    effective_client_key = client_key or config.fanart_client_key
    if not effective_api_key:
        return None

    cache_key = f"fanart_{kind}:{identifier}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return {field: tuple(cached.get(field, ())) for field in fields}
    headers = {"api-key": effective_api_key}
    if effective_client_key:
        headers["client-key"] = effective_client_key
    section = {"artist": "", "album": "albums/", "label": "labels/"}[kind]
    try:
        response = SESSION.get(
            f"{_BASE_URL}/{section}{identifier}", headers=headers, timeout=10
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        payload = require_api_object(response.json(), "Fanart response")
        entries = [payload]
        if kind == "album":
            albums = payload.get("albums")
            if isinstance(albums, dict):
                entry = albums.get(identifier)
                entries = [entry] if isinstance(entry, dict) else []
            elif isinstance(albums, list):
                entries = [
                    entry
                    for entry in albums
                    if isinstance(entry, dict)
                    and entry.get("release_group_id") == identifier
                ]
        urls = {
            field: tuple(
                url
                for entry in entries
                for image_type in image_types
                for url in _extract_image_urls(entry.get(image_type))
            )
            for field, image_types in fields.items()
        }
        set_cached_api(
            cache_key, {field: list(values) for field, values in urls.items()}
        )
        return urls
    except (httpx.HTTPError, OSError) as error:
        LOG.debug(f"Fanart.tv {kind} fetch failed for {identifier}: {error}")
        return None


@coalesce_lookup
def fetch_fanart_artist(
    artist_mbid: str,
    api_key: str | None = None,
    client_key: str | None = None,
) -> FanartArtistArtwork | None:
    """Fetch artist logos, backgrounds, banners and thumbnails by MusicBrainz ID."""
    urls = _fetch_artwork_urls(
        artist_mbid,
        "artist",
        {
            "logo_urls": ("hdmusiclogo", "musiclogo"),
            "background_urls": ("artistbackground",),
            "banner_urls": ("musicbanner",),
            "thumb_urls": ("artistthumb",),
        },
        api_key,
        client_key,
    )
    return FanartArtistArtwork(**urls) if urls is not None else None


@coalesce_lookup
def fetch_fanart_label(
    label_mbid: str,
    api_key: str | None = None,
    client_key: str | None = None,
) -> FanartLabelArtwork | None:
    """Fetch record label logos by MusicBrainz Label ID."""
    urls = _fetch_artwork_urls(
        label_mbid, "label", {"label_urls": ("musiclabel",)}, api_key, client_key
    )
    return FanartLabelArtwork(**urls) if urls is not None else None


@coalesce_lookup
def fetch_fanart_album(
    release_group_mbid: str,
    api_key: str | None = None,
    client_key: str | None = None,
) -> FanartAlbumArtwork | None:
    """Fetch disc art and album covers by MusicBrainz Release Group ID."""
    urls = _fetch_artwork_urls(
        release_group_mbid,
        "album",
        {"cdart_urls": ("cdart",), "cover_urls": ("albumcover",)},
        api_key,
        client_key,
    )
    return FanartAlbumArtwork(**urls) if urls is not None else None
