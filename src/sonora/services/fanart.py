from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.config import get_config
from sonora.core.constants import RATE_LIMIT_FANART
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.utils import RateLimiter, is_valid_uuid, safe_int

_FANART_LIMITER = RateLimiter(interval_seconds=RATE_LIMIT_FANART)
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
    if not isinstance(raw_items, list):
        return []

    sorted_items = sorted(
        raw_items,
        key=lambda image_entry: (
            safe_int(image_entry.get("likes")) or 0 if isinstance(image_entry, dict) else 0
        ),
        reverse=True,
    )
    urls: list[str] = []
    for image_entry in sorted_items:
        if isinstance(image_entry, dict):
            url = image_entry.get("url")
            if isinstance(url, str) and url.startswith("http"):
                urls.append(url)
    return urls


def download_fanart_image_bytes(url: str | None) -> bytes | None:
    """Download image payload from Fanart.tv with rate limiting and robust error handling."""
    if not url:
        return None
    _FANART_LIMITER.wait()
    try:
        response = SESSION.get(url, timeout=12)
        if response.status_code == 200:
            return response.content
    except (httpx.HTTPError, OSError) as error:
        LOG.debug(f"Failed to download fanart image from {url}: {error}")
    return None


def fetch_fanart_artist(
    artist_mbid: str,
    api_key: str | None = None,
    client_key: str | None = None,
) -> FanartArtistArtwork | None:
    """
    Fetch high-res artist logos, backgrounds, banners, and thumbnails from Fanart.tv.
    Returns FanartArtistArtwork or None if unavailable or unauthenticated.
    """
    if not artist_mbid or not is_valid_uuid(artist_mbid):
        return None

    config = get_config()
    effective_api_key = api_key or config.fanart_api_key
    effective_client_key = client_key or config.fanart_client_key

    if not effective_api_key:
        return None

    cache_key = f"fanart_artist:{artist_mbid}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return FanartArtistArtwork(
            logo_urls=tuple(cached.get("logo_urls", ())),
            background_urls=tuple(cached.get("background_urls", ())),
            banner_urls=tuple(cached.get("banner_urls", ())),
            thumb_urls=tuple(cached.get("thumb_urls", ())),
        )

    headers = {"api-key": effective_api_key}
    if effective_client_key:
        headers["client-key"] = effective_client_key

    url = f"{_BASE_URL}/{artist_mbid}"
    _FANART_LIMITER.wait()

    try:
        response = SESSION.get(url, headers=headers, timeout=10)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        fanart_payload = response.json()
        if not isinstance(fanart_payload, dict):
            return None

        logos = _extract_image_urls(fanart_payload.get("hdmusiclogo")) + _extract_image_urls(
            fanart_payload.get("musiclogo")
        )
        backgrounds = _extract_image_urls(fanart_payload.get("artistbackground"))
        banners = _extract_image_urls(fanart_payload.get("musicbanner"))
        thumbs = _extract_image_urls(fanart_payload.get("artistthumb"))

        artwork = FanartArtistArtwork(
            logo_urls=tuple(logos),
            background_urls=tuple(backgrounds),
            banner_urls=tuple(banners),
            thumb_urls=tuple(thumbs),
        )

        set_cached_api(
            cache_key,
            {
                "logo_urls": list(artwork.logo_urls),
                "background_urls": list(artwork.background_urls),
                "banner_urls": list(artwork.banner_urls),
                "thumb_urls": list(artwork.thumb_urls),
            },
        )
        return artwork
    except (httpx.HTTPError, OSError) as error:
        LOG.debug(f"Fanart.tv artist fetch failed for {artist_mbid}: {error}")
        return None


def fetch_fanart_label(
    label_mbid: str,
    api_key: str | None = None,
    client_key: str | None = None,
) -> FanartLabelArtwork | None:
    """
    Fetch record label logos from Fanart.tv by MusicBrainz Label ID.
    Returns FanartLabelArtwork or None if unavailable or unauthenticated.
    """
    if not label_mbid or not is_valid_uuid(label_mbid):
        return None

    config = get_config()
    effective_api_key = api_key or config.fanart_api_key
    effective_client_key = client_key or config.fanart_client_key

    if not effective_api_key:
        return None

    cache_key = f"fanart_label:{label_mbid}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return FanartLabelArtwork(
            label_urls=tuple(cached.get("label_urls", ())),
        )

    headers = {"api-key": effective_api_key}
    if effective_client_key:
        headers["client-key"] = effective_client_key

    url = f"{_BASE_URL}/labels/{label_mbid}"
    _FANART_LIMITER.wait()

    try:
        response = SESSION.get(url, headers=headers, timeout=10)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        fanart_payload = response.json()
        if not isinstance(fanart_payload, dict):
            return None

        labels = _extract_image_urls(fanart_payload.get("musiclabel"))
        artwork = FanartLabelArtwork(label_urls=tuple(labels))

        set_cached_api(
            cache_key,
            {"label_urls": list(artwork.label_urls)},
        )
        return artwork
    except (httpx.HTTPError, OSError) as error:
        LOG.debug(f"Fanart.tv label fetch failed for {label_mbid}: {error}")
        return None


def fetch_fanart_album(
    release_group_mbid: str,
    api_key: str | None = None,
    client_key: str | None = None,
) -> FanartAlbumArtwork | None:
    """
    Fetch CD art discs and album covers from Fanart.tv by MusicBrainz Release Group ID.
    Returns FanartAlbumArtwork or None if unavailable or unauthenticated.
    """
    if not release_group_mbid or not is_valid_uuid(release_group_mbid):
        return None

    config = get_config()
    effective_api_key = api_key or config.fanart_api_key
    effective_client_key = client_key or config.fanart_client_key

    if not effective_api_key:
        return None

    cache_key = f"fanart_album:{release_group_mbid}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return FanartAlbumArtwork(
            cdart_urls=tuple(cached.get("cdart_urls", ())),
            cover_urls=tuple(cached.get("cover_urls", ())),
        )

    headers = {"api-key": effective_api_key}
    if effective_client_key:
        headers["client-key"] = effective_client_key

    url = f"{_BASE_URL}/albums/{release_group_mbid}"
    _FANART_LIMITER.wait()

    try:
        response = SESSION.get(url, headers=headers, timeout=10)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        fanart_payload = response.json()
        if not isinstance(fanart_payload, dict):
            return None

        cdart_urls: list[str] = []
        cover_urls: list[str] = []

        album_fanart_payload = fanart_payload.get("albums")
        if isinstance(album_fanart_payload, dict):
            entry = album_fanart_payload.get(release_group_mbid) or next(
                iter(album_fanart_payload.values()), None
            )
            if isinstance(entry, dict):
                cdart_urls.extend(_extract_image_urls(entry.get("cdart")))
                cover_urls.extend(_extract_image_urls(entry.get("albumcover")))
        elif isinstance(album_fanart_payload, list):
            for entry in album_fanart_payload:
                if isinstance(entry, dict):
                    cdart_urls.extend(_extract_image_urls(entry.get("cdart")))
                    cover_urls.extend(_extract_image_urls(entry.get("albumcover")))
        else:
            cdart_urls.extend(_extract_image_urls(fanart_payload.get("cdart")))
            cover_urls.extend(_extract_image_urls(fanart_payload.get("albumcover")))

        artwork = FanartAlbumArtwork(
            cdart_urls=tuple(cdart_urls),
            cover_urls=tuple(cover_urls),
        )

        set_cached_api(
            cache_key,
            {
                "cdart_urls": list(artwork.cdart_urls),
                "cover_urls": list(artwork.cover_urls),
            },
        )
        return artwork
    except (httpx.HTTPError, OSError) as error:
        LOG.debug(f"Fanart.tv album fetch failed for {release_group_mbid}: {error}")
        return None
