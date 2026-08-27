import re
from typing import Any

import httpx

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.constants import (
    RATE_LIMIT_DISCOGS_AUTHENTICATED,
    RATE_LIMIT_DISCOGS_UNAUTHENTICATED,
)
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.utils import RateLimiter, normalize_str

_DISCOGS_LIMITER = RateLimiter(interval_seconds=RATE_LIMIT_DISCOGS_AUTHENTICATED)


def fetch_discogs_release_details(
    release_id: int | str, user_token: str | None = None
) -> dict[str, Any] | None:
    if not release_id:
        return None

    cache_key = f"discogs_rel:{release_id}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return cached

    _DISCOGS_LIMITER.wait(
        RATE_LIMIT_DISCOGS_AUTHENTICATED
        if user_token
        else RATE_LIMIT_DISCOGS_UNAUTHENTICATED
    )
    try:
        url = f"https://api.discogs.com/releases/{release_id}"
        headers = {"Authorization": f"Discogs token={user_token}"} if user_token else {}
        response = SESSION.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
        data = response.json()
        if not isinstance(data, dict):
            return None

        # 1. Primary Artist ID
        artists = data.get("artists", [])
        artist_id = (
            str(artists[0].get("id"))
            if isinstance(artists, list)
            and artists
            and isinstance(artists[0], dict)
            and artists[0].get("id")
            else None
        )

        # 2. Labels & Catalog Number
        labels = data.get("labels", [])
        label_name = None
        catalog_number = None
        if isinstance(labels, list) and labels and isinstance(labels[0], dict):
            label_name = str(labels[0].get("name")) if labels[0].get("name") else None
            catalog_number = (
                str(labels[0].get("catno"))
                if labels[0].get("catno") and labels[0].get("catno") != "none"
                else None
            )

        # 3. Barcode & Matrix Identifiers
        barcode_value = None
        identifiers = data.get("identifiers", [])
        if isinstance(identifiers, list):
            for identifier in identifiers:
                if isinstance(identifier, dict) and identifier.get("type") == "Barcode":
                    raw_barcode = (
                        str(identifier.get("value", ""))
                        .replace(" ", "")
                        .replace("-", "")
                    )
                    if raw_barcode:
                        barcode_value = raw_barcode
                        break

        # 4. Media & Format Details (e.g., "Vinyl, LP, 180g" or "CD, Album, Deluxe Edition")
        media_format = None
        formats = data.get("formats", [])
        if isinstance(formats, list) and formats and isinstance(formats[0], dict):
            first_format = formats[0]
            format_parts: list[str] = []
            if first_format.get("name"):
                format_parts.append(str(first_format["name"]))
            descriptions = first_format.get("descriptions", [])
            if isinstance(descriptions, list):
                format_parts.extend(str(desc) for desc in descriptions if desc)
            if format_parts:
                media_format = ", ".join(format_parts)

        # 5. Production & Engineering Credits
        producers: list[str] = []
        remixers: list[str] = []
        composers: list[str] = []
        extra_artists = data.get("extraartists", [])
        if isinstance(extra_artists, list):
            for extra_artist in extra_artists:
                if (
                    isinstance(extra_artist, dict)
                    and extra_artist.get("name")
                    and extra_artist.get("role")
                ):
                    name = re.sub(r"\s*\(\d+\)$", "", str(extra_artist["name"])).strip()
                    role = str(extra_artist["role"]).lower()
                    if (
                        "producer" in role or "produced by" in role
                    ) and name not in producers:
                        producers.append(name)
                    elif "remix" in role and name not in remixers:
                        remixers.append(name)
                    elif (
                        any(
                            kw in role
                            for kw in (
                                "written",
                                "composer",
                                "music by",
                                "words by",
                            )
                        )
                        and name not in composers
                    ):
                        composers.append(name)

        # 6. Tracklist Level Credits
        track_credits: dict[str, dict[str, str]] = {}
        tracklist = data.get("tracklist", [])
        if isinstance(tracklist, list):
            for track in tracklist:
                if isinstance(track, dict):
                    position = str(track.get("position", "")).strip()
                    track_title = str(track.get("title", "")).strip()
                    track_producers: list[str] = []
                    track_remixers: list[str] = []
                    for track_extra_artist in track.get("extraartists", []):
                        if (
                            isinstance(track_extra_artist, dict)
                            and track_extra_artist.get("name")
                            and track_extra_artist.get("role")
                        ):
                            track_artist_name = re.sub(
                                r"\s*\(\d+\)$", "", str(track_extra_artist["name"])
                            ).strip()
                            track_artist_role = str(track_extra_artist["role"]).lower()
                            if (
                                "producer" in track_artist_role
                                or "produced by" in track_artist_role
                            ) and track_artist_name not in track_producers:
                                track_producers.append(track_artist_name)
                            elif (
                                "remix" in track_artist_role
                                and track_artist_name not in track_remixers
                            ):
                                track_remixers.append(track_artist_name)
                    credits_dict: dict[str, str] = {}
                    if track_producers:
                        credits_dict["producers"] = ", ".join(track_producers)
                    if track_remixers:
                        credits_dict["remixer"] = ", ".join(track_remixers)
                    if position:
                        track_credits[position] = credits_dict
                    if track_title:
                        track_credits[track_title.lower()] = credits_dict

        release_result: dict[str, Any] = {
            "id": data.get("id"),
            "artist_id": artist_id,
            "title": data.get("title"),
            "year": data.get("year"),
            "released": data.get("released"),
            "genres": list(data.get("genres", []) or []),
            "styles": list(data.get("styles", []) or []),
            "country": data.get("country"),
            "label": label_name,
            "catalog_number": catalog_number,
            "barcode": barcode_value,
            "media": media_format,
            "producers": ", ".join(producers) if producers else None,
            "remixer": ", ".join(remixers) if remixers else None,
            "composer": ", ".join(composers) if composers else None,
            "track_credits": track_credits,
        }
        set_cached_api(cache_key, release_result)
        return release_result
    except (httpx.HTTPError, OSError, ValueError, KeyError) as error:
        LOG.debug(f"Discogs release fetch failed for ID {release_id}: {error}")
        return None


def search_discogs_release(
    artist: str, album: str, user_token: str | None = None
) -> dict[str, Any] | None:
    """
    Search Discogs for release metadata using official REST API and User token.
    Fetches full canonical release metadata if matched.
    Returns a dict with complete metadata if found, otherwise None.
    """
    if not user_token or not artist or not album:
        return None

    if normalize_str(album) in ["unknown album", "unknown"]:
        return None

    artist_key = normalize_str(artist)
    cache_key = f"discogs_search:{artist_key}:{normalize_str(album)}"

    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return cached

    _DISCOGS_LIMITER.wait(
        RATE_LIMIT_DISCOGS_AUTHENTICATED
        if user_token
        else RATE_LIMIT_DISCOGS_UNAUTHENTICATED
    )
    try:
        url = "https://api.discogs.com/database/search"
        headers = {"Authorization": f"Discogs token={user_token}"}
        params = {
            "release_title": album,
            "artist": artist,
            "type": "release",
            "per_page": "5",
        }
        response = SESSION.get(url, params=params, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
        data = response.json()
        results = data.get("results", []) if isinstance(data, dict) else []
        if not results or not isinstance(results[0], dict):
            return None

        first = results[0]
        release_id = first.get("id")
        if not release_id:
            return None

        # Enrich with full release details
        full_details = fetch_discogs_release_details(release_id, user_token)
        if full_details:
            set_cached_api(cache_key, full_details)
            return full_details

        # Fallback to search result fields if full fetch failed
        labels = first.get("label", [])
        label_name = str(labels[0]) if isinstance(labels, list) and labels else None
        catalog_number = str(first.get("catno")) if first.get("catno") else None
        barcodes = first.get("barcode", [])
        barcode_value = (
            str(barcodes[0]) if isinstance(barcodes, list) and barcodes else None
        )
        formats = first.get("format", [])
        media_format = (
            ", ".join(str(format_item) for format_item in formats)
            if isinstance(formats, list) and formats
            else None
        )

        release_result: dict[str, Any] = {
            "id": release_id,
            "artist_id": None,
            "title": first.get("title"),
            "year": first.get("year"),
            "released": first.get("year"),
            "genres": list(first.get("genre", []) or []),
            "styles": list(first.get("style", []) or []),
            "country": first.get("country"),
            "label": label_name,
            "catalog_number": catalog_number,
            "barcode": barcode_value,
            "media": media_format,
            "producers": None,
            "remixer": None,
            "composer": None,
            "track_credits": {},
        }
        set_cached_api(cache_key, release_result)
        return release_result
    except (httpx.HTTPError, OSError, ValueError, KeyError) as error:
        LOG.debug(f"Discogs search failed for {artist} - {album}: {error}")
        return None
