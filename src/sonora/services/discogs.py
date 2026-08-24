import re
import threading
from typing import Any

import httpx

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.utils import RateLimiter, normalize_str

_DISCOGS_LIMITER = RateLimiter(interval_seconds=1.0)

_discogs_locks: dict[str, threading.Lock] = {}
_discogs_meta_lock = threading.Lock()


def _get_discogs_lock(key: str) -> threading.Lock:
    with _discogs_meta_lock:
        if len(_discogs_locks) > 1000:
            _discogs_locks.clear()
        if key not in _discogs_locks:
            _discogs_locks[key] = threading.Lock()
        return _discogs_locks[key]


def fetch_discogs_release_details(release_id: int | str, user_token: str) -> dict[str, Any] | None:
    if not release_id or not user_token:
        return None

    cache_key = f"discogs_rel:{release_id}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return cached

    with _get_discogs_lock(f"rel_{release_id}"):
        cached = get_cached_api(cache_key)
        if isinstance(cached, dict):
            return cached

        _DISCOGS_LIMITER.wait()
        try:
            url = f"https://api.discogs.com/releases/{release_id}"
            headers = {"Authorization": f"Discogs token={user_token}"}
            resp = SESSION.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                return None
            data = resp.json()
            if not isinstance(data, dict):
                return None

            # 1. Primary Artist ID
            artists = data.get("artists", [])
            artist_id = str(artists[0].get("id")) if isinstance(artists, list) and artists and isinstance(artists[0], dict) and artists[0].get("id") else None

            # 2. Labels & Catalog Number
            labels = data.get("labels", [])
            label_name = None
            cat_no = None
            if isinstance(labels, list) and labels and isinstance(labels[0], dict):
                label_name = str(labels[0].get("name")) if labels[0].get("name") else None
                cat_no = str(labels[0].get("catno")) if labels[0].get("catno") and labels[0].get("catno") != "none" else None

            # 3. Barcode & Matrix Identifiers
            barcode_val = None
            identifiers = data.get("identifiers", [])
            if isinstance(identifiers, list):
                for ident in identifiers:
                    if isinstance(ident, dict) and ident.get("type") == "Barcode":
                        raw_bc = str(ident.get("value", "")).replace(" ", "").replace("-", "")
                        if raw_bc:
                            barcode_val = raw_bc
                            break

            # 4. Media & Format Details (e.g., "Vinyl, LP, 180g" or "CD, Album, Deluxe Edition")
            media_format = None
            formats = data.get("formats", [])
            if isinstance(formats, list) and formats and isinstance(formats[0], dict):
                f0 = formats[0]
                fmt_parts: list[str] = []
                if f0.get("name"):
                    fmt_parts.append(str(f0["name"]))
                descs = f0.get("descriptions", [])
                if isinstance(descs, list):
                    fmt_parts.extend(str(d) for d in descs if d)
                if fmt_parts:
                    media_format = ", ".join(fmt_parts)

            # 5. Production & Engineering Credits
            producers: list[str] = []
            remixers: list[str] = []
            composers: list[str] = []
            extra_artists = data.get("extraartists", [])
            if isinstance(extra_artists, list):
                for ea in extra_artists:
                    if isinstance(ea, dict) and ea.get("name") and ea.get("role"):
                        name = re.sub(r"\s*\(\d+\)$", "", str(ea["name"])).strip()
                        role = str(ea["role"]).lower()
                        if ("producer" in role or "produced by" in role) and name not in producers:
                            producers.append(name)
                        elif "remix" in role and name not in remixers:
                            remixers.append(name)
                        elif any(kw in role for kw in ("written", "composer", "music by", "words by")) and name not in composers:
                            composers.append(name)

            # 6. Tracklist Level Credits
            track_credits: dict[str, dict[str, str]] = {}
            tracklist = data.get("tracklist", [])
            if isinstance(tracklist, list):
                for t in tracklist:
                    if isinstance(t, dict):
                        pos = str(t.get("position", "")).strip()
                        t_title = str(t.get("title", "")).strip()
                        t_prods: list[str] = []
                        t_remix: list[str] = []
                        for tea in t.get("extraartists", []):
                            if isinstance(tea, dict) and tea.get("name") and tea.get("role"):
                                tea_name = re.sub(r"\s*\(\d+\)$", "", str(tea["name"])).strip()
                                tea_role = str(tea["role"]).lower()
                                if ("producer" in tea_role or "produced by" in tea_role) and tea_name not in t_prods:
                                    t_prods.append(tea_name)
                                elif "remix" in tea_role and tea_name not in t_remix:
                                    t_remix.append(tea_name)
                        c_dict: dict[str, str] = {}
                        if t_prods:
                            c_dict["producers"] = ", ".join(t_prods)
                        if t_remix:
                            c_dict["remixer"] = ", ".join(t_remix)
                        if pos:
                            track_credits[pos] = c_dict
                        if t_title:
                            track_credits[t_title.lower()] = c_dict

            res: dict[str, Any] = {
                "id": data.get("id"),
                "artist_id": artist_id,
                "title": data.get("title"),
                "year": data.get("year"),
                "released": data.get("released"),
                "genres": list(data.get("genres", []) or []),
                "styles": list(data.get("styles", []) or []),
                "country": data.get("country"),
                "label": label_name,
                "catalog_number": cat_no,
                "barcode": barcode_val,
                "media": media_format,
                "producers": ", ".join(producers) if producers else None,
                "remixer": ", ".join(remixers) if remixers else None,
                "composer": ", ".join(composers) if composers else None,
                "track_credits": track_credits,
            }
            set_cached_api(cache_key, res, expire_seconds=2419200)  # 30 days
            return res
        except (httpx.HTTPError, OSError, ValueError, KeyError) as e:
            LOG.debug(f"Discogs release fetch failed for ID {release_id}: {e}")
            return None


def search_discogs_release(artist: str, album: str, user_token: str | None = None) -> dict[str, Any] | None:
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

    with _get_discogs_lock(artist_key):
        cached = get_cached_api(cache_key)
        if isinstance(cached, dict):
            return cached

        _DISCOGS_LIMITER.wait()
        try:
            url = "https://api.discogs.com/database/search"
            headers = {"Authorization": f"Discogs token={user_token}"}
            params = {
                "release_title": album,
                "artist": artist,
                "type": "release",
                "per_page": "5",
            }
            resp = SESSION.get(url, params=params, headers=headers, timeout=10)
            if resp.status_code != 200:
                return None
            data = resp.json()
            results = data.get("results", []) if isinstance(data, dict) else []
            if not results or not isinstance(results[0], dict):
                return None

            first = results[0]
            rel_id = first.get("id")
            if not rel_id:
                return None

            # Enrich with full release details
            full_details = fetch_discogs_release_details(rel_id, user_token)
            if full_details:
                set_cached_api(cache_key, full_details, expire_seconds=2419200)
                return full_details

            # Fallback to search result fields if full fetch failed
            labels = first.get("label", [])
            label_name = str(labels[0]) if isinstance(labels, list) and labels else None
            cat_no = str(first.get("catno")) if first.get("catno") else None
            barcodes = first.get("barcode", [])
            barcode_val = str(barcodes[0]) if isinstance(barcodes, list) and barcodes else None
            formats = first.get("format", [])
            media_format = ", ".join(str(f) for f in formats) if isinstance(formats, list) and formats else None

            res: dict[str, Any] = {
                "id": rel_id,
                "artist_id": None,
                "title": first.get("title"),
                "year": first.get("year"),
                "released": first.get("year"),
                "genres": list(first.get("genre", []) or []),
                "styles": list(first.get("style", []) or []),
                "country": first.get("country"),
                "label": label_name,
                "catalog_number": cat_no,
                "barcode": barcode_val,
                "media": media_format,
                "producers": None,
                "remixer": None,
                "composer": None,
                "track_credits": {},
            }
            set_cached_api(cache_key, res, expire_seconds=2419200)
            return res
        except (httpx.HTTPError, OSError, ValueError, KeyError) as e:
            LOG.debug(f"Discogs search failed for {artist} - {album}: {e}")
            return None
