from collections import Counter
from typing import Any

import httpx
from rich.markup import escape

from sonora.core.api_payloads import (
    require_api_object,
    require_api_records,
    require_api_strings,
)
from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.config import get_config
from sonora.core.coordination import coalesce_lookup
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.text import clean_disambiguation, normalize_country_name, normalize_str
from sonora.core.titles import score_artist_title_match


def _count_discogs_audio_tracks(tracklist: list[dict[str, Any]]) -> int:
    """Count playable entries, excluding index headings and documented video positions."""
    audio_track_count = 0
    for track in tracklist:
        sub_tracks = track.get("sub_tracks")
        if sub_tracks:
            audio_track_count += _count_discogs_audio_tracks(
                require_api_records(sub_tracks, "Discogs sub-tracks")
            )
        elif (
            track.get("type_", "track") == "track"
            and str(track.get("position") or "").strip()
            and not str(track["position"]).casefold().startswith("video")
        ):
            audio_track_count += 1
    return audio_track_count


def _discogs_credit_positions(scope: str, positions: list[str]) -> set[str]:
    """Resolve documented credit scopes against actual, unambiguous tracklist positions."""
    position_counts = Counter(positions)
    selected_positions: set[str] = set()
    for selection in scope.split(","):
        start, separator, end = selection.strip().partition(" to ")
        if position_counts[start] != 1 or (separator and position_counts[end] != 1):
            return set()
        if not separator:
            selected_positions.add(start)
            continue
        start_index = positions.index(start)
        end_index = positions.index(end)
        if end_index < start_index:
            return set()
        position_range = positions[start_index : end_index + 1]
        if any(position_counts[position] != 1 for position in position_range):
            return set()
        selected_positions.update(position_range)
    return selected_positions


def _add_discogs_credit(
    credit: dict[str, Any],
    producers: list[str],
    remixers: list[str],
    composers: list[str],
) -> None:
    name = clean_disambiguation(str(credit["name"]))
    role = str(credit["role"]).lower()
    if ("producer" in role or "produced by" in role) and name not in producers:
        producers.append(name)
    elif "remix" in role and name not in remixers:
        remixers.append(name)
    elif (
        any(
            keyword in role
            for keyword in ("written", "composer", "music by", "words by")
        )
        and name not in composers
    ):
        composers.append(name)


@coalesce_lookup
def fetch_discogs_release_details(
    release_id: int | str, user_token: str | None = None
) -> dict[str, Any] | None:
    if not release_id:
        return None

    cache_key = f"discogs_rel:v3:{release_id}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return cached
    if cached == "":
        return None

    try:
        url = f"https://api.discogs.com/releases/{release_id}"
        headers = {"Authorization": f"Discogs token={user_token}"} if user_token else {}
        response = SESSION.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
        discogs_payload = require_api_object(
            response.json(), "Discogs release response"
        )

        artists = require_api_records(
            discogs_payload.get("artists", []), "Discogs artists"
        )
        artist_id = (
            str(artists[0].get("id")) if artists and artists[0].get("id") else None
        )

        labels = require_api_records(
            discogs_payload.get("labels", []), "Discogs labels"
        )
        label_name = None
        catalog_number = None
        if labels:
            label_name = str(labels[0].get("name")) if labels[0].get("name") else None
            catalog_number = (
                str(labels[0].get("catno"))
                if labels[0].get("catno") and labels[0].get("catno") != "none"
                else None
            )

        barcode_value = None
        identifiers = require_api_records(
            discogs_payload.get("identifiers", []), "Discogs identifiers"
        )
        for identifier in identifiers:
            if identifier.get("type") == "Barcode":
                raw_barcode = (
                    str(identifier.get("value", "")).replace(" ", "").replace("-", "")
                )
                if raw_barcode:
                    barcode_value = raw_barcode
                    break

        media_format = None
        formats = require_api_records(
            discogs_payload.get("formats", []), "Discogs formats"
        )
        if formats:
            first_format = formats[0]
            format_parts: list[str] = []
            if first_format.get("name"):
                format_parts.append(str(first_format["name"]))
            descriptions = require_api_strings(
                first_format.get("descriptions", []), "Discogs format descriptions"
            )
            format_parts.extend(
                description for description in descriptions if description
            )
            if format_parts:
                media_format = ", ".join(format_parts)

        producers: list[str] = []
        remixers: list[str] = []
        composers: list[str] = []
        tracklist = require_api_records(
            discogs_payload.get("tracklist", []), "Discogs tracklist"
        )
        track_positions = [
            str(track.get("position") or "").strip()
            for track in tracklist
            if str(track.get("position") or "").strip()
        ]
        scoped_credits: dict[str, list[dict[str, Any]]] = {}
        extra_artists = require_api_records(
            discogs_payload.get("extraartists", []), "Discogs extraartists"
        )
        for extra_artist in extra_artists:
            if extra_artist.get("name") and extra_artist.get("role"):
                track_scope = str(extra_artist.get("tracks") or "").strip()
                if track_scope:
                    for position in _discogs_credit_positions(
                        track_scope, track_positions
                    ):
                        scoped_credits.setdefault(position, []).append(extra_artist)
                    continue
                _add_discogs_credit(extra_artist, producers, remixers, composers)

        track_credits: dict[str, dict[str, str]] = {}
        repeated_titles: set[str] = set()
        for track in tracklist:
            position = str(track.get("position", "")).strip()
            track_title = str(track.get("title", "")).strip()
            track_producers: list[str] = []
            track_remixers: list[str] = []
            track_composers: list[str] = []
            track_extra_artists = require_api_records(
                track.get("extraartists", []), "Discogs track credits"
            )
            track_extra_artists = [
                *track_extra_artists,
                *scoped_credits.get(position, []),
            ]
            for track_extra_artist in track_extra_artists:
                if track_extra_artist.get("name") and track_extra_artist.get("role"):
                    _add_discogs_credit(
                        track_extra_artist,
                        track_producers,
                        track_remixers,
                        track_composers,
                    )
            credits_dict: dict[str, str] = {"title": track_title}
            if track_producers:
                credits_dict["producers"] = ", ".join(track_producers)
            if track_remixers:
                credits_dict["remixer"] = ", ".join(track_remixers)
            if track_composers:
                credits_dict["composer"] = ", ".join(track_composers)
            if position:
                track_credits[position] = credits_dict
            if track_title:
                title_key = track_title.lower()
                if title_key in track_credits or title_key in repeated_titles:
                    repeated_titles.add(title_key)
                    track_credits.pop(title_key, None)
                else:
                    track_credits[title_key] = credits_dict

        release_result: dict[str, Any] = {
            "id": discogs_payload.get("id"),
            "artist_id": artist_id,
            "title": discogs_payload.get("title"),
            "year": discogs_payload.get("year"),
            "released": discogs_payload.get("released"),
            "genres": list(discogs_payload.get("genres", []) or []),
            "styles": list(discogs_payload.get("styles", []) or []),
            "country": normalize_country_name(discogs_payload.get("country")),
            "label": label_name,
            "catalog_number": catalog_number,
            "barcode": barcode_value,
            "media": media_format,
            "producers": ", ".join(producers) if producers else None,
            "remixer": ", ".join(remixers) if remixers else None,
            "composer": ", ".join(composers) if composers else None,
            "track_credits": track_credits,
            "audio_track_count": _count_discogs_audio_tracks(tracklist),
        }
        set_cached_api(cache_key, release_result)
        return release_result
    except (httpx.HTTPError, OSError) as error:
        LOG.debug(f"Discogs release fetch failed for ID {release_id}: {error}")
        return None


def _score_discogs_candidate(
    release_candidate: dict[str, Any], artist: str, album: str
) -> float:
    release_title = str(release_candidate.get("title", ""))
    if " - " in release_title:
        candidate_artist, candidate_album = release_title.split(" - ", 1)
    else:
        candidate_artist, candidate_album = artist, release_title
    base_score = score_artist_title_match(
        artist, album, candidate_artist, candidate_album
    )
    if base_score <= 0.0:
        return 0.0

    score = base_score
    raw_formats = release_candidate.get("format", [])
    formats: list[str] = (
        [str(f).lower() for f in raw_formats]
        if isinstance(raw_formats, list)
        else [str(raw_formats).lower()]
    )

    if any(f in ("file", "digital", "aac", "mp3", "flac") for f in formats):
        score += 30.0
    elif any(f in ("cd", "cd-r", "hdcd") for f in formats):
        score += 15.0
    elif any(
        f in ("vinyl", "lp", "cassette", "acetate", "flexi-disc") for f in formats
    ):
        score -= 20.0

    if any(
        f in ("unofficial", "bootleg", "misprint", "promo", "promotional")
        for f in formats
    ):
        score -= 35.0

    album_norm = normalize_str(album)
    for credit_keyword in ("instrumental", "karaoke", "chopped", "remix"):
        if (
            any(credit_keyword in f for f in formats)
            and credit_keyword not in album_norm
        ):
            score -= 40.0

    return score


@coalesce_lookup
def search_discogs_release(
    artist: str,
    album: str,
    user_token: str | None = None,
    expected_track_count: int | None = None,
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

    count_suffix = f":{expected_track_count}" if expected_track_count else ""
    artist_key = normalize_str(artist)
    cache_key = f"discogs_search:v5:{artist_key}:{normalize_str(album)}{count_suffix}"

    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return cached
    if cached == "":
        return None

    try:
        url = "https://api.discogs.com/database/search"
        headers = {"Authorization": f"Discogs token={user_token}"}
        params = {
            "release_title": album,
            "artist": artist,
            "type": "release",
            "per_page": "10",
        }
        response = SESSION.get(url, params=params, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
        discogs_payload = require_api_object(response.json(), "Discogs search response")
        results = require_api_records(
            discogs_payload.get("results"), "Discogs search results"
        )
        if not results:
            return None

        best_score = 0.0
        best_candidate: dict[str, Any] | None = None
        tied_candidates = False
        for release_candidate in results:
            score = _score_discogs_candidate(release_candidate, artist, album)
            if (
                score > best_score
                and score >= get_config().catalog_candidate_rank_threshold
            ):
                best_score = score
                best_candidate = release_candidate
                tied_candidates = False
            elif (
                best_candidate is not None
                and score == best_score
                and release_candidate.get("id") != best_candidate.get("id")
            ):
                tied_candidates = True

        if tied_candidates:
            LOG.detail(
                f"Ambiguous Discogs release for {escape(artist)} - {escape(album)}"
            )
            return None
        if not best_candidate:
            return None

        selected_release = best_candidate
        release_id = selected_release.get("id")
        if not release_id:
            return None

        full_details = fetch_discogs_release_details(release_id, user_token)
        if full_details:
            if (
                expected_track_count is not None
                and full_details.get("audio_track_count") != expected_track_count
            ):
                LOG.detail(
                    f"Discogs release track count conflicts with {escape(artist)} - {escape(album)}"
                )
                return None
            set_cached_api(cache_key, full_details)
            return full_details

        if expected_track_count is not None:
            return None

        # Fallback to search result fields if full fetch failed
        labels = selected_release.get("label", [])
        label_name = str(labels[0]) if isinstance(labels, list) and labels else None
        catalog_number = (
            str(selected_release.get("catno"))
            if selected_release.get("catno")
            else None
        )
        barcodes = selected_release.get("barcode", [])
        barcode_value = (
            str(barcodes[0]) if isinstance(barcodes, list) and barcodes else None
        )
        formats = selected_release.get("format", [])
        media_format = (
            ", ".join(str(format_entry) for format_entry in formats)
            if isinstance(formats, list) and formats
            else None
        )

        release_result: dict[str, Any] = {
            "id": release_id,
            "artist_id": None,
            "title": selected_release.get("title"),
            "year": selected_release.get("year"),
            "released": selected_release.get("year"),
            "genres": list(selected_release.get("genre", []) or []),
            "styles": list(selected_release.get("style", []) or []),
            "country": normalize_country_name(selected_release.get("country")),
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
    except (httpx.HTTPError, OSError) as error:
        LOG.debug(f"Discogs search failed for {artist} - {album}: {error}")
        return None
