import urllib.error
from collections.abc import Callable
from typing import Any, TypeVar

import httpx
import musicbrainzngs
from musicbrainzngs import MusicBrainzError, NetworkError, ResponseError

from sonora import __version__
from sonora.core.api_payloads import require_api_object, require_api_records
from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.config import get_config
from sonora.core.constants import RATE_LIMIT_MUSICBRAINZ
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.rate_limit import RateLimiter
from sonora.core.text import (
    is_valid_uuid,
    normalize_country_name,
    normalize_language_name,
    normalize_script_name,
    normalize_str,
    safe_int,
)
from sonora.core.titles import clean_title, score_artist_title_match
from sonora.core.utils import retry_network_call

_MB_LIMITER = RateLimiter(interval_seconds=RATE_LIMIT_MUSICBRAINZ)
_T = TypeVar("_T")


def _is_retryable_mb_error(error: Exception) -> bool:
    if isinstance(error, NetworkError):
        return True
    if isinstance(error, ResponseError):
        cause = getattr(error, "cause", None)
        if isinstance(cause, urllib.error.HTTPError):
            return cause.code not in (400, 404)
        return True
    return isinstance(error, (OSError, TimeoutError))


def _call_musicbrainz(func: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
    return retry_network_call(
        lambda: func(*args, **kwargs),
        max_retries=3,
        initial_delay=1.5,
        rate_limiter=_MB_LIMITER,
        transient_exceptions=(
            MusicBrainzError,
            OSError,
            TimeoutError,
        ),
        retry_condition=_is_retryable_mb_error,
    )


def init_musicbrainz(
    app_name: str = "Sonora",
    version: str = __version__,
    contact: str = "https://github.com/dvmizew/Sonora",
) -> None:
    try:
        musicbrainzngs.set_useragent(app_name, version, contact)
        musicbrainzngs.set_rate_limit(limit_or_interval=1.0, new_requests=1)
    except ValueError as error:
        LOG.debug(f"MusicBrainz User-Agent initialization failed: {error}")


init_musicbrainz()


def fetch_artist_discography(artist: str) -> list[dict[str, object]]:
    """
    Fetch and cache the entire discography (releases) of an artist from MusicBrainz in a single API call.
    Returns list of release dicts.
    """
    artist_key = normalize_str(artist)
    cache_key = f"mb_discography:{artist_key}"

    cached = get_cached_api(cache_key)
    if isinstance(cached, list):
        return cached

    try:
        result = _call_musicbrainz(
            musicbrainzngs.search_releases, artist=artist, limit=100
        )
        musicbrainz_payload = require_api_object(
            result, "MusicBrainz discography response"
        )
        releases = require_api_records(
            musicbrainz_payload.get("release-list", []),
            "MusicBrainz discography releases",
        )
        set_cached_api(cache_key, releases)
        return releases
    except (
        MusicBrainzError,
        httpx.HTTPError,
        OSError,
    ) as error:
        LOG.debug(f"MusicBrainz discography fetch failed for {artist}: {error}")
        return []


def _score_musicbrainz_candidate(
    release: dict[str, object],
    artist: str,
    album: str,
    expected_track_count: int | None = None,
) -> float:
    release_title = str(release.get("title", ""))
    release_artist = ""
    artist_credit = release.get("artist-credit", [])
    if (
        isinstance(artist_credit, list)
        and artist_credit
        and isinstance(artist_credit[0], dict)
    ):
        release_artist_credit = require_api_object(
            artist_credit[0].get("artist", {}), "MusicBrainz release artist"
        )
        release_artist = str(release_artist_credit.get("name", ""))

    base_score = score_artist_title_match(artist, album, release_artist, release_title)
    if base_score <= 0.0:
        return 0.0

    score = base_score
    album_norm = normalize_str(album)
    disambig = str(release.get("disambiguation") or "").strip().lower()

    for kw in (
        "chopped",
        "instrumental",
        "karaoke",
        "commentary",
        "acapella",
        "remix",
    ):
        if kw in disambig and kw not in album_norm:
            score -= 50.0

    if not disambig:
        score += 15.0
    elif disambig in ("clean", "explicit", "digital", "deluxe"):
        score += 10.0

    if expected_track_count is not None and expected_track_count > 0:
        med_track_count = safe_int(release.get("medium-track-count"))
        if med_track_count is not None:
            if med_track_count == expected_track_count:
                score += 40.0
            else:
                diff = abs(med_track_count - expected_track_count)
                ratio_diff = diff / max(expected_track_count, 1)
                if ratio_diff > 0.25:
                    score -= 60.0
                else:
                    score -= min(40.0, diff * 5.0)

    status = str(release.get("status") or "").strip().lower()
    if status == "official":
        score += 10.0
    elif status in ("promotion", "bootleg", "pseudo-release"):
        score -= 25.0

    return score


def search_musicbrainz_release(
    artist: str, album: str, expected_track_count: int | None = None
) -> dict[str, object] | None:
    """Search MusicBrainz for an album release matching artist and album name."""
    if not album or normalize_str(album) in ["unknown album", "unknown"]:
        return None

    count_suffix = f":{expected_track_count}" if expected_track_count else ""
    cache_key = (
        f"mb_release:{normalize_str(artist)}:{normalize_str(album)}{count_suffix}"
    )
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return cached

    best_score = 0.0
    target_release: dict[str, object] | None = None

    # Batch strategy: Check artist discography cache first
    discography = fetch_artist_discography(artist)
    album_lower = normalize_str(album)
    for release in discography:
        release_title_value = release.get("title", "")
        release_title = normalize_str(str(release_title_value))
        if release_title == album_lower or (
            album_lower in release_title and len(album_lower) >= 4
        ):
            candidate_score = _score_musicbrainz_candidate(
                release, artist, album, expected_track_count
            )
            if (
                candidate_score > best_score
                and candidate_score >= get_config().catalog_candidate_rank_threshold
            ):
                best_score = candidate_score
                target_release = release

    if target_release and (best_score >= 120.0 or expected_track_count is None):
        set_cached_api(cache_key, target_release)
        return target_release

    try:
        result = _call_musicbrainz(
            musicbrainzngs.search_releases, artist=artist, release=album, limit=10
        )
        musicbrainz_payload = require_api_object(
            result, "MusicBrainz release search response"
        )
        releases = require_api_records(
            musicbrainz_payload.get("release-list", []), "MusicBrainz releases"
        )
        for release in releases:
            candidate_score = _score_musicbrainz_candidate(
                release, artist, album, expected_track_count
            )
            if (
                candidate_score > best_score
                and candidate_score >= get_config().catalog_candidate_rank_threshold
            ):
                best_score = candidate_score
                target_release = release

        cleaned_album = clean_title(album)
        if (
            (target_release is None or best_score < 100.0)
            and cleaned_album
            and normalize_str(cleaned_album) != normalize_str(album)
        ):
            alternative_result = _call_musicbrainz(
                musicbrainzngs.search_releases,
                artist=artist,
                release=cleaned_album,
                limit=10,
            )
            alternative_payload = require_api_object(
                alternative_result, "MusicBrainz alternative release response"
            )
            alt_releases = require_api_records(
                alternative_payload.get("release-list", []),
                "MusicBrainz alternative releases",
            )
            for release in alt_releases:
                candidate_score = _score_musicbrainz_candidate(
                    release, artist, album, expected_track_count
                )
                if (
                    candidate_score > best_score
                    and candidate_score >= get_config().catalog_candidate_rank_threshold
                ):
                    best_score = candidate_score
                    target_release = release

        set_cached_api(cache_key, target_release)
        return target_release
    except (
        MusicBrainzError,
        OSError,
    ) as error:
        LOG.debug(f"MusicBrainz search failed for {artist} - {album}: {error}")
        return None


def fetch_track_mbid(artist: str, title: str) -> str | None:
    cleaned_title = clean_title(title)
    cache_key = f"mb_mbid:{normalize_str(artist)}:{normalize_str(cleaned_title)}"
    cached = get_cached_api(cache_key)
    if cached is not None:
        return str(cached) if cached else None

    try:
        result = _call_musicbrainz(
            musicbrainzngs.search_recordings,
            artist=artist,
            recording=cleaned_title,
            limit=5,
        )
        musicbrainz_payload = require_api_object(
            result, "MusicBrainz recording search response"
        )
        recordings = require_api_records(
            musicbrainz_payload.get("recording-list", []), "MusicBrainz recordings"
        )
        if not recordings:
            set_cached_api(cache_key, None)
            return None

        best_mbid = None
        best_score = 0.0

        for recording in recordings:
            recording_title = str(recording.get("title", ""))
            recording_artist = ""
            artist_credit: Any = recording.get("artist-credit", [])
            if (
                isinstance(artist_credit, list)
                and artist_credit
                and isinstance(artist_credit[0], dict)
            ):
                recording_artist = str(
                    artist_credit[0].get("artist", {}).get("name", "")
                )

            score = score_artist_title_match(
                artist, cleaned_title, recording_artist, recording_title
            )
            if score > best_score:
                best_score = score
                best_mbid = str(recording.get("id"))

        if best_score <= 0.0 or not is_valid_uuid(best_mbid):
            best_mbid = None

        set_cached_api(cache_key, best_mbid)
        return best_mbid
    except (
        MusicBrainzError,
        OSError,
    ) as error:
        LOG.debug(f"MusicBrainz track lookup failed for {artist} - {title}: {error}")
        return None


def fetch_cover_art_archive_url(release_mbid: str) -> str | None:
    """
    Check if Cover Art Archive has front cover art for the given MusicBrainz release MBID.
    Returns front cover image URL or None.
    """
    if not release_mbid or not is_valid_uuid(release_mbid):
        return None
    cache_key = f"caa_url:{release_mbid}"
    cached = get_cached_api(cache_key)
    if cached is not None:
        return str(cached) if cached else None

    url = f"https://coverartarchive.org/release/{release_mbid}/front"
    try:
        response = SESSION.head(url, timeout=5)
        if response.status_code == 200:
            result_url = str(response.url) or url
            set_cached_api(cache_key, result_url)
            return result_url
        set_cached_api(cache_key, None)
    except (httpx.HTTPError, OSError) as error:
        LOG.debug(f"Cover Art Archive lookup failed: {error}")
    return None


def fetch_album_track_mbids(release_mbid: str) -> dict[Any, str]:
    """
    Fetch all track recording MBIDs for an entire album release in ONE single API call.
    Returns mapping of track_position (or (disc, pos) tuple) -> recording_mbid.
    """
    if not release_mbid or not is_valid_uuid(release_mbid):
        return {}

    cache_key = f"mb_album_tracks:{release_mbid}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict) and any(isinstance(k, tuple) for k in cached):
        return {k: str(v) for k, v in cached.items()}

    try:
        release_data = _call_musicbrainz(
            musicbrainzngs.get_release_by_id,
            release_mbid,
            includes=["recordings", "media", "artist-credits"],
        )
        musicbrainz_payload = require_api_object(
            release_data, "MusicBrainz release tracks response"
        )
        release = require_api_object(
            musicbrainz_payload.get("release"), "MusicBrainz release"
        )
        mediums = require_api_records(
            release.get("medium-list", []), "MusicBrainz media"
        )
        mapping: dict[Any, str] = {}
        for medium_idx, medium in enumerate(mediums, start=1):
            disc_num = safe_int(medium.get("position")) or medium_idx
            for track in require_api_records(
                medium.get("track-list", []), "MusicBrainz medium tracks"
            ):
                position = track.get("position")
                recording_id = track.get("recording", {}).get("id")
                if position and recording_id:
                    pos_int = safe_int(position)
                    if pos_int is not None:
                        recording_mbid = str(recording_id)
                        mapping[(disc_num, pos_int)] = recording_mbid
                        mapping[f"{disc_num}-{pos_int}"] = recording_mbid
                        if disc_num == 1 or len(mediums) <= 1:
                            mapping[pos_int] = recording_mbid
        set_cached_api(cache_key, mapping)
        return mapping
    except (
        MusicBrainzError,
        OSError,
    ) as error:
        LOG.debug(f"MusicBrainz album track fetch failed for {release_mbid}: {error}")
        return {}


def fetch_musicbrainz_recording_details(
    recording_mbid: str,
) -> dict[str, object] | None:
    """
    Fetch comprehensive recording credits and relationships from MusicBrainz
    including composers, lyricists, producers, remixers, and ISRCs.
    """
    if not recording_mbid or not is_valid_uuid(recording_mbid):
        return None

    cache_key = f"mb_rec_details:{recording_mbid}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return cached

    try:
        musicbrainz_payload = _call_musicbrainz(
            musicbrainzngs.get_recording_by_id,
            recording_mbid,
            includes=[
                "artists",
                "releases",
                "isrcs",
                "work-rels",
                "work-level-rels",
                "artist-rels",
                "tags",
            ],
        )
        musicbrainz_payload = require_api_object(
            musicbrainz_payload, "MusicBrainz recording response"
        )
        recording_dict = require_api_object(
            musicbrainz_payload.get("recording"), "MusicBrainz recording"
        )
        if not recording_dict:
            return None

        isrc_list = recording_dict.get("isrc-list", [])
        isrc = str(isrc_list[0]) if isrc_list else None

        composers: list[str] = []
        lyricists: list[str] = []
        producers: list[str] = []
        remixers: list[str] = []

        artist_rels = require_api_records(
            recording_dict.get("artist-relation-list", []),
            "MusicBrainz artist relationships",
        )
        for relationship in artist_rels:
            relationship_type = str(relationship.get("type", "")).lower()
            artist_name = relationship.get("artist", {}).get("name")
            if not artist_name:
                continue
            if relationship_type in ("composer", "writer"):
                composers.append(artist_name)
            if relationship_type in ("lyricist", "writer"):
                lyricists.append(artist_name)
            if "producer" in relationship_type:
                producers.append(artist_name)
            if relationship_type == "remixer":
                remixers.append(artist_name)

        work_rels = require_api_records(
            recording_dict.get("work-relation-list", []),
            "MusicBrainz work relationships",
        )
        work_id = None
        if work_rels:
            work_id = work_rels[0].get("work", {}).get("id")

        artist_credits = recording_dict.get("artist-credit", [])
        artist_id = (
            artist_credits[0].get("artist", {}).get("id")
            if artist_credits
            and isinstance(artist_credits, list)
            and isinstance(artist_credits[0], dict)
            else None
        )

        details: dict[str, object] = {
            "title": recording_dict.get("title"),
            "artist": recording_dict.get("artist-credit-phrase"),
            "first-release-date": recording_dict.get("first-release-date"),
            "isrc": isrc,
            "isrcs": isrc_list,
            "disambiguation": recording_dict.get("disambiguation"),
            "composer": ", ".join(dict.fromkeys(composers)) if composers else None,
            "lyricist": ", ".join(dict.fromkeys(lyricists)) if lyricists else None,
            "producers": ", ".join(dict.fromkeys(producers)) if producers else None,
            "remixer": ", ".join(dict.fromkeys(remixers)) if remixers else None,
            "musicbrainz_workid": str(work_id) if is_valid_uuid(work_id) else None,
            "musicbrainz_artistid": str(artist_id)
            if is_valid_uuid(artist_id)
            else None,
        }
        set_cached_api(cache_key, details)
        return details
    except (
        MusicBrainzError,
        OSError,
    ) as error:
        LOG.debug(
            f"MusicBrainz recording details fetch failed for {recording_mbid}: {error}"
        )
        return None


def _release_track_indexes(
    release_dict: dict[str, Any],
    release_artist: str | None,
    album_artist_id: str | None,
) -> dict[str, object]:
    mediums = require_api_records(
        release_dict.get("medium-list", []), "MusicBrainz media"
    )
    media_format = None
    total_tracks = 0
    total_discs = len(mediums) if mediums else 1
    tracks_by_position: dict[Any, dict[str, object]] = {}
    tracks_by_disc_and_position: dict[tuple[int, int], dict[str, object]] = {}
    tracks_by_mbid: dict[str, dict[str, object]] = {}

    if mediums:
        media_format = mediums[0].get("format")
        for medium_idx, medium in enumerate(mediums, start=1):
            disc_num = safe_int(medium.get("position")) or medium_idx
            track_count_value = medium.get("track-count")
            disc_total_tracks = (
                int(track_count_value)
                if track_count_value and str(track_count_value).isdigit()
                else None
            )
            if disc_total_tracks:
                total_tracks += disc_total_tracks
            disc_subtitle = medium.get("title")

            for track_item in require_api_records(
                medium.get("track-list", []), "MusicBrainz medium tracks"
            ):
                pos_int = safe_int(track_item.get("position"))
                if pos_int is None:
                    continue
                recording = require_api_object(
                    track_item.get("recording", {}), "MusicBrainz track recording"
                )
                recording_id = str(recording.get("id", "")).strip()
                isrc_list = recording.get("isrc-list", [])
                recording_isrc = str(isrc_list[0]) if isrc_list else None

                rec_composers: list[str] = []
                rec_lyricists: list[str] = []
                rec_producers: list[str] = []
                rec_remixers: list[str] = []

                for relationship in require_api_records(
                    recording.get("artist-relation-list", []),
                    "MusicBrainz recording relationships",
                ):
                    relationship_type = str(relationship.get("type", "")).lower()
                    credit_artist_name = relationship.get("artist", {}).get("name")
                    if not credit_artist_name:
                        continue
                    if relationship_type in ("composer", "writer"):
                        rec_composers.append(credit_artist_name)
                    if relationship_type in ("lyricist", "writer"):
                        rec_lyricists.append(credit_artist_name)
                    if "producer" in relationship_type:
                        rec_producers.append(credit_artist_name)
                    if relationship_type == "remixer":
                        rec_remixers.append(credit_artist_name)

                work_rels = require_api_records(
                    recording.get("work-relation-list", []),
                    "MusicBrainz recording works",
                )
                rec_work_id = None
                if work_rels:
                    w_id = work_rels[0].get("work", {}).get("id")
                    if is_valid_uuid(w_id):
                        rec_work_id = str(w_id)

                track_artist = (
                    track_item.get("artist-credit-phrase")
                    or recording.get("artist-credit-phrase")
                    or release_artist
                )
                rec_artist_credits = (
                    track_item.get("artist-credit")
                    or recording.get("artist-credit")
                    or []
                )
                rec_artist_id = (
                    rec_artist_credits[0].get("artist", {}).get("id")
                    if rec_artist_credits
                    and isinstance(rec_artist_credits, list)
                    and isinstance(rec_artist_credits[0], dict)
                    else album_artist_id
                )
                rec_details: dict[str, object] = {
                    "position": pos_int,
                    "disc_number": disc_num,
                    "total_discs": total_discs,
                    "disc_total_tracks": disc_total_tracks,
                    "disc_subtitle": disc_subtitle,
                    "title": track_item.get("title") or recording.get("title"),
                    "artist": track_artist,
                    "recording_mbid": recording_id
                    if is_valid_uuid(recording_id)
                    else None,
                    "musicbrainz_artistid": str(rec_artist_id)
                    if is_valid_uuid(rec_artist_id)
                    else None,
                    "musicbrainz_albumartistid": str(album_artist_id)
                    if is_valid_uuid(album_artist_id)
                    else None,
                    "isrc": recording_isrc,
                    "isrcs": isrc_list,
                    "disambiguation": recording.get("disambiguation"),
                    "composer": ", ".join(dict.fromkeys(rec_composers))
                    if rec_composers
                    else None,
                    "lyricist": ", ".join(dict.fromkeys(rec_lyricists))
                    if rec_lyricists
                    else None,
                    "producers": ", ".join(dict.fromkeys(rec_producers))
                    if rec_producers
                    else None,
                    "remixer": ", ".join(dict.fromkeys(rec_remixers))
                    if rec_remixers
                    else None,
                    "musicbrainz_workid": rec_work_id,
                }
                tracks_by_disc_and_position[(disc_num, pos_int)] = rec_details
                tracks_by_position[(disc_num, pos_int)] = rec_details
                tracks_by_position[f"{disc_num}-{pos_int}"] = rec_details
                if (
                    disc_num == 1
                    or total_discs == 1
                    or pos_int not in tracks_by_position
                ):
                    tracks_by_position[pos_int] = rec_details
                if is_valid_uuid(recording_id):
                    tracks_by_mbid[recording_id] = rec_details

    return {
        "media": media_format,
        "total_tracks": total_tracks if total_tracks > 0 else None,
        "total_discs": total_discs,
        "tracks_by_position": tracks_by_position,
        "tracks_by_disc_and_position": tracks_by_disc_and_position,
        "tracks_by_mbid": tracks_by_mbid,
    }


def fetch_musicbrainz_release_details(
    release_mbid: str,
) -> dict[str, object] | None:
    """
    Fetch comprehensive release metadata from MusicBrainz including barcode,
    label, catalog number, media format, country, language, and release group.
    """
    if not release_mbid or not is_valid_uuid(release_mbid):
        return None

    cache_key = f"mb_rel_details:{release_mbid}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict) and "tracks_by_disc_and_position" in cached:
        return cached

    try:
        musicbrainz_payload = _call_musicbrainz(
            musicbrainzngs.get_release_by_id,
            release_mbid,
            includes=[
                "recordings",
                "media",
                "artist-credits",
                "release-groups",
                "labels",
                "isrcs",
                "artist-rels",
                "recording-level-rels",
                "work-rels",
                "work-level-rels",
                "tags",
            ],
        )
        musicbrainz_payload = require_api_object(
            musicbrainz_payload, "MusicBrainz release response"
        )
        release_dict = require_api_object(
            musicbrainz_payload.get("release"), "MusicBrainz release"
        )
        if not release_dict:
            return None

        barcode = release_dict.get("barcode")
        country = release_dict.get("country")
        status = release_dict.get("status")

        release_group = release_dict.get("release-group", {})
        release_group_id = release_group.get("id")
        release_type = release_group.get("primary-type")

        label_name = None
        catalog_number = None
        label_mbid = None
        label_info_list = require_api_records(
            release_dict.get("label-info-list", []), "MusicBrainz label credits"
        )
        if label_info_list:
            first_label = label_info_list[0]
            label_name = first_label.get("label", {}).get("name")
            catalog_number = first_label.get("catalog-number")
            raw_label_id = first_label.get("label", {}).get("id")
            if is_valid_uuid(raw_label_id):
                label_mbid = str(raw_label_id)

        release_title = release_dict.get("title")
        release_artist = release_dict.get("artist-credit-phrase")
        artist_credits = release_dict.get("artist-credit", [])
        album_artist_id = (
            artist_credits[0].get("artist", {}).get("id")
            if artist_credits
            and isinstance(artist_credits, list)
            and isinstance(artist_credits[0], dict)
            else None
        )
        if (
            not release_artist
            and artist_credits
            and isinstance(artist_credits, list)
            and isinstance(artist_credits[0], dict)
        ):
            release_artist = artist_credits[0].get("artist", {}).get("name")

        track_indexes = _release_track_indexes(
            release_dict, release_artist, album_artist_id
        )

        text_representation = release_dict.get("text-representation", {})
        language = text_representation.get("language")
        script = text_representation.get("script")

        artist_credits = release_dict.get("artist-credit", [])
        artist_sort = None
        if (
            artist_credits
            and isinstance(artist_credits, list)
            and isinstance(artist_credits[0], dict)
        ):
            artist_sort = artist_credits[0].get("artist", {}).get("sort-name")

        details: dict[str, object] = {
            "title": release_title,
            "album_artist": release_artist,
            "artist": release_artist,
            "barcode": barcode,
            "release_country": normalize_country_name(country),
            "release_status": status,
            "release_type": release_type,
            "musicbrainz_releasegroupid": str(release_group_id)
            if is_valid_uuid(release_group_id)
            else None,
            "musicbrainz_artistid": str(album_artist_id)
            if is_valid_uuid(album_artist_id)
            else None,
            "musicbrainz_albumartistid": str(album_artist_id)
            if is_valid_uuid(album_artist_id)
            else None,
            "label": label_name,
            "label_mbid": label_mbid,
            "catalog_number": catalog_number,
            "language": normalize_language_name(language),
            "script": normalize_script_name(script),
            "artist_sort": artist_sort,
            "date": release_dict.get("date"),
            "original_date": release_group.get("first-release-date")
            or release_dict.get("date"),
        }
        details.update(track_indexes)
        set_cached_api(cache_key, details)
        return details
    except (
        MusicBrainzError,
        OSError,
    ) as error:
        LOG.debug(
            f"MusicBrainz release details fetch failed for {release_mbid}: {error}"
        )
        return None


def search_musicbrainz_artists(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """
    Search MusicBrainz for artists matching query with strict rate limiting and error resilience.
    Returns list of artist dictionaries.
    """
    init_musicbrainz()
    try:
        artist_search_payload: Any = _call_musicbrainz(
            musicbrainzngs.search_artists, query=query, limit=limit
        )
        artist_search_payload = require_api_object(
            artist_search_payload, "MusicBrainz artist search response"
        )
        return require_api_records(
            artist_search_payload.get("artist-list", []), "MusicBrainz artists"
        )
    except (
        MusicBrainzError,
        OSError,
    ) as error:
        LOG.debug(f"MusicBrainz artist search failed for '{query}': {error}")
        return []
