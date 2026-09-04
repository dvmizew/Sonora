from typing import Any

import httpx
import musicbrainzngs
from musicbrainzngs import MusicBrainzError

from sonora import __version__
from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.constants import RATE_LIMIT_MUSICBRAINZ
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.utils import (
    RateLimiter,
    clean_title,
    is_valid_uuid,
    match_score,
    normalize_str,
)

_MB_LIMITER = RateLimiter(interval_seconds=RATE_LIMIT_MUSICBRAINZ)


def init_musicbrainz(
    app_name: str = "Sonora",
    version: str = __version__,
    contact: str = "https://github.com/dvmizew/Sonora",
) -> None:
    try:
        musicbrainzngs.set_useragent(app_name, version, contact)
    except (ValueError, AttributeError, RuntimeError) as error:
        LOG.debug(f"MusicBrainz User-Agent initialization failed: {error}")


init_musicbrainz()


def fetch_artist_discography(artist: str) -> list[dict[str, object]]:
    """
    Fetch and cache the entire discography (releases) of an artist from MusicBrainz in a single API call.
    Returns list of release dicts.
    """
    if musicbrainzngs is not None:
        init_musicbrainz()
    artist_key = normalize_str(artist)
    cache_key = f"mb_discography:{artist_key}"

    cached = get_cached_api(cache_key)
    if isinstance(cached, list):
        return cached

    _MB_LIMITER.wait()
    try:
        result = musicbrainzngs.search_releases(artist=artist, limit=100)
        releases: list[dict[str, object]] = (
            result.get("release-list", []) if isinstance(result, dict) else []
        )
        set_cached_api(cache_key, releases)
        return releases
    except (
        MusicBrainzError,
        httpx.HTTPError,
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
    ) as error:
        LOG.debug(f"MusicBrainz discography fetch failed for {artist}: {error}")
        return []


def search_musicbrainz_release(artist: str, album: str) -> dict[str, object] | None:
    """Search MusicBrainz for an album release matching artist and album name."""
    if not album or normalize_str(album) in ["unknown album", "unknown"]:
        return None

    cache_key = f"mb_release:{normalize_str(artist)}:{normalize_str(album)}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return cached

    # Batch strategy: Check artist discography cache first
    discography = fetch_artist_discography(artist)
    album_lower = normalize_str(album)
    for release in discography:
        release_title_value = release.get("title", "")
        release_title = normalize_str(str(release_title_value))
        if release_title == album_lower or album_lower in release_title:
            set_cached_api(cache_key, release)
            return release

    _MB_LIMITER.wait()
    try:
        result = musicbrainzngs.search_releases(artist=artist, release=album, limit=5)
        raw_releases = (
            result.get("release-list", []) if isinstance(result, dict) else []
        )
        releases: list[dict[str, object]] = [
            release for release in raw_releases if isinstance(release, dict)
        ]
        target_release: dict[str, object] | None = None
        best_score = 0.0
        for release in releases:
            rel_title = str(release.get("title", ""))
            rel_artist = ""
            artist_credit = release.get("artist-credit", [])
            if (
                isinstance(artist_credit, list)
                and artist_credit
                and isinstance(artist_credit[0], dict)
            ):
                art_dict = artist_credit[0].get("artist", {})
                if isinstance(art_dict, dict):
                    rel_artist = str(art_dict.get("name", ""))
            score = match_score(artist, album, rel_artist, rel_title)
            if score > best_score and score >= 80.0:
                best_score = score
                target_release = release

        set_cached_api(cache_key, target_release)
        return target_release
    except (
        MusicBrainzError,
        httpx.HTTPError,
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
    ) as error:
        LOG.debug(f"MusicBrainz search failed for {artist} - {album}: {error}")
        return None


def fetch_track_mbid(artist: str, title: str) -> str | None:
    cleaned_title = clean_title(title)
    cache_key = f"mb_mbid:{normalize_str(artist)}:{normalize_str(cleaned_title)}"
    cached = get_cached_api(cache_key)
    if cached is not None:
        return str(cached) if cached else None

    _MB_LIMITER.wait()
    try:
        result = musicbrainzngs.search_recordings(
            artist=artist, recording=cleaned_title, limit=5
        )
        recordings = result.get("recording-list", [])
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

            score = match_score(
                artist, cleaned_title, recording_artist, recording_title
            )
            if score > best_score:
                best_score = score
                best_mbid = str(recording.get("id"))

        # Threshold check: require score >= 65 to avoid wrong MBIDs
        if best_score < 65.0 or not is_valid_uuid(best_mbid):
            best_mbid = None

        set_cached_api(cache_key, best_mbid)
        return best_mbid
    except (
        MusicBrainzError,
        httpx.HTTPError,
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
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
    except (httpx.HTTPError, OSError, ValueError, KeyError, RuntimeError) as error:
        LOG.debug(f"Cover Art Archive lookup failed: {error}")
    return None


def fetch_album_track_mbids(release_mbid: str) -> dict[int, str]:
    """
    Fetch all track recording MBIDs for an entire album release in ONE single API call.
    Returns mapping of track_position (1-indexed) -> recording_mbid.
    """
    if not release_mbid or not is_valid_uuid(release_mbid):
        return {}

    cache_key = f"mb_album_tracks:{release_mbid}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, dict):
        return {
            int(track_position_key): str(recording_mbid_value)
            for track_position_key, recording_mbid_value in cached.items()
        }

    _MB_LIMITER.wait()
    try:
        release_data = musicbrainzngs.get_release_by_id(
            release_mbid, includes=["recordings", "media", "artist-credits"]
        )
        mediums = (
            release_data.get("release", {}).get("medium-list", [])
            if isinstance(release_data, dict)
            else []
        )
        mapping: dict[int, str] = {}
        for medium in mediums:
            if isinstance(medium, dict):
                for track in medium.get("track-list", []):
                    if isinstance(track, dict):
                        position = track.get("position")
                        recording_id = track.get("recording", {}).get("id")
                        if position and recording_id:
                            mapping[int(position)] = str(recording_id)
        set_cached_api(cache_key, mapping)
        return mapping
    except (
        MusicBrainzError,
        OSError,
        ValueError,
        KeyError,
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

    _MB_LIMITER.wait()
    try:
        data = musicbrainzngs.get_recording_by_id(
            recording_mbid,
            includes=[
                "artists",
                "releases",
                "isrcs",
                "work-rels",
                "artist-rels",
                "tags",
            ],
        )
        recording_dict = data.get("recording", {}) if isinstance(data, dict) else {}
        if not recording_dict:
            return None

        isrc_list = recording_dict.get("isrc-list", [])
        isrc = str(isrc_list[0]) if isrc_list else None

        composers: list[str] = []
        lyricists: list[str] = []
        producers: list[str] = []
        remixers: list[str] = []

        artist_rels = recording_dict.get("artist-relation-list", [])
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

        work_rels = recording_dict.get("work-relation-list", [])
        work_id = None
        if work_rels and isinstance(work_rels, list):
            work_id = work_rels[0].get("work", {}).get("id")

        details: dict[str, object] = {
            "title": recording_dict.get("title"),
            "artist": recording_dict.get("artist-credit-phrase"),
            "first-release-date": recording_dict.get("first-release-date"),
            "isrc": isrc,
            "disambiguation": recording_dict.get("disambiguation"),
            "composer": ", ".join(dict.fromkeys(composers)) if composers else None,
            "lyricist": ", ".join(dict.fromkeys(lyricists)) if lyricists else None,
            "producers": ", ".join(dict.fromkeys(producers)) if producers else None,
            "remixer": ", ".join(dict.fromkeys(remixers)) if remixers else None,
            "musicbrainz_workid": str(work_id) if is_valid_uuid(work_id) else None,
        }
        set_cached_api(cache_key, details)
        return details
    except (
        MusicBrainzError,
        OSError,
        ValueError,
        KeyError,
    ) as error:
        LOG.debug(
            f"MusicBrainz recording details fetch failed for {recording_mbid}: {error}"
        )
        return None


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
    if isinstance(cached, dict):
        return cached

    _MB_LIMITER.wait()
    try:
        data = musicbrainzngs.get_release_by_id(
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
        release_dict = data.get("release", {}) if isinstance(data, dict) else {}
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
        label_info_list = release_dict.get("label-info-list", [])
        if label_info_list and isinstance(label_info_list, list):
            first_label = label_info_list[0]
            label_name = first_label.get("label", {}).get("name")
            catalog_number = first_label.get("catalog-number")

        release_title = release_dict.get("title")
        release_artist = release_dict.get("artist-credit-phrase")
        if not release_artist:
            artist_credits = release_dict.get("artist-credit", [])
            if (
                artist_credits
                and isinstance(artist_credits, list)
                and isinstance(artist_credits[0], dict)
            ):
                release_artist = artist_credits[0].get("artist", {}).get("name")

        mediums = release_dict.get("medium-list", [])
        media_format = None
        total_tracks = 0
        total_discs = len(mediums) if mediums else None
        tracks_by_position: dict[int, dict[str, object]] = {}
        tracks_by_mbid: dict[str, dict[str, object]] = {}

        if mediums and isinstance(mediums, list):
            media_format = mediums[0].get("format")
            for medium in mediums:
                track_count_value = medium.get("track-count")
                if track_count_value and str(track_count_value).isdigit():
                    total_tracks += int(track_count_value)
                for track_item in medium.get("track-list", []):
                    if not isinstance(track_item, dict):
                        continue
                    pos = track_item.get("position")
                    rec = track_item.get("recording", {})
                    if not isinstance(rec, dict):
                        continue
                    rec_id = str(rec.get("id", "")).strip()
                    isrc_list = rec.get("isrc-list", [])
                    rec_isrc = str(isrc_list[0]) if isrc_list else None

                    rec_composers: list[str] = []
                    rec_lyricists: list[str] = []
                    rec_producers: list[str] = []
                    rec_remixers: list[str] = []

                    for rel in rec.get("artist-relation-list", []):
                        if not isinstance(rel, dict):
                            continue
                        r_type = str(rel.get("type", "")).lower()
                        a_name = rel.get("artist", {}).get("name")
                        if not a_name:
                            continue
                        if r_type in ("composer", "writer"):
                            rec_composers.append(a_name)
                        if r_type in ("lyricist", "writer"):
                            rec_lyricists.append(a_name)
                        if "producer" in r_type:
                            rec_producers.append(a_name)
                        if r_type == "remixer":
                            rec_remixers.append(a_name)

                    work_rels = rec.get("work-relation-list", [])
                    rec_work_id = None
                    if (
                        work_rels
                        and isinstance(work_rels, list)
                        and isinstance(work_rels[0], dict)
                    ):
                        w_id = work_rels[0].get("work", {}).get("id")
                        if is_valid_uuid(w_id):
                            rec_work_id = str(w_id)

                    track_artist = (
                        track_item.get("artist-credit-phrase")
                        or rec.get("artist-credit-phrase")
                        or release_artist
                    )
                    rec_details: dict[str, object] = {
                        "title": rec.get("title") or track_item.get("title"),
                        "artist": track_artist,
                        "recording_mbid": rec_id if is_valid_uuid(rec_id) else None,
                        "isrc": rec_isrc,
                        "disambiguation": rec.get("disambiguation"),
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
                    if isinstance(pos, int):
                        tracks_by_position[pos] = rec_details
                    elif pos and str(pos).isdigit():
                        tracks_by_position[int(pos)] = rec_details
                    if is_valid_uuid(rec_id):
                        tracks_by_mbid[rec_id] = rec_details

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
            "release_country": country,
            "release_status": status,
            "release_type": release_type,
            "musicbrainz_releasegroupid": str(release_group_id)
            if is_valid_uuid(release_group_id)
            else None,
            "label": label_name,
            "catalog_number": catalog_number,
            "media": media_format,
            "total_tracks": total_tracks if total_tracks > 0 else None,
            "total_discs": total_discs,
            "language": language,
            "script": script,
            "artist_sort": artist_sort,
            "date": release_dict.get("date"),
            "original_date": release_group.get("first-release-date")
            or release_dict.get("date"),
            "tracks_by_position": tracks_by_position,
            "tracks_by_mbid": tracks_by_mbid,
        }
        set_cached_api(cache_key, details)
        return details
    except (
        MusicBrainzError,
        OSError,
        ValueError,
        KeyError,
    ) as error:
        LOG.debug(
            f"MusicBrainz release details fetch failed for {release_mbid}: {error}"
        )
        return None
