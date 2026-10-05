import dataclasses
import functools
import os
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, NamedTuple

import acoustid
import httpx
from musicbrainzngs import MusicBrainzError
from rapidfuzz import fuzz
from rich.markup import escape

from sonora.audio.art import (
    process_album_cover_art,
    process_artist_artwork,
    process_label_artwork,
)
from sonora.audio.bpm import calculate_bpm
from sonora.audio.cuesheet import find_companion_cuesheet, read_cuesheet_content
from sonora.audio.key import (
    detect_musical_key,
    key_to_camelot,
)
from sonora.audio.metadata import (
    get_audio_duration,
    read_track_metadata,
    write_track_metadata,
)
from sonora.audio.replaygain import calculate_album_replaygain
from sonora.core.config import clear_config_cache, get_config
from sonora.core.constants import (
    ALBUM_MATCH_THRESHOLD,
    ARTIST_MATCH_THRESHOLD,
    MIN_COVER_ART_DIMENSION,
)
from sonora.core.logger import (
    LOG,
    create_progress,
    interactive_pause_listener,
    wait_if_paused,
)
from sonora.core.models import NormalizeReport, TrackInfo
from sonora.core.state import get_library_state
from sonora.core.utils import (
    InterruptedOperationError,
    clean_disambiguation,
    clean_title,
    clean_unicode_punct,
    extract_artist_features,
    extract_disc_number_from_folder,
    extract_title_features,
    find_audio_files,
    get_album_root_directory,
    get_primary_artist,
    group_files_by_album_root,
    harmonize_artist_casing,
    is_interruption,
    is_noise_genre,
    is_valid_uuid,
    match_score,
    normalize_country_name,
    normalize_date,
    normalize_featured_artists,
    normalize_genre,
    normalize_language_name,
    normalize_script_name,
    normalize_str,
    parse_track_filename,
    preserve_unicode_repertoire,
    resolve_artist_name,
    safe_float,
    safe_int,
    strip_corrupt_brackets,
)
from sonora.services.acoustid import lookup_acoustid
from sonora.services.deezer import (
    fetch_deezer_album_details,
    fetch_deezer_track_by_isrc,
    fetch_deezer_track_details,
)
from sonora.services.discogs import search_discogs_release
from sonora.services.genius import fetch_genius_song_details
from sonora.services.itunes import (
    fetch_itunes_album_details,
    fetch_itunes_track_metadata,
)
from sonora.services.lastfm import fetch_lastfm_tags
from sonora.services.lyrics import process_track_lyrics
from sonora.services.musicbrainz import (
    fetch_album_track_mbids,
    fetch_musicbrainz_recording_details,
    fetch_musicbrainz_release_details,
    fetch_track_mbid,
    search_musicbrainz_release,
)
from sonora.services.shazam import recognize_audio_track
from sonora.services.theaudiodb import (
    fetch_theaudiodb_track_details,
)

_NETWORK_EXCEPTIONS = (
    MusicBrainzError,
    acoustid.AcoustidError,
    acoustid.WebServiceError,
    httpx.HTTPError,
    OSError,
    TimeoutError,
)


_PROD_BRACKET_PATTERN = re.compile(
    r"\s*[\(\[\{]\s*(?:prod(?:\.|uced|uction)?\s*(?:by|:)?|produced\s+by)\s*([^()\[\]{}]+?)[\)\]\}]",
    re.IGNORECASE,
)

_SKIP_DIFF_FIELDS: frozenset[str] = frozenset(
    {
        "file_path",
        "lyrics",
        "synced_lyrics",
        "sample_rate",
        "bitrate",
        "channels",
        "is_lossless",
        "art_width",
        "art_height",
        "is_alien",
    }
)


def _apply_mapping(
    track_info: TrackInfo,
    metadata_dict: dict[str, Any],
    field_map: dict[str, str],
    force: bool = False,
) -> None:
    for src_key, target_attr in field_map.items():
        val = metadata_dict.get(src_key)
        if val is None:
            continue
        val_str = str(val).strip()
        if not val_str or val_str.lower() in ("none", "null"):
            continue

        if target_attr == "genre":
            norm_genre = normalize_genre(val_str)
            if not norm_genre:
                continue
            val_str = norm_genre
        elif target_attr in ("date", "original_date"):
            normalized_d = normalize_date(val_str)
            if not normalized_d:
                continue
            val_str = normalized_d
        elif target_attr == "release_country":
            normalized_country = normalize_country_name(val_str)
            if not normalized_country:
                continue
            val_str = normalized_country
        elif target_attr == "language":
            normalized_lang = normalize_language_name(val_str)
            if not normalized_lang:
                continue
            val_str = normalized_lang
        elif target_attr == "script":
            normalized_script = normalize_script_name(val_str)
            if not normalized_script:
                continue
            val_str = normalized_script
        elif target_attr == "title":
            val_str = clean_unicode_punct(val_str)
            clean_t, extracted_feats = extract_title_features(
                val_str, primary_artist=track_info.artist
            )
            if extracted_feats:
                track_info.featured_artists = normalize_featured_artists(
                    [track_info.featured_artists, *extracted_feats],
                    primary_artist=track_info.artist,
                )
            existing_title = getattr(track_info, "title", None)
            val_str = preserve_unicode_repertoire(
                str(existing_title) if existing_title else None, clean_t or val_str
            )
        elif target_attr in ("artist", "album", "album_artist"):
            val_str = clean_unicode_punct(val_str)
            existing_val = getattr(track_info, target_attr, None)
            val_str = preserve_unicode_repertoire(
                str(existing_val) if existing_val else None, val_str
            )
        elif target_attr == "advisory":
            if str(val_str).strip().capitalize() != "Explicit":
                continue
            val_str = "Explicit"
        elif target_attr == "featured_artists":
            existing_featured = getattr(track_info, "featured_artists", None)
            if existing_featured and not force:
                normalized_result = normalize_featured_artists(
                    [existing_featured, val_str],
                    primary_artist=track_info.artist,
                )
            else:
                normalized_result = normalize_featured_artists(
                    val_str,
                    primary_artist=track_info.artist,
                )
            if not normalized_result:
                continue
            val_str = normalized_result

        if target_attr in (
            "disc_number",
            "total_discs",
            "track_number",
            "total_tracks",
        ):
            val_int = safe_int(val_str)
            if val_int is None:
                continue
            cur_int = safe_int(getattr(track_info, target_attr, None))
            if target_attr in ("disc_number", "total_discs"):
                if cur_int is None or cur_int != val_int:
                    setattr(track_info, target_attr, val_int)
            elif cur_int is None or force:
                setattr(track_info, target_attr, val_int)
            continue

        if not getattr(track_info, target_attr) or force:
            setattr(track_info, target_attr, val_str)


def _enrich_acoustid(
    track_info: TrackInfo,
    file_path: Path,
    acoustid_api_key: str | None,
    album_track_mbids: dict[int, str] | None = None,
    force: bool = False,
) -> None:
    if not acoustid_api_key:
        return
    # If the album match already resolved an authoritative MBID for this track, skip expensive audio fingerprinting
    album_mbids = album_track_mbids or {}
    if (
        track_info.track_number
        and track_info.track_number in album_mbids
        and is_valid_uuid(album_mbids[track_info.track_number])
    ):
        return
    if is_valid_uuid(track_info.musicbrainz_trackid) and not force:
        return
    try:
        acoustid_mbid = lookup_acoustid(
            file_path,
            api_key=acoustid_api_key,
            expected_artist=track_info.artist,
            expected_title=track_info.title,
        )
        if is_valid_uuid(acoustid_mbid):
            if not _is_generic_title(track_info.title) and not _is_generic(
                track_info.artist, "artist"
            ):
                rec_details = fetch_musicbrainz_recording_details(acoustid_mbid)
                if not rec_details or not isinstance(rec_details, dict):
                    return
                cand_artist = str(rec_details.get("artist") or "")
                cand_title = str(rec_details.get("title") or "")
                rec_isrc = rec_details.get("isrc")
                isrc_matches = bool(
                    track_info.isrc
                    and rec_isrc
                    and str(track_info.isrc).strip().upper()
                    == str(rec_isrc).strip().upper()
                )
                score = match_score(
                    track_info.artist or "",
                    track_info.title or "",
                    cand_artist,
                    cand_title,
                )
                if not isrc_matches and score < 85.0:
                    LOG.debug(
                        f"Rejecting AcoustID MBID {acoustid_mbid} ('{cand_artist} - {cand_title}') "
                        f"conflicting with ('{track_info.artist} - {track_info.title}')"
                    )
                    return
            track_info.musicbrainz_trackid = acoustid_mbid
            LOG.info(f"   ∟ 🎯 [acoustid] Matched MBID: {acoustid_mbid[:8]}...")
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"AcoustID lookup failed for {track_info.title}: {error}")


def _is_generic(text: str | None, placeholder: str) -> bool:
    if not text:
        return True
    val = text.strip().lower()
    return val in (
        "",
        placeholder.lower(),
        "unknown",
        f"unknown {placeholder.lower()}",
        "unknown track",
        "untitled",
    )


def _is_generic_title(title: str | None) -> bool:
    if _is_generic(title, "title"):
        return True
    val = str(title).strip().lower()
    if val.isdigit():
        return True
    clean_val = re.sub(r"[\(\[\{].*?[\)\]\}]", "", val).strip()
    return clean_val.isdigit() or bool(
        re.match(r"^(track|audio\s*track|audiotrack|title)\s*\d*$", clean_val)
    )


def _find_track_in_album_mapping(
    tracks_by_title: Any,
    tracks_by_pos: Any,
    title: str | None,
    track_number: int | None,
    disc_number: int | None = None,
    tracks_by_disc_and_position: Any = None,
) -> tuple[dict[str, Any] | None, bool]:
    if (
        isinstance(tracks_by_disc_and_position, dict)
        and track_number is not None
        and disc_number is not None
        and (disc_number, track_number) in tracks_by_disc_and_position
    ):
        disc_entry = tracks_by_disc_and_position[(disc_number, track_number)]
        if isinstance(disc_entry, dict):
            return disc_entry, True

    if (
        isinstance(tracks_by_pos, dict)
        and track_number is not None
        and disc_number is not None
    ):
        disc_pos_key = (disc_number, track_number)
        str_disc_key = f"{disc_number}-{track_number}"
        disc_entry = tracks_by_pos.get(disc_pos_key) or tracks_by_pos.get(str_disc_key)
        if isinstance(disc_entry, dict):
            return disc_entry, True

    if isinstance(tracks_by_title, dict) and title and not _is_generic_title(title):
        clean_key = normalize_str(clean_title(title))
        if clean_key in tracks_by_title and isinstance(
            tracks_by_title[clean_key], dict
        ):
            return tracks_by_title[clean_key], True
        norm_key = normalize_str(title)
        if norm_key in tracks_by_title and isinstance(tracks_by_title[norm_key], dict):
            return tracks_by_title[norm_key], True

    if isinstance(tracks_by_pos, dict) and track_number in tracks_by_pos:
        track_entry = tracks_by_pos[track_number]
        if isinstance(track_entry, dict):
            return track_entry, True
    return None, False


def _enrich_shazam(
    track_info: TrackInfo,
    file_path: Path,
    target_album_artist: str | None = None,
    target_album_title: str | None = None,
    has_album_context: bool = False,
    force: bool = False,
) -> None:
    """Identify track via Shazam acoustic recognition if tags are missing, generic, or unverified."""
    if not get_config().enable_shazam:
        return
    is_missing_metadata = (
        not track_info.title
        or _is_generic_title(track_info.title)
        or not track_info.artist
        or _is_generic(track_info.artist, "artist")
    )
    if not is_missing_metadata and not force:
        return

    try:
        shazam_match = recognize_audio_track(file_path)
        if not shazam_match:
            return

        # If target_album_artist is known, do not allow Shazam to overwrite a track whose artist already
        # matches the album artist with a completely different artist (sample hijacking)!
        if (
            target_album_artist
            and track_info.artist
            and fuzz.ratio(
                normalize_str(target_album_artist), normalize_str(track_info.artist)
            )
            >= 60
            and fuzz.ratio(
                normalize_str(target_album_artist),
                normalize_str(shazam_match.artist),
            )
            < 50
        ):
            return

        if has_album_context:
            if is_missing_metadata:
                track_info.title = preserve_unicode_repertoire(
                    track_info.title, shazam_match.title
                )
                if not target_album_artist or _is_generic(
                    target_album_artist, "artist"
                ):
                    track_info.artist = resolve_artist_name(shazam_match.artist)
                else:
                    track_info.artist = target_album_artist
            if target_album_title and not _is_generic(target_album_title, "album"):
                track_info.album = target_album_title
            elif not track_info.album or _is_generic(track_info.album, "album"):
                resolved_album = shazam_match.album or target_album_title
                if resolved_album:
                    track_info.album = preserve_unicode_repertoire(
                        track_info.album, resolved_album
                    )
        else:
            if is_missing_metadata or force:
                track_info.title = preserve_unicode_repertoire(
                    track_info.title, shazam_match.title
                )
                track_info.artist = resolve_artist_name(shazam_match.artist)
            if shazam_match.album and (
                not track_info.album or _is_generic(track_info.album, "album") or force
            ):
                track_info.album = preserve_unicode_repertoire(
                    track_info.album, shazam_match.album
                )
        if shazam_match.genre and not track_info.genre:
            track_info.genre = shazam_match.genre
        if shazam_match.apple_music_id and not track_info.itunes_trackid:
            track_info.itunes_trackid = shazam_match.apple_music_id
        if shazam_match.isrc and not track_info.isrc:
            track_info.isrc = shazam_match.isrc
        if shazam_match.release_date and not track_info.date:
            track_info.date = shazam_match.release_date
        if shazam_match.label and not track_info.label:
            track_info.label = shazam_match.label
        if shazam_match.lyrics and not track_info.lyrics:
            track_info.lyrics = shazam_match.lyrics
        LOG.info(
            f"   ∟ ⚡ [Shazam] Identified: [white]{escape(shazam_match.artist)} - {escape(shazam_match.title)}[/]"
        )
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"Shazam acoustic recognition failed: {error}")


def _is_strong_track_match(
    cur_artist: str | None,
    cur_title: str | None,
    cand_artist: str | None,
    cand_title: str | None,
) -> bool:
    """Verifies track match respecting Rule 7 short-string fuzzy thresholds."""
    if not cand_title or not cur_title:
        return False
    clean_cur = clean_title(cur_title).lower()
    clean_cand = clean_title(cand_title).lower()
    if clean_cur == clean_cand:
        return True
    score = match_score(
        cur_artist or "",
        cur_title,
        cand_artist or "",
        cand_title,
    )
    min_len = min(len(clean_cur), len(clean_cand))
    thresh = 95.0 if min_len < 8 else ARTIST_MATCH_THRESHOLD
    return score >= thresh


def _find_album_release_recording(
    track_info: TrackInfo,
    album_mb_release_details: dict[str, Any] | None,
    disc_num: int,
) -> dict[str, Any] | None:
    """Lookup candidate recording in pre-fetched album release details."""
    if not album_mb_release_details or not track_info.track_number:
        return None
    tracks_by_disc = album_mb_release_details.get("tracks_by_disc_and_position")
    if (
        isinstance(tracks_by_disc, dict)
        and (disc_num, track_info.track_number) in tracks_by_disc
    ):
        candidate = tracks_by_disc[(disc_num, track_info.track_number)]
        return candidate if isinstance(candidate, dict) else None

    tracks_by_pos = album_mb_release_details.get("tracks_by_position")
    if isinstance(tracks_by_pos, dict):
        candidate = (
            tracks_by_pos.get((disc_num, track_info.track_number))
            or tracks_by_pos.get(f"{disc_num}-{track_info.track_number}")
            or tracks_by_pos.get(track_info.track_number)
        )
        return candidate if isinstance(candidate, dict) else None
    return None


def _heal_track_identity_from_release(
    track_info: TrackInfo,
    candidate_rec: dict[str, Any] | None,
) -> bool:
    """Heals corrupted track title if ISRC matches album recording."""
    if not candidate_rec or not track_info.isrc:
        return False
    rec_isrc = candidate_rec.get("isrc")
    rec_title = candidate_rec.get("title")
    if (
        rec_isrc
        and str(track_info.isrc).strip().upper() == str(rec_isrc).strip().upper()
        and rec_title
        and clean_title(track_info.title).lower() != clean_title(str(rec_title)).lower()
    ):
        LOG.info(
            f"   ∟ 🩹 [Healer] Track {track_info.track_number} verified by ISRC match. "
            f"Healing corrupted title '{escape(str(track_info.title))}' -> '{escape(str(rec_title))}'"
        )
        track_info.title = preserve_unicode_repertoire(track_info.title, str(rec_title))
        return True
    return False


def _resolve_musicbrainz_track_id(
    track_info: TrackInfo,
    album_track_mbids: dict[Any, str] | None,
    album_mb_release_details: dict[str, Any] | None,
    candidate_rec: dict[str, Any] | None,
    has_corrupt_identity: bool,
    force: bool = False,
) -> None:
    """Resolve track MBID via album tracks position mapping or track search."""
    album_mbids = album_track_mbids or {}
    disc_num = track_info.disc_number or 1
    disc_pos_key = (disc_num, track_info.track_number)
    str_disc_pos_key = f"{disc_num}-{track_info.track_number}"
    has_pos_match = bool(
        track_info.track_number
        and (
            disc_pos_key in album_mbids
            or str_disc_pos_key in album_mbids
            or track_info.track_number in album_mbids
        )
    )
    candidate_mbid: str | None = None
    if (
        not is_valid_uuid(track_info.musicbrainz_trackid)
        or force
        or has_corrupt_identity
    ) and has_pos_match:
        potential_mbid = (
            album_mbids.get(disc_pos_key)
            or album_mbids.get(str_disc_pos_key)
            or album_mbids.get(track_info.track_number)
        )
        is_match = (
            has_corrupt_identity
            or not track_info.title
            or _is_generic_title(track_info.title)
        )
        if not is_match and candidate_rec:
            cand_title = str(candidate_rec.get("title") or "")
            cand_artist = str(candidate_rec.get("artist") or "")
            if _is_strong_track_match(
                track_info.artist, track_info.title, cand_artist, cand_title
            ):
                is_match = True
        elif not album_mb_release_details:
            is_match = True

        if is_match and is_valid_uuid(potential_mbid):
            candidate_mbid = potential_mbid
            track_info.musicbrainz_trackid = candidate_mbid
            LOG.info(
                f"   ∟ 🏷️ [MusicBrainz Album Match] Found MBID: {candidate_mbid[:8]}..."
            )
    elif not is_valid_uuid(track_info.musicbrainz_trackid) or (
        force and not track_info.is_alien and not candidate_mbid
    ):
        mbid = fetch_track_mbid(track_info.artist, track_info.title)
        if is_valid_uuid(mbid):
            track_info.musicbrainz_trackid = mbid
            LOG.info(f"   ∟ 🏷️ [MusicBrainz] Found MBID: {mbid[:8]}...")


def _resolve_musicbrainz_album_id(
    track_info: TrackInfo,
    album_mbid: str | None,
    album_mb_release_details: dict[str, Any] | None,
    force: bool = False,
) -> None:
    """Resolve album MBID from release context or catalog search."""
    if is_valid_uuid(album_mbid):
        if (
            force
            or not is_valid_uuid(track_info.musicbrainz_albumid)
            or album_mb_release_details is not None
        ):
            track_info.musicbrainz_albumid = str(album_mbid)
    elif not is_valid_uuid(track_info.musicbrainz_albumid):
        search_artist = track_info.album_artist or track_info.artist
        release = search_musicbrainz_release(
            search_artist,
            track_info.album,
            expected_track_count=track_info.total_tracks,
        )
        if release:
            musicbrainz_id = release.get("id")
            if is_valid_uuid(str(musicbrainz_id)):
                track_info.musicbrainz_albumid = str(musicbrainz_id)
            if not track_info.date:
                date_str = release.get("date")
                if isinstance(date_str, str) and len(date_str) >= 4:
                    track_info.date = normalize_date(date_str)


def _extract_release_recording(
    track_info: TrackInfo,
    album_mb_release_details: dict[str, Any] | None,
    candidate_rec: dict[str, Any] | None,
    has_corrupt_identity: bool,
) -> dict[str, Any] | None:
    """Extracts candidate recording from pre-fetched album release details."""
    if not album_mb_release_details:
        return None
    tracks_by_id = album_mb_release_details.get("tracks_by_mbid", {})
    if (
        isinstance(tracks_by_id, dict)
        and track_info.musicbrainz_trackid in tracks_by_id
    ):
        rec = tracks_by_id[track_info.musicbrainz_trackid]
        if isinstance(rec, dict):
            return rec

    if candidate_rec:
        cand_title = str(candidate_rec.get("title") or "")
        cand_artist = str(candidate_rec.get("artist") or "")
        cand_isrc = candidate_rec.get("isrc")
        isrc_match = bool(
            track_info.isrc
            and cand_isrc
            and str(track_info.isrc).strip().upper() == str(cand_isrc).strip().upper()
        )
        title_match = _is_strong_track_match(
            track_info.artist, track_info.title, cand_artist, cand_title
        )
        if (
            has_corrupt_identity
            or isrc_match
            or title_match
            or not track_info.title
            or _is_generic_title(track_info.title)
        ):
            return candidate_rec
    return None


def _fetch_standalone_recording(
    track_info: TrackInfo,
    force: bool = False,
) -> dict[str, Any] | None:
    """Fetches standalone recording details from MusicBrainz API with identity verification."""
    if not is_valid_uuid(track_info.musicbrainz_trackid):
        return None
    fetched_rec = fetch_musicbrainz_recording_details(track_info.musicbrainz_trackid)
    if not fetched_rec or not isinstance(fetched_rec, dict):
        return None
    rec_isrc = fetched_rec.get("isrc")
    isrc_matches = bool(
        track_info.isrc
        and rec_isrc
        and str(track_info.isrc).strip().upper() == str(rec_isrc).strip().upper()
    )
    rec_artist = str(fetched_rec.get("artist") or "")
    rec_title = str(fetched_rec.get("title") or "")
    cand_artists = [
        a
        for a in (
            track_info.artist,
            track_info.album_artist,
            track_info.featured_artists,
        )
        if a
    ]
    title_sim = max(
        (match_score(a, track_info.title, rec_artist, rec_title) for a in cand_artists),
        default=0.0,
    )
    min_len = (
        min(len(clean_title(track_info.title)), len(clean_title(rec_title)))
        if rec_title
        else 0
    )
    thresh = 95.0 if min_len < 8 else ARTIST_MATCH_THRESHOLD
    if not rec_title or isrc_matches or title_sim >= thresh:
        return fetched_rec

    LOG.debug(
        f"Discarding mismatched pre-existing MBID {escape(str(track_info.musicbrainz_trackid))} "
        f"('{escape(str(rec_artist))} - {escape(str(rec_title))}') "
        f"for track '{escape(str(track_info.artist))} - {escape(str(track_info.title))}'"
    )
    track_info.musicbrainz_trackid = None
    if force:
        new_mbid = fetch_track_mbid(track_info.artist, track_info.title)
        if is_valid_uuid(new_mbid):
            track_info.musicbrainz_trackid = new_mbid
            fresh_rec = fetch_musicbrainz_recording_details(new_mbid)
            if fresh_rec and isinstance(fresh_rec, dict):
                return fresh_rec
    return None


def _fetch_or_extract_mb_recording(
    track_info: TrackInfo,
    album_mb_release_details: dict[str, Any] | None,
    candidate_rec: dict[str, Any] | None,
    has_corrupt_identity: bool,
    force: bool = False,
) -> tuple[dict[str, Any] | None, bool]:
    """
    Extracts recording details from pre-fetched album release details,
    or fetches standalone recording details via network.
    Returns (mb_rec, is_album_rec).
    """
    release_rec = _extract_release_recording(
        track_info, album_mb_release_details, candidate_rec, has_corrupt_identity
    )
    if release_rec is not None:
        return release_rec, True
    standalone_rec = _fetch_standalone_recording(track_info, force=force)
    return standalone_rec, False


def _should_update_track_artist(
    track_info: TrackInfo,
    mb_rec: dict[str, Any],
    cleaned_artist: str,
    is_album_rec: bool,
    target_album_artist: str | None,
    has_corrupt_identity: bool,
    force: bool = False,
) -> bool:
    """Evaluates whether track artist should be updated from MusicBrainz recording."""
    if has_corrupt_identity or not track_info.artist:
        return True
    if track_info.artist.lower() in ("unknown", "unknown artist"):
        return True
    if track_info.is_alien:
        return True

    cur_primary = re.sub(
        r"^(?:the|a|an)\s+",
        "",
        clean_title(get_primary_artist(track_info.artist or "")).lower(),
    ).strip()
    cand_primary = re.sub(
        r"^(?:the|a|an)\s+",
        "",
        clean_title(get_primary_artist(cleaned_artist)).lower(),
    ).strip()
    primary_matches = bool(
        cur_primary
        and cand_primary
        and (
            cur_primary == cand_primary or fuzz.ratio(cur_primary, cand_primary) >= 80.0
        )
    )

    rec_title_str = str(mb_rec.get("title") or track_info.title)
    is_strong = _is_strong_track_match(
        track_info.artist, track_info.title, cleaned_artist, rec_title_str
    )

    if force and primary_matches and is_strong:
        return True

    if is_album_rec and track_info.artist != cleaned_artist:
        alb_artist = target_album_artist or track_info.album_artist
        is_various = bool(
            alb_artist and alb_artist.lower() in ("various artists", "soundtrack")
        )
        matches_album_artist = bool(
            alb_artist
            and fuzz.ratio(clean_title(alb_artist).lower(), cand_primary) >= 80.0
        )
        if primary_matches or is_various or matches_album_artist or is_strong:
            return True

    return False


def _apply_mb_recording_metadata(
    track_info: TrackInfo,
    mb_rec: dict[str, Any],
    is_album_rec: bool,
    target_album_artist: str | None,
    has_corrupt_identity: bool,
    force: bool = False,
) -> None:
    """Applies recording metadata and resolves track artist updates."""
    mb_map = {
        "isrc": "isrc",
        "disambiguation": "disambiguation",
        "composer": "composer",
        "lyricist": "lyricist",
        "producers": "producers",
        "remixer": "remixer",
        "musicbrainz_workid": "musicbrainz_workid",
        "musicbrainz_artistid": "musicbrainz_artistid",
        "musicbrainz_albumartistid": "musicbrainz_albumartistid",
        "disc_number": "disc_number",
        "total_discs": "total_discs",
        "disc_subtitle": "disc_subtitle",
    }
    rec_title = str(mb_rec.get("title") or "")
    rec_artist = str(mb_rec.get("artist") or "")
    title_matches = _is_strong_track_match(
        track_info.artist, track_info.title, rec_artist, rec_title
    )
    fn_title_matches = _is_strong_track_match(
        track_info.artist,
        clean_title(track_info.file_path.stem),
        rec_artist,
        rec_title,
    )
    if (
        has_corrupt_identity
        or not track_info.title
        or track_info.title == "Untitled"
        or _is_generic_title(track_info.title)
        or (force and (title_matches or fn_title_matches))
    ) and rec_title:
        mb_map["title"] = "title"

    if mb_rec.get("first-release-date") and (force or not track_info.date):
        mb_map["first-release-date"] = "date"

    _apply_mapping(track_info, mb_rec, mb_map, force=force or has_corrupt_identity)

    if mb_rec.get("disc_total_tracks") and (force or not track_info.total_tracks):
        disc_tot = safe_int(mb_rec["disc_total_tracks"])
        if disc_tot is not None:
            track_info.total_tracks = disc_tot

    mb_artist_val = mb_rec.get("artist")
    if mb_artist_val and isinstance(mb_artist_val, str):
        cleaned_artist = preserve_unicode_repertoire(
            track_info.artist,
            clean_unicode_punct(resolve_artist_name(mb_artist_val)),
        )
        if _should_update_track_artist(
            track_info,
            mb_rec,
            cleaned_artist,
            is_album_rec,
            target_album_artist,
            has_corrupt_identity,
            force=force,
        ):
            track_info.artist = cleaned_artist


def _validate_mb_release_artist(rel_art: str, cmp_art: str) -> bool:
    """Validates that release artist matches album artist or track artist."""
    if not rel_art or not cmp_art or _is_generic(cmp_art, "artist"):
        return True
    rel_primary = (
        re.sub(r"^(?:the|a|an)\s+", "", clean_title(get_primary_artist(rel_art)))
        .strip()
        .lower()
    )
    cmp_primary = (
        re.sub(r"^(?:the|a|an)\s+", "", clean_title(get_primary_artist(cmp_art)))
        .strip()
        .lower()
    )
    return (
        max(
            fuzz.ratio(rel_primary, cmp_primary),
            fuzz.token_sort_ratio(rel_primary, cmp_primary),
        )
        >= ALBUM_MATCH_THRESHOLD
    )


def _validate_mb_release_title(rel_title: str, cur_album: str | None) -> bool:
    """Validates that release title matches current album title."""
    if not rel_title or not cur_album or _is_generic(cur_album, "album"):
        return True
    norm_rel = clean_title(rel_title).lower()
    norm_alb = clean_title(cur_album).lower()
    thresh = 95.0 if min(len(norm_rel), len(norm_alb)) < 8 else ALBUM_MATCH_THRESHOLD
    return (
        max(
            fuzz.ratio(norm_rel, norm_alb),
            fuzz.token_sort_ratio(norm_rel, norm_alb),
        )
        >= thresh
    )


def _apply_mb_release_metadata(
    track_info: TrackInfo,
    album_mbid: str | None,
    album_mb_release_details: dict[str, Any] | None,
    force: bool = False,
) -> None:
    """Applies release-level metadata, validating album artist and album title."""
    if not is_valid_uuid(track_info.musicbrainz_albumid):
        return

    mb_rel = (
        album_mb_release_details
        if (
            album_mb_release_details
            and album_mbid
            and track_info.musicbrainz_albumid == str(album_mbid)
        )
        else fetch_musicbrainz_release_details(track_info.musicbrainz_albumid)
    )
    if not mb_rel or not isinstance(mb_rel, dict):
        return

    rel_art = str(mb_rel.get("album_artist") or mb_rel.get("artist") or "")
    cmp_art = track_info.album_artist or track_info.artist or ""
    if not _validate_mb_release_artist(rel_art, cmp_art):
        track_info.musicbrainz_albumid = None
        track_info.musicbrainz_releasegroupid = None
        return

    rel_title = str(mb_rel.get("title") or "").strip()
    if not _validate_mb_release_title(rel_title, track_info.album):
        track_info.musicbrainz_albumid = None
        track_info.musicbrainz_releasegroupid = None
        return

    mb_rel_map = {
        "barcode": "barcode",
        "release_country": "release_country",
        "release_status": "release_status",
        "release_type": "release_type",
        "musicbrainz_releasegroupid": "musicbrainz_releasegroupid",
        "label": "label",
        "catalog_number": "catalog_number",
        "media": "media",
        "language": "language",
        "script": "script",
        "artist_sort": "artist_sort",
        "total_discs": "total_discs",
        "musicbrainz_albumartistid": "musicbrainz_albumartistid",
    }
    if not track_info.musicbrainz_artistid and mb_rel.get("musicbrainz_artistid"):
        mb_rel_map["musicbrainz_artistid"] = "musicbrainz_artistid"
    if mb_rel.get("date") and (force or not track_info.date):
        mb_rel_map["date"] = "date"
    if mb_rel.get("original_date"):
        mb_rel_map["original_date"] = "original_date"

    _apply_mapping(track_info, mb_rel, mb_rel_map, force=force)

    if mb_rel.get("title") and (
        (
            album_mb_release_details is not None
            and (force or track_info.album != mb_rel.get("title"))
        )
        or not track_info.album
        or track_info.album.lower() in ("unknown", "unknown album")
    ):
        track_info.album = preserve_unicode_repertoire(
            track_info.album, clean_unicode_punct(str(mb_rel["title"]))
        )

    if mb_rel.get("album_artist") and (
        (
            album_mb_release_details is not None
            and (force or track_info.album_artist != mb_rel.get("album_artist"))
        )
        or not track_info.album_artist
        or track_info.album_artist.lower() in ("unknown", "unknown artist")
    ):
        track_info.album_artist = preserve_unicode_repertoire(
            track_info.album_artist,
            clean_unicode_punct(resolve_artist_name(str(mb_rel["album_artist"]))),
        )

    if album_mb_release_details is not None:
        if force or not track_info.total_tracks:
            total_tracks = safe_int(mb_rel.get("total_tracks"))
            if total_tracks is not None:
                cur_tot = track_info.total_tracks or 0
                cur_num = track_info.track_number or 0
                if total_tracks >= cur_tot and total_tracks >= cur_num:
                    track_info.total_tracks = total_tracks
        if force or not track_info.total_discs:
            total_discs = safe_int(mb_rel.get("total_discs"))
            if total_discs is not None:
                track_info.total_discs = total_discs


def _enrich_musicbrainz(
    track_info: TrackInfo,
    album_mbid: str | None,
    album_track_mbids: dict[Any, str] | None,
    album_mb_release_details: dict[str, Any] | None,
    target_album_artist: str | None = None,
    force: bool = False,
) -> None:
    try:
        disc_num = track_info.disc_number or 1
        candidate_rec = _find_album_release_recording(
            track_info, album_mb_release_details, disc_num
        )
        has_corrupt_identity = _heal_track_identity_from_release(
            track_info, candidate_rec
        )

        _resolve_musicbrainz_track_id(
            track_info,
            album_track_mbids,
            album_mb_release_details,
            candidate_rec,
            has_corrupt_identity,
            force=force,
        )
        _resolve_musicbrainz_album_id(
            track_info, album_mbid, album_mb_release_details, force=force
        )

        mb_rec, is_album_rec = _fetch_or_extract_mb_recording(
            track_info,
            album_mb_release_details,
            candidate_rec,
            has_corrupt_identity,
            force=force,
        )
        if mb_rec:
            _apply_mb_recording_metadata(
                track_info,
                mb_rec,
                is_album_rec,
                target_album_artist,
                has_corrupt_identity,
                force=force,
            )

        _apply_mb_release_metadata(
            track_info, album_mbid, album_mb_release_details, force=force
        )
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"MusicBrainz enrichment failed for {track_info.title}: {error}")


def _enrich_itunes(
    track_info: TrackInfo,
    album_itunes_details: dict[str, Any] | None = None,
    force: bool = False,
) -> None:
    try:
        api_payload: dict[str, Any] | None = None
        is_album_match = False
        if album_itunes_details:
            api_payload, is_album_match = _find_track_in_album_mapping(
                album_itunes_details.get("tracks_by_title"),
                album_itunes_details.get("tracks_by_number"),
                track_info.title,
                track_info.track_number,
                disc_number=track_info.disc_number,
                tracks_by_disc_and_position=album_itunes_details.get(
                    "tracks_by_disc_and_number"
                ),
            )

        if not api_payload and (not track_info.genre or not track_info.advisory):
            api_payload = fetch_itunes_track_metadata(
                track_info.artist, track_info.title
            )

        if api_payload and isinstance(api_payload, dict):
            itunes_map = {
                "genre": "genre",
                "advisory": "advisory",
                "copyright": "copyright",
                "itunes_trackid": "itunes_trackid",
                "itunes_collectionid": "itunes_collectionid",
                "itunes_artistid": "itunes_artistid",
                "release_country": "release_country",
            }
            if is_album_match and not is_valid_uuid(track_info.musicbrainz_albumid):
                if not track_info.disc_number:
                    itunes_map["disc_number"] = "disc_number"
                if not track_info.total_discs:
                    itunes_map["total_discs"] = "total_discs"
            if not track_info.date or (
                force and not is_valid_uuid(track_info.musicbrainz_albumid)
            ):
                itunes_map["date"] = "date"
            if (
                not track_info.title or track_info.title == "Untitled"
            ) and api_payload.get("trackName"):
                itunes_map["trackName"] = "title"
            _apply_mapping(
                track_info,
                api_payload,
                itunes_map,
                force=force,
            )
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"iTunes enrichment failed for {track_info.title}: {error}")


def _enrich_lastfm(track_info: TrackInfo, lastfm_api_key: str | None) -> None:
    if not lastfm_api_key or track_info.genre:
        return
    try:
        tags = fetch_lastfm_tags(
            track_info.artist,
            track_info.title,
            api_key=lastfm_api_key,
            mbid=track_info.musicbrainz_trackid,
        )
        if tags:
            if not track_info.genre:
                raw_genre = tags[0]
                normalized_genre = normalize_genre(raw_genre)
                if normalized_genre:
                    track_info.genre = normalized_genre
                    LOG.info(f"   ∟ 🏷️ [Last.fm] Genre: [cyan]{normalized_genre}[/]")
            if len(tags) > 1 and not track_info.style:
                subgenres = [
                    t
                    for t in tags[1:4]
                    if t.lower() != (track_info.genre or "").lower()
                ]
                if subgenres:
                    track_info.style = ", ".join(subgenres)
            if not track_info.mood:
                mood_keywords = {
                    "chill",
                    "dark",
                    "sad",
                    "happy",
                    "energetic",
                    "melancholic",
                    "relax",
                    "relaxing",
                    "aggressive",
                    "ambient",
                    "party",
                    "romantic",
                    "hype",
                    "mellow",
                    "atmospheric",
                    "epic",
                    "somber",
                    "upbeat",
                }
                for t in tags:
                    if t.lower() in mood_keywords:
                        track_info.mood = t.title()
                        break
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"Last.fm enrichment failed for {track_info.title}: {error}")


def _enrich_discogs(
    track_info: TrackInfo,
    discogs_user_token: str | None,
    album_discogs_release: dict[str, Any] | None,
    has_album_context: bool = False,
    force: bool = False,
) -> None:
    if not discogs_user_token:
        return
    try:
        release = album_discogs_release or (
            search_discogs_release(
                track_info.artist,
                track_info.album,
                user_token=discogs_user_token,
                expected_track_count=track_info.total_tracks,
            )
            if not has_album_context
            else None
        )
        if not release:
            return

        styles_val = release.get("styles")
        if isinstance(styles_val, list) and styles_val and not track_info.style:
            track_info.style = ", ".join(str(s) for s in styles_val[:3])

        genres_val = release.get("genres")
        if isinstance(genres_val, list) and genres_val and not track_info.genre:
            track_info.genre = normalize_genre(str(genres_val[0])) or track_info.genre

        discogs_rel_map = {
            "id": "discogs_release_id",
            "artist_id": "discogs_artist_id",
            "country": "release_country",
            "label": "label",
            "catalog_number": "catalog_number",
            "barcode": "barcode",
            "media": "media",
            "composer": "composer",
            "producers": "producers",
            "remixer": "remixer",
        }
        if not track_info.date or (
            force and not is_valid_uuid(track_info.musicbrainz_albumid)
        ):
            discogs_rel_map["released"] = "date"
            discogs_rel_map["year"] = "date"
        has_authoritative_mb = is_valid_uuid(
            track_info.musicbrainz_trackid
        ) or is_valid_uuid(track_info.musicbrainz_albumid)
        _apply_mapping(
            track_info,
            release,
            discogs_rel_map,
            force=False if has_authoritative_mb else force,
        )

        track_credits_dict = release.get("track_credits")
        if isinstance(track_credits_dict, dict):
            track_key = (
                str(track_info.track_number) if track_info.track_number else None
            )
            specific = None
            if track_key and track_key in track_credits_dict:
                specific = track_credits_dict[track_key]
            elif track_info.title and track_info.title.lower() in track_credits_dict:
                specific = track_credits_dict[track_info.title.lower()]
            if isinstance(specific, dict):
                discogs_track_map = {
                    "producers": "producers",
                    "remixer": "remixer",
                    "composer": "composer",
                }
                if (
                    not track_info.title or track_info.title == "Untitled"
                ) and specific.get("title"):
                    discogs_track_map["title"] = "title"
                _apply_mapping(
                    track_info,
                    specific,
                    discogs_track_map,
                    force=False if has_authoritative_mb else force,
                )
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"Discogs enrichment failed for {track_info.title}: {error}")


def _enrich_deezer(
    track_info: TrackInfo,
    album_deezer_details: dict[str, Any] | None,
    has_album_context: bool = False,
    force: bool = False,
) -> None:
    try:
        album = album_deezer_details or (
            fetch_deezer_album_details(track_info.artist, track_info.album)
            if not has_album_context
            else None
        )
        if album and isinstance(album, dict):
            deezer_album_map = {
                "genre": "genre",
                "label": "label",
                "barcode": "barcode",
            }
            if not track_info.date or (
                force and not is_valid_uuid(track_info.musicbrainz_albumid)
            ):
                deezer_album_map["release_date"] = "date"
            _apply_mapping(
                track_info,
                album,
                deezer_album_map,
                force=force,
            )
            if (
                album_deezer_details is not None
                and not is_valid_uuid(track_info.musicbrainz_albumid)
                and album.get("title")
                and (
                    force
                    or not track_info.album
                    or track_info.album.lower() in ("unknown", "unknown album")
                    or track_info.album != album.get("title")
                )
            ):
                track_info.album = preserve_unicode_repertoire(
                    track_info.album, clean_unicode_punct(str(album["title"]))
                )
            if (
                album_deezer_details is not None
                and not is_valid_uuid(track_info.musicbrainz_albumid)
                and (force or not track_info.total_tracks)
            ):
                nb_tracks = safe_int(album.get("nb_tracks"))
                if nb_tracks is not None:
                    cur_tot = track_info.total_tracks or 0
                    cur_num = track_info.track_number or 0
                    if nb_tracks >= cur_tot and nb_tracks >= cur_num:
                        track_info.total_tracks = nb_tracks

        track: dict[str, Any] | None = None
        track_is_from_album = False
        if album and isinstance(album, dict):
            track, track_is_from_album = _find_track_in_album_mapping(
                album.get("tracks_by_title"),
                album.get("tracks_by_position"),
                track_info.title,
                track_info.track_number,
                disc_number=track_info.disc_number,
                tracks_by_disc_and_position=album.get("tracks_by_disc_and_position"),
            )

        if not track and track_info.isrc:
            track = fetch_deezer_track_by_isrc(track_info.isrc)

        if not track and (not track_info.isrc or not track_info.producers or force):
            track = fetch_deezer_track_details(track_info.artist, track_info.title)

        if not track or not isinstance(track, dict):
            return

        track_pos = safe_int(track.get("track_position"))
        if (
            track_pos is not None
            and track_info.track_number is None
            and (track_is_from_album or not has_album_context)
        ):
            track_info.track_number = track_pos
        disk_num = safe_int(track.get("disk_number"))
        if (
            disk_num is not None
            and track_info.disc_number is None
            and (track_is_from_album or not has_album_context)
        ):
            track_info.disc_number = disk_num
        if track.get("explicit_lyrics"):
            track_info.advisory = "Explicit"
        deezer_track_map = {
            "isrc": "isrc",
            "composer": "composer",
            "lyricist": "lyricist",
            "featured_artists": "featured_artists",
            "producers": "producers",
        }
        if not track_info.date:
            deezer_track_map["release_date"] = "date"
        if (not track_info.title or track_info.title == "Untitled") and track.get(
            "title"
        ):
            deezer_track_map["title"] = "title"
        if not track_info.bpm and track.get("bpm"):
            safe_bpm_val = safe_float(track.get("bpm"))
            if safe_bpm_val and safe_bpm_val > 0:
                track_info.bpm = round(safe_bpm_val, 1)

        has_authoritative_mb = is_valid_uuid(
            track_info.musicbrainz_trackid
        ) or is_valid_uuid(track_info.musicbrainz_albumid)
        _apply_mapping(
            track_info,
            track,
            deezer_track_map,
            force=False if has_authoritative_mb else force,
        )
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"Deezer enrichment failed for {track_info.title}: {error}")


def _enrich_genius(
    track_info: TrackInfo,
    genius_api_token: str | None,
    force: bool = False,
) -> None:
    if not genius_api_token:
        return
    # Short-circuit if composer and producers are already populated (unless force)
    if track_info.composer and track_info.producers and not force:
        return
    try:
        genius_details = fetch_genius_song_details(
            track_info.artist, track_info.title, api_token=genius_api_token
        )
        if not genius_details:
            return
        genius_map = {
            "genius_song_id": "genius_song_id",
            "writers": "composer",
            "description": "comment",
            "featured_artists": "featured_artists",
            "producers": "producers",
        }
        if not track_info.date:
            genius_map["release_date"] = "date"
        _apply_mapping(
            track_info,
            genius_details,
            genius_map,
            force=force,
        )
        if genius_details.get("writers") and not track_info.lyricist:
            track_info.lyricist = str(genius_details["writers"])
        LOG.debug("   ∟ 📝 [Genius] Fetched song details & credits")
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"Genius enrichment failed for {track_info.title}: {error}")


def _enrich_theaudiodb(track_info: TrackInfo, force: bool = False) -> None:
    if not (
        force
        or not track_info.music_video_url
        or not track_info.mood
        or not track_info.initial_key
        or track_info.rating is None
        or not track_info.comment
    ):
        return
    try:
        tadb_details = fetch_theaudiodb_track_details(
            track_info.artist, track_info.title
        )
        if not tadb_details:
            return
        if track_info.rating is None:
            rating_val = safe_float(tadb_details.get("rating"))
            if rating_val is not None:
                track_info.rating = rating_val
        tadb_map = {
            "music_video_url": "music_video_url",
            "mood": "mood",
            "style": "style",
            "initial_key": "initial_key",
            "description": "comment",
        }
        if not track_info.genre:
            tadb_map["genre"] = "genre"
        _apply_mapping(
            track_info,
            tadb_details,
            tadb_map,
            force=force,
        )
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"TheAudioDB enrichment failed for {track_info.title}: {error}")


def _enrich_cuesheet(
    track_info: TrackInfo,
    file_path: Path,
    cuesheet_content: str | None,
) -> None:
    if cuesheet_content:
        track_info.cuesheet = cuesheet_content
    elif not track_info.cuesheet:
        companion_cue = find_companion_cuesheet(file_path.parent)
        if companion_cue:
            track_info.cuesheet = read_cuesheet_content(companion_cue)


def _enrich_bpm(
    track_info: TrackInfo,
    file_path: Path,
    fetch_bpm: bool,
    force: bool = False,
) -> None:
    if fetch_bpm and (track_info.bpm is None or force):
        try:
            bpm = calculate_bpm(file_path)
            if bpm is not None:
                track_info.bpm = bpm
                LOG.info(f"   ∟ 🎵 BPM Calculated: [green]{bpm}[/]")
        except OSError as error:
            LOG.debug(f"BPM calculation failed for {track_info.title}: {error}")


def _enrich_key(
    track_info: TrackInfo,
    file_path: Path,
    fetch_key: bool,
    force: bool = False,
) -> None:
    is_key_invalid = bool(
        track_info.initial_key and key_to_camelot(track_info.initial_key) is None
    )
    if fetch_key and (track_info.initial_key is None or force or is_key_invalid):
        try:
            detected_key = detect_musical_key(file_path)
            if detected_key is not None:
                track_info.initial_key = detected_key
                camelot = key_to_camelot(detected_key) or "Unknown"
                LOG.info(
                    f"   ∟ 🎵 Musical Key: [green]{escape(detected_key)}[/] ({escape(camelot)})"
                )
        except OSError as error:
            LOG.debug(f"Key calculation failed for {track_info.title}: {error}")


def _enrich_artwork(
    track_info: TrackInfo,
    file_path: Path,
    fetch_itunes_art: bool,
    force: bool = False,
    dry_run: bool = False,
) -> Path | None:
    if not fetch_itunes_art:
        return None
    try:
        return process_album_cover_art(
            file_path.parent,
            track_info.artist,
            track_info.album,
            musicbrainz_album_id=track_info.musicbrainz_albumid,
            force=force,
            dry_run=dry_run,
        )
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"Cover art downloading failed for {track_info.title}: {error}")
        return None


def _enrich_lyrics(
    track_info: TrackInfo,
    file_path: Path,
    fetch_lyrics: bool,
    force: bool = False,
    dry_run: bool = False,
) -> None:
    if not fetch_lyrics:
        return
    try:
        lrc_path = file_path.with_suffix(".lrc")
        had_lrc = lrc_path.exists() and lrc_path.stat().st_size > 0
        resolved_duration = get_audio_duration(file_path)
        lyrics_text, tag_type = process_track_lyrics(
            file_path,
            track_info.artist,
            track_info.title,
            force=force,
            dry_run=dry_run,
            isrc=track_info.isrc,
            album_name=track_info.album,
            duration=resolved_duration,
        )
        if lyrics_text and tag_type:
            track_info.lyrics = lyrics_text
            if not had_lrc:
                LOG.info(
                    f"   ∟ [green]✅ Saved {tag_type} lyrics for {escape(file_path.name)}[/]"
                )
            elif force:
                LOG.info(
                    f"   ∟ [yellow]🔄 Updated {tag_type} lyrics for {escape(file_path.name)}[/]"
                )
        elif not had_lrc and track_info.lyrics and not dry_run:
            try:
                lrc_path.write_text(track_info.lyrics, encoding="utf-8")
                LOG.info(
                    f"   ∟ [green]✅ Saved plain lyrics for {escape(file_path.name)}[/]"
                )
            except OSError as write_err:
                LOG.debug(f"Failed to write fallback lyrics: {write_err}")
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"Lyrics fetch failed for {track_info.title}: {error}")


def _compute_tag_diffs(
    orig_info: TrackInfo, track_info: TrackInfo
) -> list[tuple[str, Any, Any]]:
    diff_entries: list[tuple[str, Any, Any]] = []
    for field_info in dataclasses.fields(TrackInfo):
        if field_info.name in _SKIP_DIFF_FIELDS or field_info.name.startswith("_"):
            continue
        old_val = getattr(orig_info, field_info.name)
        new_val = getattr(track_info, field_info.name)
        old_clean = None if old_val in (None, "", [], ()) else old_val
        new_clean = None if new_val in (None, "", [], ()) else new_val
        if old_clean != new_clean:
            diff_entries.append((field_info.name, old_clean, new_clean))
    return diff_entries


def _render_tag_diffs(orig_info: TrackInfo, track_info: TrackInfo) -> list[str]:
    diff_lines: list[str] = []
    for field_name, old_clean, new_clean in _compute_tag_diffs(orig_info, track_info):
        if old_clean is None:
            color, sym = "green", "+"
        elif new_clean is None:
            color, sym = "red", "-"
        else:
            color, sym = "yellow", "*"
        diff_lines.append(
            f"\n       [{color}][{sym}] {field_name}: {escape(str(old_clean))} -> {escape(str(new_clean))}[/]"
        )
    return diff_lines


def _candidate_title_matches(
    trk_candidate: dict[str, Any], cur_title: str | None, file_stem: str
) -> bool:
    c_title = str(trk_candidate.get("title") or trk_candidate.get("trackName") or "")
    if not c_title:
        return True
    norm_c = normalize_str(clean_title(c_title))
    norm_cur = (
        normalize_str(clean_title(cur_title))
        if cur_title and not _is_generic_title(cur_title)
        else None
    )
    norm_fn = normalize_str(clean_title(file_stem))
    if norm_cur and (
        fuzz.ratio(norm_c, norm_cur) >= 40
        or fuzz.token_set_ratio(norm_c, norm_cur) >= 45
    ):
        return True
    return bool(
        norm_fn
        and (
            fuzz.ratio(norm_c, norm_fn) >= 40
            or fuzz.token_set_ratio(norm_c, norm_fn) >= 45
        )
    )


def _get_album_track_details(
    pos: int,
    disc_number: int | None = None,
    album_mb_release_details: dict[str, Any] | None = None,
    album_deezer_details: dict[str, Any] | None = None,
    album_itunes_details: dict[str, Any] | None = None,
) -> tuple[str | None, str | None]:
    for details in (
        album_mb_release_details,
        album_deezer_details,
        album_itunes_details,
    ):
        if not details or not isinstance(details, dict):
            continue
        if disc_number is not None:
            disc_tracks = details.get("tracks_by_disc_and_position")
            if isinstance(disc_tracks, dict) and (disc_number, pos) in disc_tracks:
                rec = disc_tracks[(disc_number, pos)]
                if isinstance(rec, dict):
                    cand_title = (
                        str(rec.get("title") or rec.get("trackName") or "") or None
                    )
                    cand_artist = (
                        str(rec.get("artist") or rec.get("artistName") or "") or None
                    )
                    if cand_title:
                        return cand_artist, cand_title

        tracks = details.get("tracks_by_position")
        if isinstance(tracks, dict):
            rec = None
            if disc_number is not None and (disc_number, pos) in tracks:
                rec = tracks[(disc_number, pos)]
            elif disc_number is not None and f"{disc_number}-{pos}" in tracks:
                rec = tracks[f"{disc_number}-{pos}"]
            elif pos in tracks:
                rec = tracks[pos]
            if isinstance(rec, dict):
                cand_title = str(rec.get("title") or rec.get("trackName") or "") or None
                cand_artist = (
                    str(rec.get("artist") or rec.get("artistName") or "") or None
                )
                if cand_title:
                    return cand_artist, cand_title
    return None, None


def _pos_in_album(
    pos: int,
    disc_number: int | None = None,
    album_track_mbids: dict[Any, str] | None = None,
    album_deezer_details: dict[str, Any] | None = None,
    album_itunes_details: dict[str, Any] | None = None,
    album_mb_release_details: dict[str, Any] | None = None,
) -> bool:
    if album_track_mbids:
        if disc_number is not None and (
            (disc_number, pos) in album_track_mbids
            or f"{disc_number}-{pos}" in album_track_mbids
        ):
            return True
        if pos in album_track_mbids:
            return True
    _, cand_title = _get_album_track_details(
        pos,
        disc_number=disc_number,
        album_mb_release_details=album_mb_release_details,
        album_deezer_details=album_deezer_details,
        album_itunes_details=album_itunes_details,
    )
    return cand_title is not None


def _match_position_by_mbid(
    track_info: TrackInfo,
    clean_mbid: str,
    album_track_mbids: dict[Any, str] | None,
    album_mb_release_details: dict[str, Any] | None,
) -> int | None:
    """Matches track position and disc number directly from recording MBID."""
    if album_track_mbids:
        for key, rec_mbid in album_track_mbids.items():
            if is_valid_uuid(rec_mbid) and str(rec_mbid).strip().lower() == clean_mbid:
                if isinstance(key, tuple) and len(key) == 2:
                    disc_num = safe_int(key[0])
                    pos_num = safe_int(key[1])
                    if disc_num is not None:
                        track_info.disc_number = disc_num
                    if pos_num is not None:
                        return pos_num
                pos_val = safe_int(key)
                if pos_val is not None:
                    return pos_val
    if album_mb_release_details:
        t_by_mbid = album_mb_release_details.get("tracks_by_mbid")
        if isinstance(t_by_mbid, dict) and clean_mbid in t_by_mbid:
            rec = t_by_mbid[clean_mbid]
            if isinstance(rec, dict):
                if rec.get("disc_number"):
                    rec_disc = safe_int(rec["disc_number"])
                    if rec_disc is not None:
                        track_info.disc_number = rec_disc
                if rec.get("position") is not None:
                    return safe_int(rec["position"])
    return None


def _match_position_by_authoritative_id(
    track_info: TrackInfo,
    cur_pos: int | None,
    id_field: str,
    id_value: str,
    release_dicts: list[dict[str, Any] | None],
    cand_matcher: Callable[[dict[str, Any]], bool],
) -> int | None:
    """Matches track position and disc number from ISRC or iTunes ID across release metadata."""
    for details in release_dicts:
        if not details or not isinstance(details, dict):
            continue
        disc_tracks = details.get("tracks_by_disc_and_position")
        if isinstance(disc_tracks, dict):
            for (d_key, p_key), trk in disc_tracks.items():
                if isinstance(trk, dict):
                    val = trk.get(id_field) or (
                        trk.get("trackId") if id_field == "itunes_trackid" else None
                    )
                    if val and str(val).strip().upper() == id_value.upper():
                        matched_pos = safe_int(p_key) or safe_int(trk.get("position"))
                        matched_disc = safe_int(d_key) or safe_int(
                            trk.get("disc_number")
                        )
                        if matched_pos is not None and (
                            (cur_pos is not None and matched_pos == cur_pos)
                            or cand_matcher(trk)
                        ):
                            if matched_disc is not None:
                                track_info.disc_number = matched_disc
                            return matched_pos

        pos_tracks = details.get("tracks_by_position")
        if isinstance(pos_tracks, dict):
            for pos_key, trk in pos_tracks.items():
                if isinstance(trk, dict):
                    val = trk.get(id_field) or (
                        trk.get("trackId") if id_field == "itunes_trackid" else None
                    )
                    if val and str(val).strip().upper() == id_value.upper():
                        matched_pos = safe_int(pos_key)
                        if matched_pos is not None and (
                            (cur_pos is not None and matched_pos == cur_pos)
                            or cand_matcher(trk)
                        ):
                            if trk.get("disc_number"):
                                d_num = safe_int(trk["disc_number"])
                                if d_num is not None:
                                    track_info.disc_number = d_num
                            return matched_pos
    return None


def _build_album_tracks_by_disc_map(
    eff_disc: int | None,
    album_track_mbids: dict[Any, str] | None,
    release_details_list: list[dict[str, Any] | None],
) -> dict[int, set[int]]:
    """Builds a mapping of disc_number -> set of track positions present in the release."""
    tracks_by_disc: dict[int, set[int]] = {}
    default_disc = eff_disc or 1

    def _add_pos(d: Any, p: Any) -> None:
        p_int = safe_int(p)
        if p_int is not None:
            d_int = safe_int(d) or default_disc
            tracks_by_disc.setdefault(d_int, set()).add(p_int)

    if album_track_mbids:
        for k in album_track_mbids:
            if isinstance(k, tuple) and len(k) == 2:
                _add_pos(k[0], k[1])
            else:
                _add_pos(default_disc, k)

    for details in release_details_list:
        if details and isinstance(details, dict):
            disc_tracks = details.get("tracks_by_disc_and_position")
            if isinstance(disc_tracks, dict):
                for d, p in disc_tracks:
                    _add_pos(d, p)
            pos_tracks = details.get("tracks_by_position")
            if isinstance(pos_tracks, dict):
                for p in pos_tracks:
                    if isinstance(p, tuple) and len(p) == 2:
                        _add_pos(p[0], p[1])
                    else:
                        _add_pos(default_disc, p)
    return tracks_by_disc


def _search_release_positions_by_title(
    track_info: TrackInfo,
    discs_to_search: list[int],
    tracks_by_disc_map: dict[int, set[int]],
    album_mb_release_details: dict[str, Any] | None,
    album_deezer_details: dict[str, Any] | None,
    album_itunes_details: dict[str, Any] | None,
) -> int | None:
    """Searches across candidate album track titles for fuzzy or exact title matches."""
    best_match_disc: int | None = None
    best_pos: int | None = None
    best_score: float = 0.0

    for disc in discs_to_search:
        for pos in sorted(tracks_by_disc_map.get(disc, ())):
            cand_artist, cand_title = _get_album_track_details(
                pos,
                disc_number=disc,
                album_mb_release_details=album_mb_release_details,
                album_deezer_details=album_deezer_details,
                album_itunes_details=album_itunes_details,
            )
            if cand_title:
                clean_cand = clean_title(cand_title).lower()
                clean_eff = clean_title(track_info.title).lower()
                if clean_cand == clean_eff:
                    track_info.disc_number = disc
                    return pos
                score = max(
                    match_score(
                        track_info.artist,
                        track_info.title,
                        str(cand_artist or ""),
                        cand_title,
                    ),
                    float(fuzz.ratio(clean_eff, clean_cand)),
                    float(fuzz.token_sort_ratio(clean_eff, clean_cand)),
                )
                min_len = min(len(clean_eff), len(clean_cand))
                req_thresh = 95.0 if min_len < 8 else 85.0
                if score >= req_thresh and score > best_score:
                    best_score = score
                    best_pos = pos
                    best_match_disc = disc

    if best_pos is not None:
        if best_match_disc is not None:
            track_info.disc_number = best_match_disc
        return best_pos
    return None


def _resolve_album_track_position(
    track_info: TrackInfo,
    file_path: Path,
    album_mb_release_details: dict[str, Any] | None = None,
    album_deezer_details: dict[str, Any] | None = None,
    album_itunes_details: dict[str, Any] | None = None,
    album_track_mbids: dict[Any, str] | None = None,
) -> int | None:
    """
    Resolve the legitimate track position (1-indexed) of an audio file within an album release.
    Validates candidates against authoritative identifiers and title similarity.
    """
    fn_disc, fn_pos, _ = parse_track_filename(file_path.name)
    folder_disc = extract_disc_number_from_folder(file_path.parent.name)
    eff_disc = track_info.disc_number or fn_disc or folder_disc
    if track_info.disc_number is None and eff_disc is not None:
        track_info.disc_number = eff_disc

    # 1. Authoritative MusicBrainz identifier match
    if is_valid_uuid(track_info.musicbrainz_trackid):
        clean_mbid = str(track_info.musicbrainz_trackid).strip().lower()
        pos = _match_position_by_mbid(
            track_info, clean_mbid, album_track_mbids, album_mb_release_details
        )
        if pos is not None:
            return pos

    track_details_getter = functools.partial(
        _get_album_track_details,
        disc_number=eff_disc,
        album_mb_release_details=album_mb_release_details,
        album_deezer_details=album_deezer_details,
        album_itunes_details=album_itunes_details,
    )
    pos_checker = functools.partial(
        _pos_in_album,
        disc_number=eff_disc,
        album_track_mbids=album_track_mbids,
        album_deezer_details=album_deezer_details,
        album_itunes_details=album_itunes_details,
        album_mb_release_details=album_mb_release_details,
    )
    cand_matcher = functools.partial(
        _candidate_title_matches,
        cur_title=track_info.title,
        file_stem=file_path.stem,
    )
    cur_pos = (
        track_info.track_number
        if track_info.track_number and track_info.track_number > 0
        else fn_pos
    )

    # 2. Authoritative identifier match (ISRC / iTunes ID)
    clean_isrc = (
        str(track_info.isrc).strip().upper()
        if track_info.isrc
        and str(track_info.isrc).strip().upper() not in ("", "NONE", "NULL", "0")
        else None
    )
    if clean_isrc:
        matched = _match_position_by_authoritative_id(
            track_info,
            cur_pos,
            "isrc",
            clean_isrc,
            [album_deezer_details, album_mb_release_details],
            cand_matcher,
        )
        if matched is not None:
            return matched

    clean_itunes_id = (
        str(track_info.itunes_trackid).strip()
        if track_info.itunes_trackid
        and str(track_info.itunes_trackid).strip() not in ("", "0", "None", "null")
        else None
    )
    if clean_itunes_id and album_itunes_details:
        matched = _match_position_by_authoritative_id(
            track_info,
            cur_pos,
            "itunes_trackid",
            clean_itunes_id,
            [album_itunes_details],
            cand_matcher,
        )
        if matched is not None:
            return matched

    # 3. Candidate positions from filename prefix and existing track_number
    candidates: list[int] = []
    if fn_pos is not None and pos_checker(fn_pos):
        candidates.append(fn_pos)
    if (
        track_info.track_number is not None
        and pos_checker(track_info.track_number)
        and track_info.track_number not in candidates
    ):
        candidates.append(track_info.track_number)

    if not track_info.title or _is_generic_title(track_info.title):
        return candidates[0] if candidates else None

    has_any_cand_titles = False
    for cand_pos in candidates:
        cand_artist, cand_title = track_details_getter(cand_pos)
        if cand_title:
            has_any_cand_titles = True
            clean_cand = clean_title(cand_title).lower()
            clean_eff = clean_title(track_info.title).lower()
            min_len = min(len(clean_eff), len(clean_cand))
            req_thresh = 95.0 if min_len < 8 else 85.0
            score = max(
                match_score(
                    track_info.artist,
                    track_info.title,
                    str(cand_artist or ""),
                    cand_title,
                ),
                float(fuzz.ratio(clean_eff, clean_cand)),
                float(fuzz.token_sort_ratio(clean_eff, clean_cand)),
            )
            if clean_cand == clean_eff or score >= req_thresh:
                return cand_pos

    if not has_any_cand_titles and candidates:
        return candidates[0]

    # 4. Search all positions in album release for matching title across discs
    tracks_by_disc_map = _build_album_tracks_by_disc_map(
        eff_disc,
        album_track_mbids,
        [album_mb_release_details, album_deezer_details, album_itunes_details],
    )
    discs_to_search: list[int] = []
    if eff_disc and eff_disc in tracks_by_disc_map:
        discs_to_search.append(eff_disc)
    for d in sorted(tracks_by_disc_map):
        if d not in discs_to_search:
            discs_to_search.append(d)

    return _search_release_positions_by_title(
        track_info,
        discs_to_search,
        tracks_by_disc_map,
        album_mb_release_details,
        album_deezer_details,
        album_itunes_details,
    )


class _KnownAlbumEntities(NamedTuple):
    artists: set[str]
    titles: set[str]
    albums: set[str]
    isrcs: set[str]
    itunes_ids: set[str]
    mbids: set[str]
    positions: set[int]


def _extract_known_album_entities(
    target_album_artist: str | None,
    target_album_title: str | None,
    release_details_list: list[dict[str, Any] | None],
) -> _KnownAlbumEntities:
    """Aggregates known artists, track titles, album titles, and authoritative IDs for album releases."""
    artists: set[str] = set()
    titles: set[str] = set()
    albums: set[str] = set()
    isrcs: set[str] = set()
    itunes_ids: set[str] = set()
    mbids: set[str] = set()
    positions: set[int] = set()

    if target_album_artist:
        artists.add(target_album_artist)
    if target_album_title:
        albums.add(target_album_title)

    for details in release_details_list:
        if not details or not isinstance(details, dict):
            continue
        art = details.get("album_artist") or details.get("artist")
        if art and isinstance(art, str):
            artists.add(art)
        alb = details.get("title")
        if alb and isinstance(alb, str):
            albums.add(alb)

        for tracks_map in (
            details.get("tracks_by_position"),
            details.get("tracks_by_number"),
        ):
            if isinstance(tracks_map, dict):
                for trk in tracks_map.values():
                    if isinstance(trk, dict):
                        if trk.get("title"):
                            titles.add(str(trk["title"]))
                        if trk.get("trackName"):
                            titles.add(str(trk["trackName"]))
                        if trk.get("artist"):
                            artists.add(str(trk["artist"]))
                        if trk.get("isrc"):
                            isrcs.add(str(trk["isrc"]).strip().upper())
                        if trk.get("recording_mbid"):
                            mbids.add(str(trk["recording_mbid"]).strip().lower())
                        t_id = trk.get("itunes_trackid") or trk.get("trackId")
                        if t_id:
                            itunes_ids.add(str(t_id).strip())

        tracks_by_mbid = details.get("tracks_by_mbid")
        if isinstance(tracks_by_mbid, dict):
            for mbid_key in tracks_by_mbid:
                mbids.add(str(mbid_key).strip().lower())

        tracks_by_pos = details.get("tracks_by_position")
        if isinstance(tracks_by_pos, dict):
            for p in tracks_by_pos:
                p_int = safe_int(p)
                if p_int is not None:
                    positions.add(p_int)

    return _KnownAlbumEntities(
        artists, titles, albums, isrcs, itunes_ids, mbids, positions
    )


def is_alien_album_track(
    track_info: TrackInfo,
    file_path: Path,
    target_album_artist: str | None,
    target_album_title: str | None,
    album_mb_release_details: dict[str, Any] | None = None,
    album_deezer_details: dict[str, Any] | None = None,
    album_itunes_details: dict[str, Any] | None = None,
) -> bool:
    """
    Determine if a track located in an album directory is an outlier / alien track
    belonging to a completely different artist/song with no correlation to the album release.
    """
    if (
        not target_album_artist
        and not target_album_title
        and not album_mb_release_details
        and not album_deezer_details
        and not album_itunes_details
    ):
        return False

    _, _, fn_clean = parse_track_filename(file_path.name)
    fn_parts = [p.strip() for p in fn_clean.split(" - ") if p.strip()]
    fn_cand_artist = fn_parts[0] if len(fn_parts) >= 2 else None
    fn_cand_title = (
        " - ".join(fn_parts[1:])
        if len(fn_parts) >= 2
        else (fn_clean if fn_clean and not fn_clean.isdigit() else None)
    )

    eff_artist = (
        track_info.artist
        if not _is_generic(track_info.artist, "artist")
        else fn_cand_artist
    )
    eff_title = (
        track_info.title if not _is_generic_title(track_info.title) else fn_cand_title
    )
    eff_album = track_info.album if not _is_generic(track_info.album, "album") else None

    if _is_generic(eff_artist, "artist") and _is_generic_title(eff_title):
        return False

    entities = _extract_known_album_entities(
        target_album_artist,
        target_album_title,
        [album_mb_release_details, album_deezer_details, album_itunes_details],
    )

    # 1. Authoritative identifiers match
    clean_isrc = (
        str(track_info.isrc).strip().upper()
        if track_info.isrc
        and str(track_info.isrc).strip().upper() not in ("", "NONE", "NULL", "0")
        else None
    )
    if clean_isrc and clean_isrc in entities.isrcs:
        return False

    clean_itunes_id = (
        str(track_info.itunes_trackid).strip()
        if track_info.itunes_trackid
        and str(track_info.itunes_trackid).strip() not in ("", "0", "None", "null")
        else None
    )
    if clean_itunes_id and clean_itunes_id in entities.itunes_ids:
        return False

    if (
        is_valid_uuid(track_info.musicbrainz_trackid)
        and str(track_info.musicbrainz_trackid).strip().lower() in entities.mbids
    ):
        return False

    # 2. Check title match across album tracklist
    title_matches = False
    if eff_title and not _is_generic_title(eff_title):
        clean_eff_t = clean_title(eff_title).lower()
        for kt in entities.titles:
            clean_kt = clean_title(kt).lower()
            if clean_eff_t == clean_kt:
                title_matches = True
                break
            if len(clean_eff_t) > 3 and len(clean_kt) > 3:
                sim = max(
                    fuzz.ratio(clean_eff_t, clean_kt),
                    fuzz.token_sort_ratio(clean_eff_t, clean_kt),
                )
                thresh = 95.0 if min(len(clean_eff_t), len(clean_kt)) < 8 else 85.0
                if sim >= thresh:
                    title_matches = True
                    break

    if title_matches:
        return False

    # 3. Check artist match
    artist_matches = False
    if eff_artist and not _is_generic(eff_artist, "artist"):
        norm_eff_a = normalize_str(eff_artist)
        pri_eff_a = normalize_str(get_primary_artist(eff_artist))

        for ka in entities.artists:
            norm_ka = normalize_str(ka)
            pri_ka = normalize_str(get_primary_artist(ka))

            if norm_eff_a == norm_ka or pri_eff_a == pri_ka:
                artist_matches = True
                break
            if (
                pri_eff_a
                and pri_ka
                and (pri_eff_a in pri_ka or pri_ka in pri_eff_a)
                and len(pri_eff_a) >= 4
                and len(pri_ka) >= 4
            ):
                artist_matches = True
                break
            if (
                max(
                    fuzz.ratio(norm_eff_a, norm_ka),
                    fuzz.token_set_ratio(norm_eff_a, norm_ka),
                )
                >= 65
            ):
                artist_matches = True
                break

    if eff_artist and not _is_generic(eff_artist, "artist") and not artist_matches:
        return True

    # 4. Outlier check when album tracklist is well known
    if (
        len(entities.titles) >= 3
        and eff_title
        and not _is_generic_title(eff_title)
        and not title_matches
    ):
        album_matches = False
        if eff_album and not _is_generic(eff_album, "album"):
            clean_eff_alb = clean_title(eff_album).lower()
            for kalb in entities.albums:
                clean_kalb = clean_title(kalb).lower()
                if (
                    clean_eff_alb == clean_kalb
                    or fuzz.ratio(clean_eff_alb, clean_kalb) >= 70
                ):
                    album_matches = True
                    break

        _, fn_pos, _ = parse_track_filename(file_path.name)
        track_pos = (
            track_info.track_number
            if track_info.track_number and track_info.track_number > 0
            else fn_pos
        )
        title_lower = eff_title.lower()
        fn_lower = file_path.name.lower()
        has_bonus_indicator = any(
            b in title_lower or b in fn_lower
            for b in ("bonus", "deluxe", "exclusive", "unreleased")
        )
        is_beyond_release = bool(
            entities.positions
            and track_pos is not None
            and track_pos > max(entities.positions)
        )

        return not (
            artist_matches
            and album_matches
            and (has_bonus_indicator or is_beyond_release)
        )

    return False


def normalize_track_metadata(
    track_info: TrackInfo,
    *,
    allow_network: bool = False,
) -> TrackInfo:
    """
    Apply canonical, deterministic normalization to all metadata fields of a TrackInfo object.
    Unifies offline tag normalization and online tag post-processing into a single source of truth.
    """
    if track_info.title:
        raw_t_lower = track_info.title.lower()
        if not track_info.advisory and (
            re.search(r"[\(\[\{]\s*explicit\s*[\)\]\}]", raw_t_lower)
            or "album version (explicit)" in raw_t_lower
        ):
            track_info.advisory = "Explicit"

        prod_matches = list(_PROD_BRACKET_PATTERN.finditer(track_info.title))
        if prod_matches:
            if not track_info.producers:
                prods = [m.group(1).strip() for m in prod_matches if m.group(1).strip()]
                if prods:
                    track_info.producers = ", ".join(prods)
            track_info.title = _PROD_BRACKET_PATTERN.sub("", track_info.title).strip()

        track_info.title = strip_corrupt_brackets(track_info.title)
        clean_base, title_feats = extract_title_features(
            track_info.title, primary_artist=track_info.artist
        )
        if title_feats:
            track_info.featured_artists = normalize_featured_artists(
                [track_info.featured_artists, *title_feats],
                primary_artist=track_info.artist,
                allow_network=allow_network,
            )
        track_info.title = clean_unicode_punct(clean_base)

    if track_info.artist:
        track_info.artist = strip_corrupt_brackets(track_info.artist)
        base_artist, extracted_features = extract_artist_features(track_info.artist)
        if base_artist:
            track_info.artist = base_artist
        if extracted_features:
            track_info.featured_artists = normalize_featured_artists(
                [track_info.featured_artists, *extracted_features],
                primary_artist=track_info.artist,
                allow_network=allow_network,
            )
        track_info.artist = clean_disambiguation(track_info.artist)
        if allow_network:
            track_info.artist = resolve_artist_name(track_info.artist)
        track_info.artist = clean_unicode_punct(track_info.artist)

    if track_info.featured_artists:
        track_info.featured_artists = normalize_featured_artists(
            track_info.featured_artists,
            primary_artist=track_info.artist,
            allow_network=allow_network,
        )

    if track_info.album:
        track_info.album = clean_unicode_punct(track_info.album)

    if track_info.album_artist:
        track_info.album_artist = clean_disambiguation(track_info.album_artist)
        if allow_network:
            track_info.album_artist = resolve_artist_name(track_info.album_artist)
        track_info.album_artist = clean_unicode_punct(track_info.album_artist)

    if (
        track_info.artist
        and track_info.album_artist
        and normalize_str(track_info.artist) == normalize_str(track_info.album_artist)
        and track_info.artist != track_info.album_artist
    ):
        harmonized = harmonize_artist_casing(
            candidate_artist=track_info.artist,
            reference_artist=track_info.album_artist,
        )
        track_info.artist = harmonized
        track_info.album_artist = harmonized

    is_singles_track = bool(
        (track_info.total_tracks == 1 and (track_info.track_number or 1) > 1)
        or (
            track_info.release_type
            and track_info.release_type.lower() == "single"
            and (track_info.total_tracks or 1) <= 1
        )
        or "singles" in [p.lower() for p in track_info.file_path.parts]
        or get_config().is_generic_container(track_info.file_path.parent.name)
    )
    if is_singles_track and (track_info.total_tracks or 1) <= 1:
        track_info.track_number = 1
        track_info.total_tracks = 1

    if track_info.artist_sort and track_info.artist:
        sort_norm = normalize_str(track_info.artist_sort)
        art_candidates = [normalize_str(track_info.artist)]
        primary_artist_val = get_primary_artist(track_info.artist)
        if primary_artist_val:
            art_candidates.append(normalize_str(primary_artist_val))
        sort_score = max(
            (
                max(
                    fuzz.ratio(sort_norm, cand),
                    fuzz.token_sort_ratio(sort_norm, cand),
                    fuzz.token_set_ratio(sort_norm, cand),
                )
                for cand in art_candidates
                if cand
            ),
            default=0.0,
        )
        if sort_score < 75.0:
            track_info.artist_sort = None

    cmp_sort_art = track_info.album_artist or track_info.artist or ""
    if track_info.album_artist_sort and cmp_sort_art:
        sort_norm = normalize_str(track_info.album_artist_sort)
        art_candidates = [normalize_str(cmp_sort_art)]
        primary_album_art = get_primary_artist(cmp_sort_art)
        if primary_album_art:
            art_candidates.append(normalize_str(primary_album_art))
        sort_score = max(
            (
                max(
                    fuzz.ratio(sort_norm, cand),
                    fuzz.token_sort_ratio(sort_norm, cand),
                    fuzz.token_set_ratio(sort_norm, cand),
                )
                for cand in art_candidates
                if cand
            ),
            default=0.0,
        )
        if sort_score < 75.0:
            track_info.album_artist_sort = None

    track_info.genre = normalize_genre(track_info.genre)
    track_info.date = normalize_date(track_info.date)
    track_info.original_date = normalize_date(track_info.original_date)
    track_info.release_country = normalize_country_name(track_info.release_country)
    track_info.language = normalize_language_name(track_info.language)
    track_info.script = normalize_script_name(track_info.script)
    if track_info.advisory != "Explicit":
        track_info.advisory = None

    return track_info


def _heal_and_align_album_track(
    track_info: TrackInfo,
    file_path: Path,
    album_mb_release_details: dict[str, Any] | None,
    album_deezer_details: dict[str, Any] | None,
    album_itunes_details: dict[str, Any] | None,
    album_track_mbids: dict[Any, str] | None,
    target_album_artist: str | None,
    target_album_title: str | None,
    force: bool,
) -> None:
    """Aligns track position and heals corrupted track artist/album identity from release context."""
    resolved_pos = _resolve_album_track_position(
        track_info=track_info,
        file_path=file_path,
        album_mb_release_details=album_mb_release_details,
        album_deezer_details=album_deezer_details,
        album_itunes_details=album_itunes_details,
        album_track_mbids=album_track_mbids,
    )
    if resolved_pos is not None and (
        force
        or track_info.track_number is None
        or track_info.track_number != resolved_pos
    ):
        track_info.track_number = resolved_pos

    if (
        target_album_artist
        and not _is_generic(target_album_artist, "artist")
        and (not track_info.album_artist or force)
    ):
        track_info.album_artist = preserve_unicode_repertoire(
            track_info.album_artist, target_album_artist
        )

    known_album_track_artists: set[str] = set()
    for details in (
        album_mb_release_details,
        album_deezer_details,
        album_itunes_details,
    ):
        if details:
            raw_tracks = details.get("tracks_by_position")
            if isinstance(raw_tracks, dict):
                for trk in raw_tracks.values():
                    if isinstance(trk, dict) and trk.get("artist"):
                        known_album_track_artists.add(normalize_str(str(trk["artist"])))

    is_legitimate_track_artist = False
    norm_cur = normalize_str(track_info.artist) if track_info.artist else ""
    if (
        norm_cur
        and known_album_track_artists
        and (
            norm_cur in known_album_track_artists
            or any(
                fuzz.ratio(norm_cur, a) >= 75 or a in norm_cur or norm_cur in a
                for a in known_album_track_artists
            )
        )
    ):
        is_legitimate_track_artist = True

    if (
        not is_legitimate_track_artist
        and target_album_artist
        and not _is_generic(target_album_artist, "artist")
        and track_info.artist
    ):
        norm_tgt = normalize_str(target_album_artist)
        if (
            norm_tgt != norm_cur
            and norm_tgt not in norm_cur
            and norm_cur not in norm_tgt
            and fuzz.ratio(norm_tgt, norm_cur) < 50
        ):
            LOG.info(
                f"   ∟ 🩹 [Healer] Healing corrupted artist '[bold red]{escape(track_info.artist)}[/]' -> '[bold green]{escape(target_album_artist)}[/]'"
            )
            track_info.artist = preserve_unicode_repertoire(
                track_info.artist, target_album_artist
            )

    if (
        target_album_title
        and not _is_generic(target_album_title, "album")
        and (not track_info.album or _is_generic(track_info.album, "album") or force)
    ):
        track_info.album = preserve_unicode_repertoire(
            track_info.album, target_album_title
        )


def _run_metadata_enrichment_pipeline(
    track_info: TrackInfo,
    file_path: Path,
    album_mbid: str | None,
    album_track_mbids: dict[Any, str] | None,
    album_mb_release_details: dict[str, Any] | None,
    album_discogs_release: dict[str, Any] | None,
    album_deezer_details: dict[str, Any] | None,
    album_itunes_details: dict[str, Any] | None,
    target_album_artist: str | None,
    target_album_title: str | None,
    has_album_context: bool,
    lastfm_api_key: str | None,
    acoustid_api_key: str | None,
    discogs_user_token: str | None,
    genius_api_token: str | None,
    force: bool,
) -> None:
    """Sequentially enriches track metadata across external providers."""
    _enrich_musicbrainz(
        track_info,
        album_mbid,
        album_track_mbids,
        album_mb_release_details,
        target_album_artist=target_album_artist,
        force=force,
    )
    if (
        _is_generic_title(track_info.title)
        or _is_generic(track_info.artist, "artist")
        or (not has_album_context and (force or not track_info.musicbrainz_trackid))
    ):
        _enrich_shazam(
            track_info,
            file_path,
            target_album_artist=target_album_artist,
            target_album_title=target_album_title,
            has_album_context=has_album_context,
            force=force,
        )
    _enrich_acoustid(
        track_info,
        file_path,
        acoustid_api_key,
        album_track_mbids=album_track_mbids,
        force=force,
    )
    _enrich_itunes(track_info, album_itunes_details=album_itunes_details, force=force)
    _enrich_lastfm(track_info, lastfm_api_key)
    _enrich_discogs(
        track_info,
        discogs_user_token,
        album_discogs_release,
        has_album_context=has_album_context,
        force=force,
    )
    _enrich_deezer(
        track_info,
        album_deezer_details,
        has_album_context=has_album_context,
        force=force,
    )
    _enrich_genius(track_info, genius_api_token, force=force)
    _enrich_theaudiodb(track_info, force=force)


def _run_audio_and_tags_persistence(
    track_info: TrackInfo,
    orig_info: TrackInfo,
    file_path: Path,
    cuesheet_content: str | None,
    album_cover_path: Path | None,
    fetch_bpm: bool,
    fetch_key: bool,
    fetch_lyrics: bool,
    fetch_itunes_art: bool,
    force: bool,
    dry_run: bool,
) -> TrackInfo:
    """Computes audio DSP features, fetches lyrics/artwork, and writes updated tags."""
    _enrich_cuesheet(track_info, file_path, cuesheet_content)
    _enrich_bpm(track_info, file_path, fetch_bpm, force=force)
    _enrich_key(track_info, file_path, fetch_key, force=force)
    cover_image = (
        album_cover_path
        if album_cover_path and album_cover_path.exists()
        else _enrich_artwork(
            track_info,
            file_path,
            fetch_itunes_art,
            force=force,
            dry_run=dry_run,
        )
    )
    _enrich_lyrics(
        track_info,
        file_path,
        fetch_lyrics,
        force=force,
        dry_run=dry_run,
    )
    normalize_track_metadata(track_info, allow_network=True)

    has_art_upgrade = bool(
        cover_image
        and cover_image.exists()
        and (
            not orig_info.art_width
            or orig_info.art_width < MIN_COVER_ART_DIMENSION
            or (orig_info.art_height and orig_info.art_height < MIN_COVER_ART_DIMENSION)
        )
    )
    diff_lines = _render_tag_diffs(orig_info, track_info)
    if has_art_upgrade:
        cur_dim_str = (
            f"{orig_info.art_width}x{orig_info.art_height}"
            if orig_info.art_width and orig_info.art_height
            else "None"
        )
        diff_lines.append(
            f"\n       [green][+] cover_art: upgraded to high-resolution (was {cur_dim_str})[/]"
        )

    if diff_lines or force or has_art_upgrade:
        if not dry_run:
            write_track_metadata(track_info, cover_art_path=cover_image)
            get_library_state().record_track_state(file_path, status="TAGGED_OK")
            LOG.info(
                f"   ∟ [green]✓[/] {escape(file_path.name)}: {len(diff_lines)} tag(s) updated.{''.join(diff_lines)}"
            )
        else:
            LOG.info(f"   ∟ [DRY-RUN] {escape(file_path.name)}{''.join(diff_lines)}")
    else:
        get_library_state().record_track_state(file_path, status="TAGGED_OK")
        LOG.info(
            f"   ∟ [bold dim]✨ SKIPPED:[/] [dim]{escape(file_path.name)}[/] [dim]is already perfect.[/]"
        )

    return track_info


def process_single_track(
    file_path: Path,
    fetch_bpm: bool = True,
    fetch_key: bool = True,
    fetch_lyrics: bool = True,
    fetch_itunes_art: bool = True,
    lastfm_api_key: str | None = None,
    acoustid_api_key: str | None = None,
    discogs_user_token: str | None = None,
    genius_api_token: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    album_mbid: str | None = None,
    album_track_mbids: dict[Any, str] | None = None,
    cuesheet_content: str | None = None,
    album_mb_release_details: dict[str, Any] | None = None,
    album_discogs_release: dict[str, Any] | None = None,
    album_deezer_details: dict[str, Any] | None = None,
    album_itunes_details: dict[str, Any] | None = None,
    album_cover_path: Path | None = None,
    target_album_artist: str | None = None,
    target_album_title: str | None = None,
) -> TrackInfo:
    wait_if_paused()
    LOG.start_buffering()
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        track_info = read_track_metadata(file_path)
        orig_info = dataclasses.replace(track_info)
        track_info.artist = resolve_artist_name(track_info.artist)

        if not bool(
            album_track_mbids or album_mb_release_details or target_album_title
        ) and (
            _is_generic_title(track_info.title)
            or _is_generic(track_info.artist, "artist")
        ):
            _enrich_shazam(track_info, file_path, force=False)

        has_album_context = bool(
            album_track_mbids
            or album_mb_release_details
            or album_deezer_details
            or album_itunes_details
            or target_album_title
        )
        if has_album_context and is_alien_album_track(
            track_info=track_info,
            file_path=file_path,
            target_album_artist=target_album_artist,
            target_album_title=target_album_title,
            album_mb_release_details=album_mb_release_details,
            album_deezer_details=album_deezer_details,
            album_itunes_details=album_itunes_details,
        ):
            track_info.is_alien = True
            album_desc = (
                f"{target_album_artist} - {target_album_title}"
                if target_album_artist and target_album_title
                else (target_album_title or target_album_artist or "Album")
            )
            disp_artist = str(track_info.artist or "Unknown Artist")
            disp_title = str(track_info.title or file_path.stem)
            LOG.warning(
                f"⚠️  [bold yellow]Alien track detected in album folder:[/] [white]{escape(file_path.name)}[/]\n"
                f"   ∟ Found '[bold]{escape(disp_artist)} - {escape(disp_title)}'[/] inside '[bold]{escape(album_desc)}[/]'.\n"
                f"   ∟ [cyan]Shielding track from album metadata poisoning. Tagging independently as standalone track.[/]"
            )
            album_mbid = None
            album_track_mbids = None
            album_mb_release_details = None
            album_discogs_release = None
            album_deezer_details = None
            album_itunes_details = None
            album_cover_path = None
            cuesheet_content = None
            has_album_context = False

        if has_album_context:
            _heal_and_align_album_track(
                track_info=track_info,
                file_path=file_path,
                album_mb_release_details=album_mb_release_details,
                album_deezer_details=album_deezer_details,
                album_itunes_details=album_itunes_details,
                album_track_mbids=album_track_mbids,
                target_album_artist=target_album_artist,
                target_album_title=target_album_title,
                force=force,
            )

        LOG.info(f"🎧 Processing track: [white]{escape(file_path.name)}[/]")

        _run_metadata_enrichment_pipeline(
            track_info=track_info,
            file_path=file_path,
            album_mbid=album_mbid,
            album_track_mbids=album_track_mbids,
            album_mb_release_details=album_mb_release_details,
            album_discogs_release=album_discogs_release,
            album_deezer_details=album_deezer_details,
            album_itunes_details=album_itunes_details,
            target_album_artist=target_album_artist,
            target_album_title=target_album_title,
            has_album_context=has_album_context,
            lastfm_api_key=lastfm_api_key,
            acoustid_api_key=acoustid_api_key,
            discogs_user_token=discogs_user_token,
            genius_api_token=genius_api_token,
            force=force,
        )

        return _run_audio_and_tags_persistence(
            track_info=track_info,
            orig_info=orig_info,
            file_path=file_path,
            cuesheet_content=cuesheet_content,
            album_cover_path=album_cover_path,
            fetch_bpm=fetch_bpm,
            fetch_key=fetch_key,
            fetch_lyrics=fetch_lyrics,
            fetch_itunes_art=fetch_itunes_art,
            force=force,
            dry_run=dry_run,
        )
    finally:
        LOG.stop_buffering()


def _resolve_album_folder_identity(
    album_dir: Path, audio_files: list[Path]
) -> tuple[str | None, str | None]:
    """
    Resolve canonical artist and album name for a directory by combining
    directory structure ("Artist - Album" or parent artist directory) with
    metadata consensus across audio files.
    """
    effective_dir = get_album_root_directory(album_dir)

    folder_name = effective_dir.name
    folder_artist: str | None = None
    folder_album: str | None = None

    if " - " in folder_name:
        parts = folder_name.split(" - ", 1)
        folder_artist, folder_album = parts[0].strip(), parts[1].strip()
    elif (
        effective_dir.parent
        and effective_dir.parent.name
        and not get_config().is_generic_container(effective_dir.parent.name)
    ):
        folder_artist = effective_dir.parent.name
        folder_album = folder_name

    if folder_artist and get_config().is_generic_container(folder_artist):
        folder_artist = None
    if folder_album and get_config().is_generic_container(folder_album):
        folder_album = None

    album_counts: dict[str, int] = {}
    artist_counts: dict[str, int] = {}

    for audio_file in audio_files[:10]:
        try:
            track_metadata = read_track_metadata(audio_file)
            if track_metadata.album and not get_config().is_generic_container(
                track_metadata.album
            ):
                album_counts[track_metadata.album] = (
                    album_counts.get(track_metadata.album, 0) + 1
                )
            candidate_artist = track_metadata.album_artist or track_metadata.artist
            if candidate_artist and not get_config().is_generic_container(
                candidate_artist
            ):
                artist_counts[candidate_artist] = (
                    artist_counts.get(candidate_artist, 0) + 1
                )
        except OSError:
            pass

    consensus_album = (
        max(album_counts, key=lambda k: album_counts[k]) if album_counts else None
    )
    consensus_artist = (
        max(artist_counts, key=lambda k: artist_counts[k]) if artist_counts else None
    )

    resolved_artist: str | None
    if folder_artist and consensus_artist:
        resolved_artist = preserve_unicode_repertoire(consensus_artist, folder_artist)
    else:
        resolved_artist = folder_artist or consensus_artist

    resolved_album: str | None
    if folder_album and consensus_album:
        resolved_album = preserve_unicode_repertoire(consensus_album, folder_album)
    else:
        resolved_album = folder_album or consensus_album

    return resolved_artist, resolved_album


@dataclasses.dataclass(slots=True)
class _AlbumMetadataContext:
    musicbrainz_id: str | None = None
    track_mbids: dict[Any, str] | None = None
    mb_release_details: dict[str, Any] | None = None
    discogs_release: dict[str, Any] | None = None
    deezer_details: dict[str, Any] | None = None
    itunes_details: dict[str, Any] | None = None
    cover_path: Path | None = None
    sample_artist: str | None = None
    sample_album: str | None = None


def _resolve_album_metadata_context(
    album_dir: Path,
    audio_files: list[Path],
    discogs_user_token: str | None,
    fetch_itunes_art: bool,
    force: bool,
    dry_run: bool,
    failures: list[dict[str, str]] | None = None,
) -> _AlbumMetadataContext:
    """Pre-fetches release-level metadata from external providers for an album folder."""
    ctx = _AlbumMetadataContext()
    try:
        sample_artist, sample_album = _resolve_album_folder_identity(
            album_dir, audio_files
        )
        ctx.sample_artist = sample_artist
        ctx.sample_album = sample_album
        if not (sample_artist and sample_album):
            return ctx

        release_info = search_musicbrainz_release(
            sample_artist, sample_album, expected_track_count=len(audio_files)
        )
        if release_info and release_info.get("id"):
            ctx.musicbrainz_id = str(release_info["id"])
            ctx.mb_release_details = fetch_musicbrainz_release_details(
                ctx.musicbrainz_id
            )
            if ctx.mb_release_details:
                raw_tracks = ctx.mb_release_details.get("tracks_by_position")
                tracks_by_disc = ctx.mb_release_details.get(
                    "tracks_by_disc_and_position"
                )
                track_mbids: dict[Any, str] = {}
                if isinstance(tracks_by_disc, dict):
                    for d_pos, rec in tracks_by_disc.items():
                        if isinstance(rec, dict) and rec.get("recording_mbid"):
                            track_mbids[d_pos] = str(rec["recording_mbid"])
                if isinstance(raw_tracks, dict):
                    for pos, rec in raw_tracks.items():
                        if isinstance(rec, dict) and rec.get("recording_mbid"):
                            track_mbids[pos] = str(rec["recording_mbid"])
                ctx.track_mbids = track_mbids
            if not ctx.track_mbids:
                ctx.track_mbids = fetch_album_track_mbids(ctx.musicbrainz_id)

        ctx.deezer_details = fetch_deezer_album_details(
            sample_artist, sample_album, expected_track_count=len(audio_files)
        )
        ctx.itunes_details = fetch_itunes_album_details(
            sample_artist, sample_album, expected_track_count=len(audio_files)
        )
        if discogs_user_token:
            ctx.discogs_release = search_discogs_release(
                sample_artist,
                sample_album,
                user_token=discogs_user_token,
                expected_track_count=len(audio_files),
            )
        if fetch_itunes_art:
            ctx.cover_path = process_album_cover_art(
                album_dir,
                sample_artist,
                sample_album,
                musicbrainz_album_id=ctx.musicbrainz_id,
                force=force,
                dry_run=dry_run,
            )
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"Pre-fetching album metadata failed: {error}")

    return ctx


def _harmonize_album_genres(
    valid_tracks: list[TrackInfo],
    album_deezer_details: dict[str, Any] | None,
    album_itunes_details: dict[str, Any] | None,
    dry_run: bool = False,
) -> None:
    """Harmonizes outlier/noise genres across valid album tracks to the consensus genre."""
    if len(valid_tracks) < 2:
        return

    album_level_genre = None
    if album_deezer_details and album_deezer_details.get("genre"):
        album_level_genre = normalize_genre(str(album_deezer_details["genre"]))
    elif album_itunes_details and album_itunes_details.get("genre"):
        album_level_genre = normalize_genre(str(album_itunes_details["genre"]))

    genre_counts: dict[str, int] = {}
    for vt in valid_tracks:
        if vt.genre and not is_noise_genre(vt.genre):
            genre_counts[vt.genre] = genre_counts.get(vt.genre, 0) + 1

    dominant_genre = album_level_genre
    if not dominant_genre and genre_counts:
        top_genre, top_count = max(genre_counts.items(), key=lambda x: x[1])
        if top_count * 2 >= len(valid_tracks):
            dominant_genre = top_genre

    if not dominant_genre:
        return

    for vt in valid_tracks:
        if (not vt.genre or is_noise_genre(vt.genre)) and vt.genre != dominant_genre:
            LOG.info(
                f"   ∟ 🏷️ [Consensus] Harmonizing genre '[bold red]{escape(str(vt.genre))}[/]' -> '[bold green]{escape(dominant_genre)}[/]' on '{escape(vt.file_path.name)}'"
            )
            vt.genre = dominant_genre
            if not dry_run:
                try:
                    write_track_metadata(vt)
                except OSError as err:
                    LOG.debug(f"Failed to save harmonized genre: {err}")


def _harmonize_album_artists(
    valid_tracks: list[TrackInfo], dry_run: bool = False
) -> None:
    """Harmonizes album artist and track artist casing across valid album tracks."""
    if len(valid_tracks) < 2:
        return

    album_artist_counts: dict[str, int] = {}
    for vt in valid_tracks:
        if vt.album_artist and vt.album_artist.strip():
            art = vt.album_artist.strip()
            album_artist_counts[art] = album_artist_counts.get(art, 0) + 1

    dominant_album_artist: str | None = None
    if album_artist_counts:
        groups: dict[str, dict[str, int]] = {}
        for cand, count in album_artist_counts.items():
            norm = normalize_str(cand)
            groups.setdefault(norm, {})[cand] = count
        dominant_norm = max(groups.keys(), key=lambda k: sum(groups[k].values()))
        rep_artist = max(groups[dominant_norm].items(), key=lambda x: x[1])[0]
        resolved = resolve_artist_name(rep_artist, allow_network=False)
        dominant_album_artist = (
            resolved if normalize_str(resolved) == dominant_norm else rep_artist
        )

    track_artist_counts: dict[str, dict[str, int]] = {}
    for vt in valid_tracks:
        if vt.artist and vt.artist.strip():
            art = vt.artist.strip()
            norm = normalize_str(art)
            track_artist_counts.setdefault(norm, {})[art] = (
                track_artist_counts.setdefault(norm, {}).get(art, 0) + 1
            )

    canonical_track_artists: dict[str, str] = {}
    for norm, casing_map in track_artist_counts.items():
        dom_casing = max(casing_map.items(), key=lambda x: x[1])[0]
        resolved = resolve_artist_name(dom_casing, allow_network=False)
        canonical_track_artists[norm] = (
            resolved if normalize_str(resolved) == norm else dom_casing
        )

    if dominant_album_artist:
        norm_dom = normalize_str(dominant_album_artist)
        if norm_dom in canonical_track_artists:
            cand_trk = canonical_track_artists[norm_dom]
            harmonized = harmonize_artist_casing(cand_trk, dominant_album_artist)
            dominant_album_artist = harmonized
            canonical_track_artists[norm_dom] = harmonized
        else:
            canonical_track_artists[norm_dom] = dominant_album_artist

    for vt in valid_tracks:
        modified = False
        if (
            dominant_album_artist
            and vt.album_artist
            and normalize_str(vt.album_artist) == normalize_str(dominant_album_artist)
            and vt.album_artist != dominant_album_artist
        ):
            LOG.info(
                f"   ∟ 🏷️ [Consensus] Harmonizing album artist casing '[bold red]{escape(vt.album_artist)}[/]' -> '[bold green]{escape(dominant_album_artist)}[/]' on '{escape(vt.file_path.name)}'"
            )
            vt.album_artist = dominant_album_artist
            modified = True

        if vt.artist:
            norm_art = normalize_str(vt.artist)
            if norm_art in canonical_track_artists:
                canonical_name = canonical_track_artists[norm_art]
                if vt.artist != canonical_name:
                    LOG.info(
                        f"   ∟ 🏷️ [Consensus] Harmonizing track artist casing '[bold red]{escape(vt.artist)}[/]' -> '[bold green]{escape(canonical_name)}[/]' on '{escape(vt.file_path.name)}'"
                    )
                    vt.artist = canonical_name
                    modified = True

        if (
            vt.album_artist
            and vt.artist
            and normalize_str(vt.album_artist) == normalize_str(vt.artist)
            and vt.album_artist != vt.artist
        ):
            target = dominant_album_artist or vt.album_artist
            vt.album_artist = target
            vt.artist = target
            modified = True

        if modified and not dry_run:
            try:
                write_track_metadata(vt)
            except OSError as write_error:
                LOG.debug(f"Failed to save harmonized artist metadata: {write_error}")


def _finalize_album_assets_and_replaygain(
    album_dir: Path,
    current_album_results: list[TrackInfo],
    valid_tracks: list[TrackInfo],
    sample_artist: str | None,
    album_mb_release_details: dict[str, Any] | None,
    fetch_replaygain: bool,
    force: bool,
    dry_run: bool,
    max_threads: int,
) -> None:
    """Calculates album ReplayGain and downloads companion artist and label assets."""
    valid_album_files = [t.file_path for t in valid_tracks]
    if fetch_replaygain and valid_album_files:
        wait_if_paused()
        calculate_album_replaygain(
            valid_album_files,
            force=force,
            dry_run=dry_run,
            max_threads=max_threads,
        )

    if not current_album_results:
        return

    wait_if_paused()
    rep_track = valid_tracks[0] if valid_tracks else current_album_results[0]
    primary_artist = sample_artist or rep_track.album_artist or rep_track.artist
    try:
        process_artist_artwork(
            album_dir,
            primary_artist,
            artist_mbid=rep_track.musicbrainz_artistid,
            dry_run=dry_run,
        )
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"Artist art download failed: {error}")

    try:
        label_mbid = (
            str(album_mb_release_details.get("label_mbid"))
            if album_mb_release_details and album_mb_release_details.get("label_mbid")
            else None
        )
        label_name = (
            str(album_mb_release_details.get("label"))
            if album_mb_release_details and album_mb_release_details.get("label")
            else rep_track.label
        )
        if label_mbid:
            process_label_artwork(
                album_dir,
                label_mbid=label_mbid,
                label_name=label_name,
                dry_run=dry_run,
            )
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"Label logo download failed: {error}")


def tag_album_folder(
    folder_path: Path,
    max_threads: int = 4,
    fetch_bpm: bool = True,
    fetch_key: bool = True,
    fetch_replaygain: bool = True,
    fetch_lyrics: bool = True,
    fetch_itunes_art: bool = True,
    lastfm_api_key: str | None = None,
    acoustid_api_key: str | None = None,
    discogs_user_token: str | None = None,
    genius_api_token: str | None = None,
    fanart_api_key: str | None = None,
    fanart_client_key: str | None = None,
    enable_shazam: bool | None = None,
    force: bool = False,
    dry_run: bool = False,
    failures: list[dict[str, str]] | None = None,
) -> list[TrackInfo]:
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Album folder not found: {folder_path}")

    if fanart_api_key:
        os.environ["FANART_API_KEY"] = fanart_api_key
        clear_config_cache()
    if fanart_client_key:
        os.environ["FANART_CLIENT_KEY"] = fanart_client_key
        clear_config_cache()
    if enable_shazam is not None:
        os.environ["ENABLE_SHAZAM"] = str(enable_shazam).lower()
        clear_config_cache()

    all_audio_files = find_audio_files(folder_path, recursive=True)
    if not all_audio_files:
        return []

    # Incremental state index check
    state_mgr = get_library_state()
    if not force:
        has_structural_anomaly = False
        seen_positions: set[tuple[int, int]] = set()
        for f in all_audio_files:
            m = read_track_metadata(f)
            pos_key = (safe_int(m.disc_number) or 1, safe_int(m.track_number) or 0)
            if pos_key[1] > 0 and pos_key in seen_positions:
                has_structural_anomaly = True
                break
            seen_positions.add(pos_key)

        if not has_structural_anomaly:
            outdated_files = set(state_mgr.filter_outdated_tracks(all_audio_files))
            if not outdated_files:
                LOG.info(
                    f"✨ All {len(all_audio_files)} tracks are already up to date in library state index."
                )
                return [read_track_metadata(f) for f in all_audio_files]

    album_groups = group_files_by_album_root(all_audio_files)

    results: list[TrackInfo] = []
    current_album_results: list[TrackInfo] = []

    with create_progress() as progress:
        task = progress.add_task("[cyan]Tagging tracks...", total=len(all_audio_files))

        with (
            interactive_pause_listener(progress, task),
            ThreadPoolExecutor(max_workers=max_threads) as executor,
        ):
            try:
                for album_dir, audio_files in album_groups.items():
                    wait_if_paused()
                    folder_name = album_dir.name
                    LOG.force_info(
                        f"📁 [bold cyan]Album:[/] [white]{escape(folder_name)}[/] [dim]({len(audio_files)} tracks)[/]"
                    )

                    # Pre-resolve Cuesheet content once for entire album
                    companion_cue = find_companion_cuesheet(album_dir)
                    album_cue_content = (
                        read_cuesheet_content(companion_cue) if companion_cue else None
                    )

                    ctx = _resolve_album_metadata_context(
                        album_dir,
                        audio_files,
                        discogs_user_token=discogs_user_token,
                        fetch_itunes_art=fetch_itunes_art,
                        force=force,
                        dry_run=dry_run,
                        failures=failures,
                    )

                    current_album_results = []
                    future_to_file = {
                        executor.submit(
                            process_single_track,
                            file_path=audio_file,
                            fetch_bpm=fetch_bpm,
                            fetch_key=fetch_key,
                            fetch_lyrics=fetch_lyrics,
                            fetch_itunes_art=fetch_itunes_art,
                            lastfm_api_key=lastfm_api_key,
                            acoustid_api_key=acoustid_api_key,
                            discogs_user_token=discogs_user_token,
                            genius_api_token=genius_api_token,
                            force=force,
                            dry_run=dry_run,
                            album_mbid=ctx.musicbrainz_id,
                            album_track_mbids=ctx.track_mbids,
                            cuesheet_content=album_cue_content,
                            album_mb_release_details=ctx.mb_release_details,
                            album_discogs_release=ctx.discogs_release,
                            album_deezer_details=ctx.deezer_details,
                            album_itunes_details=ctx.itunes_details,
                            album_cover_path=ctx.cover_path,
                            target_album_artist=ctx.sample_artist,
                            target_album_title=ctx.sample_album,
                        ): audio_file
                        for audio_file in audio_files
                    }

                    for future in as_completed(future_to_file):
                        wait_if_paused()
                        audio_file = future_to_file[future]
                        try:
                            track_info = future.result()
                            current_album_results.append(track_info)
                        except _NETWORK_EXCEPTIONS as error:
                            LOG.warning(
                                f"Failed to process {escape(audio_file.name)}: {error}"
                            )
                            if failures is not None:
                                failures.append(
                                    {
                                        "file": str(audio_file.resolve()),
                                        "filename": audio_file.name,
                                        "error": str(error),
                                        "error_type": type(error).__name__,
                                    }
                                )
                        progress.advance(task)

                    valid_tracks = [t for t in current_album_results if not t.is_alien]
                    _harmonize_album_genres(
                        valid_tracks,
                        ctx.deezer_details,
                        ctx.itunes_details,
                        dry_run=dry_run,
                    )
                    _harmonize_album_artists(valid_tracks, dry_run=dry_run)
                    _finalize_album_assets_and_replaygain(
                        album_dir,
                        current_album_results,
                        valid_tracks,
                        ctx.sample_artist,
                        ctx.mb_release_details,
                        fetch_replaygain=fetch_replaygain,
                        force=force,
                        dry_run=dry_run,
                        max_threads=max_threads,
                    )

                    results.extend(current_album_results)
                    current_album_results = []
                    future_to_file.clear()
            except (KeyboardInterrupt, RuntimeError) as exc:
                if not is_interruption(exc):
                    raise
                executor.shutdown(wait=True, cancel_futures=True)
                results.extend(current_album_results)
                raise InterruptedOperationError(results) from None

    return results


def normalize_single_track(
    file_path: Path,
    fetch_bpm: bool = False,
    fetch_key: bool = False,
    force: bool = False,
    dry_run: bool = False,
) -> TrackInfo | None:
    """
    Locally cleans and normalizes metadata for a single audio file without any API requests.
    - Repairs mojibake/UTF-8 encoding via ftfy
    - Strips bracket junk: (Official Video), [FLAC], [320kbps], (2011 Remaster), [Explicit]
    - Cleans disambiguation suffixes: 'Armin (ROU)' -> 'Armin'
    - Canonicalizes genres via normalize_genre
    - Standardizes dates via normalize_date
    - Optionally calculates audio BPM locally
    - Optionally detects musical key locally
    """
    if not file_path.exists():
        return None

    try:
        current_info = read_track_metadata(file_path)
    except OSError as error:
        LOG.debug(f"Failed to read metadata for {file_path}: {error}")
        return None

    orig_info = dataclasses.replace(current_info)

    # 1. Local DSP calculation (BPM & Key)
    _enrich_bpm(current_info, file_path, fetch_bpm, force=force)
    _enrich_key(current_info, file_path, fetch_key, force=force)

    # 2. Canonical metadata normalization (offline)
    normalize_track_metadata(current_info, allow_network=False)

    diff_entries = _compute_tag_diffs(orig_info, current_info)
    diff_descriptions: list[str] = []
    for field_name, old_val, new_val in diff_entries:
        if old_val is None:
            diff_descriptions.append(f"{field_name}: added {new_val!r}")
        elif new_val is None:
            diff_descriptions.append(f"{field_name}: removed {old_val!r}")
        else:
            diff_descriptions.append(f"{field_name}: {old_val!r} -> {new_val!r}")

    if diff_entries or force:
        rendered_diffs = _render_tag_diffs(orig_info, current_info)
        if not dry_run:
            try:
                write_track_metadata(current_info)
                get_library_state().record_track_state(file_path, "TAGGED_OK")
            except OSError as error:
                LOG.warning(
                    f"Failed to save normalized tags for {escape(file_path.name)}: {error}"
                )
                return None
            LOG.info(
                f"   ∟ [green]✓[/] {escape(file_path.name)}: {len(diff_entries)} tag(s) normalized.{''.join(rendered_diffs)}"
            )
        else:
            LOG.info(
                f"   ∟ [DRY-RUN] {escape(file_path.name)}: {len(diff_entries)} tag(s) to normalize.{''.join(rendered_diffs)}"
            )
    else:
        get_library_state().record_track_state(file_path, "TAGGED_OK")
        LOG.debug(f"Track {escape(file_path.name)} is already normalized.")

    current_info._diff_descriptions = diff_descriptions
    return current_info


def normalize_library(
    directory: Path,
    fetch_bpm: bool = False,
    fetch_key: bool = False,
    fetch_replaygain: bool = False,
    force: bool = False,
    dry_run: bool = False,
    max_threads: int = 4,
) -> NormalizeReport:
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Directory not found: {directory}")

    LOG.info(f"Scanning for audio files in {escape(str(directory))}...")
    audio_files = find_audio_files(directory, recursive=True)
    if not audio_files:
        LOG.warning("No audio files found to normalize.")
        return NormalizeReport([])

    album_groups = group_files_by_album_root(audio_files)
    results: list[TrackInfo] = []
    modified_files: dict[str, list[str]] = {}

    with create_progress() as progress:
        task = progress.add_task(
            "[cyan]Normalizing tracks (offline)...", total=len(audio_files)
        )
        with (
            interactive_pause_listener(progress, task),
            ThreadPoolExecutor(max_workers=max_threads) as executor,
        ):
            try:
                for files in album_groups.values():
                    wait_if_paused()
                    futures = {
                        executor.submit(
                            normalize_single_track,
                            file_path=f,
                            fetch_bpm=fetch_bpm,
                            fetch_key=fetch_key,
                            force=force,
                            dry_run=dry_run,
                        ): f
                        for f in files
                    }
                    for future in as_completed(futures):
                        wait_if_paused()
                        worker_track = future.result()
                        if worker_track is not None:
                            results.append(worker_track)
                            if worker_track._diff_descriptions:
                                modified_files[str(futures[future])] = (
                                    worker_track._diff_descriptions
                                )
                        progress.advance(task)

                    if fetch_replaygain:
                        wait_if_paused()
                        calculate_album_replaygain(
                            files,
                            force=force,
                            dry_run=dry_run,
                            max_threads=max_threads,
                        )
                    futures.clear()
            except (KeyboardInterrupt, RuntimeError) as exc:
                if not is_interruption(exc):
                    raise
                executor.shutdown(wait=True, cancel_futures=True)
                partial_report = NormalizeReport(results, modified_files=modified_files)
                raise InterruptedOperationError(partial_report) from None

    return NormalizeReport(results, modified_files=modified_files)
