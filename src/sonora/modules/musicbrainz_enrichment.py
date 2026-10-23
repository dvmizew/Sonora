import re
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz
from rich.markup import escape

from sonora.core.api_payloads import require_api_object
from sonora.core.artists import get_primary_artist
from sonora.core.config import get_config
from sonora.core.logger import LOG
from sonora.core.models import TrackInfo
from sonora.core.text import (
    clean_unicode_punct,
    is_valid_uuid,
    normalize_date,
    normalize_str,
    preserve_unicode_repertoire,
    safe_float,
    safe_int,
)
from sonora.core.titles import (
    clean_title,
    recording_identity_matches,
    score_artist_title_match,
)
from sonora.modules.catalog_enrichment import _NETWORK_EXCEPTIONS
from sonora.modules.matching import (
    _compatible_release_layout,
    _get_album_recording,
    _is_generic,
    _is_generic_title,
    _is_strong_track_match,
    _recording_isrc_matches,
    _recording_matches_track_identity,
)
from sonora.modules.metadata_consensus import apply_metadata_mapping
from sonora.services import acoustid as acoustid_service
from sonora.services import artist_resolution
from sonora.services import musicbrainz as musicbrainz_service


def _enrich_acoustid(
    track_info: TrackInfo,
    file_path: Path,
    acoustid_api_key: str | None,
    album_track_mbids: dict[tuple[int, int], str] | None = None,
    force: bool = False,
) -> None:
    if not acoustid_api_key:
        return
    # A verified album recording avoids another expensive audio fingerprint.
    album_mbids = album_track_mbids or {}
    if (
        track_info.track_number
        and (track_info.disc_number or 1, track_info.track_number) in album_mbids
        and track_info.musicbrainz_trackid
        == album_mbids[(track_info.disc_number or 1, track_info.track_number)]
        and track_info.metadata_sources.get("musicbrainz_trackid") == "MusicBrainz"
    ):
        return
    if is_valid_uuid(track_info.musicbrainz_trackid) and not force:
        return
    try:
        acoustid_mbid = acoustid_service.lookup_acoustid(
            file_path,
            api_key=acoustid_api_key,
            expected_artist=None
            if _is_generic(track_info.artist, "artist")
            else track_info.artist,
            expected_title=None
            if _is_generic_title(track_info.title)
            else track_info.title,
        )
        if is_valid_uuid(acoustid_mbid):
            if not _is_generic_title(track_info.title) and not _is_generic(
                track_info.artist, "artist"
            ):
                recording_details = (
                    musicbrainz_service.fetch_musicbrainz_recording_details(
                        acoustid_mbid
                    )
                )
                if not recording_details or not isinstance(recording_details, dict):
                    return
                candidate_artist = str(recording_details.get("artist") or "")
                candidate_title = str(recording_details.get("title") or "")
                score = score_artist_title_match(
                    track_info.artist or "",
                    track_info.title or "",
                    candidate_artist,
                    candidate_title,
                )
                if score <= 0.0:
                    LOG.debug(
                        f"Rejecting AcoustID MBID {acoustid_mbid} ('{candidate_artist} - {candidate_title}') "
                        f"conflicting with ('{track_info.artist} - {track_info.title}')"
                    )
                    return
            track_info.musicbrainz_trackid = acoustid_mbid
            track_info.metadata_sources["musicbrainz_trackid"] = "AcoustID"
            LOG.detail(f"   ∟ 🎯 [acoustid] Matched MBID: {acoustid_mbid[:8]}...")
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"AcoustID lookup failed for {track_info.title}: {error}")


def _find_album_release_recording(
    track_info: TrackInfo,
    album_mb_release_details: dict[str, Any] | None,
    disc_num: int,
) -> dict[str, Any] | None:
    """Lookup candidate recording in pre-fetched album release details."""
    if not album_mb_release_details or not track_info.track_number:
        return None
    return _get_album_recording(
        track_info.track_number,
        disc_num,
        album_mb_release_details=album_mb_release_details,
    )


def _heal_track_identity_from_release(
    track_info: TrackInfo,
    candidate_rec: dict[str, Any] | None,
) -> bool:
    """Fill a missing title when the release agrees with the remaining identity."""
    if not candidate_rec or not track_info.isrc:
        return False
    rec_title = candidate_rec.get("title")
    if (
        _recording_isrc_matches(track_info.isrc, candidate_rec)
        and rec_title
        and _is_generic_title(track_info.title)
        and _recording_matches_track_identity(track_info, candidate_rec)
        and clean_title(track_info.title).lower() != clean_title(str(rec_title)).lower()
    ):
        LOG.detail(
            f"   ∟ 🩹 [Healer] Track {track_info.track_number} verified by ISRC match. "
            f"Healing corrupted title '{escape(str(track_info.title))}' -> '{escape(str(rec_title))}'"
        )
        track_info.title = preserve_unicode_repertoire(track_info.title, str(rec_title))
        return True
    return False


def _resolve_musicbrainz_track_id(
    track_info: TrackInfo,
    album_track_mbids: dict[tuple[int, int], str] | None,
    album_mb_release_details: dict[str, Any] | None,
    candidate_rec: dict[str, Any] | None,
    has_corrupt_identity: bool,
    force: bool = False,
) -> None:
    """Resolve track MBID via album tracks position mapping or track search."""
    album_mbids = album_track_mbids or {}
    disc_num = track_info.disc_number or 1
    potential_mbid = (
        album_mbids.get((disc_num, track_info.track_number))
        if track_info.track_number is not None
        else None
    )
    candidate_mbid: str | None = None
    if potential_mbid is not None:
        is_match = bool(
            candidate_rec
            and _recording_matches_track_identity(track_info, candidate_rec)
            and candidate_rec.get("recording_mbid") == potential_mbid
        )

        if is_match and is_valid_uuid(potential_mbid):
            candidate_mbid = potential_mbid
            track_info.musicbrainz_trackid = candidate_mbid
            track_info.metadata_sources["musicbrainz_trackid"] = "MusicBrainz"
            LOG.detail(
                f"   ∟ 🏷️ [MusicBrainz Album Match] Found MBID: {candidate_mbid[:8]}..."
            )
    elif not is_valid_uuid(track_info.musicbrainz_trackid) or (
        force and not track_info.is_alien
    ):
        mbid = musicbrainz_service.fetch_track_mbid(track_info.artist, track_info.title)
        if is_valid_uuid(mbid):
            track_info.musicbrainz_trackid = mbid
            track_info.metadata_sources.pop("musicbrainz_trackid", None)
            LOG.detail(f"   ∟ 🏷️ [MusicBrainz] Found MBID: {mbid[:8]}...")


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
        release = musicbrainz_service.search_musicbrainz_release(
            search_artist,
            track_info.album,
            expected_track_count=track_info.total_tracks,
            expected_advisory=track_info.advisory,
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
        indexed_recording = require_api_object(
            tracks_by_id[track_info.musicbrainz_trackid], "Album MusicBrainz recording"
        )
        if _recording_matches_track_identity(track_info, indexed_recording):
            track_info.metadata_sources["musicbrainz_trackid"] = "MusicBrainz"
            return indexed_recording

    if candidate_rec and _recording_matches_track_identity(track_info, candidate_rec):
        recording_id = candidate_rec.get("recording_mbid")
        if is_valid_uuid(recording_id):
            track_info.musicbrainz_trackid = str(recording_id)
            track_info.metadata_sources["musicbrainz_trackid"] = "MusicBrainz"
            return candidate_rec
    return None


def _fetch_standalone_recording(
    track_info: TrackInfo,
    force: bool = False,
) -> dict[str, Any] | None:
    """Fetches standalone recording details from MusicBrainz API with identity verification."""
    if not is_valid_uuid(track_info.musicbrainz_trackid):
        return None
    fetched_rec = musicbrainz_service.fetch_musicbrainz_recording_details(
        track_info.musicbrainz_trackid
    )
    if not fetched_rec or not isinstance(fetched_rec, dict):
        return None
    isrc_matches = _recording_isrc_matches(track_info.isrc, fetched_rec)
    rec_artist = str(fetched_rec.get("artist") or "")
    rec_title = str(fetched_rec.get("title") or "")
    expected_artist = (
        track_info.album_artist
        if _is_generic(track_info.artist, "artist")
        else track_info.artist
    )
    title_sim = score_artist_title_match(
        expected_artist or "", track_info.title, rec_artist, rec_title
    )
    min_len = (
        min(len(clean_title(track_info.title)), len(clean_title(rec_title)))
        if rec_title
        else 0
    )
    thresh = (
        get_config().short_title_match_threshold
        if min_len < get_config().short_title_length
        else get_config().artist_match_threshold
    )
    # Stored identifiers can be wrong; they cannot override a coherent standalone identity.
    has_missing_identity = _is_generic_title(track_info.title) or _is_generic(
        track_info.artist, "artist"
    )
    acoustic_identity = (
        has_missing_identity
        and track_info.metadata_sources.get("musicbrainz_trackid") == "AcoustID"
        and recording_identity_matches(
            None if _is_generic(track_info.artist, "artist") else track_info.artist,
            rec_title if _is_generic_title(track_info.title) else track_info.title,
            rec_artist,
            rec_title,
            expected_duration=track_info.duration,
            candidate_duration=safe_float(fetched_rec.get("duration")),
        )
    )
    candidate_duration = safe_float(fetched_rec.get("duration"))
    duration_matches = not (
        track_info.duration is not None
        and candidate_duration is not None
        and abs(track_info.duration - candidate_duration)
        > get_config().single_duration_tolerance_seconds
    )
    if (
        duration_matches
        and rec_title
        and (
            acoustic_identity
            or title_sim == 100
            or title_sim > thresh
            or (
                isrc_matches
                and has_missing_identity
                and _recording_matches_track_identity(track_info, fetched_rec)
            )
        )
    ):
        if track_info.metadata_sources.get("musicbrainz_trackid") != "AcoustID":
            track_info.metadata_sources["musicbrainz_trackid"] = "MusicBrainz"
        return fetched_rec

    LOG.debug(
        f"Discarding mismatched pre-existing MBID {escape(str(track_info.musicbrainz_trackid))} "
        f"('{escape(str(rec_artist))} - {escape(str(rec_title))}') "
        f"for track '{escape(str(track_info.artist))} - {escape(str(track_info.title))}'"
    )
    track_info.musicbrainz_trackid = None
    track_info.metadata_sources.pop("musicbrainz_trackid", None)
    if (
        track_info.musicbrainz_workid
        and fetched_rec.get("musicbrainz_workid") == track_info.musicbrainz_workid
    ):
        track_info.musicbrainz_workid = None
        track_info.metadata_sources.pop("musicbrainz_workid", None)
    if force:
        new_mbid = musicbrainz_service.fetch_track_mbid(
            track_info.artist, track_info.title
        )
        if is_valid_uuid(new_mbid):
            track_info.musicbrainz_trackid = new_mbid
            return _fetch_standalone_recording(track_info)
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
    recording_details: dict[str, Any],
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
            cur_primary == cand_primary
            or fuzz.ratio(cur_primary, cand_primary)
            >= get_config().artist_credit_match_threshold
        )
    )

    rec_title_str = str(recording_details.get("title") or track_info.title)
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
            and fuzz.ratio(clean_title(alb_artist).lower(), cand_primary)
            >= get_config().artist_credit_match_threshold
        )
        if primary_matches or is_various or matches_album_artist or is_strong:
            return True

    return False


def _apply_mb_recording_metadata(
    track_info: TrackInfo,
    recording_details: dict[str, Any],
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
    }
    if is_album_rec:
        mb_map.update(
            {name: name for name in ("disc_number", "total_discs", "disc_subtitle")}
        )
    rec_title = str(recording_details.get("title") or "")
    rec_artist = str(recording_details.get("artist") or "")
    has_corrupt_identity = has_corrupt_identity or bool(
        is_album_rec
        and rec_title
        and rec_artist
        and (
            _is_generic_title(track_info.title)
            or _is_generic(track_info.artist, "artist")
        )
        and _recording_isrc_matches(track_info.isrc, recording_details)
        and not _is_strong_track_match(
            track_info.artist, track_info.title, rec_artist, rec_title
        )
    )
    title_matches = _is_strong_track_match(
        track_info.artist, track_info.title, rec_artist, rec_title
    )
    filename_title_matches = _is_strong_track_match(
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
        or (force and (title_matches or filename_title_matches))
    ) and rec_title:
        mb_map["title"] = "title"

    if recording_details.get("first-release-date") and (force or not track_info.date):
        mb_map["first-release-date"] = "date"

    apply_metadata_mapping(
        track_info,
        recording_details,
        mb_map,
        force=force or has_corrupt_identity,
        source="MusicBrainz",
    )

    if is_album_rec and recording_details.get("disc_total_tracks"):
        apply_metadata_mapping(
            track_info,
            recording_details,
            {"disc_total_tracks": "total_tracks"},
            force=force,
            source="MusicBrainz",
        )

    recording_artist = recording_details.get("artist")
    if recording_artist and isinstance(recording_artist, str):
        cleaned_artist = preserve_unicode_repertoire(
            track_info.artist,
            clean_unicode_punct(
                artist_resolution.resolve_artist_name(recording_artist)
            ),
        )
        if _should_update_track_artist(
            track_info,
            recording_details,
            cleaned_artist,
            is_album_rec,
            target_album_artist,
            has_corrupt_identity,
            force=force,
        ):
            track_info.artist = cleaned_artist


def _validate_mb_release_artist(release_artist: str, expected_artist: str) -> bool:
    """Validates that release artist matches album artist or track artist."""
    if (
        not release_artist
        or not expected_artist
        or _is_generic(expected_artist, "artist")
    ):
        return True
    rel_primary = (
        re.sub(r"^(?:the|a|an)\s+", "", clean_title(get_primary_artist(release_artist)))
        .strip()
        .lower()
    )
    cmp_primary = (
        re.sub(
            r"^(?:the|a|an)\s+", "", clean_title(get_primary_artist(expected_artist))
        )
        .strip()
        .lower()
    )
    return (
        max(
            fuzz.ratio(rel_primary, cmp_primary),
            fuzz.token_sort_ratio(rel_primary, cmp_primary),
        )
        >= get_config().album_match_threshold
    )


def _validate_mb_release_title(release_title: str, cur_album: str | None) -> bool:
    """Validates that release title matches current album title."""
    if not release_title or not cur_album or _is_generic(cur_album, "album"):
        return True
    norm_rel = clean_title(release_title).lower()
    norm_alb = clean_title(cur_album).lower()
    thresh = (
        get_config().short_title_match_threshold
        if min(len(norm_rel), len(norm_alb)) < get_config().short_title_length
        else get_config().album_match_threshold
    )
    return (
        max(
            fuzz.ratio(norm_rel, norm_alb),
            fuzz.token_sort_ratio(norm_rel, norm_alb),
        )
        >= thresh
    )


def _release_edition_conflicts(
    track_info: TrackInfo, release: dict[str, Any] | None
) -> bool:
    if not release or track_info.advisory != "Explicit":
        return False
    edition_tokens = set(
        normalize_str(str(release.get("disambiguation") or "")).split()
    )
    if {"clean", "cleaned"}.intersection(
        edition_tokens
    ) and "explicit" not in edition_tokens:
        track_info.metadata_conflicts["identity"] = (
            "MusicBrainz release is described as clean while local tags "
            "mark the recording explicit; edition verification required."
        )
        return True
    return False


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
        else musicbrainz_service.fetch_musicbrainz_release_details(
            track_info.musicbrainz_albumid
        )
    )
    if not mb_rel or not isinstance(mb_rel, dict):
        return

    if _release_edition_conflicts(track_info, mb_rel):
        return
    if _compatible_release_layout(track_info, mb_rel) is None:
        return

    release_artist = str(mb_rel.get("album_artist") or mb_rel.get("artist") or "")
    expected_artist = track_info.album_artist or track_info.artist or ""
    if not _validate_mb_release_artist(release_artist, expected_artist):
        track_info.musicbrainz_albumid = None
        track_info.musicbrainz_releasegroupid = None
        return

    release_title = str(mb_rel.get("title") or "").strip()
    if not _validate_mb_release_title(release_title, track_info.album):
        track_info.musicbrainz_albumid = None
        track_info.musicbrainz_releasegroupid = None
        return

    release_recordings = mb_rel.get("tracks_by_mbid")
    if (
        isinstance(release_recordings, dict)
        and not _is_generic(track_info.artist, "artist")
        and not _is_generic_title(track_info.title)
        and not any(
            _recording_matches_track_identity(track_info, recording)
            for recording in release_recordings.values()
            if isinstance(recording, dict)
        )
    ):
        track_info.musicbrainz_albumid = None
        track_info.metadata_sources.pop("musicbrainz_albumid", None)
        track_info.metadata_conflicts["musicbrainz_albumid"] = (
            "MusicBrainz release does not contain the verified recording; "
            "release identifier removed and release metadata withheld."
        )
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
    original_date = normalize_date(str(mb_rel.get("original_date") or ""))
    if original_date and (
        not track_info.original_date or original_date <= track_info.original_date
    ):
        mb_rel_map["original_date"] = "original_date"

    apply_metadata_mapping(
        track_info, mb_rel, mb_rel_map, force=force, source="MusicBrainz"
    )

    # Equivalent base titles do not prove that a release has the same edition.
    preserves_album_edition = bool(
        track_info.album
        and clean_title(track_info.album).casefold()
        == clean_title(release_title).casefold()
        and track_info.album.casefold() != release_title.casefold()
    )
    if (
        not preserves_album_edition
        and mb_rel.get("title")
        and (
            (
                album_mb_release_details is not None
                and (force or track_info.album != mb_rel.get("title"))
            )
            or not track_info.album
            or track_info.album.lower() in ("unknown", "unknown album")
        )
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
            clean_unicode_punct(
                artist_resolution.resolve_artist_name(str(mb_rel["album_artist"]))
            ),
        )

    if album_mb_release_details is not None and (force or not track_info.total_tracks):
        release_recording = _find_album_release_recording(
            track_info, mb_rel, track_info.disc_number or 1
        )
        # TOTALTRACKS belongs to the current medium, not the entire box set.
        total_tracks = safe_int(
            release_recording.get("disc_total_tracks") if release_recording else None
        )
        if total_tracks is None and safe_int(mb_rel.get("total_discs")) in (None, 1):
            total_tracks = safe_int(mb_rel.get("total_tracks"))
        if total_tracks is not None and total_tracks >= (track_info.track_number or 0):
            track_info.total_tracks = total_tracks


def _enrich_musicbrainz(
    track_info: TrackInfo,
    album_mbid: str | None,
    album_track_mbids: dict[tuple[int, int], str] | None,
    album_mb_release_details: dict[str, Any] | None,
    target_album_artist: str | None = None,
    force: bool = False,
) -> None:
    try:
        if _release_edition_conflicts(track_info, album_mb_release_details):
            return
        if (
            album_mb_release_details is not None
            and _compatible_release_layout(track_info, album_mb_release_details) is None
        ):
            album_mbid = None
            album_track_mbids = None
            album_mb_release_details = None
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

        recording_details, is_album_rec = _fetch_or_extract_mb_recording(
            track_info,
            album_mb_release_details,
            candidate_rec,
            has_corrupt_identity,
            force=force,
        )
        if recording_details:
            _apply_mb_recording_metadata(
                track_info,
                recording_details,
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
