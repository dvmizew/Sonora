import functools
import re
from pathlib import Path
from typing import Any, NamedTuple

from rapidfuzz import fuzz

from sonora.core.api_payloads import require_api_object
from sonora.core.artists import get_primary_artist
from sonora.core.config import get_config
from sonora.core.library_paths import (
    extract_disc_number_from_folder,
    parse_track_filename,
)
from sonora.core.models import TrackInfo
from sonora.core.text import is_valid_uuid, normalize_str, safe_float, safe_int
from sonora.core.titles import (
    clean_title,
    recording_identity_matches,
    score_artist_title_match,
)


def _is_generic(text: str | None, placeholder: str) -> bool:
    if not text:
        return True
    normalized_text = text.strip().lower()
    return normalized_text in (
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
    normalized_text = str(title).strip().lower()
    unbracketed_title = re.sub(r"[\(\[\{].*?[\)\]\}]", "", normalized_text).strip()
    return bool(
        re.match(r"^(track|audio\s*track|audiotrack|title)\s*\d*$", unbracketed_title)
    )


def _is_strong_track_match(
    cur_artist: str | None,
    cur_title: str | None,
    candidate_artist: str | None,
    candidate_title: str | None,
) -> bool:
    """Verifies track match respecting Rule 7 short-string fuzzy thresholds."""
    if not candidate_title or not cur_title:
        return False
    normalized_current_title = clean_title(cur_title).lower()
    normalized_candidate_title = clean_title(candidate_title).lower()
    score = score_artist_title_match(
        cur_artist or "",
        cur_title,
        candidate_artist or "",
        candidate_title,
    )
    min_len = min(len(normalized_current_title), len(normalized_candidate_title))
    required_score = (
        get_config().short_title_match_threshold
        if min_len < get_config().short_title_length
        else get_config().artist_match_threshold
    )
    return score >= required_score


def _recording_matches_track_identity(
    track_info: TrackInfo, recording: dict[str, Any]
) -> bool:
    candidate_title = str(recording.get("title") or "")
    candidate_duration = safe_float(recording.get("duration"))
    candidate_artist = recording.get("artist")
    if not candidate_title:
        return False
    return recording_identity_matches(
        track_info.artist
        if candidate_artist and not _is_generic(track_info.artist, "artist")
        else None,
        candidate_title if _is_generic_title(track_info.title) else track_info.title,
        str(candidate_artist) if candidate_artist else None,
        candidate_title,
        expected_duration=track_info.duration,
        candidate_duration=candidate_duration,
    )


def _get_album_recording(
    pos: int,
    disc_number: int | None = None,
    album_mb_release_details: dict[str, Any] | None = None,
    album_deezer_details: dict[str, Any] | None = None,
    album_itunes_details: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    for details in (
        album_mb_release_details,
        album_deezer_details,
        album_itunes_details,
    ):
        if not details:
            continue
        recordings = details.get("tracks_by_disc_and_position")
        if not isinstance(recordings, dict):
            continue
        recording: object
        if disc_number is not None:
            position_key = (disc_number, pos)
            if position_key not in recordings:
                continue
            recording = require_api_object(recordings[position_key], "Album recording")
        else:
            candidates = [
                recording
                for (_, position), recording in recordings.items()
                if position == pos
            ]
            recording = candidates[0] if len(candidates) == 1 else None
        if recording is not None:
            parsed_recording = require_api_object(recording, "Album recording")
            if parsed_recording.get("title"):
                return parsed_recording
    return None


def _pos_in_album(
    pos: int,
    disc_number: int | None = None,
    album_track_mbids: dict[tuple[int, int], str] | None = None,
    album_deezer_details: dict[str, Any] | None = None,
    album_itunes_details: dict[str, Any] | None = None,
    album_mb_release_details: dict[str, Any] | None = None,
) -> bool:
    if album_track_mbids:
        if disc_number is not None:
            if (disc_number, pos) in album_track_mbids:
                return True
        elif sum(position == pos for _, position in album_track_mbids) == 1:
            return True
    recording = _get_album_recording(
        pos,
        disc_number=disc_number,
        album_mb_release_details=album_mb_release_details,
        album_deezer_details=album_deezer_details,
        album_itunes_details=album_itunes_details,
    )
    return recording is not None


def _match_position_by_mbid(
    track_info: TrackInfo,
    clean_mbid: str,
    album_track_mbids: dict[tuple[int, int], str] | None,
    album_mb_release_details: dict[str, Any] | None,
) -> int | None:
    """Resolve a recording only when its compatible release position is unique."""
    if not album_mb_release_details:
        return None
    positions = album_mb_release_details.get("tracks_by_disc_and_position") or {}
    candidates: list[tuple[Any, dict[str, Any]]] = []
    if isinstance(positions, dict):
        for position_key, recording in positions.items():
            recording = require_api_object(recording, "Album recording")
            recording_id = recording.get("recording_mbid") or (
                (album_track_mbids or {}).get(position_key)
            )
            if str(recording_id or "").strip().lower() == clean_mbid:
                candidates.append((position_key, recording))
    if not candidates:
        indexed = album_mb_release_details.get("tracks_by_mbid", {})
        if isinstance(indexed, dict) and clean_mbid in indexed:
            recording = require_api_object(indexed[clean_mbid], "Album recording")
            candidates.append((recording.get("position"), recording))
    compatible_positions: set[tuple[int, int]] = set()
    for position_key, recording in candidates:
        if not _recording_matches_track_identity(track_info, recording):
            continue
        if isinstance(position_key, tuple) and len(position_key) == 2:
            disc_number = safe_int(position_key[0])
            position = safe_int(position_key[1])
        else:
            disc_number = safe_int(recording.get("disc_number"))
            position = safe_int(recording.get("position")) or safe_int(position_key)
        if position is not None:
            compatible_positions.add((disc_number or 1, position))
    if len(compatible_positions) > 1:
        if track_info.track_number is None:
            return None
        current_position = (track_info.disc_number or 1, track_info.track_number)
        if current_position not in compatible_positions:
            return None
        compatible_positions = {current_position}
    if not compatible_positions:
        return None
    matched_disc, matched_position = compatible_positions.pop()
    track_info.disc_number = matched_disc
    return matched_position


def _recording_isrc_matches(track_isrc: str | None, recording: dict[str, Any]) -> bool:
    """MusicBrainz can associate several valid ISRCs with one recording."""
    if not track_isrc:
        return False
    recording_codes = recording.get("isrcs") or [recording.get("isrc")]
    return track_isrc.strip().upper() in {
        str(code).strip().upper() for code in recording_codes if code
    }


def _match_position_by_authoritative_id(
    track_info: TrackInfo,
    current_position: int | None,
    id_field: str,
    id_value: str,
    release_dicts: list[dict[str, Any] | None],
) -> int | None:
    """An exact recording identifier can heal corrupt titles and sequencing."""
    matches: set[tuple[int, int]] = set()
    for release_details in release_dicts:
        if not release_details:
            continue
        disc_tracks = release_details.get("tracks_by_disc_and_position", {})
        if not isinstance(disc_tracks, dict):
            continue
        candidates = list(disc_tracks.items())
        for position_key, recording in candidates:
            recording = require_api_object(recording, "Album recording")
            recording_id = recording.get(id_field)
            identifier_matches = (
                _recording_isrc_matches(id_value, recording)
                if id_field == "isrc"
                else bool(
                    recording_id
                    and str(recording_id).strip().upper() == id_value.upper()
                )
            )
            if not identifier_matches:
                continue
            if not _recording_matches_track_identity(track_info, recording):
                continue
            if isinstance(position_key, tuple) and len(position_key) == 2:
                disc = safe_int(position_key[0])
                position = safe_int(position_key[1])
            else:
                disc = safe_int(
                    recording.get("disc_number") or recording.get("disk_number")
                )
                position = safe_int(position_key)
            if position is not None:
                matches.add((disc or 1, position))
    if len(matches) > 1:
        current_matches = {
            match
            for match in matches
            if match[1] == current_position
            and match[0] == (track_info.disc_number or 1)
        }
        if len(current_matches) != 1:
            return None
        matches = current_matches
    if matches:
        matched_disc, matched_position = matches.pop()
        track_info.disc_number = matched_disc
        return matched_position
    return None


def _add_album_position(
    tracks_by_disc: dict[int, set[int]],
    disc: object,
    position: object,
    default_disc: int,
) -> None:
    parsed_position = safe_int(position)
    if parsed_position is not None:
        parsed_disc = safe_int(disc) or default_disc
        tracks_by_disc.setdefault(parsed_disc, set()).add(parsed_position)


def _build_album_tracks_by_disc_map(
    effective_disc: int | None,
    album_track_mbids: dict[tuple[int, int], str] | None,
    release_details_list: list[dict[str, Any] | None],
) -> dict[int, set[int]]:
    """Builds a mapping of disc_number -> set of track positions present in the release."""
    tracks_by_disc: dict[int, set[int]] = {}
    default_disc = effective_disc or 1

    if album_track_mbids:
        for disc_number, position in album_track_mbids:
            _add_album_position(tracks_by_disc, disc_number, position, default_disc)

    for details in release_details_list:
        if details:
            disc_tracks = details.get("tracks_by_disc_and_position")
            if isinstance(disc_tracks, dict):
                for disc_number, position in disc_tracks:
                    _add_album_position(
                        tracks_by_disc, disc_number, position, default_disc
                    )
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
    position_scores: dict[tuple[int, int], float] = {}

    for disc in discs_to_search:
        for pos in sorted(tracks_by_disc_map.get(disc, ())):
            recording = _get_album_recording(
                pos,
                disc_number=disc,
                album_mb_release_details=album_mb_release_details,
                album_deezer_details=album_deezer_details,
                album_itunes_details=album_itunes_details,
            )
            if recording and _recording_matches_track_identity(track_info, recording):
                candidate_artist = str(recording.get("artist") or "")
                candidate_title = str(recording.get("title") or "")
                score = score_artist_title_match(
                    ""
                    if _is_generic(track_info.artist, "artist")
                    else track_info.artist,
                    track_info.title,
                    candidate_artist or "",
                    candidate_title,
                )
                if score > 0:
                    position_scores[(disc, pos)] = score

    if not position_scores:
        return None
    ranked_positions = sorted(
        position_scores, key=position_scores.__getitem__, reverse=True
    )
    if len(ranked_positions) > 1 and (
        position_scores[ranked_positions[0]] == position_scores[ranked_positions[1]]
    ):
        return None
    matched_disc, matched_position = ranked_positions[0]
    track_info.disc_number = matched_disc
    return matched_position


def _compatible_release_layout(
    track_info: TrackInfo, release: dict[str, Any] | None
) -> dict[str, Any] | None:
    if not release:
        return None
    release_disc_count = safe_int(release.get("total_discs"))
    disc_positions = release.get("tracks_by_disc_and_position")
    if release_disc_count is None and isinstance(disc_positions, dict):
        release_disc_count = max(
            (
                disc
                for position in disc_positions
                if isinstance(position, tuple)
                and len(position) == 2
                and (disc := safe_int(position[0])) is not None
            ),
            default=None,
        )
    if (
        track_info.total_discs
        and release_disc_count
        and track_info.total_discs != release_disc_count
    ):
        track_info.metadata_conflicts["album_sequence"] = (
            "Provider disc count differs from the local release; "
            "sequence metadata withheld."
        )
        return None
    return release


def _resolve_album_track_position(
    track_info: TrackInfo,
    file_path: Path,
    album_mb_release_details: dict[str, Any] | None = None,
    album_deezer_details: dict[str, Any] | None = None,
    album_itunes_details: dict[str, Any] | None = None,
    album_track_mbids: dict[tuple[int, int], str] | None = None,
) -> int | None:
    """
    Resolve the legitimate track position (1-indexed) of an audio file within an album release.
    Validates candidates against authoritative identifiers and title similarity.
    """
    if album_mb_release_details is not None:
        album_mb_release_details = _compatible_release_layout(
            track_info, album_mb_release_details
        )
        if album_mb_release_details is None:
            album_track_mbids = None
    album_deezer_details = _compatible_release_layout(track_info, album_deezer_details)
    album_itunes_details = _compatible_release_layout(track_info, album_itunes_details)
    filename_disc, filename_position, _ = parse_track_filename(file_path.name)
    folder_disc = extract_disc_number_from_folder(file_path.parent.name)
    effective_disc = track_info.disc_number or filename_disc or folder_disc
    if track_info.disc_number is None and effective_disc is not None:
        track_info.disc_number = effective_disc

    if is_valid_uuid(track_info.musicbrainz_trackid):
        clean_mbid = str(track_info.musicbrainz_trackid).strip().lower()
        pos = _match_position_by_mbid(
            track_info, clean_mbid, album_track_mbids, album_mb_release_details
        )
        if pos is not None:
            return pos

    album_recording_at_position = functools.partial(
        _get_album_recording,
        disc_number=effective_disc,
        album_mb_release_details=album_mb_release_details,
        album_deezer_details=album_deezer_details,
        album_itunes_details=album_itunes_details,
    )
    pos_checker = functools.partial(
        _pos_in_album,
        disc_number=effective_disc,
        album_track_mbids=album_track_mbids,
        album_deezer_details=album_deezer_details,
        album_itunes_details=album_itunes_details,
        album_mb_release_details=album_mb_release_details,
    )
    current_position = (
        track_info.track_number
        if track_info.track_number and track_info.track_number > 0
        else filename_position
    )

    clean_isrc = (
        str(track_info.isrc).strip().upper()
        if track_info.isrc
        and str(track_info.isrc).strip().upper() not in ("", "NONE", "NULL", "0")
        else None
    )
    if clean_isrc:
        matched = _match_position_by_authoritative_id(
            track_info,
            current_position,
            "isrc",
            clean_isrc,
            [album_deezer_details, album_mb_release_details],
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
            current_position,
            "itunes_trackid",
            clean_itunes_id,
            [album_itunes_details],
        )
        if matched is not None:
            return matched

    candidates: list[int] = []
    if filename_position is not None and pos_checker(filename_position):
        candidates.append(filename_position)
    if (
        track_info.track_number is not None
        and pos_checker(track_info.track_number)
        and track_info.track_number not in candidates
    ):
        candidates.append(track_info.track_number)

    if not track_info.title or _is_generic_title(track_info.title):
        return candidates[0] if candidates else None

    has_any_cand_titles = False
    for candidate_position in candidates:
        candidate_recording = album_recording_at_position(candidate_position)
        if candidate_recording:
            has_any_cand_titles = True
            if _recording_matches_track_identity(track_info, candidate_recording):
                return candidate_position

    if not has_any_cand_titles and candidates:
        return candidates[0]

    tracks_by_disc_map = _build_album_tracks_by_disc_map(
        effective_disc,
        album_track_mbids,
        [album_mb_release_details, album_deezer_details, album_itunes_details],
    )
    discs_to_search: list[int] = []
    if effective_disc and effective_disc in tracks_by_disc_map:
        discs_to_search.append(effective_disc)
    for disc_number in sorted(tracks_by_disc_map):
        if disc_number not in discs_to_search:
            discs_to_search.append(disc_number)

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
        release_artist = details.get("album_artist") or details.get("artist")
        if release_artist and isinstance(release_artist, str):
            artists.add(release_artist)
        release_title = details.get("title")
        if release_title and isinstance(release_title, str):
            albums.add(release_title)

        tracks_map = details.get("tracks_by_disc_and_position")
        if isinstance(tracks_map, dict):
            for (_, position), album_track in tracks_map.items():
                positions.add(position)
                album_track = require_api_object(album_track, "Album recording")
                if album_track.get("title"):
                    titles.add(str(album_track["title"]))
                if album_track.get("artist"):
                    artists.add(str(album_track["artist"]))
                if album_track.get("isrc"):
                    isrcs.add(str(album_track["isrc"]).strip().upper())
                isrcs.update(
                    str(code).strip().upper()
                    for code in album_track.get("isrcs", [])
                    if code
                )
                if album_track.get("recording_mbid"):
                    mbids.add(str(album_track["recording_mbid"]).strip().lower())
                itunes_track_id = album_track.get("itunes_trackid")
                if itunes_track_id:
                    itunes_ids.add(str(itunes_track_id).strip())

        tracks_by_mbid = details.get("tracks_by_mbid")
        if isinstance(tracks_by_mbid, dict):
            for mbid_key in tracks_by_mbid:
                mbids.add(str(mbid_key).strip().lower())

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

    _, _, filename_identity = parse_track_filename(file_path.name)
    filename_parts = [
        position.strip()
        for position in filename_identity.split(" - ")
        if position.strip()
    ]
    filename_artist = filename_parts[0] if len(filename_parts) >= 2 else None
    filename_title = (
        " - ".join(filename_parts[1:])
        if len(filename_parts) >= 2
        else (
            filename_identity
            if filename_identity and not filename_identity.isdigit()
            else None
        )
    )

    effective_artist = (
        track_info.artist
        if not _is_generic(track_info.artist, "artist")
        else filename_artist
    )
    effective_title = (
        track_info.title if not _is_generic_title(track_info.title) else filename_title
    )
    effective_album = (
        track_info.album if not _is_generic(track_info.album, "album") else None
    )

    if _is_generic(effective_artist, "artist") and _is_generic_title(effective_title):
        return False

    entities = _extract_known_album_entities(
        target_album_artist,
        target_album_title,
        [album_mb_release_details, album_deezer_details, album_itunes_details],
    )

    clean_isrc = (
        str(track_info.isrc).strip().upper()
        if track_info.isrc
        and str(track_info.isrc).strip().upper() not in ("", "NONE", "NULL", "0")
        else None
    )
    if (
        _is_generic_title(effective_title)
        and clean_isrc
        and clean_isrc in entities.isrcs
    ):
        return False

    clean_itunes_id = (
        str(track_info.itunes_trackid).strip()
        if track_info.itunes_trackid
        and str(track_info.itunes_trackid).strip() not in ("", "0", "None", "null")
        else None
    )
    if (
        _is_generic_title(effective_title)
        and clean_itunes_id
        and clean_itunes_id in entities.itunes_ids
    ):
        return False

    if (
        _is_generic_title(effective_title)
        and is_valid_uuid(track_info.musicbrainz_trackid)
        and str(track_info.musicbrainz_trackid).strip().lower() in entities.mbids
    ):
        return False

    title_matches = False
    if effective_title and not _is_generic_title(effective_title):
        for known_album_title in entities.titles:
            if score_artist_title_match("", effective_title, "", known_album_title) > 0:
                title_matches = True
                break

    if title_matches and (
        _is_generic(effective_artist, "artist")
        or any(
            _is_strong_track_match(
                effective_artist, effective_title, artist, effective_title
            )
            for artist in entities.artists
        )
    ):
        return False

    artist_matches = False
    if effective_artist and not _is_generic(effective_artist, "artist"):
        normalized_track_artist = normalize_str(effective_artist)
        primary_track_artist = normalize_str(get_primary_artist(effective_artist))

        for known_album_artist in entities.artists:
            normalized_album_artist = normalize_str(known_album_artist)
            primary_album_artist = normalize_str(get_primary_artist(known_album_artist))

            if (
                normalized_track_artist == normalized_album_artist
                or primary_track_artist == primary_album_artist
            ):
                artist_matches = True
                break
            if (
                primary_track_artist
                and primary_album_artist
                and (
                    primary_track_artist in primary_album_artist
                    or primary_album_artist in primary_track_artist
                )
                and len(primary_track_artist) >= 4
                and len(primary_album_artist) >= 4
            ):
                artist_matches = True
                break
            if (
                max(
                    fuzz.ratio(normalized_track_artist, normalized_album_artist),
                    fuzz.token_set_ratio(
                        normalized_track_artist, normalized_album_artist
                    ),
                )
                >= get_config().related_artist_match_threshold
            ):
                artist_matches = True
                break

    if (
        effective_artist
        and not _is_generic(effective_artist, "artist")
        and not artist_matches
    ):
        return True

    if (
        len(entities.titles) >= 3
        and effective_title
        and not _is_generic_title(effective_title)
        and not title_matches
    ):
        album_matches = False
        if effective_album and not _is_generic(effective_album, "album"):
            clean_eff_alb = clean_title(effective_album).lower()
            for known_album_title in entities.albums:
                clean_kalb = clean_title(known_album_title).lower()
                if (
                    clean_eff_alb == clean_kalb
                    or fuzz.ratio(clean_eff_alb, clean_kalb)
                    >= get_config().related_album_match_threshold
                ):
                    album_matches = True
                    break

        _, filename_position, _ = parse_track_filename(file_path.name)
        track_pos = (
            track_info.track_number
            if track_info.track_number and track_info.track_number > 0
            else filename_position
        )
        title_lower = effective_title.lower()
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


def _catalog_recording_matches_track(
    recording: dict[str, Any],
    title: str | None,
    artist: str | None,
    isrc: str | None,
    itunes_trackid: str | None,
    duration: float | None = None,
) -> bool:
    candidate_duration = safe_float(recording.get("duration"))
    if (
        duration is not None
        and candidate_duration is not None
        and abs(duration - candidate_duration)
        > get_config().single_duration_tolerance_seconds
    ):
        return False
    matched_identifier = False
    expected_isrc = (isrc or "").strip().upper()
    provider_isrcs = recording.get("isrcs") or [recording.get("isrc")]
    if expected_isrc not in ("", "NONE", "NULL", "0") and any(provider_isrcs):
        if not _recording_isrc_matches(expected_isrc, recording):
            return False
        matched_identifier = True
    expected_itunes_id = (itunes_trackid or "").strip()
    provider_itunes_id = str(recording.get("itunes_trackid") or "").strip()
    if expected_itunes_id not in ("", "None", "null", "0") and provider_itunes_id:
        if expected_itunes_id != provider_itunes_id:
            return False
        matched_identifier = True
    candidate_artist = recording.get("artist")
    candidate_title = recording.get("title")
    if _is_generic_title(title):
        return bool(
            matched_identifier
            and candidate_title
            and recording_identity_matches(
                None if _is_generic(artist, "artist") else artist,
                str(candidate_title),
                str(candidate_artist) if candidate_artist else None,
                str(candidate_title),
                expected_duration=duration,
                candidate_duration=candidate_duration,
            )
        )
    return _is_strong_track_match(
        artist if candidate_artist and not _is_generic(artist, "artist") else None,
        title,
        str(candidate_artist) if candidate_artist else None,
        str(candidate_title) if candidate_title else None,
    )


def _find_track_in_album_mapping(
    tracks_by_title: Any,
    tracks_by_disc_and_position: Any,
    title: str | None,
    track_number: int | None,
    disc_number: int | None = None,
    *,
    artist: str | None = None,
    isrc: str | None = None,
    itunes_trackid: str | None = None,
    duration: float | None = None,
) -> tuple[dict[str, Any] | None, bool]:
    candidates: list[object] = []
    if isinstance(tracks_by_disc_and_position, dict) and track_number is not None:
        if disc_number is not None:
            position_key = (disc_number, track_number)
            if position_key in tracks_by_disc_and_position:
                candidates.append(tracks_by_disc_and_position[position_key])
        else:
            position_candidates = [
                recording
                for (_, position), recording in tracks_by_disc_and_position.items()
                if position == track_number
            ]
            if len(position_candidates) == 1:
                candidates.extend(position_candidates)
    if isinstance(tracks_by_title, dict) and title and not _is_generic_title(title):
        for title_key in dict.fromkeys(
            (normalize_str(clean_title(title)), normalize_str(title))
        ):
            if title_key in tracks_by_title:
                candidates.append(tracks_by_title[title_key])
    for candidate in candidates:
        recording = require_api_object(candidate, "Album recording")
        # Position is an edition-specific hint; it must not import another song's fields.
        if _catalog_recording_matches_track(
            recording, title, artist, isrc, itunes_trackid, duration
        ):
            return recording, True
    return None, False


def provider_identity_matches(recording: TrackInfo, candidate: TrackInfo) -> bool:
    """An enrichment provider may verify identity but cannot replace it."""
    return (
        (recording.artist, recording.title, recording.duration)
        == (candidate.artist, candidate.title, candidate.duration)
    ) or recording_identity_matches(
        recording.artist,
        recording.title,
        candidate.artist,
        candidate.title,
        expected_duration=recording.duration,
        candidate_duration=candidate.duration,
    )
