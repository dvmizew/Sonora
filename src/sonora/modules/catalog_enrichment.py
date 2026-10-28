from pathlib import Path
from typing import Any

import acoustid
import httpx
from musicbrainzngs import MusicBrainzError
from rapidfuzz import fuzz
from rich.markup import escape

from sonora.core.artists import (
    extract_featured_artist_tokens,
)
from sonora.core.config import get_config
from sonora.core.logger import LOG
from sonora.core.models import TrackInfo
from sonora.core.text import (
    clean_unicode_punct,
    is_valid_uuid,
    normalize_genre,
    normalize_str,
    preserve_unicode_repertoire,
    safe_float,
    safe_int,
)
from sonora.core.titles import (
    recording_identity_matches,
    score_artist_title_match,
)
from sonora.modules.matching import (
    _catalog_recording_matches_track,
    _compatible_release_layout,
    _find_track_in_album_mapping,
    _is_generic,
    _is_generic_title,
    _is_strong_track_match,
)
from sonora.modules.metadata_consensus import apply_metadata_mapping
from sonora.services import artist_resolution
from sonora.services import deezer as deezer_service
from sonora.services import discogs as discogs_service
from sonora.services import genius as genius_service
from sonora.services import itunes as itunes_service
from sonora.services import lastfm as lastfm_service
from sonora.services import shazam as shazam_service
from sonora.services import theaudiodb as theaudiodb_service

_NETWORK_EXCEPTIONS = (
    MusicBrainzError,
    acoustid.AcoustidError,
    acoustid.WebServiceError,
    httpx.HTTPError,
    OSError,
    TimeoutError,
)


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
        shazam_match = shazam_service.recognize_audio_track(file_path)
        if not shazam_match:
            return

        if not is_missing_metadata and not _is_strong_track_match(
            track_info.artist, track_info.title, shazam_match.artist, shazam_match.title
        ):
            track_info.metadata_conflicts["identity"] = (
                f"Shazam recognized {shazam_match.artist} - {shazam_match.title}; "
                "existing artist/title preserved pending independent verification."
            )
            LOG.debug(
                "Rejected acoustic recognition conflicting with existing track identity"
            )
            return

        # Shared samples can identify a different performer than the album artist.
        if (
            target_album_artist
            and track_info.artist
            and not _is_generic(target_album_artist, "artist")
            and not _is_generic(track_info.artist, "artist")
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
                    track_info.artist = artist_resolution.resolve_artist_name(
                        shazam_match.artist
                    )
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
            track_info.title = preserve_unicode_repertoire(
                track_info.title, shazam_match.title
            )
            track_info.artist = artist_resolution.resolve_artist_name(
                shazam_match.artist
            )
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
        LOG.detail(
            f"   ∟ ⚡ [Shazam] Identified: [white]{escape(shazam_match.artist)} - {escape(shazam_match.title)}[/]"
        )
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"Shazam acoustic recognition failed: {escape(str(error))}")


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
                album_itunes_details.get("tracks_by_disc_and_position"),
                track_info.title,
                track_info.track_number,
                disc_number=track_info.disc_number,
                artist=track_info.artist,
                isrc=track_info.isrc,
                itunes_trackid=track_info.itunes_trackid,
                duration=track_info.duration,
            )

        if (
            not api_payload
            and track_info.itunes_trackid
            and not _is_generic_title(track_info.title)
        ):
            stored_recording = itunes_service.lookup_itunes_track(
                track_info.itunes_trackid
            )
            if (
                stored_recording
                and stored_recording.get("artist")
                and stored_recording.get("title")
                and track_info.artist
                and not recording_identity_matches(
                    track_info.artist,
                    track_info.title,
                    str(stored_recording.get("artist") or ""),
                    str(stored_recording.get("title") or ""),
                    expected_duration=track_info.duration,
                    candidate_duration=safe_float(stored_recording.get("duration")),
                )
            ):
                if (
                    "*" in str(stored_recording["title"]) + track_info.title
                    and track_info.duration is not None
                    and (
                        stored_duration := safe_float(stored_recording.get("duration"))
                    )
                    is not None
                    and stored_duration > 0
                    and recording_identity_matches(
                        track_info.artist,
                        track_info.title,
                        str(stored_recording["artist"]),
                        track_info.title,
                        expected_duration=track_info.duration,
                        candidate_duration=stored_duration,
                    )
                ):
                    track_info.metadata_conflicts["itunes_trackid"] = (
                        "Apple title may be censored; artist and duration agree, "
                        "but recording identity requires independent verification."
                    )
                    return
                LOG.detail(
                    f"Discarding conflicting iTunes identifier for {escape(track_info.artist)} - {escape(track_info.title)}"
                )
                track_info.itunes_trackid = None
                track_info.itunes_collectionid = None
                track_info.itunes_artistid = None
                track_info.metadata_sources.pop("itunes_trackid", None)
                track_info.metadata_sources.pop("itunes_collectionid", None)
                track_info.metadata_sources.pop("itunes_artistid", None)
            elif (
                stored_recording
                and stored_recording.get("artist")
                and stored_recording.get("title")
                and track_info.artist
            ):
                api_payload = stored_recording

        if not api_payload and (
            force
            or not track_info.itunes_trackid
            or not track_info.genre
            or not track_info.advisory
        ):
            api_payload = itunes_service.fetch_itunes_track_metadata(
                track_info.artist,
                track_info.title,
                track_id=track_info.itunes_trackid,
                duration=track_info.duration,
            )

        if api_payload and not _catalog_recording_matches_track(
            api_payload,
            track_info.title,
            track_info.artist,
            track_info.isrc,
            track_info.itunes_trackid,
            track_info.duration,
        ):
            LOG.detail(
                f"Rejecting conflicting iTunes metadata for {escape(track_info.artist)} - {escape(track_info.title)}"
            )
            return

        if api_payload:
            if (
                track_info.advisory == "Explicit"
                and api_payload.get("explicitness") == "cleaned"
            ):
                track_info.metadata_conflicts["itunes_trackid"] = (
                    "Apple identifies a cleaned edition while local tags mark "
                    "the recording explicit; enrichment withheld."
                )
                return
            if api_payload.get("itunes_trackid"):
                apply_metadata_mapping(
                    track_info,
                    api_payload,
                    {
                        "itunes_trackid": "itunes_trackid",
                        "itunes_collectionid": "itunes_collectionid",
                        "itunes_artistid": "itunes_artistid",
                    },
                    force=True,
                    source="iTunes",
                )
            if api_payload.get("explicitness") in {
                "explicit",
                "notexplicit",
                "cleaned",
            }:
                track_info.advisory = (
                    "Explicit" if api_payload["explicitness"] == "explicit" else None
                )
                track_info.metadata_sources["advisory"] = "iTunes"
            itunes_map = {"genre": "genre", "advisory": "advisory"}
            if is_album_match or not track_info.copyright:
                itunes_map["copyright"] = "copyright"
            if is_album_match and not is_valid_uuid(track_info.musicbrainz_albumid):
                if not track_info.disc_number:
                    itunes_map["disc_number"] = "disc_number"
                if not track_info.total_discs:
                    itunes_map["total_discs"] = "total_discs"
            if not track_info.date or (
                force
                and is_album_match
                and not is_valid_uuid(track_info.musicbrainz_albumid)
            ):
                itunes_map["date"] = "date"
            if _is_generic_title(track_info.title) and api_payload.get("title"):
                itunes_map["title"] = "title"
            apply_metadata_mapping(
                track_info,
                api_payload,
                itunes_map,
                force=force,
                source="iTunes",
            )
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(
            f"iTunes enrichment failed for {escape(track_info.title)}: {escape(str(error))}"
        )


def _enrich_lastfm(track_info: TrackInfo, lastfm_api_key: str | None) -> None:
    if not lastfm_api_key or track_info.genre:
        return
    try:
        tags = lastfm_service.fetch_lastfm_tags(
            track_info.artist,
            track_info.title,
            api_key=lastfm_api_key,
            mbid=track_info.musicbrainz_trackid,
        )
        if tags:
            raw_genre = tags[0]
            normalized_genre = normalize_genre(raw_genre)
            if normalized_genre:
                track_info.genre = normalized_genre
                track_info.metadata_sources["genre"] = "Last.fm"
                LOG.detail(f"   ∟ 🏷️ [Last.fm] Genre: [cyan]{normalized_genre}[/]")
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
        LOG.debug(
            f"Last.fm enrichment failed for {escape(track_info.title)}: {escape(str(error))}"
        )


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
            discogs_service.search_discogs_release(
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

        selectors = [
            (str(track_info.discogs_release_id or ""), str(release.get("id") or "")),
            (
                normalize_str(track_info.barcode or "")
                .replace(" ", "")
                .replace("-", "")
                .lstrip("0"),
                normalize_str(str(release.get("barcode") or ""))
                .replace(" ", "")
                .replace("-", "")
                .lstrip("0"),
            ),
            (
                normalize_str(track_info.catalog_number or ""),
                normalize_str(str(release.get("catalog_number") or "")),
            ),
        ]
        comparisons = [
            local == candidate for local, candidate in selectors if local and candidate
        ]
        format_families = (
            {"cd", "hdcd", "cdr"},
            {"vinyl", "lp"},
            {"cassette"},
            {"file", "digital", "flac", "aac", "mp3"},
        )
        local_format = next(
            (
                i
                for i, names in enumerate(format_families)
                if names.intersection(normalize_str(track_info.media or "").split())
            ),
            None,
        )
        candidate_format = next(
            (
                i
                for i, names in enumerate(format_families)
                if names.intersection(
                    normalize_str(str(release.get("media") or "")).split()
                )
            ),
            None,
        )
        edition_verified = (
            bool(comparisons)
            and all(comparisons)
            and (
                local_format is None
                or candidate_format is None
                or local_format == candidate_format
            )
        )
        discogs_rel_map = {"artist_id": "discogs_artist_id"}
        if edition_verified:
            discogs_rel_map.update(
                {
                    "id": "discogs_release_id",
                    "country": "release_country",
                    "label": "label",
                    "catalog_number": "catalog_number",
                    "barcode": "barcode",
                    "media": "media",
                    "composer": "composer",
                    "producers": "producers",
                    "remixer": "remixer",
                }
            )
            if not track_info.date or (
                force and not is_valid_uuid(track_info.musicbrainz_albumid)
            ):
                discogs_rel_map["released"] = "date"
                discogs_rel_map["year"] = "date"
        elif comparisons and (
            not all(comparisons)
            or (
                local_format is not None
                and candidate_format is not None
                and local_format != candidate_format
            )
        ):
            track_info.metadata_conflicts["discogs_release_id"] = (
                "Discogs release identifiers or medium disagree with local evidence; "
                "release identifiers, dates and pressing metadata withheld."
            )
        else:
            LOG.detail(
                "Discogs release edition unverified; pressing metadata withheld."
            )
        has_authoritative_mb = is_valid_uuid(
            track_info.musicbrainz_trackid
        ) or is_valid_uuid(track_info.musicbrainz_albumid)
        apply_metadata_mapping(
            track_info,
            release,
            discogs_rel_map,
            force=False if has_authoritative_mb else force,
            source="Discogs",
        )

        track_credits_dict = release.get("track_credits")
        if isinstance(track_credits_dict, dict):
            track_key = (
                str(track_info.track_number) if track_info.track_number else None
            )
            specific = None
            for candidate_key in (track_key, track_info.title.lower()):
                candidate_credits = track_credits_dict.get(candidate_key)
                if isinstance(
                    candidate_credits, dict
                ) and _catalog_recording_matches_track(
                    candidate_credits,
                    track_info.title,
                    track_info.artist,
                    None,
                    None,
                    track_info.duration,
                ):
                    specific = candidate_credits
                    break
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
                apply_metadata_mapping(
                    track_info,
                    specific,
                    discogs_track_map,
                    force=False if has_authoritative_mb else force,
                    source="Discogs",
                )
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(
            f"Discogs enrichment failed for {escape(track_info.title)}: {escape(str(error))}"
        )


def _deezer_medium_track_count(
    track_info: TrackInfo, recording: dict[str, Any], album: dict[str, Any]
) -> int | None:
    disc_number = safe_int(recording.get("disk_number")) or track_info.disc_number or 1
    disc_positions = album.get("tracks_by_disc_and_position")
    album_track_count = safe_int(album.get("nb_tracks"))
    if isinstance(disc_positions, dict) and disc_positions:
        if album_track_count is not None and len(disc_positions) != album_track_count:
            return None
        medium_positions = {
            position[1]
            for position in disc_positions
            if isinstance(position, tuple)
            and len(position) == 2
            and position[0] == disc_number
            and isinstance(position[1], int)
        }
        if medium_positions and medium_positions == set(
            range(1, len(medium_positions) + 1)
        ):
            return len(medium_positions)
        return None
    # Deezer's nb_tracks counts the entire release, not the selected medium.
    if (
        max(
            disc_number,
            track_info.total_discs or 1,
            safe_int(album.get("total_discs")) or 1,
        )
        == 1
    ):
        return (
            album_track_count if album_track_count and album_track_count > 0 else None
        )
    return None


def _enrich_deezer(
    track_info: TrackInfo,
    album_deezer_details: dict[str, Any] | None,
    has_album_context: bool = False,
    force: bool = False,
) -> None:
    try:
        album = album_deezer_details or (
            deezer_service.fetch_deezer_album_details(
                track_info.artist, track_info.album
            )
            if not has_album_context
            else None
        )
        if album:
            deezer_album_map = {"genre": "genre"}
            same_album = _is_generic(track_info.album, "album") or (
                normalize_str(track_info.album)
                == normalize_str(str(album.get("title") or ""))
            )
            if same_album:
                deezer_album_map["label"] = "label"
                if not track_info.date or (
                    force and not is_valid_uuid(track_info.musicbrainz_albumid)
                ):
                    deezer_album_map["release_date"] = "date"
            if track_info.barcode and str(album.get("barcode") or "").lstrip(
                "0"
            ) == track_info.barcode.lstrip("0"):
                deezer_album_map["barcode"] = "barcode"
            apply_metadata_mapping(
                track_info,
                album,
                deezer_album_map,
                force=False,
                source="Deezer",
            )
            if (
                album_deezer_details is not None
                and not is_valid_uuid(track_info.musicbrainz_albumid)
                and album.get("title")
                and (
                    _is_generic(track_info.album, "album")
                    or normalize_str(track_info.album)
                    == normalize_str(str(album["title"]))
                )
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

        track: dict[str, Any] | None = None
        track_is_from_album = False
        conflicting_isrc = False
        if album:
            track, track_is_from_album = _find_track_in_album_mapping(
                album.get("tracks_by_title"),
                album.get("tracks_by_disc_and_position"),
                track_info.title,
                track_info.track_number,
                disc_number=track_info.disc_number,
                artist=track_info.artist,
                isrc=track_info.isrc,
                duration=track_info.duration,
            )

        if not track and track_info.isrc:
            candidate = deezer_service.fetch_deezer_track_by_isrc(track_info.isrc)
            if candidate and recording_identity_matches(
                track_info.artist,
                track_info.title,
                str(candidate.get("artist") or ""),
                str(candidate.get("title") or ""),
                expected_duration=track_info.duration,
                candidate_duration=safe_float(candidate.get("duration")),
            ):
                track = candidate
            elif candidate:
                conflicting_isrc = True
                track_info.metadata_conflicts["isrc"] = (
                    "Stored ISRC resolves to a conflicting Deezer recording; "
                    "existing value requires independent verification."
                )
                LOG.detail(
                    f"Rejected conflicting Deezer ISRC result for {escape(track_info.file_path.name)}"
                )

        if not track and (
            not track_info.isrc or not track_info.producers or force or conflicting_isrc
        ):
            track = deezer_service.fetch_deezer_track_details(
                track_info.artist,
                track_info.title,
                duration=track_info.duration,
            )

        if track and conflicting_isrc:
            candidate_isrc = str(track.get("isrc") or "").strip().upper()
            candidate_duration = safe_float(track.get("duration"))
            if (
                candidate_isrc
                and candidate_isrc != str(track_info.isrc or "").strip().upper()
                and track_info.duration is not None
                and candidate_duration is not None
                and abs(track_info.duration - candidate_duration)
                <= get_config().single_duration_tolerance_seconds
                and score_artist_title_match(
                    track_info.artist,
                    track_info.title,
                    str(track.get("artist") or ""),
                    str(track.get("title") or ""),
                )
                == 100
            ):
                track_info.isrc = candidate_isrc
                track_info.metadata_sources["isrc"] = "Deezer"
                track_info.metadata_conflicts.pop("isrc", None)

        if not track or not _catalog_recording_matches_track(
            track,
            track_info.title,
            track_info.artist,
            track_info.isrc,
            None,
            track_info.duration,
        ):
            return

        sequence_album = _compatible_release_layout(track_info, album_deezer_details)
        track_is_from_album = track_is_from_album and sequence_album is not None
        if (
            track_is_from_album
            and sequence_album is not None
            and not is_valid_uuid(track_info.musicbrainz_albumid)
            and (force or not track_info.total_tracks)
        ):
            medium_track_count = _deezer_medium_track_count(
                track_info, track, sequence_album
            )
            if medium_track_count is not None and medium_track_count >= max(
                track_info.total_tracks or 0, track_info.track_number or 0
            ):
                track_info.total_tracks = medium_track_count
                track_info.metadata_sources["total_tracks"] = "Deezer"

        track_pos = safe_int(track.get("track_position"))
        if (
            track_pos is not None
            and track_info.track_number is None
            and track_is_from_album
        ):
            track_info.track_number = track_pos
        disk_num = safe_int(track.get("disk_number"))
        if (
            disk_num is not None
            and track_info.disc_number is None
            and track_is_from_album
        ):
            track_info.disc_number = disk_num
        explicit_lyrics = track.get("explicit_lyrics")
        if isinstance(explicit_lyrics, bool):
            track_info.advisory = "Explicit" if explicit_lyrics else None
            track_info.metadata_sources["advisory"] = "Deezer"
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
        apply_metadata_mapping(
            track_info,
            track,
            deezer_track_map,
            force=False if has_authoritative_mb else force,
            source="Deezer",
        )
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(
            f"Deezer enrichment failed for {escape(track_info.title)}: {escape(str(error))}"
        )


def _enrich_genius(track_info: TrackInfo, genius_api_token: str | None) -> None:
    if not genius_api_token:
        return
    credited_artist = track_info.artist
    featured_artists = extract_featured_artist_tokens(
        track_info.featured_artists, primary_artist=track_info.artist
    )
    if featured_artists:
        credited_artist = f"{credited_artist} feat. {', '.join(featured_artists)}"
    try:
        genius_details = genius_service.fetch_genius_song_details(
            credited_artist,
            track_info.title,
            api_token=genius_api_token,
            album=track_info.album,
        )
        if not genius_details:
            if not track_info.genius_song_id:
                return
            stored_song = genius_service.lookup_genius_song_details(
                track_info.genius_song_id, api_token=genius_api_token
            )
            if (
                not stored_song
                or _is_generic(track_info.artist, "artist")
                or _is_generic_title(track_info.title)
                or not stored_song.get("artist")
                or not stored_song.get("title")
            ):
                return
            identity_score = max(
                score_artist_title_match(
                    credited_artist,
                    track_info.title,
                    str(stored_song["artist"]),
                    str(stored_song["title"]),
                )
                for credited_artist in (
                    extract_featured_artist_tokens(track_info.artist)
                    or [track_info.artist]
                )
            )
            validated_score = genius_service.score_genius_song_match(
                credited_artist, track_info.title, stored_song, album=track_info.album
            )
            if identity_score <= 0 and validated_score <= 0:
                track_info.genius_song_id = None
                track_info.metadata_sources.pop("genius_song_id", None)
                track_info.metadata_conflicts["genius_song_id_validation"] = (
                    "Stored Genius ID identifies a different artist or title; ID removed."
                )
                return
            if (
                validated_score <= 0
                or validated_score < get_config().genius_match_threshold
            ):
                track_info.metadata_conflicts["genius_song_id_validation"] = (
                    "Stored Genius title or collaboration differs; credits "
                    "withheld pending recording verification."
                )
                return
            genius_details = stored_song
        previous_genius_id = track_info.genius_song_id
        replacement_genius_id = genius_details.get("genius_song_id")
        if (
            previous_genius_id
            and replacement_genius_id
            and str(replacement_genius_id) != previous_genius_id
        ):
            # A repaired identifier does not repair legacy fields retained by
            # the non-destructive merge. Their original provenance may be lost.
            for field_name, provider_field in (
                ("comment", "description"),
                ("composer", "writers"),
                ("lyricist", "writers"),
                ("producers", "producers"),
                ("featured_artists", "featured_artists"),
            ):
                existing = getattr(track_info, field_name)
                if (
                    existing
                    and existing != genius_details.get(provider_field)
                    and track_info.metadata_sources.get(field_name) in (None, "Genius")
                ):
                    track_info.metadata_conflicts.setdefault(
                        field_name,
                        "Genius identifier changed; existing metadata may belong "
                        "to the previous song and requires review. Value preserved.",
                    )
        apply_metadata_mapping(
            track_info,
            genius_details,
            {"genius_song_id": "genius_song_id"},
            force=True,
            source="Genius",
        )
        genius_map = {
            "writers": "composer",
            "featured_artists": "featured_artists",
            "producers": "producers",
        }
        if get_config().import_genius_descriptions:
            genius_map["description"] = "comment"
        if not track_info.date:
            genius_map["release_date"] = "date"
        apply_metadata_mapping(
            track_info,
            genius_details,
            genius_map,
            force=False,
            source="Genius",
        )
        if genius_details.get("writers") and not track_info.lyricist:
            track_info.lyricist = str(genius_details["writers"])
        LOG.debug("   ∟ 📝 [Genius] Fetched song details & credits")
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(
            f"Genius enrichment failed for {escape(track_info.title)}: {escape(str(error))}"
        )


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
        tadb_details = theaudiodb_service.fetch_theaudiodb_track_details(
            track_info.artist,
            track_info.title,
            recording_mbid=track_info.musicbrainz_trackid,
            duration=track_info.duration,
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
        apply_metadata_mapping(
            track_info,
            tadb_details,
            tadb_map,
            force=False,
            source="TheAudioDB",
        )
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(
            f"TheAudioDB enrichment failed for {escape(track_info.title)}: {escape(str(error))}"
        )
