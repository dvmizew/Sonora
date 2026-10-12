from pathlib import Path
from typing import Any

import acoustid
import httpx
from musicbrainzngs import MusicBrainzError
from rapidfuzz import fuzz
from rich.markup import escape

from sonora.audio import key as key_module
from sonora.core import artists as artists_module
from sonora.core.api_payloads import require_api_object
from sonora.core.artists import normalize_featured_artists
from sonora.core.config import get_config
from sonora.core.logger import LOG
from sonora.core.models import MetadataSource, TrackInfo
from sonora.core.text import (
    clean_unicode_punct,
    is_valid_uuid,
    normalize_country_name,
    normalize_date,
    normalize_genre,
    normalize_language_name,
    normalize_script_name,
    normalize_str,
    preserve_unicode_repertoire,
    safe_float,
    safe_int,
)
from sonora.core.titles import clean_title, extract_title_features
from sonora.modules.matching import (
    _is_generic,
    _is_generic_title,
    _is_strong_track_match,
    _recording_isrc_matches,
)
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


def apply_metadata_mapping(
    track_info: TrackInfo,
    metadata_dict: dict[str, Any],
    field_map: dict[str, str],
    force: bool = False,
    source: MetadataSource | None = None,
) -> None:
    for src_key, target_attr in field_map.items():
        provider_value = metadata_dict.get(src_key)
        if provider_value is None:
            continue
        normalized_value: str | None = str(provider_value).strip()
        if not normalized_value or normalized_value.lower() in ("none", "null"):
            continue

        if target_attr == "isrc" and _recording_isrc_matches(
            track_info.isrc, metadata_dict
        ):
            normalized_value = str(track_info.isrc).strip().upper()
        if target_attr == "genre":
            normalized_value = normalize_genre(normalized_value)
        elif target_attr in ("date", "original_date"):
            normalized_value = normalize_date(normalized_value)
        elif target_attr == "release_country":
            normalized_value = normalize_country_name(normalized_value)
        elif target_attr == "language":
            normalized_value = normalize_language_name(normalized_value)
        elif target_attr == "script":
            normalized_value = normalize_script_name(normalized_value)
        if not normalized_value:
            continue

        if target_attr == "title":
            normalized_value = clean_unicode_punct(normalized_value)
            clean_t, extracted_feats = extract_title_features(
                normalized_value, primary_artist=track_info.artist
            )
            if extracted_feats:
                track_info.featured_artists = normalize_featured_artists(
                    [track_info.featured_artists, *extracted_feats],
                    primary_artist=track_info.artist,
                )
            existing_title = getattr(track_info, "title", None)
            normalized_value = preserve_unicode_repertoire(
                str(existing_title) if existing_title else None,
                clean_t or normalized_value,
            )
        elif target_attr in ("artist", "album", "album_artist"):
            normalized_value = clean_unicode_punct(normalized_value)
            existing_val = getattr(track_info, target_attr, None)
            normalized_value = preserve_unicode_repertoire(
                str(existing_val) if existing_val else None, normalized_value
            )
        elif target_attr == "advisory":
            if str(normalized_value).strip().capitalize() != "Explicit":
                continue
            normalized_value = "Explicit"
        elif (
            target_attr == "initial_key"
            and key_module.key_to_camelot(normalized_value) is None
        ):
            continue
        elif target_attr == "featured_artists":
            existing_featured = getattr(track_info, "featured_artists", None)
            if existing_featured and not force:
                normalized_result = normalize_featured_artists(
                    [existing_featured, normalized_value],
                    primary_artist=track_info.artist,
                )
            else:
                normalized_result = normalize_featured_artists(
                    normalized_value,
                    primary_artist=track_info.artist,
                )
            if not normalized_result:
                continue
            normalized_value = normalized_result

        if target_attr in (
            "disc_number",
            "total_discs",
            "track_number",
            "total_tracks",
        ):
            val_int = safe_int(normalized_value)
            if val_int is None:
                continue
            cur_int = safe_int(getattr(track_info, target_attr, None))
            if target_attr in ("disc_number", "total_discs"):
                if cur_int is None or cur_int != val_int:
                    setattr(track_info, target_attr, val_int)
            elif cur_int is None or force:
                setattr(track_info, target_attr, val_int)
            continue

        if target_attr in ("label", "copyright", "composer", "lyricist", "producers"):
            existing_credit = getattr(track_info, target_attr)
            if existing_credit and set(normalize_str(normalized_value).split()) < set(
                normalize_str(str(existing_credit)).split()
            ):
                continue

        current_value = getattr(track_info, target_attr)
        missing_identity = (
            target_attr == "title" and _is_generic_title(current_value)
        ) or (
            target_attr in ("artist", "album", "album_artist")
            and _is_generic(current_value, target_attr)
        )
        if (
            not current_value
            or missing_identity
            or target_attr == "featured_artists"
            or force
        ):
            setattr(track_info, target_attr, normalized_value)
            if source is not None:
                track_info.metadata_sources[target_attr] = source
        elif (
            getattr(track_info, target_attr) == normalized_value and source is not None
        ):
            track_info.metadata_sources.setdefault(target_attr, source)


def _catalog_recording_matches_track(
    recording: dict[str, Any],
    title: str | None,
    artist: str | None,
    isrc: str | None,
    itunes_trackid: str | None,
) -> bool:
    matched_identifier = False
    expected_isrc = (isrc or "").strip().upper()
    provider_isrcs = recording.get("isrcs") or [recording.get("isrc")]
    if expected_isrc not in ("", "NONE", "NULL", "0") and any(provider_isrcs):
        if not _recording_isrc_matches(expected_isrc, recording):
            return False
        matched_identifier = True
    expected_itunes_id = (itunes_trackid or "").strip()
    provider_itunes_id = str(
        recording.get("itunes_trackid") or recording.get("trackId") or ""
    ).strip()
    if expected_itunes_id not in ("", "None", "null", "0") and provider_itunes_id:
        if expected_itunes_id != provider_itunes_id:
            return False
        matched_identifier = True
    if matched_identifier:
        return True
    if _is_generic_title(title):
        return False
    candidate_artist = recording.get("artist") or recording.get("artistName")
    candidate_title = recording.get("title") or recording.get("trackName")
    return _is_strong_track_match(
        artist if candidate_artist and not _is_generic(artist, "artist") else None,
        title,
        str(candidate_artist) if candidate_artist else None,
        str(candidate_title) if candidate_title else None,
    )


def _find_track_in_album_mapping(
    tracks_by_title: Any,
    tracks_by_pos: Any,
    title: str | None,
    track_number: int | None,
    disc_number: int | None = None,
    tracks_by_disc_and_position: Any = None,
    *,
    artist: str | None = None,
    isrc: str | None = None,
    itunes_trackid: str | None = None,
) -> tuple[dict[str, Any] | None, bool]:
    disc_keys: tuple[object, ...] = (
        ((disc_number, track_number),)
        if disc_number is not None and track_number is not None
        else ()
    )
    combined_position_keys = (
        (*disc_keys, f"{disc_number}-{track_number}") if disc_keys else ()
    )
    title_keys = (
        tuple(dict.fromkeys((normalize_str(clean_title(title)), normalize_str(title))))
        if title and not _is_generic_title(title)
        else ()
    )
    candidate_indexes = (
        (tracks_by_disc_and_position, disc_keys),
        (tracks_by_pos, combined_position_keys),
        (tracks_by_title, title_keys),
        (tracks_by_pos, (track_number,) if track_number is not None else ()),
    )
    for recording_index, candidate_keys in candidate_indexes:
        if not isinstance(recording_index, dict):
            continue
        for candidate_key in candidate_keys:
            if candidate_key not in recording_index:
                continue
            recording = require_api_object(
                recording_index[candidate_key], "Album recording"
            )
            # Position is an edition-specific hint; it must not import another song's fields.
            if _catalog_recording_matches_track(
                recording, title, artist, isrc, itunes_trackid
            ):
                return recording, True
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
        shazam_match = shazam_service.recognize_audio_track(file_path)
        if not shazam_match:
            return

        # If target_album_artist is known, do not allow Shazam to overwrite a track whose artist already
        # matches the album artist with a completely different artist (sample hijacking)!
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
                    track_info.artist = artists_module.resolve_artist_name(
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
            track_info.artist = artists_module.resolve_artist_name(shazam_match.artist)
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
                    "tracks_by_disc_and_position"
                ),
                artist=track_info.artist,
                isrc=track_info.isrc,
                itunes_trackid=track_info.itunes_trackid,
            )

        if not api_payload and (not track_info.genre or not track_info.advisory):
            api_payload = itunes_service.fetch_itunes_track_metadata(
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
            apply_metadata_mapping(
                track_info,
                api_payload,
                itunes_map,
                force=force,
                source="iTunes",
            )
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"iTunes enrichment failed for {track_info.title}: {error}")


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
                apply_metadata_mapping(
                    track_info,
                    specific,
                    discogs_track_map,
                    force=False if has_authoritative_mb else force,
                    source="Discogs",
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
            deezer_service.fetch_deezer_album_details(
                track_info.artist, track_info.album
            )
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
                artist=track_info.artist,
                isrc=track_info.isrc,
            )

        if not track and track_info.isrc:
            track = deezer_service.fetch_deezer_track_by_isrc(track_info.isrc)

        if not track and (not track_info.isrc or not track_info.producers or force):
            track = deezer_service.fetch_deezer_track_details(
                track_info.artist, track_info.title
            )

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
        apply_metadata_mapping(
            track_info,
            track,
            deezer_track_map,
            force=False if has_authoritative_mb else force,
            source="Deezer",
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
    if track_info.composer and track_info.producers and not force:
        return
    try:
        genius_details = genius_service.fetch_genius_song_details(
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
        tadb_details = theaudiodb_service.fetch_theaudiodb_track_details(
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
        apply_metadata_mapping(
            track_info,
            tadb_details,
            tadb_map,
            force=False,
            source="TheAudioDB",
        )
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"TheAudioDB enrichment failed for {track_info.title}: {error}")
