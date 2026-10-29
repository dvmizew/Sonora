import io
import json
import os
import threading
from pathlib import Path
from typing import Any

import taglib
from mutagen import MutagenError
from mutagen.flac import FLAC
from mutagen.mp4 import MP4
from mutagen.wave import WAVE
from PIL import Image

from sonora.core.artists import (
    extract_featured_artist_tokens,
    normalize_featured_artists,
)
from sonora.core.constants import SUPPORTED_EXTS
from sonora.core.files import atomic_audio_edit
from sonora.core.library_paths import (
    extract_disc_number_from_folder,
    parse_track_filename,
)
from sonora.core.logger import LOG
from sonora.core.models import TrackInfo
from sonora.core.text import (
    is_valid_uuid,
    normalize_advisory,
    normalize_date,
    normalize_genre,
    normalize_str,
    safe_float,
    safe_int,
    safe_int_pair,
)
from sonora.core.titles import recording_identity_matches

_METADATA_CACHE: dict[tuple[str, int, int], TrackInfo] = {}
_METADATA_CACHE_LOCK = threading.RLock()
_MAX_METADATA_CACHE_SIZE = 1024

_UUID_FIELDS = {
    "musicbrainz_trackid",
    "musicbrainz_albumid",
    "musicbrainz_releasegroupid",
    "musicbrainz_artistid",
    "musicbrainz_workid",
    "musicbrainz_albumartistid",
}

# field_name -> (canonical_write_tag, *read_aliases)
_TAG_SCHEMA: dict[str, tuple[str, ...]] = {
    "artist_sort": ("ARTISTSORT",),
    "album_artist_sort": ("ALBUMARTISTSORT",),
    "isrc": ("ISRC", "TSRC"),
    "musicbrainz_trackid": (
        "MUSICBRAINZ_TRACKID",
        "MUSICBRAINZ TRACK ID",
        "TXXX:MUSICBRAINZ TRACK ID",
    ),
    "musicbrainz_albumid": (
        "MUSICBRAINZ_ALBUMID",
        "MUSICBRAINZ ALBUM ID",
        "TXXX:MUSICBRAINZ ALBUM ID",
    ),
    "musicbrainz_releasegroupid": (
        "MUSICBRAINZ_RELEASEGROUPID",
        "MUSICBRAINZ RELEASEGROUP ID",
    ),
    "musicbrainz_artistid": ("MUSICBRAINZ_ARTISTID", "MUSICBRAINZ ARTIST ID"),
    "musicbrainz_workid": ("MUSICBRAINZ_WORKID",),
    "musicbrainz_albumartistid": (
        "MUSICBRAINZ_ALBUMARTISTID",
        "MUSICBRAINZ ALBUM ARTIST ID",
    ),
    "acoustid_id": ("ACOUSTID_ID", "ACOUSTID ID"),
    "discogs_release_id": ("DISCOGS_RELEASE_ID", "DISCOGS RELEASE ID"),
    "discogs_artist_id": ("DISCOGS_ARTIST_ID", "DISCOGS ARTIST ID"),
    "itunes_trackid": ("ITUNESTRACKID", "ITUNES_TRACK_ID"),
    "itunes_collectionid": ("ITUNESCOLLECTIONID", "ITUNES_COLLECTION_ID"),
    "itunes_artistid": ("ITUNESARTISTID", "ITUNES_ARTIST_ID"),
    "spotify_trackid": (
        "SPOTIFY_TRACK_ID",
        "SPOTIFY_ID",
        "TXXX:SPOTIFY_TRACK_ID",
    ),
    "release_type": ("RELEASETYPE",),
    "release_status": ("RELEASESTATUS",),
    "release_country": ("RELEASECOUNTRY", "COUNTRY"),
    "label": ("LABEL", "PUBLISHER", "ORGANIZATION", "TPUB"),
    "catalog_number": ("CATALOGNUMBER", "CATALOG_NUMBER"),
    "barcode": ("BARCODE",),
    "media": ("MEDIA", "TMED"),
    "comment": ("COMMENT", "COMM"),
    "advisory": ("ITUNESADVISORY", "ADVISORY"),
    "cuesheet": ("CUESHEET",),
    "composer": ("COMPOSER", "TCOM"),
    "lyricist": ("LYRICIST", "TEXT", "WRITER"),
    "remixer": ("REMIXER", "TPE4"),
    "initial_key": ("INITIALKEY", "KEY", "TKEY"),
    "copyright": ("COPYRIGHT", "TCOP"),
    "language": ("LANGUAGE", "TLAN"),
    "script": ("SCRIPT",),
    "mood": ("MOOD",),
    "style": ("STYLE",),
    "disambiguation": ("DISAMBIGUATION", "TXXX:DISAMBIGUATION"),
    "featured_artists": ("FEATURED_ARTISTS", "TXXX:FEATURED_ARTISTS"),
    "producers": ("PRODUCERS", "TXXX:PRODUCERS"),
    "genius_song_id": ("GENIUS_SONG_ID", "TXXX:GENIUS_SONG_ID"),
    "music_video_url": ("MUSIC_VIDEO_URL", "TXXX:MUSIC_VIDEO_URL"),
    "lyrics": ("LYRICS", "UNSYNCEDLYRICS", "USLT"),
    "disc_subtitle": ("DISCSUBTITLE", "TSST", "SETSUBTITLE"),
}


def _get_tag(tags: dict[str, list[str]], *keys: str) -> str | None:
    for key in keys:
        values = tags.get(key) or tags.get(key.upper()) or tags.get(key.lower())
        if values:
            return str(values[0]).strip()
    return None


def _read_advisory(
    file_path: Path, tags: dict[str, list[str]]
) -> tuple[str | None, bool]:
    advisory = normalize_advisory(_get_tag(tags, *_TAG_SCHEMA["advisory"]))
    existing = {key: tags[key] for key in _TAG_SCHEMA["advisory"] if key in tags}
    if file_path.suffix.lower() in {".m4a", ".mp4"}:
        # TagLib's property map does not expose the native MP4 content-rating atom.
        rating = MP4(file_path).get("rtng")
        if rating is not None:
            advisory = "Explicit" if rating == [4] else None
        return advisory, bool(existing) or rating != ([4] if advisory else None)
    expected = {"ITUNESADVISORY": ["1"], "ADVISORY": ["Explicit"]} if advisory else {}
    return advisory, existing != expected


def _write_mp4_advisory(file_path: Path, advisory: str | None) -> None:
    recording = MP4(file_path)
    if normalize_advisory(advisory):
        recording["rtng"] = [4]
    else:
        recording.pop("rtng", None)
    recording.save()


def _read_audio_encoding(file_path: Path) -> tuple[bool, int | None]:
    file_ext = file_path.suffix.lower()
    if file_ext in {".flac", ".wav", ".aiff", ".alac", ".ape", ".wv"}:
        is_lossless = True
    elif file_ext in {".mp3", ".ogg", ".opus", ".mpc", ".wma"}:
        is_lossless = False
    elif file_ext in {".m4a", ".mp4"}:
        try:
            mp4_audio = MP4(file_path)
            is_lossless = str(getattr(mp4_audio.info, "codec", "")).lower() == "alac"
        except (MutagenError, OSError):
            is_lossless = False
    else:
        is_lossless = True

    bits_per_sample: int | None = None
    if file_ext in {".flac", ".wav"}:
        try:
            audio = FLAC(file_path) if file_ext == ".flac" else WAVE(file_path)
            bits_per_sample = safe_int(getattr(audio.info, "bits_per_sample", None))
        except (MutagenError, OSError):
            pass

    return is_lossless, bits_per_sample


def _featured_artists_from_tags(
    tags: dict[str, list[str]], primary_artist: str
) -> str | None:
    raw_artists = [*tags.get("ARTISTS", []), *tags.get("ARTIST", [])]
    primary_tokens = extract_featured_artist_tokens(primary_artist)
    primary_keys = {normalize_str(artist) for artist in primary_tokens if artist}
    primary_keys.add(normalize_str(primary_artist))
    extra_artists = [
        artist.strip()
        for artist in raw_artists
        if artist
        and artist.strip()
        and normalize_str(artist.strip()) not in primary_keys
    ]
    if not extra_artists:
        return None
    return normalize_featured_artists(extra_artists, primary_artist=primary_artist)


def _read_numbering_total(
    tags: dict[str, list[str]],
    position: int | None,
    inline_total: int | None,
    total_keys: tuple[str, str],
) -> tuple[int | None, bool, bool]:
    """Reconcile stored totals without inventing a release count or position."""
    totals = [
        value
        for value in (
            *(safe_int(_get_tag(tags, key)) for key in total_keys),
            inline_total,
        )
        if value is not None and value > 0
    ]
    if not totals:
        return None, False, False
    valid = {total for total in totals if position is None or total >= position}
    if len(valid) != 1:
        # Aliases are redundant representations, not independent edition evidence.
        return totals[0], False, True
    selected = next(iter(valid))
    return selected, len(set(totals)) > 1, False


def read_track_metadata(file_path: Path) -> TrackInfo:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported audio format: {file_path}")

    stat = file_path.stat()
    cache_key = (str(file_path.resolve()), stat.st_mtime_ns, stat.st_size)
    with _METADATA_CACHE_LOCK:
        cached = _METADATA_CACHE.get(cache_key)
        if cached is not None:
            return cached.copy()

    try:
        with taglib.File(str(file_path)) as song:
            tags = song.tags

            artist = _get_tag(tags, "ARTIST", "TPE1", "AUTHOR") or "Unknown Artist"
            title = _get_tag(tags, "TITLE", "TIT2") or "Unknown Title"
            album = _get_tag(tags, "ALBUM", "TALB", "WM/ALBUMTITLE") or "Unknown Album"
            album_artist = _get_tag(
                tags, "ALBUMARTIST", "ALBUM ARTIST", "TPE2", "WM/ALBUMARTIST"
            )
            date = normalize_date(_get_tag(tags, "DATE", "TDRC", "YEAR", "WM/YEAR"))
            original_date = _get_tag(tags, "ORIGINALDATE", "ORIGINALYEAR")
            genre_values: list[str] = next(
                (tags[key] for key in ("GENRE", "TCON", "WM/GENRE") if tags.get(key)),
                [],
            )
            genre = next(
                (
                    normalized
                    for value in genre_values
                    if (normalized := normalize_genre(value))
                ),
                None,
            )

            track_num_tag = _get_tag(tags, "TRACKNUMBER", "TRCK", "TRACK")
            track_number, track_total_fallback = safe_int_pair(track_num_tag)

            disc_num_tag = _get_tag(tags, "DISCNUMBER", "TPOS", "DISC")
            disc_number, disc_total_fallback = safe_int_pair(disc_num_tag)

            total_tracks, track_totals_need_normalization, track_totals_conflict = (
                _read_numbering_total(
                    tags,
                    track_number,
                    track_total_fallback,
                    ("TRACKTOTAL", "TOTALTRACKS"),
                )
            )
            total_discs, disc_totals_need_normalization, disc_totals_conflict = (
                _read_numbering_total(
                    tags, disc_number, disc_total_fallback, ("DISCTOTAL", "TOTALDISCS")
                )
            )

            # Filename and folder numbers fill absent tags; explicit disc 1 is evidence too.
            if disc_number is None and file_path.parent:
                folder_disc = extract_disc_number_from_folder(file_path.parent.name)
                if folder_disc is not None:
                    disc_number = folder_disc

            fn_disc, fn_track, _ = parse_track_filename(file_path.name)
            if fn_disc is not None and disc_number is None:
                disc_number = fn_disc
            if track_number is None and fn_track is not None:
                track_number = fn_track

            if disc_number is None:
                disc_number = 1
            if total_discs is None and disc_number > 1:
                total_discs = disc_number

            bpm = safe_float(_get_tag(tags, "BPM", "TBPM", "WM/BEATSPERMINUTE"))
            if bpm is not None and bpm <= 0:
                bpm = None
            raw_rating = _get_tag(tags, "RATING", "POPM")
            rating = safe_float(raw_rating)

            raw_compilation = _get_tag(tags, "COMPILATION", "TCMP")
            compilation = (
                True
                if raw_compilation in ("1", "true", "True")
                else False
                if raw_compilation in ("0", "false", "False")
                else None
            )

            replaygain_track_gain = safe_float(
                _get_tag(tags, "REPLAYGAIN_TRACK_GAIN", "TXXX:REPLAYGAIN_TRACK_GAIN")
            )
            replaygain_track_peak = safe_float(
                _get_tag(tags, "REPLAYGAIN_TRACK_PEAK", "TXXX:REPLAYGAIN_TRACK_PEAK")
            )
            replaygain_album_gain = safe_float(
                _get_tag(tags, "REPLAYGAIN_ALBUM_GAIN", "TXXX:REPLAYGAIN_ALBUM_GAIN")
            )
            replaygain_album_peak = safe_float(
                _get_tag(tags, "REPLAYGAIN_ALBUM_PEAK", "TXXX:REPLAYGAIN_ALBUM_PEAK")
            )

            art_width, art_height = None, None
            if hasattr(song, "pictures") and song.pictures:
                first_picture = next(
                    (
                        picture
                        for picture in song.pictures
                        if picture.picture_type == "Front Cover"
                    ),
                    song.pictures[0],
                )
                art_width = getattr(first_picture, "width", None) or None
                art_height = getattr(first_picture, "height", None) or None
                if (not art_width or not art_height) and getattr(
                    first_picture, "data", None
                ):
                    try:
                        with Image.open(io.BytesIO(first_picture.data)) as img:
                            art_width, art_height = img.size
                    except OSError as e:
                        LOG.debug(f"Failed to read image dimensions: {e}")

            mapped_fields: dict[str, Any] = {
                field: _get_tag(tags, *tag_keys)
                for field, tag_keys in _TAG_SCHEMA.items()
            }
            advisory, advisory_needs_normalization = _read_advisory(file_path, tags)
            tidal_payload = None
            try:
                value = _get_tag(tags, "TIDAL_DATA")
                payload = json.loads(value) if value else None
                if isinstance(payload, dict):
                    provider_artist = payload.get("artist")
                    if isinstance(provider_artist, dict):
                        provider_artist = provider_artist.get("name")
                    provider_album = payload.get("album")
                    if isinstance(provider_album, dict):
                        provider_album = provider_album.get("title")
                    version = str(payload.get("version") or "").strip()
                    edition_text = normalize_str(f"{title} {album}")
                    if (
                        (
                            not provider_album
                            or normalize_str(str(provider_album))
                            == normalize_str(album)
                        )
                        and (not version or normalize_str(version) in edition_text)
                        and not (
                            advisory == "Explicit" and payload.get("explicit") is False
                        )
                        and recording_identity_matches(
                            artist,
                            title,
                            str(provider_artist or ""),
                            str(payload.get("title") or ""),
                            expected_duration=(
                                float(song.length) if song.length is not None else None
                            ),
                            candidate_duration=safe_float(payload.get("duration")),
                        )
                        and payload.get("isrc")
                    ):
                        tidal_payload = payload
            except (TypeError, ValueError):
                pass
            if not mapped_fields.get("featured_artists"):
                additional_features = _featured_artists_from_tags(tags, artist)
                if additional_features:
                    mapped_fields["featured_artists"] = additional_features
            mapped_fields["advisory"] = advisory

            is_lossless, bits_per_sample = _read_audio_encoding(file_path)

            track_info = TrackInfo(
                file_path=file_path,
                _advisory_needs_normalization=advisory_needs_normalization,
                _numbering_needs_normalization=(
                    track_totals_need_normalization or disc_totals_need_normalization
                ),
                metadata_conflicts={
                    "album_sequence": (
                        "Stored numbering totals conflict with each other or the position; "
                        "release verification required."
                    )
                }
                if track_totals_conflict or disc_totals_conflict
                else {},
                artist=artist,
                title=title,
                album=album,
                album_artist=album_artist,
                track_number=track_number,
                disc_number=disc_number,
                total_tracks=total_tracks,
                total_discs=total_discs,
                date=date,
                original_date=original_date,
                genre=genre,
                bpm=bpm,
                rating=rating,
                compilation=compilation,
                sample_rate=song.sampleRate,
                bitrate=song.bitrate,
                channels=song.channels,
                duration=float(song.length) if song.length is not None else None,
                bits_per_sample=bits_per_sample,
                is_lossless=is_lossless,
                replaygain_track_gain=replaygain_track_gain,
                replaygain_track_peak=replaygain_track_peak,
                replaygain_album_gain=replaygain_album_gain,
                replaygain_album_peak=replaygain_album_peak,
                art_width=art_width,
                art_height=art_height,
                **mapped_fields,
            )
            if tidal_payload:
                track_info.isrc = str(tidal_payload["isrc"]).strip().upper()
                track_info.metadata_sources["isrc"] = "Tidal"
            with _METADATA_CACHE_LOCK:
                if _METADATA_CACHE and len(_METADATA_CACHE) >= _MAX_METADATA_CACHE_SIZE:
                    _METADATA_CACHE.pop(next(iter(_METADATA_CACHE)))
                _METADATA_CACHE[cache_key] = track_info.copy()
            return track_info
    except FileNotFoundError:
        raise
    except (OSError, ValueError) as error:
        raise OSError(f"Failed to read metadata for {file_path}: {error}") from error


def _write_artist_tags(tags: dict[str, list[str]], track_info: TrackInfo) -> None:
    tags["ARTIST"] = [track_info.artist]
    tags["TITLE"] = [track_info.title]
    tags["ALBUM"] = [track_info.album]

    primary_tokens = (
        extract_featured_artist_tokens(track_info.artist) if track_info.artist else []
    )
    if not primary_tokens and track_info.artist:
        primary_tokens = [track_info.artist]

    feat_tokens = (
        extract_featured_artist_tokens(
            track_info.featured_artists,
            primary_artist=track_info.artist,
        )
        if track_info.featured_artists
        else []
    )

    all_track_artists: list[str] = []
    for art_candidate in [*primary_tokens, *feat_tokens]:
        clean_art = str(art_candidate).strip()
        if clean_art and clean_art not in all_track_artists:
            all_track_artists.append(clean_art)

    if len(all_track_artists) > 1:
        tags["ARTISTS"] = all_track_artists
    else:
        tags.pop("ARTISTS", None)
        tags.pop("TXXX:ARTISTS", None)

    if track_info.album_artist:
        tags["ALBUMARTIST"] = [track_info.album_artist]
        album_tokens = extract_featured_artist_tokens(track_info.album_artist)
        if not album_tokens:
            album_tokens = [track_info.album_artist]
        unique_album_artists: list[str] = []
        for alb_candidate in album_tokens:
            clean_alb = str(alb_candidate).strip()
            if clean_alb and clean_alb not in unique_album_artists:
                unique_album_artists.append(clean_alb)
        if len(unique_album_artists) > 1 and normalize_str(
            track_info.album_artist
        ) not in ("various artists", "soundtrack"):
            tags["ALBUMARTISTS"] = unique_album_artists
        else:
            tags.pop("ALBUMARTISTS", None)
            tags.pop("TXXX:ALBUMARTISTS", None)
    else:
        tags.pop("ALBUMARTIST", None)
        tags.pop("ALBUM ARTIST", None)
        tags.pop("ALBUMARTISTS", None)
        tags.pop("TXXX:ALBUMARTISTS", None)


def _write_numbering_and_dates(
    tags: dict[str, list[str]], track_info: TrackInfo
) -> None:
    for name, number_tag, total_tag, total_alias in (
        ("track", "TRACKNUMBER", "TRACKTOTAL", "TOTALTRACKS"),
        ("disc", "DISCNUMBER", "DISCTOTAL", "TOTALDISCS"),
    ):
        number = getattr(track_info, f"{name}_number")
        total = getattr(track_info, f"total_{name}s")
        stored_number, stored_inline_total = safe_int_pair(_get_tag(tags, number_tag))
        stored_total, _, unresolved = _read_numbering_total(
            tags, stored_number, stored_inline_total, (total_tag, total_alias)
        )
        if (
            unresolved
            and number == stored_number
            and total == stored_total
            and not any(
                key in track_info.metadata_sources
                for key in (f"{name}_number", f"total_{name}s")
            )
        ):
            # An unrelated tag update must not discard conflicting release evidence.
            continue
        if number is None:
            tags.pop(number_tag, None)
        else:
            tags[number_tag] = [f"{number}/{total}" if total else str(number)]
        for key in (total_tag, total_alias):
            if total:
                tags[key] = [str(total)]
            else:
                tags.pop(key, None)

    if track_info.date:
        tags["DATE"] = [track_info.date]
    else:
        tags.pop("DATE", None)
    tags.pop("YEAR", None)

    if track_info.original_date:
        tags["ORIGINALDATE"] = [track_info.original_date]
        tags["ORIGINALYEAR"] = [track_info.original_date[:4]]
    else:
        tags.pop("ORIGINALDATE", None)
        tags.pop("ORIGINALYEAR", None)

    if track_info.genre:
        tags["GENRE"] = [track_info.genre]
    else:
        tags.pop("GENRE", None)


def _write_audio_properties_tags(
    tags: dict[str, list[str]], track_info: TrackInfo
) -> None:
    bpm = safe_float(track_info.bpm)
    if bpm is not None and bpm > 0:
        tags["BPM"] = [f"{bpm:.1f}"]
    else:
        tags.pop("BPM", None)

    if track_info.rating is not None:
        tags["RATING"] = [f"{track_info.rating:.1f}"]
    else:
        tags.pop("RATING", None)

    if track_info.compilation is not None:
        tags["COMPILATION"] = ["1" if track_info.compilation else "0"]
    elif track_info.album_artist and normalize_str(track_info.album_artist) in (
        "various artists",
        "soundtrack",
    ):
        tags["COMPILATION"] = ["1"]
    else:
        tags.pop("COMPILATION", None)

    for tag_key, measurement, value_format in (
        ("REPLAYGAIN_TRACK_GAIN", track_info.replaygain_track_gain, "+.2f"),
        ("REPLAYGAIN_TRACK_PEAK", track_info.replaygain_track_peak, ".6f"),
        ("REPLAYGAIN_ALBUM_GAIN", track_info.replaygain_album_gain, "+.2f"),
        ("REPLAYGAIN_ALBUM_PEAK", track_info.replaygain_album_peak, ".6f"),
    ):
        if measurement is None:
            tags.pop(tag_key, None)
        else:
            unit = " dB" if tag_key.endswith("_GAIN") else ""
            tags[tag_key] = [f"{measurement:{value_format}}{unit}"]


def _write_schema_tags(tags: dict[str, list[str]], track_info: TrackInfo) -> None:
    for field, tag_keys in _TAG_SCHEMA.items():
        tag_value = getattr(track_info, field, None)
        if not tag_value or (
            field in _UUID_FIELDS
            and not is_valid_uuid(tag_value, allow_multivalue=True)
        ):
            for key in tag_keys:
                tags.pop(key, None)
            continue
        if field == "advisory":
            if normalize_advisory(str(tag_value)):
                tags["ITUNESADVISORY"] = ["1"]
                tags["ADVISORY"] = ["Explicit"]
            else:
                tags.pop("ITUNESADVISORY", None)
                tags.pop("ADVISORY", None)
        elif field == "disc_subtitle":
            for key in tag_keys:
                tags[key] = [str(tag_value)]
        elif field == "lyrics":
            tags["LYRICS"] = [str(tag_value)]
            tags["UNSYNCEDLYRICS"] = [str(tag_value)]
        else:
            tags[tag_keys[0]] = [str(tag_value)]
            for alias in tag_keys[1:]:
                tags.pop(alias, None)


def _preserve_unchanged_multivalues(
    original_tags: dict[str, list[str]],
    updated_tags: dict[str, list[str]],
) -> None:
    original_genres = [
        normalized
        for value in original_tags.get("GENRE", [])
        if (normalized := normalize_genre(value))
    ]
    updated_genres = updated_tags.get("GENRE")
    if (
        original_genres
        and updated_genres
        and len(updated_genres) == 1
        and normalize_genre(updated_genres[0]) == original_genres[0]
    ):
        updated_tags["GENRE"] = [updated_genres[0], *original_genres[1:]]
    for tag_key, original_values in original_tags.items():
        updated_values = updated_tags.get(tag_key)
        if (
            len(original_values) > 1
            and updated_values
            and len(updated_values) == 1
            and normalize_str(updated_values[0]) == normalize_str(original_values[0])
        ):
            updated_tags[tag_key] = [updated_values[0], *original_values[1:]]
    for tag_keys in _TAG_SCHEMA.values():
        original_values = next(
            (original_tags[key] for key in tag_keys if original_tags.get(key)), []
        )
        updated_values = updated_tags.get(tag_keys[0])
        if (
            len(original_values) > 1
            and updated_values
            and len(updated_values) == 1
            and normalize_str(updated_values[0]) == normalize_str(original_values[0])
        ):
            updated_tags[tag_keys[0]] = [updated_values[0], *original_values[1:]]
    original_album_artist = _get_tag(
        original_tags, "ALBUMARTIST", "ALBUM ARTIST", "ALBUMARTISTS"
    )
    updated_album_artist = _get_tag(updated_tags, "ALBUMARTIST", "ALBUM ARTIST")
    if (
        original_album_artist
        and updated_album_artist
        and normalize_str(original_album_artist) == normalize_str(updated_album_artist)
    ):
        for tag_key in ("ALBUMARTISTS", "TXXX:ALBUMARTISTS"):
            if tag_key not in updated_tags and original_tags.get(tag_key):
                updated_tags[tag_key] = original_tags[tag_key]


def write_track_metadata(
    track_info: TrackInfo,
    cover_art_path: Path | None = None,
    *,
    lyrics_only: bool = False,
) -> None:
    if not track_info.file_path.exists():
        raise FileNotFoundError(f"File not found: {track_info.file_path}")
    if not os.access(track_info.file_path, os.W_OK):
        raise PermissionError(
            f"Permission denied: File is read-only '{track_info.file_path}'"
        )

    try:
        with atomic_audio_edit(track_info.file_path) as staging_path:
            with taglib.File(str(staging_path)) as song:
                original_tags = {key: list(values) for key, values in song.tags.items()}
                _write_artist_tags(song.tags, track_info)
                _write_numbering_and_dates(song.tags, track_info)
                _write_audio_properties_tags(song.tags, track_info)
                _write_schema_tags(song.tags, track_info)
                if track_info.file_path.suffix.lower() in {".m4a", ".mp4"}:
                    for key in _TAG_SCHEMA["advisory"]:
                        song.tags.pop(key, None)
                _preserve_unchanged_multivalues(original_tags, song.tags)

                if lyrics_only:
                    lyrics_tags = {
                        key: song.tags[key]
                        for key in _TAG_SCHEMA["lyrics"]
                        if key in song.tags
                    }
                    song.tags.clear()
                    song.tags.update(original_tags)
                    for key in _TAG_SCHEMA["lyrics"]:
                        song.tags.pop(key, None)
                    song.tags.update(lyrics_tags)

                if not lyrics_only and cover_art_path and cover_art_path.exists():
                    cover_bytes = cover_art_path.read_bytes()
                    with Image.open(io.BytesIO(cover_bytes)) as cover_image:
                        cover_image.load()
                        cover_dimensions = cover_image.size
                        mime = Image.MIME.get(cover_image.format or "", "image/jpeg")
                    existing_pictures = song.pictures
                    front_covers = [
                        picture
                        for picture in existing_pictures
                        if picture.picture_type == "Front Cover"
                    ]
                    existing_dimensions: list[tuple[int, int]] = []
                    for picture in front_covers:
                        try:
                            with Image.open(io.BytesIO(picture.data)) as embedded_image:
                                existing_dimensions.append(embedded_image.size)
                        except OSError:
                            continue
                    if not any(
                        width > cover_dimensions[0] or height > cover_dimensions[1]
                        for width, height in existing_dimensions
                    ):
                        song.pictures = [
                            picture
                            for picture in existing_pictures
                            if picture.picture_type != "Front Cover"
                        ] + [
                            taglib.Picture(
                                data=cover_bytes,
                                mime_type=mime,
                                description="Cover",
                                picture_type="Front Cover",
                            )
                        ]

                unsaved = song.save()
                if unsaved:
                    raise OSError(
                        f"Unsupported tags for {track_info.file_path}: {', '.join(unsaved)}"
                    )
            if not lyrics_only and track_info.file_path.suffix.lower() in {
                ".m4a",
                ".mp4",
            }:
                _write_mp4_advisory(staging_path, track_info.advisory)
        if not lyrics_only:
            track_info._advisory_needs_normalization = False
            track_info._numbering_needs_normalization = False
        clear_metadata_cache(track_info.file_path)
    except FileNotFoundError:
        raise
    except (OSError, ValueError, MutagenError) as error:
        raise OSError(
            f"Failed to write metadata for {track_info.file_path}: {error}"
        ) from error


def clear_metadata_cache(file_path: Path | None = None) -> int:
    """Invalidate one file, or the entire cache, and return the entry count removed."""
    with _METADATA_CACHE_LOCK:
        if file_path is None:
            count = len(_METADATA_CACHE)
            _METADATA_CACHE.clear()
            return count
        resolved = str(file_path.resolve())
        keys = [key for key in _METADATA_CACHE if key[0] == resolved]
        for key in keys:
            del _METADATA_CACHE[key]
        return len(keys)


def get_metadata_cache_size() -> int:
    """Return the number of entries currently cached in memory."""
    with _METADATA_CACHE_LOCK:
        return len(_METADATA_CACHE)


def get_audio_duration(file_path: Path) -> float | None:
    try:
        import mutagen

        m_file = mutagen.File(file_path)
        if m_file and getattr(m_file, "info", None):
            return safe_float(m_file.info.length)
    except (OSError, MutagenError):
        pass
    return None
