import dataclasses
import io
import os
import threading
from pathlib import Path
from typing import Any, cast

import mutagen.flac
import mutagen.mp4
import mutagen.wave
import taglib
from mutagen._util import MutagenError
from PIL import Image

from sonora.core.constants import SUPPORTED_EXTS
from sonora.core.logger import LOG
from sonora.core.models import TrackInfo
from sonora.core.utils import (
    extract_disc_number_from_folder,
    extract_featured_artist_tokens,
    is_valid_uuid,
    normalize_date,
    normalize_featured_artists,
    normalize_genre,
    normalize_str,
    parse_track_filename,
    safe_float,
    safe_int,
    safe_int_pair,
)

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
        if values and len(values) > 0 and values[0] is not None:
            return str(values[0]).strip()
    return None


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
            return dataclasses.replace(cached)

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
            genre = normalize_genre(_get_tag(tags, "GENRE", "TCON", "WM/GENRE"))

            # Track and disc numbering
            track_num_tag = _get_tag(tags, "TRACKNUMBER", "TRCK", "TRACK")
            track_number, track_total_fallback = safe_int_pair(track_num_tag)

            disc_num_tag = _get_tag(tags, "DISCNUMBER", "TPOS", "DISC")
            disc_number, disc_total_fallback = safe_int_pair(disc_num_tag)

            raw_total_tracks = _get_tag(tags, "TRACKTOTAL", "TOTALTRACKS")
            total_tracks = safe_int(raw_total_tracks) or track_total_fallback

            raw_total_discs = _get_tag(tags, "DISCTOTAL", "TOTALDISCS")
            total_discs = safe_int(raw_total_discs) or disc_total_fallback

            # Deduce disc and track numbers from folder structure or filename when missing or default
            if (disc_number is None or disc_number == 1) and file_path.parent:
                folder_disc = extract_disc_number_from_folder(file_path.parent.name)
                if folder_disc is not None:
                    disc_number = folder_disc

            fn_disc, fn_track, _ = parse_track_filename(file_path.name)
            if fn_disc is not None and (disc_number is None or disc_number == 1):
                disc_number = fn_disc
            if track_number is None and fn_track is not None:
                track_number = fn_track

            if disc_number is None:
                disc_number = 1
            if total_discs is None and disc_number > 1:
                total_discs = disc_number

            # Numerical and audio stats
            bpm = safe_float(_get_tag(tags, "BPM", "TBPM", "WM/BEATSPERMINUTE"))
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

            # ReplayGain
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
                first_picture = song.pictures[0]
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
            if not mapped_fields.get("featured_artists"):
                raw_artists_list = tags.get("ARTISTS") or []
                if raw_artists_list:
                    primary_tokens = extract_featured_artist_tokens(
                        artist, allow_network=False
                    )
                    primary_norms = {
                        normalize_str(tok) for tok in primary_tokens if tok
                    }
                    extra_artists = [
                        art_candidate.strip()
                        for art_candidate in raw_artists_list
                        if art_candidate
                        and art_candidate.strip()
                        and normalize_str(art_candidate.strip()) not in primary_norms
                    ]
                    if extra_artists:
                        mapped_fields["featured_artists"] = normalize_featured_artists(
                            extra_artists, primary_artist=artist, allow_network=False
                        )
            raw_advisory = mapped_fields.get("advisory")
            if raw_advisory:
                raw_str = str(raw_advisory).strip().lower()
                if raw_str in ("1", "explicit"):
                    mapped_fields["advisory"] = "Explicit"
                else:
                    mapped_fields["advisory"] = None

            file_ext = file_path.suffix.lower()
            if file_ext in {".flac", ".wav", ".aiff", ".alac", ".ape", ".wv"}:
                is_lossless = True
            elif file_ext in {".mp3", ".ogg", ".opus", ".mpc", ".wma"}:
                is_lossless = False
            elif file_ext in {".m4a", ".mp4"}:
                try:
                    mp4_audio = cast(Any, mutagen.mp4.MP4)(file_path)
                    is_lossless = (
                        str(getattr(mp4_audio.info, "codec", "")).lower() == "alac"
                    )
                except (MutagenError, OSError):
                    is_lossless = False
            else:
                is_lossless = True

            bits_per_sample: int | None = None
            if file_ext == ".flac":
                try:
                    flac_info = cast(Any, mutagen.flac.FLAC)(file_path).info
                    bits_per_sample = safe_int(
                        getattr(flac_info, "bits_per_sample", None)
                    )
                except (MutagenError, OSError):
                    pass
            elif file_ext == ".wav":
                try:
                    wav_info = cast(Any, mutagen.wave.WAVE)(file_path).info
                    bits_per_sample = safe_int(
                        getattr(wav_info, "bits_per_sample", None)
                    )
                except (MutagenError, OSError):
                    pass

            track_info = TrackInfo(
                file_path=file_path,
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
            with _METADATA_CACHE_LOCK:
                if len(_METADATA_CACHE) >= _MAX_METADATA_CACHE_SIZE:
                    _METADATA_CACHE.clear()
                _METADATA_CACHE[cache_key] = dataclasses.replace(track_info)
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
        extract_featured_artist_tokens(track_info.artist, allow_network=False)
        if track_info.artist
        else []
    )
    if not primary_tokens and track_info.artist:
        primary_tokens = [track_info.artist]

    feat_tokens = (
        extract_featured_artist_tokens(
            track_info.featured_artists,
            primary_artist=track_info.artist,
            allow_network=False,
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
        album_tokens = extract_featured_artist_tokens(
            track_info.album_artist, allow_network=False
        )
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
    total_tracks_str = str(track_info.total_tracks) if track_info.total_tracks else None
    if track_info.track_number is not None:
        tags["TRACKNUMBER"] = [
            f"{track_info.track_number}/{total_tracks_str}"
            if total_tracks_str
            else str(track_info.track_number)
        ]
    else:
        tags.pop("TRACKNUMBER", None)

    if total_tracks_str:
        tags["TRACKTOTAL"] = [total_tracks_str]
        tags["TOTALTRACKS"] = [total_tracks_str]
    else:
        tags.pop("TRACKTOTAL", None)
        tags.pop("TOTALTRACKS", None)

    total_discs_str = str(track_info.total_discs) if track_info.total_discs else None
    if track_info.disc_number is not None:
        tags["DISCNUMBER"] = [
            f"{track_info.disc_number}/{total_discs_str}"
            if total_discs_str
            else str(track_info.disc_number)
        ]
    else:
        tags.pop("DISCNUMBER", None)

    if total_discs_str:
        tags["DISCTOTAL"] = [total_discs_str]
        tags["TOTALDISCS"] = [total_discs_str]
    else:
        tags.pop("DISCTOTAL", None)
        tags.pop("TOTALDISCS", None)

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
    if track_info.bpm is not None:
        tags["BPM"] = [f"{track_info.bpm:.1f}"]
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

    # ReplayGain
    if track_info.replaygain_track_gain is not None:
        tags["REPLAYGAIN_TRACK_GAIN"] = [f"{track_info.replaygain_track_gain:+.2f} dB"]
    else:
        tags.pop("REPLAYGAIN_TRACK_GAIN", None)

    if track_info.replaygain_track_peak is not None:
        tags["REPLAYGAIN_TRACK_PEAK"] = [f"{track_info.replaygain_track_peak:.6f}"]
    else:
        tags.pop("REPLAYGAIN_TRACK_PEAK", None)

    if track_info.replaygain_album_gain is not None:
        tags["REPLAYGAIN_ALBUM_GAIN"] = [f"{track_info.replaygain_album_gain:+.2f} dB"]
    else:
        tags.pop("REPLAYGAIN_ALBUM_GAIN", None)

    if track_info.replaygain_album_peak is not None:
        tags["REPLAYGAIN_ALBUM_PEAK"] = [f"{track_info.replaygain_album_peak:.6f}"]
    else:
        tags.pop("REPLAYGAIN_ALBUM_PEAK", None)


def _write_schema_tags(tags: dict[str, list[str]], track_info: TrackInfo) -> None:
    for field, tag_keys in _TAG_SCHEMA.items():
        tag_value = getattr(track_info, field, None)
        canonical_key = tag_keys[0]
        if tag_value:
            if field in _UUID_FIELDS and not is_valid_uuid(
                tag_value, allow_multivalue=True
            ):
                for k in tag_keys:
                    tags.pop(k, None)
                continue
            if field == "advisory":
                norm_adv = str(tag_value).strip().capitalize() if tag_value else None
                if norm_adv == "Explicit":
                    tags["ITUNESADVISORY"] = ["1"]
                    tags["ADVISORY"] = ["Explicit"]
                else:
                    tags.pop("ITUNESADVISORY", None)
                    tags.pop("ADVISORY", None)
                continue
            if field == "disc_subtitle":
                for k in tag_keys:
                    tags[k] = [str(tag_value)]
                continue
            if field == "lyrics":
                tags["LYRICS"] = [str(tag_value)]
                tags["UNSYNCEDLYRICS"] = [str(tag_value)]
                continue
            tags[canonical_key] = [str(tag_value)]
            for alias_key in tag_keys[1:]:
                tags.pop(alias_key, None)
        else:
            for k in tag_keys:
                tags.pop(k, None)
            if field == "lyrics":
                tags.pop("UNSYNCEDLYRICS", None)


def write_track_metadata(
    track_info: TrackInfo, cover_art_path: Path | None = None
) -> None:
    if not track_info.file_path.exists():
        raise FileNotFoundError(f"File not found: {track_info.file_path}")
    if not os.access(track_info.file_path, os.W_OK):
        raise PermissionError(
            f"Permission denied: File is read-only '{track_info.file_path}'"
        )

    try:
        with taglib.File(str(track_info.file_path)) as song:
            _write_artist_tags(song.tags, track_info)
            _write_numbering_and_dates(song.tags, track_info)
            _write_audio_properties_tags(song.tags, track_info)
            _write_schema_tags(song.tags, track_info)

            # Front cover
            if cover_art_path and cover_art_path.exists():
                mime = (
                    "image/jpeg"
                    if cover_art_path.suffix.lower() in [".jpg", ".jpeg"]
                    else "image/png"
                )
                if hasattr(song, "pictures"):
                    song.pictures = [
                        taglib.Picture(
                            data=cover_art_path.read_bytes(),
                            mime_type=mime,
                            description="Cover",
                            picture_type="Front Cover",
                        )
                    ]

            unsaved = song.save()
            if unsaved:
                LOG.debug(f"TagLib unsaved tags for {track_info.file_path}: {unsaved}")

            try:
                new_stat = track_info.file_path.stat()
                new_key = (
                    str(track_info.file_path.resolve()),
                    new_stat.st_mtime_ns,
                    new_stat.st_size,
                )
                with _METADATA_CACHE_LOCK:
                    if len(_METADATA_CACHE) >= _MAX_METADATA_CACHE_SIZE:
                        _METADATA_CACHE.clear()
                    _METADATA_CACHE[new_key] = dataclasses.replace(track_info)
            except OSError as e:
                LOG.debug(f"Failed to cache track metadata by inode: {e}")
    except FileNotFoundError:
        raise
    except (OSError, ValueError) as error:
        raise OSError(
            f"Failed to write metadata for {track_info.file_path}: {error}"
        ) from error


def clear_metadata_cache() -> int:
    """Clear the in-memory metadata cache and return the number of entries cleared."""
    with _METADATA_CACHE_LOCK:
        count = len(_METADATA_CACHE)
        _METADATA_CACHE.clear()
        return count


def get_metadata_cache_size() -> int:
    """Return the number of entries currently cached in memory."""
    with _METADATA_CACHE_LOCK:
        return len(_METADATA_CACHE)


def get_audio_duration(file_path: Path) -> float | None:
    try:
        import mutagen

        m_file = cast(Any, mutagen).File(file_path)
        if m_file and getattr(m_file, "info", None):
            return safe_float(m_file.info.length)
    except (OSError, MutagenError):
        pass
    return None
