import dataclasses
from pathlib import Path
from typing import Any

import taglib

from sonora.core.constants import SUPPORTED_EXTS
from sonora.core.logger import LOG
from sonora.core.models import TrackInfo
from sonora.core.utils import is_valid_uuid, normalize_date, normalize_genre

_METADATA_CACHE: dict[tuple[str, int, int], TrackInfo] = {}
_MAX_METADATA_CACHE_SIZE = 8192

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
    "label": ("LABEL", "PUBLISHER"),
    "catalog_number": ("CATALOGNUMBER",),
    "barcode": ("BARCODE",),
    "media": ("MEDIA",),
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
}


def _get_tag(tags: dict[str, list[str]], *keys: str) -> str | None:
    for key in keys:
        values = tags.get(key) or tags.get(key.upper()) or tags.get(key.lower())
        if values and len(values) > 0 and values[0] is not None:
            return str(values[0]).strip()
    return None


def _parse_float_tag(raw_value: str | None, tag_name: str) -> float | None:
    if not raw_value:
        return None
    try:
        return float(str(raw_value).replace(" dB", "").strip())
    except ValueError as error:
        LOG.debug(f"Failed to parse {tag_name} '{raw_value}': {error}")
        return None


def _parse_int_tag(
    raw_value: str | None, tag_name: str, default: int | None = None
) -> int | None:
    if not raw_value:
        return default
    try:
        return int(str(raw_value).split("/")[0].strip())
    except ValueError as error:
        LOG.debug(f"Failed to parse {tag_name} '{raw_value}': {error}")
        return default


def read_track_metadata(file_path: Path) -> TrackInfo:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported audio format: {file_path}")

    stat = file_path.stat()
    cache_key = (str(file_path.resolve()), stat.st_mtime_ns, stat.st_size)
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
            track_number = _parse_int_tag(
                _get_tag(tags, "TRACKNUMBER", "TRCK", "TRACK"), "track number"
            )
            disc_number = (
                _parse_int_tag(
                    _get_tag(tags, "DISCNUMBER", "TPOS", "DISC"),
                    "disc number",
                    default=1,
                )
                or 1
            )
            raw_total_tracks = _get_tag(tags, "TRACKTOTAL", "TOTALTRACKS")
            total_tracks = (
                int(raw_total_tracks)
                if raw_total_tracks and raw_total_tracks.isdigit()
                else None
            )
            raw_total_discs = _get_tag(tags, "DISCTOTAL", "TOTALDISCS")
            total_discs = (
                int(raw_total_discs)
                if raw_total_discs and raw_total_discs.isdigit()
                else None
            )

            # Numerical and audio stats
            bpm = _parse_float_tag(
                _get_tag(tags, "BPM", "TBPM", "WM/BEATSPERMINUTE"), "BPM"
            )
            raw_rating = _get_tag(tags, "RATING", "POPM")
            rating = float(raw_rating) if raw_rating else None

            raw_listeners = _get_tag(tags, "LISTENERS", "TXXX:LASTFM_LISTENERS")
            listeners = (
                int(raw_listeners)
                if raw_listeners and raw_listeners.isdigit()
                else None
            )
            raw_playcount = _get_tag(tags, "PLAYCOUNT", "TXXX:LASTFM_PLAYCOUNT")
            playcount = (
                int(raw_playcount)
                if raw_playcount and raw_playcount.isdigit()
                else None
            )

            raw_compilation = _get_tag(tags, "COMPILATION", "TCMP")
            compilation = (
                True
                if raw_compilation in ("1", "true", "True")
                else False
                if raw_compilation in ("0", "false", "False")
                else None
            )

            # ReplayGain
            replaygain_track_gain = _parse_float_tag(
                _get_tag(tags, "REPLAYGAIN_TRACK_GAIN", "TXXX:REPLAYGAIN_TRACK_GAIN"),
                "ReplayGain track gain",
            )
            replaygain_track_peak = _parse_float_tag(
                _get_tag(tags, "REPLAYGAIN_TRACK_PEAK", "TXXX:REPLAYGAIN_TRACK_PEAK"),
                "ReplayGain track peak",
            )
            replaygain_album_gain = _parse_float_tag(
                _get_tag(tags, "REPLAYGAIN_ALBUM_GAIN", "TXXX:REPLAYGAIN_ALBUM_GAIN"),
                "ReplayGain album gain",
            )
            replaygain_album_peak = _parse_float_tag(
                _get_tag(tags, "REPLAYGAIN_ALBUM_PEAK", "TXXX:REPLAYGAIN_ALBUM_PEAK"),
                "ReplayGain album peak",
            )

            art_width, art_height = None, None
            if hasattr(song, "pictures") and song.pictures:
                first_picture = song.pictures[0]
                art_width = getattr(first_picture, "width", None)
                art_height = getattr(first_picture, "height", None)

            mapped_fields: dict[str, Any] = {
                field: _get_tag(tags, *tag_keys)
                for field, tag_keys in _TAG_SCHEMA.items()
            }

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
                listeners=listeners,
                playcount=playcount,
                compilation=compilation,
                sample_rate=song.sampleRate,
                bitrate=song.bitrate,
                channels=song.channels,
                replaygain_track_gain=replaygain_track_gain,
                replaygain_track_peak=replaygain_track_peak,
                replaygain_album_gain=replaygain_album_gain,
                replaygain_album_peak=replaygain_album_peak,
                art_width=art_width,
                art_height=art_height,
                **mapped_fields,
            )
            if len(_METADATA_CACHE) >= _MAX_METADATA_CACHE_SIZE:
                _METADATA_CACHE.clear()
            _METADATA_CACHE[cache_key] = dataclasses.replace(track_info)
            return track_info
    except (RuntimeError, ValueError, FileNotFoundError):
        raise
    except (OSError, KeyError) as error:
        raise RuntimeError(
            f"Failed to read metadata for {file_path}: {error}"
        ) from error


def write_track_metadata(
    track_info: TrackInfo, cover_art_path: Path | None = None
) -> None:
    if not track_info.file_path.exists():
        raise FileNotFoundError(f"File not found: {track_info.file_path}")

    try:
        with taglib.File(str(track_info.file_path)) as song:
            # Core tags
            song.tags["ARTIST"] = [track_info.artist]
            song.tags["TITLE"] = [track_info.title]
            song.tags["ALBUM"] = [track_info.album]

            if track_info.album_artist:
                song.tags["ALBUMARTIST"] = [track_info.album_artist]

            # Track & Disc numbering
            total_tracks_str = (
                str(track_info.total_tracks) if track_info.total_tracks else None
            )
            if track_info.track_number is not None:
                song.tags["TRACKNUMBER"] = [
                    f"{track_info.track_number}/{total_tracks_str}"
                    if total_tracks_str
                    else str(track_info.track_number)
                ]
            if total_tracks_str:
                song.tags["TRACKTOTAL"] = [total_tracks_str]
                song.tags["TOTALTRACKS"] = [total_tracks_str]

            total_discs_str = (
                str(track_info.total_discs) if track_info.total_discs else None
            )
            if track_info.disc_number is not None:
                song.tags["DISCNUMBER"] = [
                    f"{track_info.disc_number}/{total_discs_str}"
                    if total_discs_str
                    else str(track_info.disc_number)
                ]
            if total_discs_str:
                song.tags["DISCTOTAL"] = [total_discs_str]
                song.tags["TOTALDISCS"] = [total_discs_str]

            if track_info.date:
                song.tags["DATE"] = [track_info.date]
            if track_info.original_date:
                song.tags["ORIGINALDATE"] = [track_info.original_date]
                song.tags["ORIGINALYEAR"] = [track_info.original_date[:4]]
            if track_info.genre:
                song.tags["GENRE"] = [track_info.genre]

            # Numeric & audio tags
            if track_info.bpm is not None:
                song.tags["BPM"] = [f"{track_info.bpm:.1f}"]
            if track_info.rating is not None:
                song.tags["RATING"] = [f"{track_info.rating:.1f}"]
            if track_info.listeners is not None:
                song.tags["LISTENERS"] = [str(track_info.listeners)]
            if track_info.playcount is not None:
                song.tags["PLAYCOUNT"] = [str(track_info.playcount)]
            if track_info.compilation is not None:
                song.tags["COMPILATION"] = ["1" if track_info.compilation else "0"]

            # ReplayGain
            if track_info.replaygain_track_gain is not None:
                song.tags["REPLAYGAIN_TRACK_GAIN"] = [
                    f"{track_info.replaygain_track_gain:+.2f} dB"
                ]
            if track_info.replaygain_track_peak is not None:
                song.tags["REPLAYGAIN_TRACK_PEAK"] = [
                    f"{track_info.replaygain_track_peak:.6f}"
                ]
            if track_info.replaygain_album_gain is not None:
                song.tags["REPLAYGAIN_ALBUM_GAIN"] = [
                    f"{track_info.replaygain_album_gain:+.2f} dB"
                ]
            if track_info.replaygain_album_peak is not None:
                song.tags["REPLAYGAIN_ALBUM_PEAK"] = [
                    f"{track_info.replaygain_album_peak:.6f}"
                ]

            # Declarative schema write
            for field, tag_keys in _TAG_SCHEMA.items():
                val = getattr(track_info, field, None)
                if val:
                    if field in _UUID_FIELDS and not is_valid_uuid(
                        val, allow_multivalue=True
                    ):
                        continue
                    song.tags[tag_keys[0]] = [str(val)]

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
                if len(_METADATA_CACHE) >= _MAX_METADATA_CACHE_SIZE:
                    _METADATA_CACHE.clear()
                _METADATA_CACHE[new_key] = dataclasses.replace(track_info)
            except OSError:
                pass
    except (RuntimeError, ValueError, FileNotFoundError):
        raise
    except (OSError, KeyError) as error:
        raise RuntimeError(
            f"Failed to write metadata for {track_info.file_path}: {error}"
        ) from error
