"""
Metadata reader and writer using Mutagen for FLAC and audio formats.
"""

from pathlib import Path
from typing import Any

import mutagen
from mutagen.easyid3 import EasyID3
from mutagen.flac import FLAC, Picture

from sonora.core.exceptions import MetadataError
from sonora.core.models import TrackInfo


def read_track_metadata(file_path: Path) -> TrackInfo:
    """
    Read metadata tags from an audio file and return a TrackInfo dataclass.
    """
    if not file_path.exists():
        raise MetadataError(f"File not found: {file_path}")

    try:
        audio: Any = mutagen.File(str(file_path))
        if audio is None:
            raise MetadataError(f"Unsupported audio format: {file_path}")

        # Extract standard tags by handling lists returned by Mutagen
        def get_tag(key: str) -> str | None:
            val = audio.get(key)
            if not val:
                return None
            if isinstance(val, list):
                return str(val[0])
            return str(val)

        artist = get_tag("artist") or get_tag("ARTIST") or "Unknown Artist"
        title = get_tag("title") or get_tag("TITLE") or "Unknown Title"
        album = get_tag("album") or get_tag("ALBUM") or "Unknown Album"
        album_artist = get_tag("albumartist") or get_tag("ALBUMARTIST")
        date = get_tag("date") or get_tag("DATE") or get_tag("year")
        genre = get_tag("genre") or get_tag("GENRE")
        isrc = get_tag("isrc") or get_tag("ISRC")

        # Parse track number
        raw_track = get_tag("tracknumber") or get_tag("TRACKNUMBER")
        track_number = None
        if raw_track:
            try:
                track_number = int(str(raw_track).split("/")[0])
            except ValueError:
                pass

        # Parse BPM
        raw_bpm = get_tag("bpm") or get_tag("BPM")
        bpm = None
        if raw_bpm:
            try:
                bpm = float(raw_bpm)
            except ValueError:
                pass

        # Audio stream properties safely accessed for static type checkers
        info = getattr(audio, "info", None)
        sample_rate = getattr(info, "sample_rate", None) if info else None
        bitrate = getattr(info, "bitrate", None) if info else None
        channels = getattr(info, "channels", None) if info else None

        return TrackInfo(
            file_path=file_path,
            artist=artist,
            title=title,
            album=album,
            album_artist=album_artist,
            track_number=track_number,
            date=date,
            genre=genre,
            isrc=isrc,
            bpm=bpm,
            sample_rate=sample_rate,
            bitrate=bitrate,
            channels=channels,
            musicbrainz_trackid=get_tag("musicbrainz_trackid"),
            musicbrainz_albumid=get_tag("musicbrainz_albumid"),
        )
    except Exception as e:
        if isinstance(e, MetadataError):
            raise
        raise MetadataError(f"Failed to read metadata for {file_path}: {e}") from e


def write_track_metadata(track_info: TrackInfo) -> None:
    """
    Write metadata tags from TrackInfo back to the audio file.
    """
    if not track_info.file_path.exists():
        raise MetadataError(f"File not found: {track_info.file_path}")

    try:
        if track_info.file_path.suffix.lower() == ".flac":
            audio: Any = FLAC(str(track_info.file_path))
            audio["ARTIST"] = [track_info.artist]
            audio["TITLE"] = [track_info.title]
            audio["ALBUM"] = [track_info.album]

            if track_info.album_artist:
                audio["ALBUMARTIST"] = [track_info.album_artist]
            if track_info.track_number:
                audio["TRACKNUMBER"] = [str(track_info.track_number)]
            if track_info.date:
                audio["DATE"] = [track_info.date]
            if track_info.genre:
                audio["GENRE"] = [track_info.genre]
            if track_info.isrc:
                audio["ISRC"] = [track_info.isrc]
            if track_info.bpm:
                audio["BPM"] = [f"{track_info.bpm:.1f}"]
            if track_info.musicbrainz_trackid:
                audio["MUSICBRAINZ_TRACKID"] = [track_info.musicbrainz_trackid]
            if track_info.musicbrainz_albumid:
                audio["MUSICBRAINZ_ALBUMID"] = [track_info.musicbrainz_albumid]
            if track_info.replaygain_track_gain is not None:
                audio["REPLAYGAIN_TRACK_GAIN"] = [f"{track_info.replaygain_track_gain:+.2f} dB"]
            if track_info.replaygain_track_peak is not None:
                audio["REPLAYGAIN_TRACK_PEAK"] = [f"{track_info.replaygain_track_peak:.6f}"]
            audio.save()
        else:
            try:
                easy_audio: Any = EasyID3(str(track_info.file_path))
            except Exception:  # noqa: BLE001
                easy_audio = EasyID3()
            easy_audio["artist"] = track_info.artist
            easy_audio["title"] = track_info.title
            easy_audio["album"] = track_info.album
            if track_info.genre:
                easy_audio["genre"] = track_info.genre
            if track_info.date:
                easy_audio["date"] = track_info.date
            if track_info.track_number:
                easy_audio["tracknumber"] = str(track_info.track_number)
            easy_audio.save(str(track_info.file_path))
    except Exception as e:
        if isinstance(e, MetadataError):
            raise
        raise MetadataError(f"Failed to write metadata for {track_info.file_path}: {e}") from e


def embed_cover_art(file_path: Path, image_path: Path) -> None:
    """
    Embed an image file as front cover art into a FLAC audio file.
    """
    if not file_path.exists():
        raise MetadataError(f"Audio file not found: {file_path}")
    if not image_path.exists():
        raise MetadataError(f"Image file not found: {image_path}")

    try:
        audio: Any = FLAC(str(file_path))
        with open(image_path, "rb") as f:
            image_data = f.read()

        picture: Any = Picture()
        picture.data = image_data
        picture.type = 3  # Front Cover
        picture.mime = "image/jpeg" if image_path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
        picture.desc = "Cover"

        audio.clear_pictures()
        audio.add_picture(picture)
        audio.save()
    except Exception as e:
        raise MetadataError(f"Failed to embed cover art into {file_path}: {e}") from e
