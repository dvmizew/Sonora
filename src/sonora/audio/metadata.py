from pathlib import Path

from mutagen._file import File
from mutagen.flac import FLAC, Picture
from mutagen.id3 import ID3
from mutagen.id3._frames import (
    APIC,
    TALB,
    TBPM,
    TCON,
    TDRC,
    TIT2,
    TPE1,
    TPE2,
    TRCK,
    TXXX,
)

from sonora.core.exceptions import MetadataError
from sonora.core.logger import LOG
from sonora.core.models import TrackInfo
from sonora.core.utils import normalize_date, normalize_genre


def read_track_metadata(file_path: Path) -> TrackInfo:
    """
    Read metadata tags from an audio file and return a TrackInfo dataclass.
    """
    if not file_path.exists():
        raise MetadataError(f"File not found: {file_path}")

    try:
        audio = File(str(file_path))
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
        date = normalize_date(get_tag("date") or get_tag("DATE") or get_tag("year"))
        genre = normalize_genre(get_tag("genre") or get_tag("GENRE"))
        isrc = get_tag("isrc") or get_tag("ISRC")

        # Parse track number
        raw_track = get_tag("tracknumber") or get_tag("TRACKNUMBER")
        track_number = None
        if raw_track:
            try:
                track_number = int(str(raw_track).split("/")[0])
            except ValueError as e:
                LOG.debug(f"Failed to parse track number '{raw_track}': {e}")

        # Parse BPM
        raw_bpm = get_tag("bpm") or get_tag("BPM")
        bpm = None
        if raw_bpm:
            try:
                bpm = float(raw_bpm)
            except ValueError as e:
                LOG.debug(f"Failed to parse BPM '{raw_bpm}': {e}")

        # Audio stream properties
        sample_rate = getattr(audio.info, "sample_rate", None)
        bitrate = getattr(audio.info, "bitrate", None)
        channels = getattr(audio.info, "channels", None)

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


def write_track_metadata(track_info: TrackInfo, cover_art_path: Path | None = None) -> None:
    """
    Write metadata tags from TrackInfo back to the audio file.
    Also embeds cover art in the same I/O operation if cover_art_path is provided, avoiding double file rewrites.
    """
    if not track_info.file_path.exists():
        raise MetadataError(f"File not found: {track_info.file_path}")

    try:
        if track_info.file_path.suffix.lower() == ".flac":
            audio = FLAC(str(track_info.file_path))
            audio["ARTIST"] = [track_info.artist]
            audio["TITLE"] = [track_info.title]
            audio["ALBUM"] = [track_info.album]

            if track_info.album_artist:
                audio["ALBUMARTIST"] = [track_info.album_artist]
            if track_info.track_number is not None:
                audio["TRACKNUMBER"] = [str(track_info.track_number)]
            if track_info.date:
                audio["DATE"] = [track_info.date]
            if track_info.genre:
                audio["GENRE"] = [track_info.genre]
            if track_info.isrc:
                audio["ISRC"] = [track_info.isrc]
            if track_info.bpm is not None:
                audio["BPM"] = [f"{track_info.bpm:.1f}"]
            if track_info.musicbrainz_trackid:
                audio["MUSICBRAINZ_TRACKID"] = [track_info.musicbrainz_trackid]
            if track_info.musicbrainz_albumid:
                audio["MUSICBRAINZ_ALBUMID"] = [track_info.musicbrainz_albumid]
            if track_info.replaygain_track_gain is not None:
                audio["REPLAYGAIN_TRACK_GAIN"] = [f"{track_info.replaygain_track_gain:+.2f} dB"]
            if track_info.replaygain_track_peak is not None:
                audio["REPLAYGAIN_TRACK_PEAK"] = [f"{track_info.replaygain_track_peak:.6f}"]

            if cover_art_path and cover_art_path.exists():
                with open(cover_art_path, "rb") as f:
                    image_data = f.read()
                picture = Picture()
                picture.data = image_data
                picture.type = 3  # Front Cover
                picture.mime = "image/jpeg" if cover_art_path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
                picture.desc = "Cover"
                audio.clear_pictures()
                audio.add_picture(picture)

            audio.save()
        else:
            try:
                id3_audio = ID3(str(track_info.file_path))
            except Exception as e:  # noqa: BLE001
                LOG.debug(f"ID3 parsing failed, creating empty ID3: {e}")
                id3_audio = ID3()
                
            id3_audio.add(TPE1(encoding=3, text=[track_info.artist]))
            id3_audio.add(TIT2(encoding=3, text=[track_info.title]))
            id3_audio.add(TALB(encoding=3, text=[track_info.album]))
            
            if track_info.album_artist:
                id3_audio.add(TPE2(encoding=3, text=[track_info.album_artist]))
            if track_info.track_number is not None:
                id3_audio.add(TRCK(encoding=3, text=[str(track_info.track_number)]))
            if track_info.date:
                id3_audio.add(TDRC(encoding=3, text=[track_info.date]))
            if track_info.genre:
                id3_audio.add(TCON(encoding=3, text=[track_info.genre]))
            if track_info.bpm is not None:
                id3_audio.add(TBPM(encoding=3, text=[str(round(track_info.bpm))]))
                
            # Custom TXXX frames for MusicBrainz and ReplayGain
            if track_info.musicbrainz_trackid:
                id3_audio.add(TXXX(encoding=3, desc="MusicBrainz Track Id", text=[track_info.musicbrainz_trackid]))
            if track_info.musicbrainz_albumid:
                id3_audio.add(TXXX(encoding=3, desc="MusicBrainz Album Id", text=[track_info.musicbrainz_albumid]))
            if track_info.replaygain_track_gain is not None:
                id3_audio.add(TXXX(encoding=3, desc="REPLAYGAIN_TRACK_GAIN", text=[f"{track_info.replaygain_track_gain:+.2f} dB"]))
            if track_info.replaygain_track_peak is not None:
                id3_audio.add(TXXX(encoding=3, desc="REPLAYGAIN_TRACK_PEAK", text=[f"{track_info.replaygain_track_peak:.6f}"]))
                
            if cover_art_path and cover_art_path.exists():
                with open(cover_art_path, "rb") as f:
                    image_data = f.read()
                mime = "image/jpeg" if cover_art_path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
                id3_audio.add(APIC(encoding=3, mime=mime, type=3, desc="Cover", data=image_data))
                
            id3_audio.save(str(track_info.file_path), v2_version=3)
    except Exception as e:
        if isinstance(e, MetadataError):
            raise
        raise MetadataError(f"Failed to write metadata for {track_info.file_path}: {e}") from e
