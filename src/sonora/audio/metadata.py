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
        raw_track = get_tag("tracknumber") or get_tag("TRACKNUMBER")
        track_number = None
        if raw_track:
            try:
                track_number = int(str(raw_track).split("/")[0])
            except ValueError as e:
                LOG.debug(f"Failed to parse track number '{raw_track}': {e}")
        raw_bpm = get_tag("bpm") or get_tag("BPM")
        bpm = None
        if raw_bpm:
            try:
                bpm = float(raw_bpm)
            except ValueError as e:
                LOG.debug(f"Failed to parse BPM '{raw_bpm}': {e}")
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
        ext = track_info.file_path.suffix.lower()
        if ext == ".flac":
            flac_audio = FLAC(str(track_info.file_path))
            flac_audio["ARTIST"] = [track_info.artist]
            flac_audio["TITLE"] = [track_info.title]
            flac_audio["ALBUM"] = [track_info.album]

            if track_info.album_artist:
                flac_audio["ALBUMARTIST"] = [track_info.album_artist]
            if track_info.track_number is not None:
                flac_audio["TRACKNUMBER"] = [str(track_info.track_number)]
            if track_info.date:
                flac_audio["DATE"] = [track_info.date]
            if track_info.genre:
                flac_audio["GENRE"] = [track_info.genre]
            if track_info.isrc:
                flac_audio["ISRC"] = [track_info.isrc]
            if track_info.bpm is not None:
                flac_audio["BPM"] = [f"{track_info.bpm:.1f}"]
            if track_info.musicbrainz_trackid:
                flac_audio["MUSICBRAINZ_TRACKID"] = [track_info.musicbrainz_trackid]
            if track_info.musicbrainz_albumid:
                flac_audio["MUSICBRAINZ_ALBUMID"] = [track_info.musicbrainz_albumid]
            if track_info.replaygain_track_gain is not None:
                flac_audio["REPLAYGAIN_TRACK_GAIN"] = [f"{track_info.replaygain_track_gain:+.2f} dB"]
            if track_info.replaygain_track_peak is not None:
                flac_audio["REPLAYGAIN_TRACK_PEAK"] = [f"{track_info.replaygain_track_peak:.6f}"]

            if cover_art_path and cover_art_path.exists():
                with open(cover_art_path, "rb") as f:
                    image_data = f.read()
                picture = Picture()
                picture.data = image_data
                picture.type = 3  # Front Cover
                picture.mime = "image/jpeg" if cover_art_path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
                picture.desc = "Cover"
                flac_audio.clear_pictures()
                flac_audio.add_picture(picture)

            flac_audio.save()
            
        elif ext == ".mp3":
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
            
        elif ext in (".m4a", ".mp4", ".alac"):
            from mutagen.mp4 import MP4, MP4Cover
            mp4_audio = MP4(str(track_info.file_path))
            mp4_audio["\xa9ART"] = [track_info.artist]
            mp4_audio["\xa9nam"] = [track_info.title]
            mp4_audio["\xa9alb"] = [track_info.album]
            if track_info.album_artist: mp4_audio["aART"] = [track_info.album_artist]
            if track_info.track_number is not None: mp4_audio["trkn"] = [(track_info.track_number, 0)]
            if track_info.date: mp4_audio["\xa9day"] = [track_info.date]
            if track_info.genre: mp4_audio["\xa9gen"] = [track_info.genre]
            if track_info.bpm is not None: mp4_audio["tmpo"] = [round(track_info.bpm)]
            
            if cover_art_path and cover_art_path.exists():
                with open(cover_art_path, "rb") as f:
                    image_data = f.read()
                fmt = MP4Cover.FORMAT_JPEG if cover_art_path.suffix.lower() in [".jpg", ".jpeg"] else MP4Cover.FORMAT_PNG
                mp4_audio["covr"] = [MP4Cover(image_data, imageformat=fmt)]
            mp4_audio.save()
            
        elif ext in (".ogg", ".opus"):
            import base64

            from mutagen.oggvorbis import OggVorbis
            ogg_audio = OggVorbis(str(track_info.file_path))
            ogg_audio["ARTIST"] = [track_info.artist]
            ogg_audio["TITLE"] = [track_info.title]
            ogg_audio["ALBUM"] = [track_info.album]
            if track_info.album_artist: ogg_audio["ALBUMARTIST"] = [track_info.album_artist]
            if track_info.track_number is not None: ogg_audio["TRACKNUMBER"] = [str(track_info.track_number)]
            if track_info.date: ogg_audio["DATE"] = [track_info.date]
            if track_info.genre: ogg_audio["GENRE"] = [track_info.genre]
            if track_info.bpm is not None: ogg_audio["BPM"] = [f"{track_info.bpm:.1f}"]
            
            if cover_art_path and cover_art_path.exists():
                with open(cover_art_path, "rb") as f:
                    image_data = f.read()
                picture = Picture()
                picture.data = image_data
                picture.type = 3
                picture.mime = "image/jpeg" if cover_art_path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
                picture.desc = "Cover"
                ogg_audio["metadata_block_picture"] = [base64.b64encode(picture.write()).decode("ascii")]
            ogg_audio.save()
            
        else:
            LOG.warning(f"Using generic fallback tagger for format {ext}. Some metadata (like Cover Art/ReplayGain) might not be saved.")
            generic_audio = File(str(track_info.file_path), easy=True)
            if generic_audio is not None:
                try:
                    generic_audio["artist"] = [track_info.artist]
                    generic_audio["title"] = [track_info.title]
                    generic_audio["album"] = [track_info.album]
                    if track_info.album_artist: generic_audio["albumartist"] = [track_info.album_artist]
                    if track_info.track_number is not None: generic_audio["tracknumber"] = [str(track_info.track_number)]
                    if track_info.date: generic_audio["date"] = [track_info.date]
                    if track_info.genre: generic_audio["genre"] = [track_info.genre]
                    if track_info.bpm is not None: generic_audio["bpm"] = [str(round(track_info.bpm))]
                    generic_audio.save()
                except Exception as e:  # noqa: BLE001
                    LOG.error(f"Generic tagger failed to write to {ext}: {e}")
            else:
                LOG.warning(f"Writing metadata to {ext} files is not supported by mutagen.")
    except Exception as e:
        if isinstance(e, MetadataError):
            raise
        raise MetadataError(f"Failed to write metadata for {track_info.file_path}: {e}") from e
