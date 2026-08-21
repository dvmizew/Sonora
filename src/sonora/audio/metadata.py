from pathlib import Path

from mutagen._file import File
from mutagen.aiff import AIFF
from mutagen.asf import ASF
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
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4, MP4Cover
from mutagen.oggvorbis import OggVorbis
from mutagen.wave import WAVE

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
        def get_first_of(*keys: str) -> str | None:
            for k in keys:
                try:
                    val = audio.get(k)
                except (ValueError, KeyError):
                    continue
                if val:
                    if isinstance(val, list):
                        if len(val) > 0 and val[0] is not None:
                            # MP4 stores track numbers as a tuple: (track, total_tracks)
                            if isinstance(val[0], tuple) and len(val[0]) > 0:
                                return str(val[0][0])
                            # Generic lists (Vorbis, Atom text arrays)
                            return str(val[0])
                    else:
                        # ID3 frames (e.g. TPE1, TIT2) have a `.text` attribute
                        if hasattr(val, "text") and isinstance(val.text, list) and len(val.text) > 0:
                            return str(val.text[0])
                        return str(val)
            return None

        artist = get_first_of("TPE1", "ARTIST", "\xa9ART", "artist", "Author", "Artist") or "Unknown Artist"
        title = get_first_of("TIT2", "TITLE", "\xa9nam", "title", "Title") or "Unknown Title"
        album = get_first_of("TALB", "ALBUM", "\xa9alb", "album", "WM/AlbumTitle", "Album") or "Unknown Album"
        album_artist = get_first_of("TPE2", "ALBUMARTIST", "aART", "albumartist", "WM/AlbumArtist", "Album Artist")
        date_str = get_first_of("TDRC", "DATE", "\xa9day", "date", "year", "WM/Year", "Year")
        date = normalize_date(date_str)
        genre = normalize_genre(get_first_of("TCON", "GENRE", "\xa9gen", "genre", "WM/Genre", "Genre"))
        isrc = get_first_of("TSRC", "ISRC", "isrc", "WM/ISRC")
        
        raw_track = get_first_of("TRCK", "TRACKNUMBER", "trkn", "tracknumber", "WM/TrackNumber", "Track")
        track_number = None
        if raw_track:
            try:
                # Handle ID3 formats like "1/12"
                track_number = int(str(raw_track).split("/")[0])
            except ValueError as e:
                LOG.debug(f"Failed to parse track number '{raw_track}': {e}")
                
        raw_bpm = get_first_of("TBPM", "BPM", "tmpo", "bpm", "WM/BeatsPerMinute")
        bpm = None
        if raw_bpm:
            try:
                bpm = float(raw_bpm)
            except ValueError as e:
                LOG.debug(f"Failed to parse BPM '{raw_bpm}': {e}")

        raw_disc = get_first_of("TPOS", "DISCNUMBER", "disk", "discnumber", "WM/PartOfSet", "Disc")
        disc_number = 1
        if raw_disc:
            try:
                disc_number = int(str(raw_disc).split("/")[0])
            except ValueError:
                disc_number = 1

        raw_rg_gain = get_first_of("TXXX:REPLAYGAIN_TRACK_GAIN", "REPLAYGAIN_TRACK_GAIN", "replaygain_track_gain")
        rg_gain = None
        if raw_rg_gain:
            try:
                rg_gain = float(str(raw_rg_gain).replace(" dB", "").strip())
            except ValueError:
                pass

        raw_rg_peak = get_first_of("TXXX:REPLAYGAIN_TRACK_PEAK", "REPLAYGAIN_TRACK_PEAK", "replaygain_track_peak")
        rg_peak = None
        if raw_rg_peak:
            try:
                rg_peak = float(str(raw_rg_peak).strip())
            except ValueError:
                pass

        sample_rate = getattr(audio.info, "sample_rate", None)
        bitrate = getattr(audio.info, "bitrate", None)
        channels = getattr(audio.info, "channels", None)

        mb_track = get_first_of("TXXX:MusicBrainz Track Id", "MUSICBRAINZ_TRACKID", "musicbrainz_trackid", "MusicBrainz/Track Id")
        mb_album = get_first_of("TXXX:MusicBrainz Album Id", "MUSICBRAINZ_ALBUMID", "musicbrainz_albumid", "MusicBrainz/Album Id")
        mb_rg = get_first_of("TXXX:MusicBrainz Release Group Id", "MUSICBRAINZ_RELEASEGROUPID", "musicbrainz_releasegroupid")
        mb_art = get_first_of("TXXX:MusicBrainz Artist Id", "MUSICBRAINZ_ARTISTID", "musicbrainz_artistid")
        mb_work = get_first_of("TXXX:MusicBrainz Work Id", "MUSICBRAINZ_WORKID", "musicbrainz_workid")
        acoustid_id = get_first_of("TXXX:Acoustid Id", "ACOUSTID_ID", "acoustid_id")
        discogs_id = get_first_of("TXXX:Discogs Release Id", "DISCOGS_RELEASE_ID", "discogs_release_id")
        itunes_trackid = get_first_of("ITUNESTRACKID", "ITUNES_TRACK_ID")
        itunes_collectionid = get_first_of("ITUNESCOLLECTIONID", "ITUNES_COLLECTION_ID")
        itunes_artistid = get_first_of("ITUNESARTISTID", "ITUNES_ARTIST_ID")
        album_artist_sort = get_first_of("ALBUMARTISTSORT", "albumartistsort")
        artist_sort = get_first_of("ARTISTSORT", "artistsort")
        
        raw_tot_tracks = get_first_of("TRACKTOTAL", "TOTALTRACKS", "totaltracks")
        total_tracks = int(raw_tot_tracks) if raw_tot_tracks and raw_tot_tracks.isdigit() else None
        
        raw_tot_discs = get_first_of("DISCTOTAL", "TOTALDISCS", "totaldiscs")
        total_discs = int(raw_tot_discs) if raw_tot_discs and raw_tot_discs.isdigit() else None

        release_type = get_first_of("RELEASETYPE", "releasetype")
        release_status = get_first_of("RELEASESTATUS", "releasestatus")
        release_country = get_first_of("RELEASECOUNTRY", "releasecountry", "COUNTRY")
        label = get_first_of("LABEL", "label", "PUBLISHER")
        catalog_number = get_first_of("CATALOGNUMBER", "catalognumber")
        barcode = get_first_of("BARCODE", "barcode")
        media = get_first_of("MEDIA", "media")
        comment = get_first_of("COMM", "COMMENT", "comment")
        advisory = get_first_of("ITUNESADVISORY", "ADVISORY")
        orig_date = get_first_of("ORIGINALDATE", "originaldate", "ORIGINALYEAR")
        cuesheet = get_first_of("CUESHEET", "cuesheet")

        return TrackInfo(
            file_path=file_path,
            artist=artist,
            title=title,
            album=album,
            album_artist=album_artist,
            track_number=track_number,
            disc_number=disc_number,
            date=date,
            genre=genre,
            isrc=isrc,
            bpm=bpm,
            replaygain_track_gain=rg_gain,
            replaygain_track_peak=rg_peak,
            sample_rate=sample_rate,
            bitrate=bitrate,
            channels=channels,
            musicbrainz_trackid=mb_track,
            musicbrainz_albumid=mb_album,
            musicbrainz_releasegroupid=mb_rg,
            musicbrainz_artistid=mb_art,
            musicbrainz_workid=mb_work,
            acoustid_id=acoustid_id,
            discogs_release_id=discogs_id,
            itunes_trackid=itunes_trackid,
            itunes_collectionid=itunes_collectionid,
            itunes_artistid=itunes_artistid,
            album_artist_sort=album_artist_sort,
            artist_sort=artist_sort,
            total_tracks=total_tracks,
            total_discs=total_discs,
            release_type=release_type,
            release_status=release_status,
            release_country=release_country,
            label=label,
            catalog_number=catalog_number,
            barcode=barcode,
            media=media,
            comment=comment,
            advisory=advisory,
            original_date=orig_date,
            cuesheet=cuesheet,
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
        import typing
        audio_container: typing.Any = None
        
        ext = track_info.file_path.suffix.lower()
        if ext in (".flac", ".ogg", ".opus"):
            if ext == ".flac":
                audio_container = FLAC(str(track_info.file_path))
            else:
                audio_container = OggVorbis(str(track_info.file_path))

            audio_container["ARTIST"] = [track_info.artist]
            audio_container["TITLE"] = [track_info.title]
            audio_container["ALBUM"] = [track_info.album]
            if track_info.album_artist: audio_container["ALBUMARTIST"] = [track_info.album_artist]
            if track_info.artist_sort: audio_container["ARTISTSORT"] = [track_info.artist_sort]
            if track_info.album_artist_sort: audio_container["ALBUMARTISTSORT"] = [track_info.album_artist_sort]
            
            tot_tr = str(track_info.total_tracks) if track_info.total_tracks else None
            if track_info.track_number is not None:
                tr_str = f"{track_info.track_number}/{tot_tr}" if tot_tr else str(track_info.track_number)
                audio_container["TRACKNUMBER"] = [tr_str]
            if tot_tr:
                audio_container["TRACKTOTAL"] = [tot_tr]
                audio_container["TOTALTRACKS"] = [tot_tr]

            tot_ds = str(track_info.total_discs) if track_info.total_discs else None
            if track_info.disc_number is not None:
                ds_str = f"{track_info.disc_number}/{tot_ds}" if tot_ds else str(track_info.disc_number)
                audio_container["DISCNUMBER"] = [ds_str]
            if tot_ds:
                audio_container["DISCTOTAL"] = [tot_ds]
                audio_container["TOTALDISCS"] = [tot_ds]

            if track_info.date: audio_container["DATE"] = [track_info.date]
            if track_info.genre: audio_container["GENRE"] = [track_info.genre]
            if track_info.bpm is not None: audio_container["BPM"] = [f"{track_info.bpm:.1f}"]
            if track_info.musicbrainz_trackid: audio_container["MUSICBRAINZ_TRACKID"] = [track_info.musicbrainz_trackid]
            if track_info.musicbrainz_albumid: audio_container["MUSICBRAINZ_ALBUMID"] = [track_info.musicbrainz_albumid]
            if track_info.musicbrainz_releasegroupid: audio_container["MUSICBRAINZ_RELEASEGROUPID"] = [track_info.musicbrainz_releasegroupid]
            if track_info.musicbrainz_artistid: audio_container["MUSICBRAINZ_ARTISTID"] = [track_info.musicbrainz_artistid]
            if track_info.musicbrainz_workid: audio_container["MUSICBRAINZ_WORKID"] = [track_info.musicbrainz_workid]
            if track_info.acoustid_id: audio_container["ACOUSTID_ID"] = [track_info.acoustid_id]
            if track_info.discogs_release_id: audio_container["DISCOGS_RELEASE_ID"] = [track_info.discogs_release_id]
            if track_info.itunes_trackid: audio_container["ITUNESTRACKID"] = [track_info.itunes_trackid]
            if track_info.itunes_collectionid: audio_container["ITUNESCOLLECTIONID"] = [track_info.itunes_collectionid]
            if track_info.itunes_artistid: audio_container["ITUNESARTISTID"] = [track_info.itunes_artistid]
            if track_info.release_type: audio_container["RELEASETYPE"] = [track_info.release_type]
            if track_info.release_status: audio_container["RELEASESTATUS"] = [track_info.release_status]
            if track_info.release_country:
                audio_container["RELEASECOUNTRY"] = [track_info.release_country]
                audio_container["COUNTRY"] = [track_info.release_country]
            if track_info.label:
                audio_container["LABEL"] = [track_info.label]
                audio_container["PUBLISHER"] = [track_info.label]
            if track_info.catalog_number: audio_container["CATALOGNUMBER"] = [track_info.catalog_number]
            if track_info.barcode: audio_container["BARCODE"] = [track_info.barcode]
            if track_info.media: audio_container["MEDIA"] = [track_info.media]
            if track_info.comment: audio_container["COMMENT"] = [track_info.comment]
            if track_info.advisory: audio_container["ITUNESADVISORY"] = [track_info.advisory]
            if track_info.original_date:
                audio_container["ORIGINALDATE"] = [track_info.original_date]
                audio_container["ORIGINALYEAR"] = [track_info.original_date[:4]]
            if track_info.cuesheet: audio_container["CUESHEET"] = [track_info.cuesheet]

            if ext == ".flac":
                if track_info.isrc: audio_container["ISRC"] = [track_info.isrc]
                if track_info.replaygain_track_gain is not None: audio_container["REPLAYGAIN_TRACK_GAIN"] = [f"{track_info.replaygain_track_gain:+.2f} dB"]
                if track_info.replaygain_track_peak is not None: audio_container["REPLAYGAIN_TRACK_PEAK"] = [f"{track_info.replaygain_track_peak:.6f}"]

            if cover_art_path and cover_art_path.exists():
                with open(cover_art_path, "rb") as f:
                    image_data = f.read()
                picture = Picture()
                picture.data = image_data
                picture.type = 3  # Front Cover
                picture.mime = "image/jpeg" if cover_art_path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
                picture.desc = "Cover"
                
                if isinstance(audio_container, FLAC):
                    audio_container.clear_pictures()
                    audio_container.add_picture(picture)
                else:
                    import base64
                    audio_container["metadata_block_picture"] = [base64.b64encode(picture.write()).decode("ascii")]

            audio_container.save()
            
        elif ext in (".mp3", ".wav", ".aiff"):
            if ext == ".mp3":
                try:
                    audio_container = MP3(str(track_info.file_path))
                    if audio_container.tags is None:
                        audio_container.add_tags()
                    id3_audio = audio_container.tags
                except (MetadataError, OSError, ValueError, KeyError) as e:
                    LOG.debug(f"MP3 ID3 parsing failed: {e}")
                    id3_audio = ID3()
            elif ext == ".wav":
                audio_container = WAVE(str(track_info.file_path))
                if audio_container.tags is None:
                    audio_container.add_tags()
                id3_audio = audio_container.tags
            else:
                audio_container = AIFF(str(track_info.file_path))
                if audio_container.tags is None:
                    audio_container.add_tags()
                id3_audio = audio_container.tags

            if id3_audio is None:
                raise MetadataError(f"Failed to initialize ID3 tags for {track_info.file_path}")
                
            id3_audio.add(TPE1(encoding=3, text=[track_info.artist]))
            id3_audio.add(TIT2(encoding=3, text=[track_info.title]))
            id3_audio.add(TALB(encoding=3, text=[track_info.album]))
            
            if track_info.album_artist: id3_audio.add(TPE2(encoding=3, text=[track_info.album_artist]))
            if track_info.track_number is not None: id3_audio.add(TRCK(encoding=3, text=[str(track_info.track_number)]))
            if track_info.date: id3_audio.add(TDRC(encoding=3, text=[track_info.date]))
            if track_info.genre: id3_audio.add(TCON(encoding=3, text=[track_info.genre]))
            if track_info.bpm is not None: id3_audio.add(TBPM(encoding=3, text=[str(round(track_info.bpm))]))
                
            if track_info.musicbrainz_trackid: id3_audio.add(TXXX(encoding=3, desc="MusicBrainz Track Id", text=[track_info.musicbrainz_trackid]))
            if track_info.musicbrainz_albumid: id3_audio.add(TXXX(encoding=3, desc="MusicBrainz Album Id", text=[track_info.musicbrainz_albumid]))
            if track_info.replaygain_track_gain is not None: id3_audio.add(TXXX(encoding=3, desc="REPLAYGAIN_TRACK_GAIN", text=[f"{track_info.replaygain_track_gain:+.2f} dB"]))
            if track_info.replaygain_track_peak is not None: id3_audio.add(TXXX(encoding=3, desc="REPLAYGAIN_TRACK_PEAK", text=[f"{track_info.replaygain_track_peak:.6f}"]))
                
            if cover_art_path and cover_art_path.exists():
                with open(cover_art_path, "rb") as f:
                    image_data = f.read()
                mime = "image/jpeg" if cover_art_path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
                id3_audio.add(APIC(encoding=3, mime=mime, type=3, desc="Cover", data=image_data))
                
            if audio_container is not None:
                audio_container.save(v2_version=3)
            else:
                id3_audio.save(str(track_info.file_path), v2_version=3)
            
        elif ext in (".m4a", ".mp4", ".alac"):
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
            

        elif ext == ".wma":
            asf_audio = ASF(str(track_info.file_path))
            asf_audio["Author"] = [track_info.artist]
            asf_audio["Title"] = [track_info.title]
            asf_audio["WM/AlbumTitle"] = [track_info.album]
            if track_info.album_artist: asf_audio["WM/AlbumArtist"] = [track_info.album_artist]
            if track_info.track_number is not None: asf_audio["WM/TrackNumber"] = [str(track_info.track_number)]
            if track_info.date: asf_audio["WM/Year"] = [track_info.date]
            if track_info.genre: asf_audio["WM/Genre"] = [track_info.genre]
            if track_info.bpm is not None: asf_audio["WM/BeatsPerMinute"] = [str(round(track_info.bpm))]
            if track_info.musicbrainz_trackid: asf_audio["MusicBrainz/Track Id"] = [track_info.musicbrainz_trackid]
            if track_info.musicbrainz_albumid: asf_audio["MusicBrainz/Album Id"] = [track_info.musicbrainz_albumid]
            asf_audio.save()

        elif ext in (".ape", ".wv", ".mpc"):
            ape_audio = File(str(track_info.file_path))
            if ape_audio is not None:
                if ape_audio.tags is None:
                    ape_audio.add_tags()
                ape_audio["Artist"] = [track_info.artist]
                ape_audio["Title"] = [track_info.title]
                ape_audio["Album"] = [track_info.album]
                if track_info.album_artist: ape_audio["Album Artist"] = [track_info.album_artist]
                if track_info.track_number is not None: ape_audio["Track"] = [str(track_info.track_number)]
                if track_info.date: ape_audio["Year"] = [track_info.date]
                if track_info.genre: ape_audio["Genre"] = [track_info.genre]
                if track_info.musicbrainz_trackid: ape_audio["MUSICBRAINZ_TRACKID"] = [track_info.musicbrainz_trackid]
                
                if cover_art_path and cover_art_path.exists():
                    with open(cover_art_path, "rb") as f:
                        image_data = f.read()
                    filename = cover_art_path.name.encode("utf-8")
                    ape_audio["Cover Art (Front)"] = filename + b"\0" + image_data
                ape_audio.save()
            
        else:
            raise MetadataError(f"Writing metadata to {ext} files is not supported.")
    except Exception as e:
        if isinstance(e, MetadataError):
            raise
        raise MetadataError(f"Failed to write metadata for {track_info.file_path}: {e}") from e
