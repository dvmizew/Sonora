from pathlib import Path

import taglib

from sonora.core.constants import SUPPORTED_EXTS
from sonora.core.logger import LOG
from sonora.core.models import TrackInfo
from sonora.core.utils import normalize_date, normalize_genre


def read_track_metadata(file_path: Path) -> TrackInfo:
    """
    Read metadata tags from an audio file using C++ TagLib (pytaglib) and return a TrackInfo dataclass.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported audio format: {file_path}")

    try:
        with taglib.File(str(file_path)) as song:
            tags = song.tags

            def get_tag(*keys: str) -> str | None:
                for k in keys:
                    vals = tags.get(k) or tags.get(k.upper()) or tags.get(k.lower())
                    if vals and len(vals) > 0 and vals[0] is not None:
                        return str(vals[0]).strip()
                return None

            artist = get_tag("ARTIST", "TPE1", "AUTHOR") or "Unknown Artist"
            title = get_tag("TITLE", "TIT2") or "Unknown Title"
            album = get_tag("ALBUM", "TALB", "WM/ALBUMTITLE") or "Unknown Album"
            album_artist = get_tag("ALBUMARTIST", "ALBUM ARTIST", "TPE2", "WM/ALBUMARTIST")
            date_str = get_tag("DATE", "TDRC", "YEAR", "WM/YEAR")
            date = normalize_date(date_str)
            genre = normalize_genre(get_tag("GENRE", "TCON", "WM/GENRE"))
            isrc = get_tag("ISRC", "TSRC")

            raw_track = get_tag("TRACKNUMBER", "TRCK", "TRACK")
            track_number = None
            if raw_track:
                try:
                    track_number = int(str(raw_track).split("/")[0])
                except ValueError as e:
                    LOG.debug(f"Failed to parse track number '{raw_track}': {e}")

            raw_bpm = get_tag("BPM", "TBPM", "WM/BEATSPERMINUTE")
            bpm = None
            if raw_bpm:
                try:
                    bpm = float(raw_bpm)
                except ValueError as e:
                    LOG.debug(f"Failed to parse BPM '{raw_bpm}': {e}")

            raw_disc = get_tag("DISCNUMBER", "TPOS", "DISC")
            disc_number = 1
            if raw_disc:
                try:
                    disc_number = int(str(raw_disc).split("/")[0])
                except ValueError:
                    disc_number = 1

            raw_rg_gain = get_tag("REPLAYGAIN_TRACK_GAIN", "TXXX:REPLAYGAIN_TRACK_GAIN")
            rg_gain = None
            if raw_rg_gain:
                try:
                    rg_gain = float(str(raw_rg_gain).replace(" dB", "").strip())
                except ValueError as e:
                    LOG.debug(f"Failed to parse ReplayGain gain '{raw_rg_gain}': {e}")

            raw_rg_peak = get_tag("REPLAYGAIN_TRACK_PEAK", "TXXX:REPLAYGAIN_TRACK_PEAK")
            rg_peak = None
            if raw_rg_peak:
                try:
                    rg_peak = float(str(raw_rg_peak).strip())
                except ValueError as e:
                    LOG.debug(f"Failed to parse ReplayGain peak '{raw_rg_peak}': {e}")

            sample_rate = song.sampleRate
            bitrate = song.bitrate
            channels = song.channels

            mb_track = get_tag("MUSICBRAINZ_TRACKID", "MUSICBRAINZ TRACK ID", "TXXX:MUSICBRAINZ TRACK ID")
            mb_album = get_tag("MUSICBRAINZ_ALBUMID", "MUSICBRAINZ ALBUM ID", "TXXX:MUSICBRAINZ ALBUM ID")
            mb_rg = get_tag("MUSICBRAINZ_RELEASEGROUPID", "MUSICBRAINZ RELEASEGROUP ID")
            mb_art = get_tag("MUSICBRAINZ_ARTISTID", "MUSICBRAINZ ARTIST ID")
            mb_work = get_tag("MUSICBRAINZ_WORKID")
            acoustid_id = get_tag("ACOUSTID_ID", "ACOUSTID ID")
            discogs_id = get_tag("DISCOGS_RELEASE_ID", "DISCOGS RELEASE ID")
            itunes_trackid = get_tag("ITUNESTRACKID", "ITUNES_TRACK_ID")
            itunes_collectionid = get_tag("ITUNESCOLLECTIONID", "ITUNES_COLLECTION_ID")
            itunes_artistid = get_tag("ITUNESARTISTID", "ITUNES_ARTIST_ID")
            album_artist_sort = get_tag("ALBUMARTISTSORT")
            artist_sort = get_tag("ARTISTSORT")

            raw_tot_tracks = get_tag("TRACKTOTAL", "TOTALTRACKS")
            total_tracks = int(raw_tot_tracks) if raw_tot_tracks and raw_tot_tracks.isdigit() else None

            raw_tot_discs = get_tag("DISCTOTAL", "TOTALDISCS")
            total_discs = int(raw_tot_discs) if raw_tot_discs and raw_tot_discs.isdigit() else None

            release_type = get_tag("RELEASETYPE")
            release_status = get_tag("RELEASESTATUS")
            release_country = get_tag("RELEASECOUNTRY", "COUNTRY")
            label = get_tag("LABEL", "PUBLISHER")
            catalog_number = get_tag("CATALOGNUMBER")
            barcode = get_tag("BARCODE")
            media = get_tag("MEDIA")
            comment = get_tag("COMMENT", "COMM")
            advisory = get_tag("ITUNESADVISORY", "ADVISORY")
            orig_date = get_tag("ORIGINALDATE", "ORIGINALYEAR")
            cuesheet = get_tag("CUESHEET")

            composer = get_tag("COMPOSER", "TCOM")
            lyricist = get_tag("LYRICIST", "TEXT", "WRITER")
            remixer = get_tag("REMIXER", "TPE4")
            initial_key = get_tag("INITIALKEY", "KEY", "TKEY")
            copyright_val = get_tag("COPYRIGHT", "TCOP")
            raw_comp = get_tag("COMPILATION", "TCMP")
            compilation = True if raw_comp in ("1", "true", "True") else False if raw_comp in ("0", "false", "False") else None
            spotify_trackid = get_tag("SPOTIFY_TRACK_ID", "SPOTIFY_ID", "TXXX:SPOTIFY_TRACK_ID")

            mb_album_artist = get_tag("MUSICBRAINZ_ALBUMARTISTID", "MUSICBRAINZ ALBUM ARTIST ID")
            discogs_artist_id = get_tag("DISCOGS_ARTIST_ID", "DISCOGS ARTIST ID")
            language = get_tag("LANGUAGE", "TLAN")
            script = get_tag("SCRIPT")
            mood = get_tag("MOOD")
            style = get_tag("STYLE")

            disambiguation = get_tag("DISAMBIGUATION", "TXXX:DISAMBIGUATION")
            raw_rat = get_tag("RATING", "POPM")
            rating = float(raw_rat) if raw_rat else None
            featured_artists = get_tag("FEATURED_ARTISTS", "TXXX:FEATURED_ARTISTS")
            producers = get_tag("PRODUCERS", "TXXX:PRODUCERS")
            genius_song_id = get_tag("GENIUS_SONG_ID", "TXXX:GENIUS_SONG_ID")
            raw_list = get_tag("LISTENERS", "TXXX:LASTFM_LISTENERS")
            listeners = int(raw_list) if raw_list and raw_list.isdigit() else None
            raw_play = get_tag("PLAYCOUNT", "TXXX:LASTFM_PLAYCOUNT")
            playcount = int(raw_play) if raw_play and raw_play.isdigit() else None
            music_video_url = get_tag("MUSIC_VIDEO_URL", "TXXX:MUSIC_VIDEO_URL")

            art_w, art_h = None, None
            if hasattr(song, "pictures") and song.pictures and len(song.pictures) > 0:
                p0 = song.pictures[0]
                art_w = getattr(p0, "width", None)
                art_h = getattr(p0, "height", None)

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
                composer=composer,
                lyricist=lyricist,
                remixer=remixer,
                initial_key=initial_key,
                copyright=copyright_val,
                compilation=compilation,
                spotify_trackid=spotify_trackid,
                musicbrainz_albumartistid=mb_album_artist,
                discogs_artist_id=discogs_artist_id,
                language=language,
                script=script,
                mood=mood,
                style=style,
                disambiguation=disambiguation,
                rating=rating,
                featured_artists=featured_artists,
                producers=producers,
                genius_song_id=genius_song_id,
                listeners=listeners,
                playcount=playcount,
                music_video_url=music_video_url,
                art_width=art_w,
                art_height=art_h,
            )
    except Exception as e:
        if isinstance(e, (RuntimeError, ValueError, FileNotFoundError)):
            raise
        raise RuntimeError(f"Failed to read metadata for {file_path}: {e}") from e


def write_track_metadata(track_info: TrackInfo, cover_art_path: Path | None = None) -> None:

    if not track_info.file_path.exists():
        raise FileNotFoundError(f"File not found: {track_info.file_path}")

    try:
        with taglib.File(str(track_info.file_path)) as song:
            song.tags["ARTIST"] = [track_info.artist]
            song.tags["TITLE"] = [track_info.title]
            song.tags["ALBUM"] = [track_info.album]

            if track_info.album_artist: song.tags["ALBUMARTIST"] = [track_info.album_artist]
            if track_info.artist_sort: song.tags["ARTISTSORT"] = [track_info.artist_sort]
            if track_info.album_artist_sort: song.tags["ALBUMARTISTSORT"] = [track_info.album_artist_sort]

            tot_tr = str(track_info.total_tracks) if track_info.total_tracks else None
            if track_info.track_number is not None:
                tr_str = f"{track_info.track_number}/{tot_tr}" if tot_tr else str(track_info.track_number)
                song.tags["TRACKNUMBER"] = [tr_str]
            if tot_tr:
                song.tags["TRACKTOTAL"] = [tot_tr]
                song.tags["TOTALTRACKS"] = [tot_tr]

            tot_ds = str(track_info.total_discs) if track_info.total_discs else None
            if track_info.disc_number is not None:
                ds_str = f"{track_info.disc_number}/{tot_ds}" if tot_ds else str(track_info.disc_number)
                song.tags["DISCNUMBER"] = [ds_str]
            if tot_ds:
                song.tags["DISCTOTAL"] = [tot_ds]
                song.tags["TOTALDISCS"] = [tot_ds]

            if track_info.date: song.tags["DATE"] = [track_info.date]
            if track_info.genre: song.tags["GENRE"] = [track_info.genre]
            if track_info.bpm is not None: song.tags["BPM"] = [f"{track_info.bpm:.1f}"]
            if track_info.isrc: song.tags["ISRC"] = [track_info.isrc]
            if track_info.replaygain_track_gain is not None:
                song.tags["REPLAYGAIN_TRACK_GAIN"] = [f"{track_info.replaygain_track_gain:+.2f} dB"]
            if track_info.replaygain_track_peak is not None:
                song.tags["REPLAYGAIN_TRACK_PEAK"] = [f"{track_info.replaygain_track_peak:.6f}"]

            if track_info.musicbrainz_trackid: song.tags["MUSICBRAINZ_TRACKID"] = [track_info.musicbrainz_trackid]
            if track_info.musicbrainz_albumid: song.tags["MUSICBRAINZ_ALBUMID"] = [track_info.musicbrainz_albumid]
            if track_info.musicbrainz_releasegroupid: song.tags["MUSICBRAINZ_RELEASEGROUPID"] = [track_info.musicbrainz_releasegroupid]
            if track_info.musicbrainz_artistid: song.tags["MUSICBRAINZ_ARTISTID"] = [track_info.musicbrainz_artistid]
            if track_info.musicbrainz_workid: song.tags["MUSICBRAINZ_WORKID"] = [track_info.musicbrainz_workid]
            if track_info.acoustid_id: song.tags["ACOUSTID_ID"] = [track_info.acoustid_id]
            if track_info.discogs_release_id: song.tags["DISCOGS_RELEASE_ID"] = [track_info.discogs_release_id]
            if track_info.itunes_trackid: song.tags["ITUNESTRACKID"] = [track_info.itunes_trackid]
            if track_info.itunes_collectionid: song.tags["ITUNESCOLLECTIONID"] = [track_info.itunes_collectionid]
            if track_info.itunes_artistid: song.tags["ITUNESARTISTID"] = [track_info.itunes_artistid]
            if track_info.release_type: song.tags["RELEASETYPE"] = [track_info.release_type]
            if track_info.release_status: song.tags["RELEASESTATUS"] = [track_info.release_status]
            if track_info.release_country:
                song.tags["RELEASECOUNTRY"] = [track_info.release_country]
                song.tags["COUNTRY"] = [track_info.release_country]
            if track_info.label:
                song.tags["LABEL"] = [track_info.label]
                song.tags["PUBLISHER"] = [track_info.label]
            if track_info.catalog_number: song.tags["CATALOGNUMBER"] = [track_info.catalog_number]
            if track_info.barcode: song.tags["BARCODE"] = [track_info.barcode]
            if track_info.media: song.tags["MEDIA"] = [track_info.media]
            if track_info.comment: song.tags["COMMENT"] = [track_info.comment]
            if track_info.advisory: song.tags["ITUNESADVISORY"] = [track_info.advisory]
            if track_info.original_date:
                song.tags["ORIGINALDATE"] = [track_info.original_date]
                song.tags["ORIGINALYEAR"] = [track_info.original_date[:4]]
            if track_info.cuesheet: song.tags["CUESHEET"] = [track_info.cuesheet]
            if track_info.composer: song.tags["COMPOSER"] = [track_info.composer]
            if track_info.lyricist: song.tags["LYRICIST"] = [track_info.lyricist]
            if track_info.remixer: song.tags["REMIXER"] = [track_info.remixer]
            if track_info.initial_key: song.tags["INITIALKEY"] = [track_info.initial_key]
            if track_info.copyright: song.tags["COPYRIGHT"] = [track_info.copyright]
            if track_info.compilation is not None: song.tags["COMPILATION"] = ["1" if track_info.compilation else "0"]
            if track_info.spotify_trackid: song.tags["SPOTIFY_TRACK_ID"] = [track_info.spotify_trackid]
            if track_info.musicbrainz_albumartistid: song.tags["MUSICBRAINZ_ALBUMARTISTID"] = [track_info.musicbrainz_albumartistid]
            if track_info.discogs_artist_id: song.tags["DISCOGS_ARTIST_ID"] = [track_info.discogs_artist_id]
            if track_info.language: song.tags["LANGUAGE"] = [track_info.language]
            if track_info.script: song.tags["SCRIPT"] = [track_info.script]
            if track_info.mood: song.tags["MOOD"] = [track_info.mood]
            if track_info.style: song.tags["STYLE"] = [track_info.style]
            if track_info.disambiguation: song.tags["DISAMBIGUATION"] = [track_info.disambiguation]
            if track_info.rating is not None: song.tags["RATING"] = [f"{track_info.rating:.1f}"]
            if track_info.featured_artists: song.tags["FEATURED_ARTISTS"] = [track_info.featured_artists]
            if track_info.producers: song.tags["PRODUCERS"] = [track_info.producers]
            if track_info.genius_song_id: song.tags["GENIUS_SONG_ID"] = [track_info.genius_song_id]
            if track_info.listeners is not None: song.tags["LISTENERS"] = [str(track_info.listeners)]
            if track_info.playcount is not None: song.tags["PLAYCOUNT"] = [str(track_info.playcount)]
            if track_info.music_video_url: song.tags["MUSIC_VIDEO_URL"] = [track_info.music_video_url]

            if cover_art_path and cover_art_path.exists():
                with open(cover_art_path, "rb") as f:
                    image_data = f.read()
                mime = "image/jpeg" if cover_art_path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
                if hasattr(song, "pictures"):
                    pic = taglib.Picture(data=image_data, mime_type=mime, description="Cover", picture_type="Front Cover")
                    song.pictures = [pic]

            unsaved = song.save()
            if unsaved:
                LOG.debug(f"TagLib unsaved tags for {track_info.file_path}: {unsaved}")
    except Exception as e:
        if isinstance(e, (RuntimeError, ValueError, FileNotFoundError)):
            raise
        raise RuntimeError(f"Failed to write metadata for {track_info.file_path}: {e}") from e
