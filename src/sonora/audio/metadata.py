from pathlib import Path

import taglib

from sonora.core.constants import SUPPORTED_EXTS
from sonora.core.logger import LOG
from sonora.core.models import TrackInfo
from sonora.core.utils import is_valid_uuid, normalize_date, normalize_genre


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
                for key in keys:
                    tag_values = (
                        tags.get(key) or tags.get(key.upper()) or tags.get(key.lower())
                    )
                    if tag_values and len(tag_values) > 0 and tag_values[0] is not None:
                        return str(tag_values[0]).strip()
                return None

            artist = get_tag("ARTIST", "TPE1", "AUTHOR") or "Unknown Artist"
            title = get_tag("TITLE", "TIT2") or "Unknown Title"
            album = get_tag("ALBUM", "TALB", "WM/ALBUMTITLE") or "Unknown Album"
            album_artist = get_tag(
                "ALBUMARTIST", "ALBUM ARTIST", "TPE2", "WM/ALBUMARTIST"
            )
            date_str = get_tag("DATE", "TDRC", "YEAR", "WM/YEAR")
            date = normalize_date(date_str)
            genre = normalize_genre(get_tag("GENRE", "TCON", "WM/GENRE"))
            isrc = get_tag("ISRC", "TSRC")

            raw_track = get_tag("TRACKNUMBER", "TRCK", "TRACK")
            track_number = None
            if raw_track:
                try:
                    track_number = int(str(raw_track).split("/")[0])
                except ValueError as error:
                    LOG.debug(f"Failed to parse track number '{raw_track}': {error}")

            raw_bpm = get_tag("BPM", "TBPM", "WM/BEATSPERMINUTE")
            bpm = None
            if raw_bpm:
                try:
                    bpm = float(raw_bpm)
                except ValueError as error:
                    LOG.debug(f"Failed to parse BPM '{raw_bpm}': {error}")

            raw_disc = get_tag("DISCNUMBER", "TPOS", "DISC")
            disc_number = 1
            if raw_disc:
                try:
                    disc_number = int(str(raw_disc).split("/")[0])
                except ValueError:
                    disc_number = 1

            raw_replaygain_gain = get_tag(
                "REPLAYGAIN_TRACK_GAIN", "TXXX:REPLAYGAIN_TRACK_GAIN"
            )
            replaygain_track_gain = None
            if raw_replaygain_gain:
                try:
                    replaygain_track_gain = float(
                        str(raw_replaygain_gain).replace(" dB", "").strip()
                    )
                except ValueError as error:
                    LOG.debug(
                        f"Failed to parse ReplayGain gain '{raw_replaygain_gain}': {error}"
                    )

            raw_replaygain_peak = get_tag(
                "REPLAYGAIN_TRACK_PEAK", "TXXX:REPLAYGAIN_TRACK_PEAK"
            )
            replaygain_track_peak = None
            if raw_replaygain_peak:
                try:
                    replaygain_track_peak = float(str(raw_replaygain_peak).strip())
                except ValueError as error:
                    LOG.debug(
                        f"Failed to parse ReplayGain peak '{raw_replaygain_peak}': {error}"
                    )

            raw_replaygain_album_gain = get_tag(
                "REPLAYGAIN_ALBUM_GAIN", "TXXX:REPLAYGAIN_ALBUM_GAIN"
            )
            replaygain_album_gain = None
            if raw_replaygain_album_gain:
                try:
                    replaygain_album_gain = float(
                        str(raw_replaygain_album_gain).replace(" dB", "").strip()
                    )
                except ValueError as error:
                    LOG.debug(
                        f"Failed to parse ReplayGain album gain '{raw_replaygain_album_gain}': {error}"
                    )

            raw_replaygain_album_peak = get_tag(
                "REPLAYGAIN_ALBUM_PEAK", "TXXX:REPLAYGAIN_ALBUM_PEAK"
            )
            replaygain_album_peak = None
            if raw_replaygain_album_peak:
                try:
                    replaygain_album_peak = float(
                        str(raw_replaygain_album_peak).strip()
                    )
                except ValueError as error:
                    LOG.debug(
                        f"Failed to parse ReplayGain album peak '{raw_replaygain_album_peak}': {error}"
                    )

            sample_rate = song.sampleRate
            bitrate = song.bitrate
            channels = song.channels

            musicbrainz_track_id = get_tag(
                "MUSICBRAINZ_TRACKID",
                "MUSICBRAINZ TRACK ID",
                "TXXX:MUSICBRAINZ TRACK ID",
            )
            musicbrainz_album_id = get_tag(
                "MUSICBRAINZ_ALBUMID",
                "MUSICBRAINZ ALBUM ID",
                "TXXX:MUSICBRAINZ ALBUM ID",
            )
            musicbrainz_release_group_id = get_tag(
                "MUSICBRAINZ_RELEASEGROUPID", "MUSICBRAINZ RELEASEGROUP ID"
            )
            musicbrainz_artist_id = get_tag(
                "MUSICBRAINZ_ARTISTID", "MUSICBRAINZ ARTIST ID"
            )
            musicbrainz_work_id = get_tag("MUSICBRAINZ_WORKID")
            acoustid_id = get_tag("ACOUSTID_ID", "ACOUSTID ID")
            discogs_release_id = get_tag("DISCOGS_RELEASE_ID", "DISCOGS RELEASE ID")
            itunes_trackid = get_tag("ITUNESTRACKID", "ITUNES_TRACK_ID")
            itunes_collectionid = get_tag("ITUNESCOLLECTIONID", "ITUNES_COLLECTION_ID")
            itunes_artistid = get_tag("ITUNESARTISTID", "ITUNES_ARTIST_ID")
            album_artist_sort = get_tag("ALBUMARTISTSORT")
            artist_sort = get_tag("ARTISTSORT")

            raw_total_tracks = get_tag("TRACKTOTAL", "TOTALTRACKS")
            total_tracks = (
                int(raw_total_tracks)
                if raw_total_tracks and raw_total_tracks.isdigit()
                else None
            )

            raw_total_discs = get_tag("DISCTOTAL", "TOTALDISCS")
            total_discs = (
                int(raw_total_discs)
                if raw_total_discs and raw_total_discs.isdigit()
                else None
            )

            release_type = get_tag("RELEASETYPE")
            release_status = get_tag("RELEASESTATUS")
            release_country = get_tag("RELEASECOUNTRY", "COUNTRY")
            label = get_tag("LABEL", "PUBLISHER")
            catalog_number = get_tag("CATALOGNUMBER")
            barcode = get_tag("BARCODE")
            media = get_tag("MEDIA")
            comment = get_tag("COMMENT", "COMM")
            advisory = get_tag("ITUNESADVISORY", "ADVISORY")
            original_date = get_tag("ORIGINALDATE", "ORIGINALYEAR")
            cuesheet = get_tag("CUESHEET")

            composer = get_tag("COMPOSER", "TCOM")
            lyricist = get_tag("LYRICIST", "TEXT", "WRITER")
            remixer = get_tag("REMIXER", "TPE4")
            initial_key = get_tag("INITIALKEY", "KEY", "TKEY")
            copyright_text = get_tag("COPYRIGHT", "TCOP")
            raw_compilation = get_tag("COMPILATION", "TCMP")
            compilation = (
                True
                if raw_compilation in ("1", "true", "True")
                else False
                if raw_compilation in ("0", "false", "False")
                else None
            )
            spotify_trackid = get_tag(
                "SPOTIFY_TRACK_ID", "SPOTIFY_ID", "TXXX:SPOTIFY_TRACK_ID"
            )

            musicbrainz_album_artist_id = get_tag(
                "MUSICBRAINZ_ALBUMARTISTID", "MUSICBRAINZ ALBUM ARTIST ID"
            )
            discogs_artist_id = get_tag("DISCOGS_ARTIST_ID", "DISCOGS ARTIST ID")
            language = get_tag("LANGUAGE", "TLAN")
            script = get_tag("SCRIPT")
            mood = get_tag("MOOD")
            style = get_tag("STYLE")

            disambiguation = get_tag("DISAMBIGUATION", "TXXX:DISAMBIGUATION")
            raw_rating = get_tag("RATING", "POPM")
            rating = float(raw_rating) if raw_rating else None
            featured_artists = get_tag("FEATURED_ARTISTS", "TXXX:FEATURED_ARTISTS")
            producers = get_tag("PRODUCERS", "TXXX:PRODUCERS")
            genius_song_id = get_tag("GENIUS_SONG_ID", "TXXX:GENIUS_SONG_ID")
            raw_listeners = get_tag("LISTENERS", "TXXX:LASTFM_LISTENERS")
            listeners = (
                int(raw_listeners)
                if raw_listeners and raw_listeners.isdigit()
                else None
            )
            raw_playcount = get_tag("PLAYCOUNT", "TXXX:LASTFM_PLAYCOUNT")
            playcount = (
                int(raw_playcount)
                if raw_playcount and raw_playcount.isdigit()
                else None
            )
            music_video_url = get_tag("MUSIC_VIDEO_URL", "TXXX:MUSIC_VIDEO_URL")

            art_width, art_height = None, None
            if hasattr(song, "pictures") and song.pictures and len(song.pictures) > 0:
                first_picture = song.pictures[0]
                art_width = getattr(first_picture, "width", None)
                art_height = getattr(first_picture, "height", None)

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
                replaygain_track_gain=replaygain_track_gain,
                replaygain_track_peak=replaygain_track_peak,
                replaygain_album_gain=replaygain_album_gain,
                replaygain_album_peak=replaygain_album_peak,
                sample_rate=sample_rate,
                bitrate=bitrate,
                channels=channels,
                musicbrainz_trackid=musicbrainz_track_id,
                musicbrainz_albumid=musicbrainz_album_id,
                musicbrainz_releasegroupid=musicbrainz_release_group_id,
                musicbrainz_artistid=musicbrainz_artist_id,
                musicbrainz_workid=musicbrainz_work_id,
                acoustid_id=acoustid_id,
                discogs_release_id=discogs_release_id,
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
                original_date=original_date,
                cuesheet=cuesheet,
                composer=composer,
                lyricist=lyricist,
                remixer=remixer,
                initial_key=initial_key,
                copyright=copyright_text,
                compilation=compilation,
                spotify_trackid=spotify_trackid,
                musicbrainz_albumartistid=musicbrainz_album_artist_id,
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
                art_width=art_width,
                art_height=art_height,
            )
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
            song.tags["ARTIST"] = [track_info.artist]
            song.tags["TITLE"] = [track_info.title]
            song.tags["ALBUM"] = [track_info.album]

            if track_info.album_artist:
                song.tags["ALBUMARTIST"] = [track_info.album_artist]
            if track_info.artist_sort:
                song.tags["ARTISTSORT"] = [track_info.artist_sort]
            if track_info.album_artist_sort:
                song.tags["ALBUMARTISTSORT"] = [track_info.album_artist_sort]

            total_tracks_str = (
                str(track_info.total_tracks) if track_info.total_tracks else None
            )
            if track_info.track_number is not None:
                track_number_str = (
                    f"{track_info.track_number}/{total_tracks_str}"
                    if total_tracks_str
                    else str(track_info.track_number)
                )
                song.tags["TRACKNUMBER"] = [track_number_str]
            if total_tracks_str:
                song.tags["TRACKTOTAL"] = [total_tracks_str]
                song.tags["TOTALTRACKS"] = [total_tracks_str]

            total_discs_str = (
                str(track_info.total_discs) if track_info.total_discs else None
            )
            if track_info.disc_number is not None:
                disc_number_str = (
                    f"{track_info.disc_number}/{total_discs_str}"
                    if total_discs_str
                    else str(track_info.disc_number)
                )
                song.tags["DISCNUMBER"] = [disc_number_str]
            if total_discs_str:
                song.tags["DISCTOTAL"] = [total_discs_str]
                song.tags["TOTALDISCS"] = [total_discs_str]

            if track_info.date:
                song.tags["DATE"] = [track_info.date]
            if track_info.genre:
                song.tags["GENRE"] = [track_info.genre]
            if track_info.bpm is not None:
                song.tags["BPM"] = [f"{track_info.bpm:.1f}"]
            if track_info.isrc:
                song.tags["ISRC"] = [track_info.isrc]
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

            if is_valid_uuid(track_info.musicbrainz_trackid):
                song.tags["MUSICBRAINZ_TRACKID"] = [track_info.musicbrainz_trackid]
            if is_valid_uuid(track_info.musicbrainz_albumid):
                song.tags["MUSICBRAINZ_ALBUMID"] = [track_info.musicbrainz_albumid]
            if is_valid_uuid(track_info.musicbrainz_releasegroupid):
                song.tags["MUSICBRAINZ_RELEASEGROUPID"] = [
                    track_info.musicbrainz_releasegroupid
                ]
            if is_valid_uuid(track_info.musicbrainz_artistid):
                song.tags["MUSICBRAINZ_ARTISTID"] = [track_info.musicbrainz_artistid]
            if is_valid_uuid(track_info.musicbrainz_workid):
                song.tags["MUSICBRAINZ_WORKID"] = [track_info.musicbrainz_workid]
            if track_info.acoustid_id:
                song.tags["ACOUSTID_ID"] = [track_info.acoustid_id]
            if track_info.discogs_release_id:
                song.tags["DISCOGS_RELEASE_ID"] = [track_info.discogs_release_id]
            if track_info.itunes_trackid:
                song.tags["ITUNESTRACKID"] = [track_info.itunes_trackid]
            if track_info.itunes_collectionid:
                song.tags["ITUNESCOLLECTIONID"] = [track_info.itunes_collectionid]
            if track_info.itunes_artistid:
                song.tags["ITUNESARTISTID"] = [track_info.itunes_artistid]
            if track_info.release_type:
                song.tags["RELEASETYPE"] = [track_info.release_type]
            if track_info.release_status:
                song.tags["RELEASESTATUS"] = [track_info.release_status]
            if track_info.release_country:
                song.tags["RELEASECOUNTRY"] = [track_info.release_country]
                song.tags["COUNTRY"] = [track_info.release_country]
            if track_info.label:
                song.tags["LABEL"] = [track_info.label]
                song.tags["PUBLISHER"] = [track_info.label]
            if track_info.catalog_number:
                song.tags["CATALOGNUMBER"] = [track_info.catalog_number]
            if track_info.barcode:
                song.tags["BARCODE"] = [track_info.barcode]
            if track_info.media:
                song.tags["MEDIA"] = [track_info.media]
            if track_info.comment:
                song.tags["COMMENT"] = [track_info.comment]
            if track_info.advisory:
                song.tags["ITUNESADVISORY"] = [track_info.advisory]
            if track_info.original_date:
                song.tags["ORIGINALDATE"] = [track_info.original_date]
                song.tags["ORIGINALYEAR"] = [track_info.original_date[:4]]
            if track_info.cuesheet:
                song.tags["CUESHEET"] = [track_info.cuesheet]
            if track_info.composer:
                song.tags["COMPOSER"] = [track_info.composer]
            if track_info.lyricist:
                song.tags["LYRICIST"] = [track_info.lyricist]
            if track_info.remixer:
                song.tags["REMIXER"] = [track_info.remixer]
            if track_info.initial_key:
                song.tags["INITIALKEY"] = [track_info.initial_key]
            if track_info.copyright:
                song.tags["COPYRIGHT"] = [track_info.copyright]
            if track_info.compilation is not None:
                song.tags["COMPILATION"] = ["1" if track_info.compilation else "0"]
            if track_info.spotify_trackid:
                song.tags["SPOTIFY_TRACK_ID"] = [track_info.spotify_trackid]
            if is_valid_uuid(track_info.musicbrainz_albumartistid):
                song.tags["MUSICBRAINZ_ALBUMARTISTID"] = [
                    track_info.musicbrainz_albumartistid
                ]
            if track_info.discogs_artist_id:
                song.tags["DISCOGS_ARTIST_ID"] = [track_info.discogs_artist_id]
            if track_info.language:
                song.tags["LANGUAGE"] = [track_info.language]
            if track_info.script:
                song.tags["SCRIPT"] = [track_info.script]
            if track_info.mood:
                song.tags["MOOD"] = [track_info.mood]
            if track_info.style:
                song.tags["STYLE"] = [track_info.style]
            if track_info.disambiguation:
                song.tags["DISAMBIGUATION"] = [track_info.disambiguation]
            if track_info.rating is not None:
                song.tags["RATING"] = [f"{track_info.rating:.1f}"]
            if track_info.featured_artists:
                song.tags["FEATURED_ARTISTS"] = [track_info.featured_artists]
            if track_info.producers:
                song.tags["PRODUCERS"] = [track_info.producers]
            if track_info.genius_song_id:
                song.tags["GENIUS_SONG_ID"] = [track_info.genius_song_id]
            if track_info.listeners is not None:
                song.tags["LISTENERS"] = [str(track_info.listeners)]
            if track_info.playcount is not None:
                song.tags["PLAYCOUNT"] = [str(track_info.playcount)]
            if track_info.music_video_url:
                song.tags["MUSIC_VIDEO_URL"] = [track_info.music_video_url]

            if cover_art_path and cover_art_path.exists():
                image_data = cover_art_path.read_bytes()
                mime_type = (
                    "image/jpeg"
                    if cover_art_path.suffix.lower() in [".jpg", ".jpeg"]
                    else "image/png"
                )
                if hasattr(song, "pictures"):
                    cover_picture = taglib.Picture(
                        data=image_data,
                        mime_type=mime_type,
                        description="Cover",
                        picture_type="Front Cover",
                    )
                    song.pictures = [cover_picture]

            unsaved = song.save()
            if unsaved:
                LOG.debug(f"TagLib unsaved tags for {track_info.file_path}: {unsaved}")
    except (RuntimeError, ValueError, FileNotFoundError):
        raise
    except (OSError, KeyError) as error:
        raise RuntimeError(
            f"Failed to write metadata for {track_info.file_path}: {error}"
        ) from error
