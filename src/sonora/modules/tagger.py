import dataclasses
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from sonora.audio.art import process_album_cover_art, process_artist_artwork
from sonora.audio.bpm import calculate_bpm
from sonora.audio.cuesheet import read_cuesheet_content
from sonora.audio.metadata import read_track_metadata, write_track_metadata
from sonora.audio.replaygain import calculate_album_replaygain
from sonora.core.constants import ARTIST_ALIASES, SUPPORTED_EXTS
from sonora.core.logger import CONSOLE, LOG
from sonora.core.models import TrackInfo
from sonora.core.utils import (
    is_valid_uuid,
    normalize_date,
    normalize_genre,
    normalize_str,
)
from sonora.services.acoustid import lookup_acoustid
from sonora.services.deezer import (
    fetch_deezer_album_details,
    fetch_deezer_track_details,
)
from sonora.services.discogs import search_discogs_release
from sonora.services.genius import fetch_genius_song_details
from sonora.services.lastfm import fetch_lastfm_tags, fetch_lastfm_track_stats
from sonora.services.lyrics import process_track_lyrics
from sonora.services.musicbrainz import (
    fetch_album_track_mbids,
    fetch_track_mbid,
    search_musicbrainz_release,
)
from sonora.services.theaudiodb import fetch_track_video_url


def normalize_artist_alias(artist: str) -> str:
    return ARTIST_ALIASES.get(normalize_str(artist), artist.strip())


def process_single_track(
    file_path: Path,
    fetch_bpm: bool = True,
    fetch_lyrics: bool = True,
    fetch_itunes_art: bool = True,
    lastfm_api_key: str | None = None,
    acoustid_api_key: str | None = None,
    discogs_user_token: str | None = None,
    genius_api_token: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    album_mbid: str | None = None,
    album_track_mbids: dict[int, str] | None = None,
    cuesheet_content: str | None = None,
) -> TrackInfo:
    LOG.start_buffering()
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        track_info = read_track_metadata(file_path)
        orig_info = dataclasses.replace(track_info)
        track_info.artist = normalize_artist_alias(track_info.artist)

        LOG.info(f"🎧 Processing track: [white]{file_path.name}[/]")

        # 1. Respect existing MBID or prioritize AcoustID (most exact) over text search
        if (
            not is_valid_uuid(track_info.musicbrainz_trackid) or force
        ) and acoustid_api_key:
            try:
                acoustid_mbid = lookup_acoustid(
                    file_path,
                    api_key=acoustid_api_key,
                    expected_artist=track_info.artist,
                    expected_title=track_info.title,
                )
                if is_valid_uuid(acoustid_mbid):
                    track_info.musicbrainz_trackid = acoustid_mbid
                    LOG.info(f"   ∟ 🎯 [acoustid] Matched MBID: {acoustid_mbid[:8]}...")
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as e:
                LOG.debug(f"AcoustID lookup failed for {track_info.title}: {e}")

        # 2. Check pre-fetched album track MBIDs map first (1 single API call per album!)
        album_mbids = album_track_mbids or {}
        if (
            (not is_valid_uuid(track_info.musicbrainz_trackid) or force)
            and track_info.track_number
            and track_info.track_number in album_mbids
        ):
            cand_mbid = album_mbids[track_info.track_number]
            if is_valid_uuid(cand_mbid):
                track_info.musicbrainz_trackid = cand_mbid
                LOG.info(
                    f"   ∟ 🏷️ [MusicBrainz Album Match] Found MBID: {cand_mbid[:8]}..."
                )
        elif not is_valid_uuid(track_info.musicbrainz_trackid) or force:
            try:
                mbid = fetch_track_mbid(track_info.artist, track_info.title)
                if is_valid_uuid(mbid):
                    track_info.musicbrainz_trackid = mbid
                    LOG.info(f"   ∟ 🏷️ [MusicBrainz] Found MBID: {mbid[:8]}...")
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as e:
                LOG.debug(f"MusicBrainz lookup failed for {track_info.title}: {e}")

        # 3. Fetch MusicBrainz Album ID
        if not is_valid_uuid(track_info.musicbrainz_albumid):
            if is_valid_uuid(album_mbid):
                track_info.musicbrainz_albumid = str(album_mbid)
            else:
                try:
                    search_artist = (
                        track_info.album_artist
                        if track_info.album_artist
                        else track_info.artist
                    )
                    release = search_musicbrainz_release(
                        search_artist, track_info.album
                    )
                    if release:
                        mb_id = release.get("id")
                        if is_valid_uuid(str(mb_id)):
                            track_info.musicbrainz_albumid = str(mb_id)
                        if not track_info.date:
                            date_str = release.get("date")
                            if isinstance(date_str, str) and len(date_str) >= 4:
                                track_info.date = normalize_date(date_str)
                except (httpx.HTTPError, OSError, ValueError, RuntimeError) as e:
                    LOG.debug(
                        f"MusicBrainz Album lookup failed for {track_info.title}: {e}"
                    )

        # 4. Fetch Last.fm genre/mood tags
        if lastfm_api_key:
            try:
                tags = fetch_lastfm_tags(
                    track_info.artist,
                    track_info.title,
                    api_key=lastfm_api_key,
                    mbid=track_info.musicbrainz_trackid,
                )
                if tags:
                    raw_genre = tags[0]
                    norm_genre = normalize_genre(raw_genre)
                    if norm_genre:
                        track_info.genre = norm_genre
                        LOG.info(f"   ∟ 🏷️ [Last.fm] Genre: [cyan]{norm_genre}[/]")
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as e:
                LOG.debug(f"Last.fm lookup failed for {track_info.title}: {e}")

        # 5. Discogs metadata enrichment
        if discogs_user_token:
            try:
                release = search_discogs_release(
                    track_info.artist, track_info.album, user_token=discogs_user_token
                )
                if release:
                    if release.get("id") and not track_info.discogs_release_id:
                        track_info.discogs_release_id = str(release["id"])
                    if release.get("artist_id") and not track_info.discogs_artist_id:
                        track_info.discogs_artist_id = str(release["artist_id"])
                    if release.get("released") and not track_info.date:
                        track_info.date = normalize_date(str(release["released"]))
                    elif release.get("year") and not track_info.date:
                        track_info.date = normalize_date(str(release["year"]))

                    genres_val = release.get("genres")
                    if (
                        isinstance(genres_val, list)
                        and genres_val
                        and not track_info.genre
                    ):
                        raw_genre = str(genres_val[0])
                        norm_genre = normalize_genre(raw_genre)
                        if norm_genre:
                            track_info.genre = norm_genre

                    styles_val = release.get("styles")
                    if (
                        isinstance(styles_val, list)
                        and styles_val
                        and not track_info.style
                    ):
                        track_info.style = ", ".join(str(s) for s in styles_val[:3])

                    if release.get("country") and not track_info.release_country:
                        track_info.release_country = str(release["country"])
                    if release.get("label") and not track_info.label:
                        track_info.label = str(release["label"])
                    if release.get("catalog_number") and not track_info.catalog_number:
                        track_info.catalog_number = str(release["catalog_number"])
                    if release.get("barcode") and not track_info.barcode:
                        track_info.barcode = str(release["barcode"])
                    if release.get("media") and not track_info.media:
                        track_info.media = str(release["media"])
                    if release.get("composer") and not track_info.composer:
                        track_info.composer = str(release["composer"])

                    # Check track-specific credits or album-level credits for producers and remixer
                    t_credits = release.get("track_credits")
                    t_spec = None
                    if isinstance(t_credits, dict):
                        track_key = (
                            str(track_info.track_number)
                            if track_info.track_number
                            else None
                        )
                        if track_key and track_key in t_credits:
                            t_spec = t_credits[track_key]
                        elif track_info.title and track_info.title.lower() in t_credits:
                            t_spec = t_credits[track_info.title.lower()]

                    if isinstance(t_spec, dict):
                        if t_spec.get("producers") and not track_info.producers:
                            track_info.producers = str(t_spec["producers"])
                        if t_spec.get("remixer") and not track_info.remixer:
                            track_info.remixer = str(t_spec["remixer"])

                    if release.get("producers") and not track_info.producers:
                        track_info.producers = str(release["producers"])
                    if release.get("remixer") and not track_info.remixer:
                        track_info.remixer = str(release["remixer"])
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as e:
                LOG.debug(f"Discogs lookup failed for {track_info.title}: {e}")

        # 5b. Deezer metadata enrichment (Label, Barcode, Release Date, Genre, ISRC)
        try:
            d_album = fetch_deezer_album_details(track_info.artist, track_info.album)
            if d_album:
                if d_album.get("label") and not track_info.label:
                    track_info.label = str(d_album["label"])
                if d_album.get("barcode") and not track_info.barcode:
                    track_info.barcode = str(d_album["barcode"])
                if d_album.get("release_date") and not track_info.date:
                    track_info.date = normalize_date(str(d_album["release_date"]))
                if d_album.get("genre") and not track_info.genre:
                    norm_g = normalize_genre(str(d_album["genre"]))
                    if norm_g:
                        track_info.genre = norm_g

            d_track = fetch_deezer_track_details(track_info.artist, track_info.title)
            if d_track and d_track.get("isrc") and not track_info.isrc:
                track_info.isrc = str(d_track["isrc"])
        except (httpx.HTTPError, OSError, ValueError, RuntimeError) as e:
            LOG.debug(f"Deezer enrichment failed for {track_info.title}: {e}")

        # 6. Genius song details (description, genius_song_id, featured_artists, producers)
        if genius_api_token:
            try:
                g_details = fetch_genius_song_details(
                    track_info.artist, track_info.title, api_token=genius_api_token
                )
                if g_details:
                    if g_details.get("description") and (
                        not track_info.comment or force
                    ):
                        track_info.comment = str(g_details["description"])
                    if g_details.get("genius_song_id"):
                        track_info.genius_song_id = str(g_details["genius_song_id"])
                    if g_details.get("featured_artists") and (
                        not track_info.featured_artists or force
                    ):
                        track_info.featured_artists = str(g_details["featured_artists"])
                    if g_details.get("producers") and (
                        not track_info.producers or force
                    ):
                        track_info.producers = str(g_details["producers"])
                    LOG.info("   ∟ 📝 [Genius] Fetched song details & credits")
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as e:
                LOG.debug(f"Genius lookup failed for {track_info.title}: {e}")

        # 6a. Last.fm stats (listeners, playcount)
        if lastfm_api_key:
            try:
                stats = fetch_lastfm_track_stats(
                    track_info.artist, track_info.title, api_key=lastfm_api_key
                )
                if stats:
                    if stats.get("listeners"):
                        track_info.listeners = stats["listeners"]
                    if stats.get("playcount"):
                        track_info.playcount = stats["playcount"]
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as e:
                LOG.debug(f"Last.fm stats lookup failed for {track_info.title}: {e}")

        # 6b. TheAudioDB video URL
        try:
            vid_url = fetch_track_video_url(track_info.artist, track_info.title)
            if vid_url:
                track_info.music_video_url = vid_url
        except (OSError, ValueError, RuntimeError) as e:
            LOG.debug(f"TheAudioDB video lookup failed for {track_info.title}: {e}")

        # 6c. Embed Cuesheet content
        if cuesheet_content:
            track_info.cuesheet = cuesheet_content
        elif not track_info.cuesheet:
            cue_files = list(file_path.parent.glob("*.cue"))
            if cue_files:
                track_info.cuesheet = read_cuesheet_content(cue_files[0])

        # 7. Calculate BPM
        if fetch_bpm and (track_info.bpm is None or force):
            try:
                bpm = calculate_bpm(file_path)
                if bpm is not None:
                    track_info.bpm = bpm
                    LOG.info(f"   ∟ 🎵 BPM Calculated: [green]{bpm}[/]")
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as e:
                LOG.debug(f"BPM calculation failed for {track_info.title}: {e}")

        # 8. Fetch iTunes Cover Art (Download only, don't embed yet)
        cover_jpg = None
        if fetch_itunes_art:
            try:
                cover_jpg = process_album_cover_art(
                    file_path.parent,
                    track_info.artist,
                    track_info.album,
                    mb_album_id=track_info.musicbrainz_albumid,
                    force=force,
                    dry_run=dry_run,
                )
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as e:
                LOG.debug(f"Cover art downloading failed for {track_info.title}: {e}")
                cover_jpg = None

        # 9. Fetch & write .lrc lyrics file (Quality Upgrade: enhanced (3) > line-synced (2) > plain (1))
        if fetch_lyrics:
            try:
                lrc_text, tag_type = process_track_lyrics(
                    file_path,
                    track_info.artist,
                    track_info.title,
                    force=force,
                    dry_run=dry_run,
                    isrc=track_info.isrc,
                )
                if lrc_text and tag_type:
                    track_info.lyrics = lrc_text
                    LOG.info(f"   ∟ ✅ Saved {tag_type} lyrics for {file_path.name}")
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as e:
                LOG.debug(f"Lyrics fetch failed for {track_info.title}: {e}")

        # Compute exact tag diffs (compare dataclass fields)
        diff_lines = []
        _SKIP_FIELDS = {
            "file_path",
            "lyrics",
            "synced_lyrics",
            "sample_rate",
            "bitrate",
            "channels",
            "is_lossless",
            "art_width",
            "art_height",
        }
        for f in dataclasses.fields(TrackInfo):
            if f.name in _SKIP_FIELDS:
                continue
            old_val = getattr(orig_info, f.name)
            new_val = getattr(track_info, f.name)
            if old_val != new_val:
                diff_lines.append(f"\n       [*] {f.name}: {old_val} -> {new_val}")

        # 10. Write metadata tags back to file only if changed
        if diff_lines or force:
            if not dry_run:
                write_track_metadata(track_info, cover_art_path=cover_jpg)
                LOG.info(
                    f"   ∟ [green]✓[/] {file_path.name}: {len(diff_lines)} tag(s) updated.{''.join(diff_lines)}"
                )
            else:
                LOG.info(f"   ∟ [DRY-RUN] {file_path.name}{''.join(diff_lines)}")
        else:
            LOG.info(
                f"   ∟ [bold dim]✨ SKIPPED:[/] [dim]{file_path.name}[/] [dim]is already perfect.[/]"
            )

        return track_info
    finally:
        LOG.stop_buffering()


def tag_album_folder(
    folder_path: Path,
    max_workers: int = 4,
    fetch_bpm: bool = True,
    fetch_replaygain: bool = True,
    fetch_lyrics: bool = True,
    fetch_itunes_art: bool = True,
    lastfm_api_key: str | None = None,
    acoustid_api_key: str | None = None,
    discogs_user_token: str | None = None,
    genius_api_token: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> list[TrackInfo]:
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Album folder not found: {folder_path}")

    # Check if folder_path contains child directories with audio files
    subdirectories = [
        directory
        for directory in sorted(folder_path.iterdir())
        if directory.is_dir()
        and any(
            path.is_file() and path.suffix.lower() in SUPPORTED_EXTS
            for path in directory.rglob("*")
        )
    ]

    # If folder_path has child album directories and NO direct audio files in its root, tag each sub-album independently
    if subdirectories and not any(
        path.is_file() and path.suffix.lower() in SUPPORTED_EXTS
        for path in folder_path.glob("*")
    ):
        all_results: list[TrackInfo] = []
        for subdirectory in subdirectories:
            sub_results = tag_album_folder(
                subdirectory,
                max_workers=max_workers,
                fetch_bpm=fetch_bpm,
                fetch_replaygain=fetch_replaygain,
                fetch_lyrics=fetch_lyrics,
                fetch_itunes_art=fetch_itunes_art,
                lastfm_api_key=lastfm_api_key,
                acoustid_api_key=acoustid_api_key,
                discogs_user_token=discogs_user_token,
                genius_api_token=genius_api_token,
                force=force,
                dry_run=dry_run,
            )
            all_results.extend(sub_results)
        return all_results

    audio_files = sorted(
        [
            path
            for path in folder_path.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS
        ]
    )

    folder_name = folder_path.name
    LOG.force_info(
        f"📁 [bold cyan]Album:[/] [white]{folder_name}[/] [dim]({len(audio_files)} tracks)[/]"
    )

    if not audio_files:
        return []

    # Pre-resolve Cuesheet content once for entire album
    cue_files = list(folder_path.glob("*.cue"))
    album_cue_content = read_cuesheet_content(cue_files[0]) if cue_files else None

    # Batch Optimization: Fetch entire album track MBIDs and album MBID in 1 single API call
    album_musicbrainz_id: str | None = None
    album_track_mbids: dict[int, str] | None = None
    try:
        sample_meta = read_track_metadata(audio_files[0])
        sample_artist = sample_meta.album_artist or sample_meta.artist
        sample_album = sample_meta.album
        if sample_artist and sample_album:
            release_info = search_musicbrainz_release(sample_artist, sample_album)
            if release_info and release_info.get("id"):
                album_musicbrainz_id = str(release_info["id"])
                album_track_mbids = fetch_album_track_mbids(album_musicbrainz_id)
    except (httpx.HTTPError, OSError, ValueError, RuntimeError) as error:
        LOG.debug(f"Pre-fetching album track MBIDs failed: {error}")

    results: list[TrackInfo] = []

    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        future_to_file = {
            executor.submit(
                process_single_track,
                file_path=audio_file,
                fetch_bpm=fetch_bpm,
                fetch_lyrics=fetch_lyrics,
                fetch_itunes_art=fetch_itunes_art,
                lastfm_api_key=lastfm_api_key,
                acoustid_api_key=acoustid_api_key,
                discogs_user_token=discogs_user_token,
                genius_api_token=genius_api_token,
                force=force,
                dry_run=dry_run,
                album_mbid=album_musicbrainz_id,
                album_track_mbids=album_track_mbids,
                cuesheet_content=album_cue_content,
            ): audio_file
            for audio_file in audio_files
        }

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("[dim]/[/dim]"),
            TimeRemainingColumn(),
            console=CONSOLE,
        ) as progress:
            task = progress.add_task("[cyan]Tagging tracks...", total=len(audio_files))
            for future in as_completed(future_to_file):
                audio_file = future_to_file[future]
                try:
                    track_info = future.result()
                    results.append(track_info)
                except (httpx.HTTPError, OSError, ValueError, RuntimeError) as error:
                    LOG.warning(f"Failed to process {audio_file.name}: {error}")
                progress.advance(task)
    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    if fetch_replaygain:
        calculate_album_replaygain(audio_files, force=force, dry_run=dry_run)

    if results:
        primary_artist = results[0].album_artist or results[0].artist
        try:
            process_artist_artwork(folder_path, primary_artist, dry_run=dry_run)
        except (OSError, ValueError, RuntimeError) as error:
            LOG.debug(f"Artist art download failed: {error}")

    return results
