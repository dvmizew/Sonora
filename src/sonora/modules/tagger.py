import dataclasses
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz
from rich.markup import escape

from sonora.audio import art as art_module
from sonora.audio import cuesheet as cuesheet_module
from sonora.audio import metadata as metadata_module
from sonora.audio import replaygain as replaygain_module
from sonora.core import artists as artists_module
from sonora.core.artists import harmonize_artist_casing
from sonora.core.cache import api_cache_session
from sonora.core.config import clear_config_cache, get_config
from sonora.core.logger import (
    LOG,
    create_progress,
    interactive_pause_listener,
    resume_workers,
    wait_if_paused,
)
from sonora.core.models import TrackInfo
from sonora.core.state import get_library_state, tagging_state_status
from sonora.core.text import (
    is_noise_genre,
    is_valid_uuid,
    normalize_genre,
    normalize_str,
    preserve_unicode_repertoire,
)
from sonora.core.utils import (
    InterruptedOperationError,
    find_audio_files,
    get_album_root_directory,
    group_files_by_album_root,
    is_interruption,
)
from sonora.modules.catalog_enrichment import (
    _NETWORK_EXCEPTIONS,
    _enrich_deezer,
    _enrich_discogs,
    _enrich_genius,
    _enrich_itunes,
    _enrich_lastfm,
    _enrich_shazam,
    _enrich_theaudiodb,
)
from sonora.modules.matching import (
    _extract_known_album_entities,
    _is_generic,
    _is_generic_title,
    _resolve_album_track_position,
    is_alien_album_track,
)
from sonora.modules.musicbrainz_enrichment import _enrich_acoustid, _enrich_musicbrainz
from sonora.modules.track_processing import (
    _record_persisted_album,
    _run_audio_and_tags_persistence,
)
from sonora.services import deezer as deezer_service
from sonora.services import discogs as discogs_service
from sonora.services import itunes as itunes_service
from sonora.services import musicbrainz as musicbrainz_service


def _heal_and_align_album_track(
    track_info: TrackInfo,
    file_path: Path,
    album_mb_release_details: dict[str, Any] | None,
    album_deezer_details: dict[str, Any] | None,
    album_itunes_details: dict[str, Any] | None,
    album_track_mbids: dict[Any, str] | None,
    target_album_artist: str | None,
    target_album_title: str | None,
    force: bool,
) -> None:
    """Aligns track position and heals corrupted track artist/album identity from release context."""
    resolved_pos = _resolve_album_track_position(
        track_info=track_info,
        file_path=file_path,
        album_mb_release_details=album_mb_release_details,
        album_deezer_details=album_deezer_details,
        album_itunes_details=album_itunes_details,
        album_track_mbids=album_track_mbids,
    )
    if resolved_pos is not None and (
        force
        or track_info.track_number is None
        or track_info.track_number != resolved_pos
    ):
        track_info.track_number = resolved_pos

    if (
        target_album_artist
        and not _is_generic(target_album_artist, "artist")
        and (not track_info.album_artist or force)
    ):
        track_info.album_artist = preserve_unicode_repertoire(
            track_info.album_artist, target_album_artist
        )

    known_album_entities = _extract_known_album_entities(
        None,
        None,
        [album_mb_release_details, album_deezer_details, album_itunes_details],
    )
    known_album_track_artists = {
        normalize_str(artist) for artist in known_album_entities.artists
    }

    is_legitimate_track_artist = False
    norm_cur = normalize_str(track_info.artist) if track_info.artist else ""
    if (
        norm_cur
        and known_album_track_artists
        and (
            norm_cur in known_album_track_artists
            or any(
                fuzz.ratio(norm_cur, a) >= get_config().artist_similarity_threshold
                or a in norm_cur
                or norm_cur in a
                for a in known_album_track_artists
            )
        )
    ):
        is_legitimate_track_artist = True

    if (
        not is_legitimate_track_artist
        and target_album_artist
        and not _is_generic(target_album_artist, "artist")
        and track_info.artist
    ):
        norm_tgt = normalize_str(target_album_artist)
        if (
            norm_tgt != norm_cur
            and norm_tgt not in norm_cur
            and norm_cur not in norm_tgt
            and fuzz.ratio(norm_tgt, norm_cur) < 50
        ):
            LOG.info(
                f"   ∟ 🩹 [Healer] Healing corrupted artist '[bold red]{escape(track_info.artist)}[/]' -> '[bold green]{escape(target_album_artist)}[/]'"
            )
            track_info.artist = preserve_unicode_repertoire(
                track_info.artist, target_album_artist
            )

    if (
        target_album_title
        and not _is_generic(target_album_title, "album")
        and (not track_info.album or _is_generic(track_info.album, "album") or force)
    ):
        track_info.album = preserve_unicode_repertoire(
            track_info.album, target_album_title
        )


def _run_metadata_enrichment_pipeline(
    track_info: TrackInfo,
    file_path: Path,
    album_mbid: str | None,
    album_track_mbids: dict[Any, str] | None,
    album_mb_release_details: dict[str, Any] | None,
    album_discogs_release: dict[str, Any] | None,
    album_deezer_details: dict[str, Any] | None,
    album_itunes_details: dict[str, Any] | None,
    target_album_artist: str | None,
    target_album_title: str | None,
    has_album_context: bool,
    lastfm_api_key: str | None,
    acoustid_api_key: str | None,
    discogs_user_token: str | None,
    genius_api_token: str | None,
    force: bool,
) -> None:
    """Sequentially enriches track metadata across external providers."""
    _enrich_musicbrainz(
        track_info,
        album_mbid,
        album_track_mbids,
        album_mb_release_details,
        target_album_artist=target_album_artist,
        force=force,
    )
    if (
        _is_generic_title(track_info.title)
        or _is_generic(track_info.artist, "artist")
        or (not has_album_context and (force or not track_info.musicbrainz_trackid))
    ):
        _enrich_shazam(
            track_info,
            file_path,
            target_album_artist=target_album_artist,
            target_album_title=target_album_title,
            has_album_context=has_album_context,
            force=force,
        )
    _enrich_acoustid(
        track_info,
        file_path,
        acoustid_api_key,
        album_track_mbids=album_track_mbids,
        force=force,
    )
    _enrich_itunes(track_info, album_itunes_details=album_itunes_details, force=force)
    _enrich_lastfm(track_info, lastfm_api_key)
    _enrich_discogs(
        track_info,
        discogs_user_token,
        album_discogs_release,
        has_album_context=has_album_context,
        force=force,
    )
    _enrich_deezer(
        track_info,
        album_deezer_details,
        has_album_context=has_album_context,
        force=force,
    )
    _enrich_genius(track_info, genius_api_token, force=force)
    _enrich_theaudiodb(track_info, force=force)


@api_cache_session()
def process_single_track(
    file_path: Path,
    fetch_bpm: bool = True,
    fetch_key: bool = True,
    fetch_lyrics: bool = True,
    fetch_itunes_art: bool = True,
    lastfm_api_key: str | None = None,
    acoustid_api_key: str | None = None,
    discogs_user_token: str | None = None,
    genius_api_token: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    album_mbid: str | None = None,
    album_track_mbids: dict[Any, str] | None = None,
    cuesheet_content: str | None = None,
    album_mb_release_details: dict[str, Any] | None = None,
    album_discogs_release: dict[str, Any] | None = None,
    album_deezer_details: dict[str, Any] | None = None,
    album_itunes_details: dict[str, Any] | None = None,
    album_cover_path: Path | None = None,
    target_album_artist: str | None = None,
    target_album_title: str | None = None,
) -> TrackInfo:
    wait_if_paused()
    LOG.start_buffering()
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        track_info = metadata_module.read_track_metadata(file_path)
        orig_info = dataclasses.replace(track_info)
        track_info.artist = artists_module.resolve_artist_name(track_info.artist)

        if not bool(
            album_track_mbids or album_mb_release_details or target_album_title
        ) and (
            _is_generic_title(track_info.title)
            or _is_generic(track_info.artist, "artist")
        ):
            _enrich_shazam(track_info, file_path, force=False)

        has_album_context = bool(
            album_track_mbids
            or album_mb_release_details
            or album_deezer_details
            or album_itunes_details
            or target_album_title
        )
        if has_album_context and is_alien_album_track(
            track_info=track_info,
            file_path=file_path,
            target_album_artist=target_album_artist,
            target_album_title=target_album_title,
            album_mb_release_details=album_mb_release_details,
            album_deezer_details=album_deezer_details,
            album_itunes_details=album_itunes_details,
        ):
            track_info.is_alien = True
            album_desc = (
                f"{target_album_artist} - {target_album_title}"
                if target_album_artist and target_album_title
                else (target_album_title or target_album_artist or "Album")
            )
            disp_artist = str(track_info.artist or "Unknown Artist")
            disp_title = str(track_info.title or file_path.stem)
            LOG.warning(
                f"⚠️  [bold yellow]Alien track detected in album folder:[/] [white]{escape(file_path.name)}[/]\n"
                f"   ∟ Found '[bold]{escape(disp_artist)} - {escape(disp_title)}'[/] inside '[bold]{escape(album_desc)}[/]'.\n"
                f"   ∟ [cyan]Shielding track from album metadata poisoning. Tagging independently as standalone track.[/]"
            )
            album_mbid = None
            album_track_mbids = None
            album_mb_release_details = None
            album_discogs_release = None
            album_deezer_details = None
            album_itunes_details = None
            album_cover_path = None
            cuesheet_content = None
            target_album_artist = None
            target_album_title = None
            has_album_context = False

        if has_album_context:
            _heal_and_align_album_track(
                track_info=track_info,
                file_path=file_path,
                album_mb_release_details=album_mb_release_details,
                album_deezer_details=album_deezer_details,
                album_itunes_details=album_itunes_details,
                album_track_mbids=album_track_mbids,
                target_album_artist=target_album_artist,
                target_album_title=target_album_title,
                force=force,
            )

        LOG.info(f"🎧 Processing track: [white]{escape(file_path.name)}[/]")

        _run_metadata_enrichment_pipeline(
            track_info=track_info,
            file_path=file_path,
            album_mbid=album_mbid,
            album_track_mbids=album_track_mbids,
            album_mb_release_details=album_mb_release_details,
            album_discogs_release=album_discogs_release,
            album_deezer_details=album_deezer_details,
            album_itunes_details=album_itunes_details,
            target_album_artist=target_album_artist,
            target_album_title=target_album_title,
            has_album_context=has_album_context,
            lastfm_api_key=lastfm_api_key,
            acoustid_api_key=acoustid_api_key,
            discogs_user_token=discogs_user_token,
            genius_api_token=genius_api_token,
            force=force,
        )

        return _run_audio_and_tags_persistence(
            track_info=track_info,
            orig_info=orig_info,
            file_path=file_path,
            cuesheet_content=cuesheet_content,
            album_cover_path=album_cover_path,
            fetch_bpm=fetch_bpm,
            fetch_key=fetch_key,
            fetch_lyrics=fetch_lyrics,
            fetch_itunes_art=fetch_itunes_art,
            force=force,
            dry_run=dry_run,
        )
    finally:
        LOG.stop_buffering()


def _resolve_album_folder_identity(
    album_dir: Path, audio_files: list[Path]
) -> tuple[str | None, str | None]:
    """
    Resolve canonical artist and album name for a directory by combining
    directory structure ("Artist - Album" or parent artist directory) with
    metadata consensus across audio files.
    """
    effective_dir = get_album_root_directory(album_dir)

    folder_name = effective_dir.name
    folder_artist: str | None = None
    folder_album: str | None = None

    if " - " in folder_name:
        parts = folder_name.split(" - ", 1)
        folder_artist, folder_album = parts[0].strip(), parts[1].strip()
    elif (
        effective_dir.parent
        and effective_dir.parent.name
        and not get_config().is_generic_container(effective_dir.parent.name)
    ):
        folder_artist = effective_dir.parent.name
        folder_album = folder_name

    if folder_artist and get_config().is_generic_container(folder_artist):
        folder_artist = None
    if folder_album and get_config().is_generic_container(folder_album):
        folder_album = None

    album_counts: dict[str, int] = {}
    artist_counts: dict[str, int] = {}

    for audio_file in audio_files[:10]:
        try:
            track_metadata = metadata_module.read_track_metadata(audio_file)
            if track_metadata.album and not get_config().is_generic_container(
                track_metadata.album
            ):
                album_counts[track_metadata.album] = (
                    album_counts.get(track_metadata.album, 0) + 1
                )
            candidate_artist = track_metadata.album_artist or track_metadata.artist
            if candidate_artist and not get_config().is_generic_container(
                candidate_artist
            ):
                artist_counts[candidate_artist] = (
                    artist_counts.get(candidate_artist, 0) + 1
                )
        except OSError:
            pass

    consensus_album = (
        max(album_counts, key=lambda k: album_counts[k]) if album_counts else None
    )
    consensus_artist = (
        max(artist_counts, key=lambda k: artist_counts[k]) if artist_counts else None
    )

    resolved_artist: str | None
    if folder_artist and consensus_artist:
        resolved_artist = preserve_unicode_repertoire(consensus_artist, folder_artist)
    else:
        resolved_artist = folder_artist or consensus_artist

    resolved_album: str | None
    if folder_album and consensus_album:
        resolved_album = preserve_unicode_repertoire(consensus_album, folder_album)
    else:
        resolved_album = folder_album or consensus_album

    return resolved_artist, resolved_album


@dataclasses.dataclass(slots=True)
class _AlbumMetadataContext:
    musicbrainz_id: str | None = None
    track_mbids: dict[Any, str] | None = None
    mb_release_details: dict[str, Any] | None = None
    discogs_release: dict[str, Any] | None = None
    deezer_details: dict[str, Any] | None = None
    itunes_details: dict[str, Any] | None = None
    cover_path: Path | None = None
    sample_artist: str | None = None
    sample_album: str | None = None


def _resolve_album_metadata_context(
    album_dir: Path,
    audio_files: list[Path],
    discogs_user_token: str | None,
    fetch_itunes_art: bool,
    force: bool,
    dry_run: bool,
    failures: list[dict[str, str]] | None = None,
) -> _AlbumMetadataContext:
    """Pre-fetches release-level metadata from external providers for an album folder."""
    album_context = _AlbumMetadataContext()
    try:
        sample_artist, sample_album = _resolve_album_folder_identity(
            album_dir, audio_files
        )
        album_context.sample_artist = sample_artist
        album_context.sample_album = sample_album
        if not (sample_artist and sample_album):
            return album_context

        existing_release_ids: Counter[str] = Counter()
        for audio_path in audio_files:
            try:
                existing_metadata = metadata_module.read_track_metadata(audio_path)
            except OSError:
                continue
            if is_valid_uuid(existing_metadata.musicbrainz_albumid):
                existing_release_ids[str(existing_metadata.musicbrainz_albumid)] += 1
        if (
            existing_release_ids
            and existing_release_ids.most_common(1)[0][1] > len(audio_files) / 2
        ):
            release_info: dict[str, object] | None = {
                "id": existing_release_ids.most_common(1)[0][0]
            }
        else:
            release_info = musicbrainz_service.search_musicbrainz_release(
                sample_artist, sample_album, expected_track_count=len(audio_files)
            )
        if release_info and release_info.get("id"):
            album_context.musicbrainz_id = str(release_info["id"])
            album_context.mb_release_details = (
                musicbrainz_service.fetch_musicbrainz_release_details(
                    album_context.musicbrainz_id
                )
            )
            if album_context.mb_release_details:
                raw_tracks = album_context.mb_release_details.get("tracks_by_position")
                tracks_by_disc = album_context.mb_release_details.get(
                    "tracks_by_disc_and_position"
                )
                track_mbids: dict[Any, str] = {}
                if isinstance(tracks_by_disc, dict):
                    for d_pos, recording in tracks_by_disc.items():
                        if isinstance(recording, dict) and recording.get(
                            "recording_mbid"
                        ):
                            track_mbids[d_pos] = str(recording["recording_mbid"])
                if isinstance(raw_tracks, dict):
                    for pos, recording in raw_tracks.items():
                        if isinstance(recording, dict) and recording.get(
                            "recording_mbid"
                        ):
                            track_mbids[pos] = str(recording["recording_mbid"])
                album_context.track_mbids = track_mbids
            if not album_context.track_mbids:
                album_context.track_mbids = musicbrainz_service.fetch_album_track_mbids(
                    album_context.musicbrainz_id
                )

        album_context.deezer_details = deezer_service.fetch_deezer_album_details(
            sample_artist, sample_album, expected_track_count=len(audio_files)
        )
        album_context.itunes_details = itunes_service.fetch_itunes_album_details(
            sample_artist, sample_album, expected_track_count=len(audio_files)
        )
        if discogs_user_token:
            album_context.discogs_release = discogs_service.search_discogs_release(
                sample_artist,
                sample_album,
                user_token=discogs_user_token,
                expected_track_count=len(audio_files),
            )
        if fetch_itunes_art:
            album_context.cover_path = art_module.process_album_cover_art(
                album_dir,
                sample_artist,
                sample_album,
                musicbrainz_album_id=album_context.musicbrainz_id,
                musicbrainz_releasegroup_id=(
                    str(album_context.mb_release_details["musicbrainz_releasegroupid"])
                    if album_context.mb_release_details
                    and album_context.mb_release_details.get(
                        "musicbrainz_releasegroupid"
                    )
                    else None
                ),
                force=force,
                dry_run=dry_run,
            )
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"Pre-fetching album metadata failed: {error}")

    return album_context


def _harmonize_album_genres(
    valid_tracks: list[TrackInfo],
    album_deezer_details: dict[str, Any] | None,
    album_itunes_details: dict[str, Any] | None,
    dry_run: bool = False,
) -> None:
    """Harmonizes outlier/noise genres across valid album tracks to the consensus genre."""
    if len(valid_tracks) < 2:
        return

    album_level_genre = None
    if album_deezer_details and album_deezer_details.get("genre"):
        album_level_genre = normalize_genre(str(album_deezer_details["genre"]))
    elif album_itunes_details and album_itunes_details.get("genre"):
        album_level_genre = normalize_genre(str(album_itunes_details["genre"]))

    genre_counts: dict[str, int] = {}
    for vt in valid_tracks:
        if vt.genre and not is_noise_genre(vt.genre):
            genre_counts[vt.genre] = genre_counts.get(vt.genre, 0) + 1

    dominant_genre = album_level_genre
    if not dominant_genre and genre_counts:
        top_genre, top_count = max(genre_counts.items(), key=lambda x: x[1])
        if top_count * 2 >= len(valid_tracks):
            dominant_genre = top_genre

    if not dominant_genre:
        return

    for vt in valid_tracks:
        if (not vt.genre or is_noise_genre(vt.genre)) and vt.genre != dominant_genre:
            LOG.info(
                f"   ∟ 🏷️ [Consensus] Harmonizing genre '[bold red]{escape(str(vt.genre))}[/]' -> '[bold green]{escape(dominant_genre)}[/]' on '{escape(vt.file_path.name)}'"
            )
            vt.genre = dominant_genre
            if not dry_run:
                try:
                    metadata_module.write_track_metadata(vt)
                except OSError as err:
                    LOG.debug(f"Failed to save harmonized genre: {err}")


def _harmonize_album_artists(
    valid_tracks: list[TrackInfo], dry_run: bool = False
) -> None:
    """Harmonizes album artist and track artist casing across valid album tracks."""
    if len(valid_tracks) < 2:
        return

    album_artist_counts: dict[str, int] = {}
    for vt in valid_tracks:
        if vt.album_artist and vt.album_artist.strip():
            art = vt.album_artist.strip()
            album_artist_counts[art] = album_artist_counts.get(art, 0) + 1

    dominant_album_artist: str | None = None
    if album_artist_counts:
        groups: dict[str, dict[str, int]] = {}
        for cand, count in album_artist_counts.items():
            norm = normalize_str(cand)
            groups.setdefault(norm, {})[cand] = count
        dominant_norm = max(groups.keys(), key=lambda k: sum(groups[k].values()))
        rep_artist = max(groups[dominant_norm].items(), key=lambda x: x[1])[0]
        resolved = artists_module.resolve_artist_name(rep_artist, allow_network=False)
        dominant_album_artist = (
            resolved if normalize_str(resolved) == dominant_norm else rep_artist
        )

    track_artist_counts: dict[str, dict[str, int]] = {}
    for vt in valid_tracks:
        if vt.artist and vt.artist.strip():
            art = vt.artist.strip()
            norm = normalize_str(art)
            track_artist_counts.setdefault(norm, {})[art] = (
                track_artist_counts.setdefault(norm, {}).get(art, 0) + 1
            )

    canonical_track_artists: dict[str, str] = {}
    for norm, casing_map in track_artist_counts.items():
        dom_casing = max(casing_map.items(), key=lambda x: x[1])[0]
        resolved = artists_module.resolve_artist_name(dom_casing, allow_network=False)
        canonical_track_artists[norm] = (
            resolved if normalize_str(resolved) == norm else dom_casing
        )

    if dominant_album_artist:
        norm_dom = normalize_str(dominant_album_artist)
        if norm_dom in canonical_track_artists:
            cand_trk = canonical_track_artists[norm_dom]
            harmonized = harmonize_artist_casing(cand_trk, dominant_album_artist)
            dominant_album_artist = harmonized
            canonical_track_artists[norm_dom] = harmonized
        else:
            canonical_track_artists[norm_dom] = dominant_album_artist

    for vt in valid_tracks:
        modified = False
        if (
            dominant_album_artist
            and vt.album_artist
            and normalize_str(vt.album_artist) == normalize_str(dominant_album_artist)
            and vt.album_artist != dominant_album_artist
        ):
            LOG.info(
                f"   ∟ 🏷️ [Consensus] Harmonizing album artist casing '[bold red]{escape(vt.album_artist)}[/]' -> '[bold green]{escape(dominant_album_artist)}[/]' on '{escape(vt.file_path.name)}'"
            )
            vt.album_artist = dominant_album_artist
            modified = True

        if vt.artist:
            norm_art = normalize_str(vt.artist)
            if norm_art in canonical_track_artists:
                canonical_name = canonical_track_artists[norm_art]
                if vt.artist != canonical_name:
                    LOG.info(
                        f"   ∟ 🏷️ [Consensus] Harmonizing track artist casing '[bold red]{escape(vt.artist)}[/]' -> '[bold green]{escape(canonical_name)}[/]' on '{escape(vt.file_path.name)}'"
                    )
                    vt.artist = canonical_name
                    modified = True

        if (
            vt.album_artist
            and vt.artist
            and normalize_str(vt.album_artist) == normalize_str(vt.artist)
            and vt.album_artist != vt.artist
        ):
            target = harmonize_artist_casing(vt.artist, vt.album_artist)
            vt.album_artist = target
            vt.artist = target
            modified = True

        if modified and not dry_run:
            try:
                metadata_module.write_track_metadata(vt)
            except OSError as write_error:
                LOG.debug(f"Failed to save harmonized artist metadata: {write_error}")


def _finalize_album_assets_and_replaygain(
    album_dir: Path,
    current_album_results: list[TrackInfo],
    valid_tracks: list[TrackInfo],
    sample_artist: str | None,
    album_mb_release_details: dict[str, Any] | None,
    fetch_replaygain: bool,
    force: bool,
    dry_run: bool,
    max_threads: int,
    fetch_artwork: bool = True,
) -> None:
    """Calculates album ReplayGain and downloads companion artist and label assets."""
    valid_album_files = [t.file_path for t in valid_tracks]
    if fetch_replaygain and valid_album_files:
        wait_if_paused()
        replaygain_module.calculate_album_replaygain(
            valid_album_files,
            force=force,
            dry_run=dry_run,
            max_threads=max_threads,
        )

    if not current_album_results or not fetch_artwork:
        return

    wait_if_paused()
    rep_track = valid_tracks[0] if valid_tracks else current_album_results[0]
    primary_artist = sample_artist or rep_track.album_artist or rep_track.artist
    try:
        art_module.process_artist_artwork(
            album_dir,
            primary_artist,
            artist_mbid=rep_track.musicbrainz_artistid,
            dry_run=dry_run,
        )
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"Artist art download failed: {error}")

    try:
        label_mbid = (
            str(album_mb_release_details.get("label_mbid"))
            if album_mb_release_details and album_mb_release_details.get("label_mbid")
            else None
        )
        label_name = (
            str(album_mb_release_details.get("label"))
            if album_mb_release_details and album_mb_release_details.get("label")
            else rep_track.label
        )
        if label_mbid:
            art_module.process_label_artwork(
                album_dir,
                label_mbid=label_mbid,
                label_name=label_name,
                dry_run=dry_run,
            )
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"Label logo download failed: {error}")


def tag_album_folder(
    folder_path: Path,
    max_threads: int = 4,
    fetch_bpm: bool = True,
    fetch_key: bool = True,
    fetch_replaygain: bool = True,
    fetch_lyrics: bool = True,
    fetch_itunes_art: bool = True,
    lastfm_api_key: str | None = None,
    acoustid_api_key: str | None = None,
    discogs_user_token: str | None = None,
    genius_api_token: str | None = None,
    fanart_api_key: str | None = None,
    fanart_client_key: str | None = None,
    enable_shazam: bool | None = None,
    force: bool = False,
    dry_run: bool = False,
    failures: list[dict[str, str]] | None = None,
) -> list[TrackInfo]:
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Album folder not found: {folder_path}")

    if fanart_api_key:
        os.environ["FANART_API_KEY"] = fanart_api_key
        clear_config_cache()
    if fanart_client_key:
        os.environ["FANART_CLIENT_KEY"] = fanart_client_key
        clear_config_cache()
    if enable_shazam is not None:
        os.environ["ENABLE_SHAZAM"] = str(enable_shazam).lower()
        clear_config_cache()

    all_audio_files = find_audio_files(folder_path, recursive=True)
    if not all_audio_files:
        return []

    state_vault = get_library_state()
    state_status = tagging_state_status(
        get_config(),
        fetch_bpm=fetch_bpm,
        fetch_key=fetch_key,
        fetch_replaygain=fetch_replaygain,
        fetch_lyrics=fetch_lyrics,
        fetch_itunes_art=fetch_itunes_art,
    )
    outdated_files = set(
        all_audio_files
        if force
        else state_vault.filter_outdated_tracks(
            all_audio_files, expected_status=state_status
        )
    )
    album_groups = group_files_by_album_root(all_audio_files)

    results: list[TrackInfo] = []
    current_album_results: list[TrackInfo] = []

    with create_progress() as progress:
        task = progress.add_task("[cyan]Tagging tracks...", total=len(all_audio_files))

        with (
            interactive_pause_listener(progress, task) as pause_stop_event,
            ThreadPoolExecutor(max_workers=max_threads) as executor,
        ):
            try:
                for album_dir, audio_files in album_groups.items():
                    wait_if_paused()
                    if not any(
                        audio_path in outdated_files for audio_path in audio_files
                    ):
                        try:
                            results.extend(
                                [
                                    metadata_module.read_track_metadata(audio_path)
                                    for audio_path in audio_files
                                ]
                            )
                        except OSError:
                            outdated_files.update(audio_files)
                        else:
                            progress.advance(task, len(audio_files))
                            continue
                    folder_name = album_dir.name
                    LOG.force_info(
                        f"📁 [bold cyan]Album:[/] [white]{escape(folder_name)}[/] [dim]({len(audio_files)} tracks)[/]"
                    )

                    companion_cue = cuesheet_module.find_companion_cuesheet(album_dir)
                    album_cue_content = (
                        cuesheet_module.read_cuesheet_content(companion_cue)
                        if companion_cue
                        else None
                    )

                    album_context = _resolve_album_metadata_context(
                        album_dir,
                        audio_files,
                        discogs_user_token=discogs_user_token,
                        fetch_itunes_art=fetch_itunes_art,
                        force=force,
                        dry_run=dry_run,
                        failures=failures,
                    )

                    current_album_results = []
                    future_to_file = {
                        executor.submit(
                            process_single_track,
                            file_path=audio_file,
                            fetch_bpm=fetch_bpm,
                            fetch_key=fetch_key,
                            fetch_lyrics=fetch_lyrics,
                            fetch_itunes_art=fetch_itunes_art,
                            lastfm_api_key=lastfm_api_key,
                            acoustid_api_key=acoustid_api_key,
                            discogs_user_token=discogs_user_token,
                            genius_api_token=genius_api_token,
                            force=force,
                            dry_run=dry_run,
                            album_mbid=album_context.musicbrainz_id,
                            album_track_mbids=album_context.track_mbids,
                            cuesheet_content=album_cue_content,
                            album_mb_release_details=album_context.mb_release_details,
                            album_discogs_release=album_context.discogs_release,
                            album_deezer_details=album_context.deezer_details,
                            album_itunes_details=album_context.itunes_details,
                            album_cover_path=album_context.cover_path,
                            target_album_artist=album_context.sample_artist,
                            target_album_title=album_context.sample_album,
                        ): audio_file
                        for audio_file in audio_files
                    }

                    for future in as_completed(future_to_file):
                        wait_if_paused()
                        audio_file = future_to_file[future]
                        try:
                            track_info = future.result()
                            current_album_results.append(track_info)
                        except _NETWORK_EXCEPTIONS as error:
                            LOG.warning(
                                f"Failed to process {escape(audio_file.name)}: {escape(str(error))}"
                            )
                            if failures is not None:
                                failures.append(
                                    {
                                        "file": str(audio_file.resolve()),
                                        "filename": audio_file.name,
                                        "error": str(error),
                                        "error_type": type(error).__name__,
                                    }
                                )
                        progress.advance(task)

                    valid_tracks = [t for t in current_album_results if not t.is_alien]
                    _harmonize_album_genres(
                        valid_tracks,
                        album_context.deezer_details,
                        album_context.itunes_details,
                        dry_run=dry_run,
                    )
                    _harmonize_album_artists(valid_tracks, dry_run=dry_run)
                    _finalize_album_assets_and_replaygain(
                        album_dir,
                        current_album_results,
                        valid_tracks,
                        album_context.sample_artist,
                        album_context.mb_release_details,
                        fetch_replaygain=fetch_replaygain,
                        force=force,
                        dry_run=dry_run,
                        max_threads=max_threads,
                        fetch_artwork=fetch_itunes_art,
                    )
                    if not dry_run:
                        current_album_results = _record_persisted_album(
                            current_album_results,
                            state_vault,
                            state_status,
                            fetch_bpm=fetch_bpm,
                            fetch_key=fetch_key,
                            fetch_replaygain=fetch_replaygain,
                            fetch_lyrics=fetch_lyrics,
                            fetch_itunes_art=fetch_itunes_art,
                        )
                    results.extend(current_album_results)
                    current_album_results = []
                    future_to_file.clear()
            except (KeyboardInterrupt, RuntimeError) as exc:
                if not is_interruption(exc):
                    raise
                resume_workers(pause_stop_event)
                executor.shutdown(wait=True, cancel_futures=True)
                results.extend(current_album_results)
                raise InterruptedOperationError(results) from None

    return results
