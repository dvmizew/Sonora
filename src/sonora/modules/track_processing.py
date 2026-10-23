import dataclasses
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx
from PIL import Image
from rich.markup import escape

from sonora.audio import art as art_module
from sonora.audio import bpm as bpm_module
from sonora.audio import cuesheet as cuesheet_module
from sonora.audio import key as key_module
from sonora.audio import metadata as metadata_module
from sonora.audio import replaygain as replaygain_module
from sonora.core.audio_files import find_audio_files
from sonora.core.cache import api_cache_session
from sonora.core.cancellation import InterruptedOperationError, is_interruption
from sonora.core.config import get_config
from sonora.core.constants import MIN_COVER_ART_DIMENSION
from sonora.core.files import atomic_write_bytes
from sonora.core.library_paths import group_files_by_album_root
from sonora.core.logger import (
    LOG,
    create_progress,
    interactive_pause_listener,
    resume_workers,
    wait_if_paused,
)
from sonora.core.models import NormalizeReport, TrackInfo
from sonora.core.sidecars import (
    extract_max_timestamp,
    get_lyrics_quality,
    is_lyrics_duration_valid,
    lyrics_identity_matches,
    lyrics_word_body,
    repair_lyrics_headers,
    strip_lyrics_headers,
)
from sonora.core.state import LibraryStateVault
from sonora.modules.catalog_enrichment import _NETWORK_EXCEPTIONS
from sonora.modules.normalization import normalize_track_metadata
from sonora.services import lyrics as lyrics_service

_SKIP_DIFF_FIELDS: frozenset[str] = frozenset(
    {
        "file_path",
        "synced_lyrics",
        "sample_rate",
        "bitrate",
        "channels",
        "is_lossless",
        "art_width",
        "art_height",
        "is_alien",
        "metadata_sources",
        "metadata_conflicts",
    }
)


def _enrich_cuesheet(
    track_info: TrackInfo,
    file_path: Path,
    cuesheet_content: str | None,
) -> None:
    if cuesheet_content:
        track_info.cuesheet = cuesheet_content
    elif not track_info.cuesheet:
        companion_cue = cuesheet_module.find_companion_cuesheet(file_path.parent)
        if companion_cue:
            track_info.cuesheet = cuesheet_module.read_cuesheet_content(companion_cue)


def _enrich_bpm(
    track_info: TrackInfo,
    file_path: Path,
    fetch_bpm: bool,
    force: bool = False,
) -> None:
    if fetch_bpm and (track_info.bpm is None or force):
        try:
            bpm = bpm_module.calculate_bpm(file_path)
            if bpm is not None:
                track_info.bpm = bpm
                LOG.detail(f"   ∟ 🎵 BPM Calculated: [green]{bpm}[/]")
        except OSError as error:
            LOG.debug(f"BPM calculation failed for {track_info.title}: {error}")


def _enrich_key(
    track_info: TrackInfo,
    file_path: Path,
    fetch_key: bool,
    force: bool = False,
) -> None:
    is_key_invalid = bool(
        track_info.initial_key
        and key_module.key_to_camelot(track_info.initial_key) is None
    )
    if fetch_key and (track_info.initial_key is None or force or is_key_invalid):
        try:
            detected_key = key_module.detect_musical_key(file_path)
            if detected_key is not None:
                track_info.initial_key = detected_key
                camelot = key_module.key_to_camelot(detected_key) or "Unknown"
                LOG.detail(
                    f"   ∟ 🎵 Musical Key: [green]{escape(detected_key)}[/] ({escape(camelot)})"
                )
        except OSError as error:
            LOG.debug(f"Key calculation failed for {track_info.title}: {error}")


def _enrich_artwork(
    track_info: TrackInfo,
    file_path: Path,
    fetch_itunes_art: bool,
    force: bool = False,
    dry_run: bool = False,
) -> Path | None:
    if not fetch_itunes_art:
        return None
    try:
        return art_module.process_album_cover_art(
            file_path.parent,
            track_info.artist,
            track_info.album,
            musicbrainz_album_id=track_info.musicbrainz_albumid,
            musicbrainz_releasegroup_id=track_info.musicbrainz_releasegroupid,
            force=force,
            dry_run=dry_run,
        )
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"Cover art downloading failed for {track_info.title}: {error}")
        return None


def _enrich_lyrics(
    track_info: TrackInfo,
    file_path: Path,
    fetch_lyrics: bool,
    force: bool = False,
    dry_run: bool = False,
    musixmatch_token: str | None = None,
) -> None:
    if not fetch_lyrics:
        return
    try:
        lrc_path = file_path.with_suffix(".lrc")
        had_lrc = lrc_path.exists() and lrc_path.stat().st_size > 0
        resolved_duration = metadata_module.get_audio_duration(file_path)
        local_lyrics = [track_info.lyrics, track_info.synced_lyrics]
        if had_lrc:
            try:
                local_lyrics.append(lrc_path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeError):
                track_info.metadata_conflicts["lyrics"] = (
                    "Existing lyrics cannot be read as UTF-8; review before reuse."
                )
        if any(
            lyric_text
            and (
                not lyrics_identity_matches(
                    lyric_text, track_info.artist, track_info.title
                )
                or not is_lyrics_duration_valid(lyric_text, resolved_duration)
            )
            for lyric_text in local_lyrics
        ):
            track_info.metadata_conflicts["lyrics"] = (
                "Existing lyrics headers or timestamps conflict with the recording; review required."
            )
        lyrics_text, tag_type = process_track_lyrics(
            file_path,
            track_info.artist,
            track_info.title,
            force=force,
            dry_run=dry_run,
            isrc=track_info.isrc,
            album_name=track_info.album,
            duration=resolved_duration,
            existing_lyrics=track_info.lyrics,
            musixmatch_token=musixmatch_token,
        )
        if lyrics_text and tag_type:
            track_info.lyrics = lyrics_text
            track_info.metadata_conflicts.pop("lyrics", None)
            if not had_lrc:
                LOG.detail(
                    f"   ∟ [green]✅ Saved {tag_type} lyrics for {escape(file_path.name)}[/]"
                )
            elif force:
                LOG.detail(
                    f"   ∟ [yellow]🔄 Updated {tag_type} lyrics for {escape(file_path.name)}[/]"
                )
    except _NETWORK_EXCEPTIONS as error:
        LOG.debug(f"Lyrics fetch failed for {track_info.title}: {error}")


def _compute_tag_diffs(
    original_metadata: TrackInfo, track_info: TrackInfo
) -> list[tuple[str, Any, Any]]:
    diff_entries: list[tuple[str, Any, Any]] = []
    for metadata_field in dataclasses.fields(TrackInfo):
        if metadata_field.name in _SKIP_DIFF_FIELDS or metadata_field.name.startswith(
            "_"
        ):
            continue
        previous_tag_value = getattr(original_metadata, metadata_field.name)
        updated_tag_value = getattr(track_info, metadata_field.name)
        old_clean = (
            None if previous_tag_value in (None, "", [], ()) else previous_tag_value
        )
        new_clean = (
            None if updated_tag_value in (None, "", [], ()) else updated_tag_value
        )
        if old_clean != new_clean:
            diff_entries.append((metadata_field.name, old_clean, new_clean))
    if original_metadata._advisory_needs_normalization and not any(
        field == "advisory" for field, _, _ in diff_entries
    ):
        diff_entries.append(
            ("advisory", "noncanonical tag encoding", track_info.advisory)
        )
    return diff_entries


def _describe_tag_diffs(diff_entries: list[tuple[str, Any, Any]]) -> list[str]:
    descriptions: list[str] = []
    for field_name, old_value, new_value in diff_entries:
        if field_name == "lyrics":
            old_value = f"{len(str(old_value or ''))} characters"
            new_value = f"{len(str(new_value or ''))} characters"
        if old_value is None:
            descriptions.append(f"{field_name}: added {new_value!r}")
        elif new_value is None:
            descriptions.append(f"{field_name}: removed {old_value!r}")
        else:
            descriptions.append(f"{field_name}: {old_value!r} -> {new_value!r}")
    return descriptions


def _render_tag_diffs(diff_entries: list[tuple[str, Any, Any]]) -> list[str]:
    diff_lines: list[str] = []
    for field_name, old_clean, new_clean in diff_entries:
        if old_clean is None:
            color, change_symbol = "green", "+"
        elif new_clean is None:
            color, change_symbol = "red", "-"
        else:
            color, change_symbol = "yellow", "*"
        if field_name == "lyrics":
            old_clean = f"{len(str(old_clean or ''))} characters"
            new_clean = f"{len(str(new_clean or ''))} characters"
        diff_lines.append(
            f"\n       [{color}][{change_symbol}] {field_name}: {escape(str(old_clean))} -> {escape(str(new_clean))}[/]"
        )
    return diff_lines


def _process_audio_and_save_tags(
    track_info: TrackInfo,
    original_metadata: TrackInfo,
    file_path: Path,
    cuesheet_content: str | None,
    album_cover_path: Path | None,
    fetch_bpm: bool,
    fetch_key: bool,
    fetch_lyrics: bool,
    fetch_itunes_art: bool,
    force: bool,
    dry_run: bool,
    musixmatch_token: str | None = None,
) -> TrackInfo:
    """Computes audio DSP features, fetches lyrics/artwork, and writes updated tags."""
    _enrich_cuesheet(track_info, file_path, cuesheet_content)
    _enrich_bpm(track_info, file_path, fetch_bpm, force=force)
    _enrich_key(track_info, file_path, fetch_key, force=force)
    cover_image = None
    if not track_info.is_alien:
        cover_image = (
            album_cover_path
            if album_cover_path and album_cover_path.exists()
            else _enrich_artwork(
                track_info, file_path, fetch_itunes_art, force=force, dry_run=dry_run
            )
        )
    _enrich_lyrics(
        track_info,
        file_path,
        fetch_lyrics,
        force=force,
        dry_run=dry_run,
        musixmatch_token=musixmatch_token,
    )
    normalize_track_metadata(track_info)

    has_art_upgrade = False
    if cover_image and cover_image.exists():
        with Image.open(cover_image) as candidate_cover:
            candidate_cover.load()
            candidate_width, candidate_height = candidate_cover.size
        embedded_area = (original_metadata.art_width or 0) * (
            original_metadata.art_height or 0
        )
        has_art_upgrade = candidate_width * candidate_height > embedded_area
    diff_entries = _compute_tag_diffs(original_metadata, track_info)
    diff_lines = _render_tag_diffs(diff_entries)
    track_info._diff_descriptions = _describe_tag_diffs(diff_entries)
    if has_art_upgrade:
        previous_artwork_dimensions = (
            f"{original_metadata.art_width}x{original_metadata.art_height}"
            if original_metadata.art_width and original_metadata.art_height
            else "None"
        )
        diff_lines.append(
            f"\n       [green][+] cover_art: upgraded to high-resolution (was {previous_artwork_dimensions})[/]"
        )
        track_info._diff_descriptions.append(
            f"cover_art: upgraded from {previous_artwork_dimensions}"
        )

    if diff_lines or force or has_art_upgrade:
        if not dry_run:
            metadata_module.write_track_metadata(track_info, cover_art_path=cover_image)
            LOG.detail(
                f"   ∟ [green]✓[/] {escape(file_path.name)}: {len(diff_lines)} tag(s) updated.{''.join(diff_lines)}"
            )
        else:
            LOG.detail(f"   ∟ [DRY-RUN] {escape(file_path.name)}{''.join(diff_lines)}")
    else:
        LOG.detail(
            f"   ∟ [bold dim]✨ SKIPPED:[/] [dim]{escape(file_path.name)}[/] [dim]tags are unchanged.[/]"
        )

    return track_info


@api_cache_session()
def normalize_single_track(
    file_path: Path,
    fetch_bpm: bool = False,
    fetch_key: bool = False,
    force: bool = False,
    dry_run: bool = False,
) -> TrackInfo | None:
    """
    Locally cleans and normalizes metadata for a single audio file without any API requests.
    - Repairs mojibake/UTF-8 encoding via ftfy
    - Strips bracket junk: (Official Video), [FLAC], [320kbps], (2011 Remaster), [Explicit]
    - Cleans disambiguation suffixes: 'Armin (ROU)' -> 'Armin'
    - Canonicalizes genres via normalize_genre
    - Standardizes dates via normalize_date
    - Optionally calculates audio BPM locally
    - Optionally detects musical key locally
    """
    if not file_path.exists():
        return None

    try:
        current_info = metadata_module.read_track_metadata(file_path)
    except OSError as error:
        LOG.debug(f"Failed to read metadata for {file_path}: {error}")
        return None

    original_metadata = current_info.copy()

    _enrich_bpm(current_info, file_path, fetch_bpm, force=force)
    _enrich_key(current_info, file_path, fetch_key, force=force)

    normalize_track_metadata(current_info)

    diff_entries = _compute_tag_diffs(original_metadata, current_info)
    diff_descriptions = _describe_tag_diffs(diff_entries)

    if diff_entries or force:
        rendered_diffs = _render_tag_diffs(diff_entries)
        if not dry_run:
            try:
                metadata_module.write_track_metadata(current_info)
            except OSError as error:
                LOG.warning(
                    f"Failed to save normalized tags for {escape(file_path.name)}: {escape(str(error))}"
                )
                return None
            LOG.detail(
                f"   ∟ [green]✓[/] {escape(file_path.name)}: {len(diff_entries)} tag(s) normalized.{''.join(rendered_diffs)}"
            )
        else:
            LOG.detail(
                f"   ∟ [DRY-RUN] {escape(file_path.name)}: {len(diff_entries)} tag(s) to normalize.{''.join(rendered_diffs)}"
            )
    else:
        LOG.debug(f"Track {escape(file_path.name)} is already normalized.")

    current_info._diff_descriptions = diff_descriptions
    return current_info


def normalize_library(
    directory: Path,
    fetch_bpm: bool = False,
    fetch_key: bool = False,
    fetch_replaygain: bool = False,
    force: bool = False,
    dry_run: bool = False,
    max_threads: int = 4,
) -> NormalizeReport:
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Directory not found: {directory}")

    LOG.info(f"Scanning for audio files in {escape(str(directory))}...")
    audio_files = find_audio_files(directory, recursive=True)
    if not audio_files:
        LOG.warning("No audio files found to normalize.")
        return NormalizeReport([])

    album_groups = group_files_by_album_root(audio_files)
    results: list[TrackInfo] = []
    modified_files: dict[str, list[str]] = {}

    with create_progress() as progress:
        task = progress.add_task(
            "[cyan]Normalizing tracks (offline)...", total=len(audio_files)
        )
        with (
            interactive_pause_listener(progress, task) as pause_stop_event,
            ThreadPoolExecutor(max_workers=max_threads) as executor,
        ):
            try:
                for files in album_groups.values():
                    wait_if_paused()
                    futures = {
                        executor.submit(
                            normalize_single_track,
                            file_path=f,
                            fetch_bpm=fetch_bpm,
                            fetch_key=fetch_key,
                            force=force,
                            dry_run=dry_run,
                        ): f
                        for f in files
                    }
                    for future in as_completed(futures):
                        wait_if_paused()
                        worker_track = future.result()
                        if worker_track is not None:
                            results.append(worker_track)
                            if worker_track._diff_descriptions:
                                modified_files[str(futures[future])] = (
                                    worker_track._diff_descriptions
                                )
                        progress.advance(task)

                    if fetch_replaygain:
                        wait_if_paused()
                        replaygain_module.calculate_album_replaygain(
                            files,
                            force=force,
                            dry_run=dry_run,
                            max_threads=max_threads,
                        )
                    futures.clear()
            except (KeyboardInterrupt, RuntimeError) as exc:
                if not is_interruption(exc):
                    raise
                resume_workers(pause_stop_event)
                executor.shutdown(wait=True, cancel_futures=True)
                partial_report = NormalizeReport(results, modified_files=modified_files)
                raise InterruptedOperationError(partial_report) from None

    return NormalizeReport(results, modified_files=modified_files)


def _record_persisted_album(
    tracks: list[TrackInfo],
    state_vault: LibraryStateVault,
    state_status: str,
    *,
    fetch_bpm: bool,
    fetch_key: bool,
    fetch_replaygain: bool,
    fetch_lyrics: bool,
    fetch_itunes_art: bool,
) -> list[TrackInfo]:
    persisted_tracks = [
        dataclasses.replace(
            metadata_module.read_track_metadata(track.file_path),
            is_alien=track.is_alien,
            metadata_sources=dict(track.metadata_sources),
            metadata_conflicts=dict(track.metadata_conflicts),
            _diff_descriptions=list(track._diff_descriptions),
        )
        for track in tracks
    ]
    state_vault.record_tracks_state_batch(
        [
            track.file_path
            for track in persisted_tracks
            if track.metadata_sources
            and not track.metadata_conflicts
            and (
                not fetch_itunes_art
                or (track.art_width or 0) >= MIN_COVER_ART_DIMENSION
                and (track.art_height or 0) >= MIN_COVER_ART_DIMENSION
            )
            and (not fetch_bpm or track.bpm is not None)
            and (not fetch_key or track.initial_key is not None)
            and (not fetch_replaygain or track.replaygain_track_gain is not None)
            and (not fetch_lyrics or track.lyrics or track.synced_lyrics)
        ],
        status=state_status,
    )
    return persisted_tracks


def process_track_lyrics(
    file_path: Path,
    artist: str,
    title: str,
    force: bool = False,
    dry_run: bool = False,
    isrc: str | None = None,
    album_name: str | None = None,
    duration: float | None = None,
    existing_lyrics: str | None = None,
    musixmatch_token: str | None = None,
) -> tuple[str | None, str | None]:
    """
    Fetches, cleans, and saves track lyrics to an accompanying .lrc file.
    Supports smart quality upgrade:
      - Enhanced (quality 3, word-synced): Maximum quality, skip network query unless force=True.
      - Line-synced (quality 2): Queries online to attempt upgrade to Enhanced (quality 3).
      - Plain text (quality 1): Queries online to attempt upgrade to Synced or Enhanced (quality 2/3).
      - Missing (quality 0): Queries online for any available lyrics.
    Returns (lyrics_text, quality_tag) or (None, None).
    """
    lrc_path = file_path.with_suffix(".lrc")
    resolved_duration = (
        duration
        if duration is not None
        else metadata_module.get_audio_duration(file_path)
    )
    current_quality = 0
    existing_content: str | None = None
    rejected_content: str | None = None

    if lrc_path.exists() and lrc_path.stat().st_size > 0:
        try:
            existing_content = lrc_path.read_text(encoding="utf-8-sig").strip()
        except UnicodeDecodeError:
            existing_content = None
            current_quality = 0
        except OSError:
            existing_content = None

    if existing_content and is_lyrics_duration_valid(
        existing_content, resolved_duration
    ):
        current_quality = get_lyrics_quality(existing_content)

    if existing_content and (
        not lyrics_identity_matches(existing_content, artist, title)
        or not is_lyrics_duration_valid(existing_content, resolved_duration)
    ):
        LOG.warning(
            f"Local lyrics require review for {escape(file_path.name)}: "
            "headers or timestamps conflict with the requested recording; reuse withheld."
        )
        rejected_content = existing_content
        existing_content = None
        current_quality = 0

    embedded_quality = get_lyrics_quality(existing_lyrics)
    if existing_lyrics and (
        not lyrics_identity_matches(existing_lyrics, artist, title)
        or (
            rejected_content is not None
            and strip_lyrics_headers(existing_lyrics)
            == strip_lyrics_headers(rejected_content)
        )
    ):
        LOG.warning(
            f"Embedded lyrics require review for {escape(file_path.name)}: "
            "recording identity conflicts with local evidence; reuse withheld."
        )
        embedded_quality = 0
    if embedded_quality > current_quality and is_lyrics_duration_valid(
        existing_lyrics, resolved_duration
    ):
        current_quality = embedded_quality
        existing_content = existing_lyrics
        if existing_content and not dry_run:
            atomic_write_bytes(lrc_path, existing_content.encode("utf-8"))

    if not force and current_quality >= 3 and existing_content:
        return existing_content, "existing"

    try:
        lyrics_text = lyrics_service.fetch_synced_lyrics(
            artist,
            title,
            enhanced=True,
            isrc=isrc,
            album_name=album_name,
            duration=resolved_duration,
            musixmatch_token=musixmatch_token,
        )
    except (
        httpx.HTTPError,
        OSError,
        ValueError,
        RuntimeError,
        TimeoutError,
    ) as error:
        LOG.debug(f"Lyrics lookup error for {title}: {error}")
        lyrics_text = None

    if (
        lyrics_text
        and resolved_duration is None
        and extract_max_timestamp(lyrics_text) is not None
    ):
        LOG.warning(
            f"Cannot validate timestamped lyrics without audio duration: {title}"
        )
        lyrics_text = None
    if lyrics_text and not is_lyrics_duration_valid(lyrics_text, resolved_duration):
        LOG.warning(
            f"   ∟ ⚠️  Rejected mismatched lyrics for {title}: "
            f"lyrics duration exceeds audio length ({int(resolved_duration or 0)}s)."
        )
        lyrics_text = None
    if lyrics_text and not lyrics_identity_matches(lyrics_text, artist, title):
        LOG.warning(
            f"Fetched lyrics require review for {escape(file_path.name)}: "
            "returned headers conflict with the requested recording."
        )
        lyrics_text = None

    if not lyrics_text:
        # Remote search returned nothing: preserve valid existing local lyrics if any
        if existing_content and current_quality > 0:
            return existing_content, "existing"
        return None, None

    new_quality = get_lyrics_quality(lyrics_text)

    # Only the strict provider path establishes identity independently of local headers.
    if (
        not get_config().allow_unverified_lyrics
        and resolved_duration is not None
        and resolved_duration > 0
        and not any(
            delimiter in artist or delimiter in title for delimiter in ("]", "\r", "\n")
        )
    ):
        verified_body = lyrics_word_body(lyrics_text)
        for rejected_lyrics in (rejected_content, existing_lyrics):
            if (
                rejected_lyrics
                and verified_body
                and not lyrics_identity_matches(rejected_lyrics, artist, title)
                and is_lyrics_duration_valid(rejected_lyrics, resolved_duration)
                and lyrics_word_body(rejected_lyrics) == verified_body
                and get_lyrics_quality(rejected_lyrics) >= new_quality
            ):
                repaired_lyrics = repair_lyrics_headers(rejected_lyrics, artist, title)
                if lyrics_identity_matches(repaired_lyrics, artist, title):
                    lyrics_text = repaired_lyrics
                    new_quality = get_lyrics_quality(lyrics_text)

    # Downgrade protection: NEVER overwrite synced or enhanced lyrics with plain text
    if current_quality >= 2 and new_quality < current_quality:
        if existing_content:
            return existing_content, "existing"
        return None, None

    if not force and current_quality > 0 and new_quality <= current_quality:
        if existing_content:
            return existing_content, "existing"
        return None, None

    if not dry_run:
        atomic_write_bytes(lrc_path, lyrics_text.encode("utf-8"))

    tag_type = (
        "enhanced" if new_quality == 3 else ("synced" if new_quality == 2 else "plain")
    )
    return lyrics_text, tag_type
