import dataclasses
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx
from mutagen import MutagenError
from PIL import Image
from rapidfuzz import fuzz
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
    clean_lyrics_text,
    extract_max_timestamp,
    get_lyrics_quality,
    is_lyrics_duration_valid,
    lyrics_identity_matches,
    lyrics_word_body,
    repair_lyrics_headers,
    strip_lyrics_headers,
)
from sonora.core.state import LibraryStateVault
from sonora.core.text import normalize_str
from sonora.core.titles import clean_title
from sonora.modules.catalog_metadata import _NETWORK_EXCEPTIONS
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


def _update_cuesheet(
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


def _update_bpm(
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
                LOG.detail(f"   ∟ BPM Calculated: [green]{bpm}[/]")
        except OSError as error:
            LOG.debug(f"BPM calculation failed for {track_info.title}: {error}")


def _update_key(
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
                    f"   ∟ Musical Key: [green]{escape(detected_key)}[/] ({escape(camelot)})"
                )
        except OSError as error:
            LOG.debug(f"Key calculation failed for {track_info.title}: {error}")


def _update_artwork(
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


def _update_lyrics(
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
            conflicts=track_info.metadata_conflicts,
        )
        if lyrics_text and tag_type:
            track_info.lyrics = lyrics_text
            track_info.metadata_conflicts.pop("lyrics", None)
            if not had_lrc:
                LOG.detail(
                    f"   ∟ [green]Saved {tag_type} lyrics for {escape(file_path.name)}[/]"
                )
            elif force:
                LOG.detail(
                    f"   ∟ [yellow]Updated {tag_type} lyrics for {escape(file_path.name)}[/]"
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
    if original_metadata._numbering_needs_normalization or (
        "album_sequence" in original_metadata.metadata_conflicts
        and any(
            field in track_info.metadata_sources
            for field in ("total_tracks", "total_discs")
        )
    ):
        diff_entries.append(
            (
                "numbering",
                "conflicting total aliases",
                f"track total {track_info.total_tracks}, disc total {track_info.total_discs}",
            )
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
    _update_cuesheet(track_info, file_path, cuesheet_content)
    _update_bpm(track_info, file_path, fetch_bpm, force=force)
    _update_key(track_info, file_path, fetch_key, force=force)
    cover_image = None
    if not track_info.is_alien:
        cover_image = (
            album_cover_path
            if album_cover_path and album_cover_path.exists()
            else _update_artwork(
                track_info, file_path, fetch_itunes_art, force=force, dry_run=dry_run
            )
        )
    _update_lyrics(
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

    if diff_lines or force:
        if not dry_run:
            metadata_module.write_track_metadata(track_info, cover_art_path=cover_image)
            LOG.detail(
                f"   ∟ [green]✓[/] {escape(file_path.name)}: {len(diff_lines)} tag(s) updated.{''.join(diff_lines)}"
            )
        else:
            LOG.detail(f"   ∟ [DRY-RUN] {escape(file_path.name)}{''.join(diff_lines)}")
    else:
        LOG.detail(
            f"   ∟ [bold dim]Unchanged:[/] [dim]{escape(file_path.name)}[/] [dim]tags are unchanged.[/]"
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
    """Normalize tags offline, optionally adding locally measured BPM and key."""
    if not file_path.exists():
        return None

    try:
        current_info = metadata_module.read_track_metadata(file_path)
    except OSError as error:
        LOG.debug(f"Failed to read metadata for {file_path}: {error}")
        return None

    original_metadata = current_info.copy()

    _update_bpm(current_info, file_path, fetch_bpm, force=force)
    _update_key(current_info, file_path, fetch_key, force=force)

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
        return NormalizeReport()

    album_groups = group_files_by_album_root(audio_files)
    report = NormalizeReport()

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
                        report.total_scanned += 1
                        if worker_track is not None:
                            if worker_track._diff_descriptions:
                                report.modified_files[str(futures[future])] = (
                                    worker_track._diff_descriptions
                                )
                        else:
                            report.failed_files += 1
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
                raise InterruptedOperationError(report) from None

    return report


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
    conflicts: dict[str, str] | None = None,
    details: dict[str, Any] | None = None,
    lookup_state: dict[str, Any] | None = None,
) -> tuple[str | None, str | None]:
    """Validate and upgrade sidecar/embedded lyrics without lowering sync quality.

    Existing text is cleaned before reuse; conflicting words require review.
    Return (text, status). Missing results do not erase existing lyric files.
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
    raw_sidecar: str | None = None
    raw_embedded = existing_lyrics
    existing_lyrics = clean_lyrics_text(existing_lyrics)
    review_required = bool(raw_embedded and not existing_lyrics)
    lookup_failed = False
    existing_source = "sidecar"
    source_disagreement = False

    if lrc_path.exists() and lrc_path.stat().st_size > 0:
        try:
            raw_sidecar = lrc_path.read_text(encoding="utf-8-sig").strip()
            existing_content = clean_lyrics_text(raw_sidecar)
            review_required |= bool(raw_sidecar and not existing_content)
        except UnicodeDecodeError:
            existing_content = None
            current_quality = 0
            if not force:
                reason = "Local lyric file is not valid UTF-8; review its encoding before replacing it."
                if conflicts is not None:
                    conflicts["lyrics"] = reason
                if details is not None:
                    details.update(action="review", source="sidecar", reason=reason)
                return None, "review"
        except OSError as error:
            LOG.warning(
                f"Cannot read lyrics for {escape(file_path.name)}: {escape(str(error))}"
            )
            lookup_failed = True

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
        review_required = True
        if details is not None:
            details["reason"] = (
                "Local lyric headers or timestamps conflict with the recording."
            )

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
        review_required = True
    embedded_valid = embedded_quality > 0 and is_lyrics_duration_valid(
        existing_lyrics, resolved_duration
    )
    if embedded_quality > 0 and not embedded_valid:
        review_required = True
        if details is not None:
            details["reason"] = "Embedded lyric timestamps conflict with the recording."
    if (
        existing_content
        and embedded_valid
        and lyrics_word_body(existing_content)
        != lyrics_word_body(existing_lyrics or "")
    ):
        source_disagreement = True
        review_required = True
        if conflicts is not None:
            conflicts["lyrics"] = (
                "Sidecar and embedded lyric words disagree; review required."
            )
        if details is not None:
            details["reason"] = "Sidecar and embedded lyric words disagree."
        embedded_valid = False
    if embedded_quality > current_quality and embedded_valid:
        current_quality = embedded_quality
        existing_content = existing_lyrics
        existing_source = "embedded"

    other_song_bodies: list[str] = []
    version_pattern = (
        r"\s*[\(\[][^\)\]]*\b(?:remix|mix|edit|live|acoustic|instrumental|"
        r"extended|version|sped up|slowed|reverb)\b[^\)\]]*[\)\]]"
        r"|\s+[-–—]\s+(?:sped up|slowed(?:\s*\+?\s*reverb)?)\s*$"
    )
    for sibling in find_audio_files(file_path.parent, recursive=False):
        if sibling == file_path:
            continue
        try:
            original = metadata_module.read_track_metadata(sibling)
        except (OSError, ValueError, MutagenError):
            continue
        current_title, sibling_title = (
            normalize_str(
                clean_title(
                    re.sub(version_pattern, "", name or "", flags=re.IGNORECASE)
                )
            )
            for name in (title, original.title)
        )
        different_song = current_title != sibling_title
        different_remix_credit = (
            "remix" in title.casefold()
            and any(
                token in artist.casefold() for token in (" & ", " feat", " ft.", ";")
            )
            and normalize_str(original.artist) != normalize_str(artist)
        )
        if different_song or different_remix_credit:
            body = lyrics_word_body(
                clean_lyrics_text(original.lyrics or original.synced_lyrics) or ""
            )
            if len(body) >= 200:
                other_song_bodies.append(body)
    if existing_content and any(
        fuzz.ratio(lyrics_word_body(existing_content), body) >= 98
        for body in other_song_bodies
    ):
        LOG.warning(
            f"Existing lyrics for {escape(file_path.name)} match another recording; reuse withheld."
        )
        if conflicts is not None:
            conflicts["lyrics"] = "Lyrics match another recording; review required."
        existing_content = None
        current_quality = 0
        review_required = True
        if details is not None:
            details["reason"] = "Existing lyric words match another recording."

    def keep_existing() -> tuple[str | None, str | None]:
        if review_required and conflicts is not None:
            conflicts.setdefault(
                "lyrics", "Existing or returned lyrics require recording verification."
            )
        if details is not None:
            details["previous_quality"] = current_quality
            details["source"] = existing_source if existing_content else None
            details["action"] = "review" if review_required else "unchanged"
        if existing_content and current_quality > 0:
            original_content = raw_sidecar if raw_sidecar is not None else raw_embedded
            status = "cleaned" if existing_content != original_content else "existing"
            if details is not None and not review_required:
                if raw_sidecar is None:
                    details["action"] = "restore"
                elif status == "cleaned":
                    details["action"] = "cleanup"
            if existing_content != raw_sidecar and not dry_run:
                atomic_write_bytes(lrc_path, existing_content.encode("utf-8"))
            return existing_content, status
        return None, "failed" if lookup_failed else None

    if not force and (
        source_disagreement or (current_quality >= 3 and existing_content)
    ):
        return keep_existing()

    try:
        lookup_options: dict[str, Any] = {}
        if details is not None:
            lookup_options["details"] = details
        if lookup_state is not None:
            lookup_options["lookup_state"] = lookup_state
        lyrics_text = lyrics_service.fetch_synced_lyrics(
            artist,
            title,
            enhanced=True,
            isrc=isrc,
            album_name=album_name,
            duration=resolved_duration,
            musixmatch_token=musixmatch_token,
            **lookup_options,
        )
    except (
        httpx.HTTPError,
        OSError,
        ValueError,
        RuntimeError,
        TimeoutError,
    ) as error:
        LOG.warning(
            f"Lyrics lookup failed for {escape(file_path.name)}: {escape(str(error))}"
        )
        lookup_failed = True
        if details is not None:
            details["lookup_status"] = "failed"
            details["error"] = str(error)
        lyrics_text = None

    returned_text = lyrics_text
    lyrics_text = clean_lyrics_text(lyrics_text)
    if returned_text and not lyrics_text:
        review_required = True

    if (
        lyrics_text
        and resolved_duration is None
        and extract_max_timestamp(lyrics_text) is not None
    ):
        LOG.warning(
            f"Cannot validate timestamped lyrics without audio duration: {title}"
        )
        review_required = True
        lyrics_text = None
        if details is not None:
            details["reason"] = (
                "Timestamped lyrics cannot be verified without audio duration."
            )
    if lyrics_text and not is_lyrics_duration_valid(lyrics_text, resolved_duration):
        LOG.warning(
            f"   ∟ Rejected mismatched lyrics for {title}: "
            f"lyrics duration exceeds audio length ({int(resolved_duration or 0)}s)."
        )
        review_required = True
        lyrics_text = None
        if details is not None:
            details["reason"] = (
                "Returned lyric timestamps exceed the recording duration."
            )
    if lyrics_text and not lyrics_identity_matches(lyrics_text, artist, title):
        LOG.warning(
            f"Fetched lyrics require review for {escape(file_path.name)}: "
            "returned headers conflict with the requested recording."
        )
        review_required = True
        lyrics_text = None
        if details is not None:
            details["reason"] = "Returned lyric headers conflict with the recording."

    if lyrics_text and any(
        fuzz.ratio(lyrics_word_body(lyrics_text), body) >= 98
        for body in other_song_bodies
    ):
        LOG.warning(
            f"Fetched lyrics for {escape(file_path.name)} match another recording; import withheld."
        )
        if conflicts is not None:
            conflicts["lyrics"] = (
                "Fetched lyrics match another recording; review required."
            )
        review_required = True
        lyrics_text = None
        if details is not None:
            details["reason"] = "Returned lyric words match another recording."

    if not lyrics_text:
        return keep_existing()

    new_quality = get_lyrics_quality(lyrics_text)
    if new_quality == 0:
        review_required = True
        if details is not None:
            details["reason"] = "Returned lyrics contain no usable lyric words."
        return keep_existing()

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

    if not force and existing_content:
        old_body = lyrics_word_body(existing_content)
        new_body = lyrics_word_body(lyrics_text)
        if old_body and old_body != new_body:
            LOG.warning(
                f"Existing and returned lyric words disagree: {escape(file_path.name)}"
            )
            if conflicts is not None:
                conflicts["lyrics"] = (
                    "Existing and returned lyric words disagree; replacement requires review."
                )
            review_required = True
            if details is not None:
                details["reason"] = "Existing and returned lyric words disagree."
                details["returned_quality"] = new_quality
            return keep_existing()

    # A forced lookup must not replace synchronized lyrics with a lower timing tier.
    if current_quality >= 2 and new_quality < current_quality:
        return keep_existing()

    if not force and current_quality > 0 and new_quality <= current_quality:
        return keep_existing()

    if not dry_run:
        atomic_write_bytes(lrc_path, lyrics_text.encode("utf-8"))

    tag_type = (
        "enhanced" if new_quality == 3 else ("synced" if new_quality == 2 else "plain")
    )
    if details is not None:
        details["previous_quality"] = current_quality
        details["returned_quality"] = new_quality
        details["action"] = "timing_upgrade" if current_quality > 0 else "download"
    return lyrics_text, tag_type
