import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.markup import escape

from sonora.audio.metadata import read_track_metadata
from sonora.core.artists import get_primary_artist
from sonora.core.audio_files import find_audio_files
from sonora.core.cancellation import InterruptedOperationError, is_interruption
from sonora.core.config import get_config
from sonora.core.files import atomic_write_bytes
from sonora.core.library_paths import (
    get_single_release_title,
    group_files_by_album_root,
    is_in_singles_hierarchy,
    resolve_unique_path,
    safe_case_rename,
)
from sonora.core.logger import (
    LOG,
    create_progress,
    interactive_pause_listener,
    resume_workers,
    wait_if_paused,
)
from sonora.core.models import RenameReport, TrackInfo
from sonora.core.playlists import update_local_playlists
from sonora.core.sidecars import find_companion_lyrics, relocate_companion_lyrics
from sonora.core.text import normalize_str, safe_int, sanitize_name
from sonora.core.titles import clean_title, strip_corrupt_brackets


def sync_lrc_metadata(lrc_path: Path, artist: str, title: str) -> bool:
    """
    Update or insert [ar: Artist] and [ti: Title] metadata headers into an .lrc file.
    Preserve synchronized lyrics and return True only when header changes are saved.
    """
    if not lrc_path.is_file():
        return False

    try:
        with lrc_path.open(encoding="utf-8", newline="") as file_handle:
            lines = file_handle.readlines()

        new_lines: list[str] = []
        identity_headers = {"ar": artist, "ti": title}
        found_headers: set[str] = set()

        for line in lines:
            line_stripped = line.strip()
            header_key = next(
                (
                    key
                    for key in identity_headers
                    if line_stripped.lower().startswith(f"[{key}:")
                ),
                None,
            )
            if header_key is not None:
                header = f"[{header_key}:{identity_headers[header_key]}]"
                line_ending = line[len(line.rstrip("\r\n")) :]
                new_lines.append(
                    line if line_stripped == header else f"{header}{line_ending}"
                )
                found_headers.add(header_key)
            else:
                new_lines.append(line)

        headers = [
            f"[{key}:{value}]\n"
            for key, value in identity_headers.items()
            if key not in found_headers and value
        ]
        updated_lyrics = "".join(headers + new_lines)
        if updated_lyrics == "".join(lines):
            return False
        atomic_write_bytes(lrc_path, updated_lyrics.encode("utf-8"))
        return True

    except (OSError, UnicodeDecodeError) as error:
        LOG.debug(
            f"Failed to sync LRC metadata for {escape(str(lrc_path))}: {escape(str(error))}"
        )
        return False


def build_new_filename(
    track_number: int | str | None,
    title: str,
    extension: str,
    disc_number: int | str | None = None,
    total_discs: int | str | None = 1,
) -> str | None:
    """
    - '01 - Title.flac' for single disc albums
    - '1-01 - Title.flac' for multi-disc albums (when disc > 1 or total_discs > 1)
    """
    if not title:
        return None

    sanitized_title = sanitize_name(strip_corrupt_brackets(clean_title(title)))
    parsed_track_number = safe_int(track_number)

    disc_prefix = ""
    parsed_disc_number = safe_int(disc_number)
    if parsed_disc_number:
        discs_count = safe_int(total_discs) or 1
        if parsed_disc_number > 1 or discs_count > 1:
            disc_prefix = f"{parsed_disc_number}-"

    if parsed_track_number is not None:
        return f"{disc_prefix}{parsed_track_number:02d} - {sanitized_title}{extension}"
    return f"{disc_prefix}{sanitized_title}{extension}"


def rename_track_file(
    file_path: Path,
    format_pattern: str | None = None,
    track_info: TrackInfo | None = None,
    dry_run: bool = False,
    relocated_lrc_collector: list[Path] | None = None,
    synced_lrc_collector: list[Path] | None = None,
) -> Path:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if track_info is None:
        track_info = read_track_metadata(file_path)

    folder = file_path.parent
    in_singles = is_in_singles_hierarchy(file_path)

    if format_pattern is None:
        if in_singles:
            folder_audios = find_audio_files(folder, recursive=False)
            if len(folder_audios) <= 1:
                filename_track_number: int | str | None = 1
            else:
                filename_track_number = track_info.track_number or 1
            filename_disc_number: int | str | None = 1
            total_discs: int | str | None = 1
        else:
            filename_track_number = track_info.track_number
            filename_disc_number = track_info.disc_number
            total_discs = track_info.total_discs

        new_name = build_new_filename(
            track_number=filename_track_number,
            title=track_info.title,
            extension=file_path.suffix,
            disc_number=filename_disc_number,
            total_discs=total_discs,
        )
        if not new_name:
            return file_path
    else:
        formatted_track_number = 1 if in_singles else (track_info.track_number or 1)
        artist_clean = sanitize_name(track_info.artist)
        title_clean = sanitize_name(clean_title(track_info.title))
        new_stem = format_pattern.format(
            track_number=formatted_track_number,
            artist=artist_clean,
            title=title_clean,
        )
        new_stem = re.sub(r"\s+", " ", new_stem).strip()
        new_name = f"{new_stem}{file_path.suffix}"

    new_path = folder / new_name

    companion_lyrics = find_companion_lyrics(
        file_path, track_number=track_info.track_number, in_singles=in_singles
    )

    for companion in companion_lyrics:
        if (
            companion.suffix.lower() == ".lrc"
            and not dry_run
            and sync_lrc_metadata(companion, track_info.artist, track_info.title)
            and synced_lrc_collector is not None
        ):
            synced_lrc_collector.append(companion)

    if file_path.name != new_name or file_path.parent != new_path.parent:
        if new_path.exists() and not file_path.samefile(new_path):
            new_path = resolve_unique_path(new_path, current_path=file_path)
            new_name = new_path.name

        if not dry_run:
            try:
                while True:
                    try:
                        safe_case_rename(file_path, new_path)
                        break
                    except FileExistsError:
                        new_path = resolve_unique_path(new_path, current_path=file_path)
                new_name = new_path.name
                LOG.detail(
                    f"   ∟ [dim]{escape(file_path.name)}[/] -> [white]{escape(new_name)}[/]"
                )
                relocated = relocate_companion_lyrics(
                    file_path, new_path, dry_run=False, sidecars=companion_lyrics
                )
                if relocated_lrc_collector is not None:
                    relocated_lrc_collector.extend(relocated)
            except OSError as error:
                LOG.warning(
                    f"Failed to rename file {escape(file_path.name)}: {escape(str(error))}"
                )
                raise
        else:
            LOG.detail(
                f"[DRY-RUN] Would rename {escape(file_path.name)} -> {escape(new_name)}"
            )

    return new_path


def rename_single_folder(
    folder_path: Path, track_info: TrackInfo, dry_run: bool = False
) -> Path:
    """Rename a single release folder while preserving library containers."""
    current_folder_name = folder_path.name
    if (
        get_config().is_generic_container(current_folder_name)
        or get_config().is_disc_folder(current_folder_name)
        or current_folder_name.lower() in ("singles", "single")
    ):
        return folder_path

    try:
        if any(entry_path.is_dir() for entry_path in folder_path.iterdir()):
            return folder_path
    except OSError:
        return folder_path

    primary_artist = get_primary_artist(track_info.artist)
    single_title = get_single_release_title(track_info)
    canonical_with_feat = (
        sanitize_name(
            f"{primary_artist} - {single_title} (feat. {track_info.featured_artists})"
        )
        if track_info.featured_artists
        and not re.search(r"\b(?:feat|ft|featuring)\b", single_title, re.IGNORECASE)
        else None
    )
    canonical_without_feat = sanitize_name(f"{primary_artist} - {single_title}")

    if current_folder_name in (canonical_with_feat, canonical_without_feat):
        return folder_path

    expected_name = canonical_with_feat or canonical_without_feat
    new_folder = folder_path.with_name(expected_name)
    if (
        new_folder.exists()
        and folder_path.resolve() != new_folder.resolve()
        and folder_path.name.lower() != new_folder.name.lower()
    ):
        return folder_path

    if not dry_run:
        try:
            safe_case_rename(folder_path, new_folder)
            LOG.detail(
                f"   ∟ Single folder renamed: [dim]{escape(current_folder_name)}[/] -> [cyan]{escape(expected_name)}[/]"
            )
            return new_folder
        except OSError as error:
            LOG.warning(
                f"Failed to rename single folder {escape(current_folder_name)}: {escape(str(error))}"
            )
            return folder_path
    else:
        LOG.detail(
            f"[DRY-RUN] Would rename single folder {escape(current_folder_name)} -> {escape(expected_name)}"
        )
        return new_folder


def rename_album_folder(
    folder_path: Path, artist: str, album: str, dry_run: bool = False
) -> Path:
    if (
        not album
        or get_config().is_generic_container(album)
        or get_config().is_disc_folder(folder_path.name)
        or is_in_singles_hierarchy(folder_path)
    ):
        return folder_path

    current_folder_name = folder_path.name

    # Shield artist container folders from being renamed to album names
    if normalize_str(current_folder_name) == normalize_str(artist) and normalize_str(
        album
    ) != normalize_str(artist):
        return folder_path

    # Do not rename if this directory contains child directories other than CD/Disc folders
    try:
        if any(
            entry_path.is_dir() and not get_config().is_disc_folder(entry_path.name)
            for entry_path in folder_path.iterdir()
        ):
            return folder_path
    except OSError:
        return folder_path

    expected_name = sanitize_name(f"{artist} - {album}")

    if current_folder_name != expected_name:
        new_folder = folder_path.with_name(expected_name)
        if (
            new_folder.exists()
            and folder_path.resolve() != new_folder.resolve()
            and folder_path.name.lower() != new_folder.name.lower()
        ):
            return folder_path

        if not dry_run:
            try:
                safe_case_rename(folder_path, new_folder)
                LOG.detail(
                    f"   ∟ Album folder renamed: [dim]{escape(current_folder_name)}[/] -> [cyan]{escape(expected_name)}[/]"
                )
                return new_folder
            except OSError as error:
                LOG.warning(
                    f"Failed to rename folder {escape(current_folder_name)}: {escape(str(error))}"
                )
                return folder_path
        else:
            LOG.detail(
                f"[DRY-RUN] Would rename album folder {escape(current_folder_name)} -> {escape(expected_name)}"
            )
            return new_folder
    return folder_path


def _rename_single_worker(
    path: Path, dry_run: bool
) -> tuple[Path, TrackInfo | None, Path | None, int]:
    synced_lyrics_paths: list[Path] = []
    try:
        extracted_track_info = read_track_metadata(path)
        new_path = rename_track_file(
            path,
            track_info=extracted_track_info,
            dry_run=dry_run,
            synced_lrc_collector=synced_lyrics_paths,
        )
        return path, extracted_track_info, new_path, len(synced_lyrics_paths)
    except OSError as error:
        LOG.warning(f"Failed to rename file {escape(str(path))}: {escape(str(error))}")
        return path, None, None, len(synced_lyrics_paths)


def rename_directory_files(
    dir_path: Path,
    dry_run: bool = False,
    max_threads: int = 4,
    report: RenameReport | None = None,
) -> list[Path]:
    """
    Scan a directory (recursively) and rename all supported audio files, their .lrc files,
    and album folders based on metadata shared by the tracks.
    """
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    renamed: list[Path] = []
    all_audio_files = find_audio_files(dir_path, recursive=True)
    total_files_count = len(all_audio_files)
    folder_files = group_files_by_album_root(all_audio_files)

    if report is None:
        report = RenameReport(total_files=total_files_count)
    else:
        report.total_files = total_files_count

    with create_progress() as progress:
        task = progress.add_task(
            "[cyan]Renaming audio files...", total=total_files_count
        )
        with (
            interactive_pause_listener(progress, task) as pause_stop_event,
            ThreadPoolExecutor(max_workers=max_threads) as executor,
        ):
            try:
                for folder, files in folder_files.items():
                    album_counts: Counter[tuple[str, str]] = Counter()
                    folder_renamed_paths: list[Path] = []
                    folder_track_infos: list[TrackInfo] = []
                    playlist_renames: dict[Path, Path] = {}

                    file_results = (
                        (
                            rename_future.result()
                            for rename_future in as_completed(
                                [
                                    executor.submit(
                                        _rename_single_worker, audio_path, dry_run
                                    )
                                    for audio_path in files
                                ]
                            )
                        )
                        if max_threads > 1 and len(files) > 1
                        else (
                            _rename_single_worker(audio_path, dry_run)
                            for audio_path in files
                        )
                    )
                    for path, track_info, new_path, lrc_count in file_results:
                        wait_if_paused()
                        report.lrc_synced += lrc_count
                        if track_info is not None and new_path is not None:
                            folder_track_infos.append(track_info)
                            search_artist = track_info.album_artist or track_info.artist
                            if (
                                search_artist != "Unknown Artist"
                                and track_info.album != "Unknown Album"
                            ):
                                album_counts[(search_artist, track_info.album)] += 1
                            folder_renamed_paths.append(new_path)
                            playlist_renames[path] = new_path
                            if (
                                new_path.name != path.name
                                or new_path.parent != path.parent
                            ):
                                report.files_renamed += 1
                            else:
                                report.unchanged_files += 1
                        else:
                            report.failed_files += 1
                        progress.advance(task)

                    for playlist_folder in {folder, *(path.parent for path in files)}:
                        report.playlists_updated += len(
                            update_local_playlists(
                                playlist_folder,
                                playlist_renames,
                                folder_track_infos,
                                dry_run=dry_run,
                            )
                        )
                    final_folder = folder
                    if is_in_singles_hierarchy(folder, root_dir=dir_path):
                        if folder_track_infos:
                            final_folder = rename_single_folder(
                                folder, folder_track_infos[0], dry_run=dry_run
                            )
                            if final_folder != folder:
                                report.folders_renamed += 1
                    elif album_counts:
                        ranked_album_credits = album_counts.most_common(1)
                        if (
                            ranked_album_credits
                            and ranked_album_credits[0][1] > len(files) / 2
                        ):
                            album_artist, album_title = ranked_album_credits[0][0]
                        elif (
                            len({album_title for _, album_title in album_counts})
                            == 1
                            and sum(album_counts.values()) > len(files) / 2
                        ):
                            album_artist, album_title = (
                                "Various Artists",
                                next(iter(album_counts))[1],
                            )
                        else:
                            album_artist, album_title = None, None

                        if album_artist and album_title:
                            final_folder = rename_album_folder(
                                folder, album_artist, album_title, dry_run=dry_run
                            )
                            if final_folder != folder:
                                report.folders_renamed += 1

                    renamed.extend(
                        (
                            final_folder / audio_path.relative_to(folder)
                            if final_folder != folder
                            else audio_path
                        )
                        for audio_path in folder_renamed_paths
                    )
            except (KeyboardInterrupt, RuntimeError) as exc:
                if not is_interruption(exc):
                    raise
                resume_workers(pause_stop_event)
                executor.shutdown(wait=True, cancel_futures=True)
                raise InterruptedOperationError(report) from None

    return renamed
