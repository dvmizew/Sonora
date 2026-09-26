import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.markup import escape

from sonora.audio.metadata import read_track_metadata
from sonora.core.config import get_config
from sonora.core.logger import (
    LOG,
    create_progress,
    interactive_pause_listener,
    wait_if_paused,
)
from sonora.core.models import RenameReport, TrackInfo
from sonora.core.utils import (
    InterruptedOperationError,
    deduplicate_title_features,
    find_audio_files,
    find_companion_lyrics,
    group_files_by_parent,
    is_interruption,
    normalize_str,
    relocate_companion_lyrics,
    safe_case_rename,
    safe_int,
    sanitize_name,
)


def sync_lrc_metadata(lrc_path: Path, artist: str, title: str) -> bool:
    """
    Update or insert [ar: Artist] and [ti: Title] metadata headers into an .lrc file.
    Preserves all synchronized timestamps [mm:ss.xx] intact.
    """
    if not lrc_path.exists() or not lrc_path.is_file():
        return False

    try:
        with lrc_path.open(encoding="utf-8", errors="ignore") as file_handle:
            lines = file_handle.readlines()

        new_lines: list[str] = []
        artist_header_found = False
        title_header_found = False

        for line in lines:
            line_stripped = line.strip()
            if line_stripped.lower().startswith("[ar:"):
                new_lines.append(f"[ar:{artist}]\n")
                artist_header_found = True
            elif line_stripped.lower().startswith("[ti:"):
                new_lines.append(f"[ti:{title}]\n")
                title_header_found = True
            else:
                new_lines.append(line)

        headers: list[str] = []
        if not artist_header_found and artist:
            headers.append(f"[ar:{artist}]\n")
        if not title_header_found and title:
            headers.append(f"[ti:{title}]\n")
        if headers:
            new_lines = headers + new_lines

        with lrc_path.open("w", encoding="utf-8") as file_handle:
            file_handle.writelines(new_lines)
        return True

    except (OSError, ValueError) as error:
        LOG.debug(f"Failed to sync LRC metadata for {lrc_path}: {error}")
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

    clean_title = sanitize_name(deduplicate_title_features(title)) or "Untitled"
    track_num_int = safe_int(track_number)

    disc_prefix = ""
    disc_num_int = safe_int(disc_number)
    if disc_num_int:
        discs_count = safe_int(total_discs) or 1
        if disc_num_int > 1 or discs_count > 1:
            disc_prefix = f"{disc_num_int}-"

    if track_num_int is not None:
        return f"{disc_prefix}{track_num_int:02d} - {clean_title}{extension}"
    return f"{disc_prefix}{clean_title}{extension}"


def rename_track_file(
    file_path: Path,
    format_pattern: str | None = None,
    track_info: TrackInfo | None = None,
    dry_run: bool = False,
    relocated_lrc_collector: list[Path] | None = None,
) -> Path:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        if track_info is None:
            track_info = read_track_metadata(file_path)
    except OSError as error:
        raise RuntimeError(f"Cannot rename file without metadata: {error}") from error

    if format_pattern is None:
        new_name = build_new_filename(
            track_number=track_info.track_number,
            title=track_info.title,
            extension=file_path.suffix,
            disc_number=track_info.disc_number,
            total_discs=track_info.total_discs,
        )
        if not new_name:
            return file_path
    else:
        num = track_info.track_number or 1
        artist_clean = sanitize_name(track_info.artist)
        title_clean = (
            sanitize_name(deduplicate_title_features(track_info.title)) or "Untitled"
        )
        new_stem = format_pattern.format(
            track_number=num,
            artist=artist_clean,
            title=title_clean,
        )
        new_stem = re.sub(r"\s+", " ", new_stem).strip()
        new_name = f"{new_stem}{file_path.suffix}"

    folder = file_path.parent
    new_path = folder / new_name

    companion_lyrics = find_companion_lyrics(file_path)

    # Fallback search by track number prefix if no exact stem match
    if not companion_lyrics and track_info.track_number is not None:
        parsed_track = safe_int(track_info.track_number)
        if parsed_track is not None:
            prefix = f"{parsed_track:02d}"
            prefix_unpadded = str(parsed_track)
            for candidate in folder.iterdir():
                if candidate.suffix.lower() == ".lrc" and (
                    candidate.name.startswith(prefix)
                    or candidate.name.startswith(prefix_unpadded)
                ):
                    companion_lyrics.append(candidate)
                    break

    for companion in companion_lyrics:
        if companion.suffix.lower() == ".lrc" and not dry_run:
            sync_lrc_metadata(companion, track_info.artist, track_info.title)

    if file_path.name != new_name or file_path.parent != new_path.parent:
        base_stem = Path(new_name).stem
        if new_path.exists() and (
            file_path.parent != new_path.parent
            or file_path.name.lower() != new_path.name.lower()
        ):
            counter = 2
            while new_path.exists() and (
                file_path.parent != new_path.parent
                or file_path.name.lower() != new_path.name.lower()
            ):
                new_name = f"{base_stem} ({counter}){file_path.suffix}"
                new_path = folder / new_name
                counter += 1

        if not dry_run:
            try:
                safe_case_rename(file_path, new_path)
                LOG.info(
                    f"   ∟ 🎵 [dim]{escape(file_path.name)}[/] -> [white]{escape(new_name)}[/]"
                )
                relocated = relocate_companion_lyrics(
                    file_path, new_path, dry_run=False
                )
                if relocated_lrc_collector is not None:
                    relocated_lrc_collector.extend(relocated)
            except OSError as error:
                LOG.warning(f"Failed to rename file {escape(file_path.name)}: {error}")
        else:
            LOG.info(
                f"[DRY-RUN] Would rename {escape(file_path.name)} -> {escape(new_name)}"
            )

    return new_path


def rename_album_folder(
    folder_path: Path, artist: str, album: str, dry_run: bool = False
) -> Path:
    if not album or get_config().is_generic_container(album):
        return folder_path

    folder_now = folder_path.name
    is_in_singles = any(get_config().is_generic_container(p) for p in folder_path.parts)

    # Shield artist container folders from being renamed to album names
    if normalize_str(folder_now) == normalize_str(artist) and normalize_str(
        album
    ) != normalize_str(artist):
        return folder_path

    # Do not rename if this directory contains child directories other than CD/Disc folders
    try:
        if any(
            p.is_dir() and not get_config().is_disc_folder(p.name)
            for p in folder_path.iterdir()
        ):
            return folder_path
    except OSError:
        return folder_path

    expected_name = sanitize_name(f"{artist} - {album}")

    if folder_now != expected_name:
        if is_in_singles:
            base_album = album.split("(")[0].split("-")[0].strip()
            if normalize_str(artist) in normalize_str(folder_now) and normalize_str(
                base_album
            ) in normalize_str(folder_now):
                return folder_path

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
                LOG.info(
                    f"   ∟ 📂 Album folder renamed: [dim]{escape(folder_now)}[/] -> [cyan]{escape(expected_name)}[/]"
                )
                return new_folder
            except OSError as error:
                LOG.warning(f"Failed to rename folder {escape(folder_now)}: {error}")
                return folder_path
        else:
            LOG.info(
                f"[DRY-RUN] Would rename album folder {escape(folder_now)} -> {escape(expected_name)}"
            )
    return folder_path


def _rename_single_worker(
    path: Path, dry_run: bool
) -> tuple[Path, TrackInfo | None, Path | None, int]:
    try:
        extracted_track_info = read_track_metadata(path)
        relocated_lrcs: list[Path] = []
        new_path = rename_track_file(
            path,
            track_info=extracted_track_info,
            dry_run=dry_run,
            relocated_lrc_collector=relocated_lrcs,
        )
        return path, extracted_track_info, new_path, len(relocated_lrcs)
    except OSError as error:
        LOG.warning(f"Failed to rename file {escape(str(path))}: {error}")
        return path, None, None, 0


def rename_directory_files(
    dir_path: Path,
    dry_run: bool = False,
    max_threads: int = 4,
    report: RenameReport | None = None,
) -> list[Path]:
    """
    Scan a directory (recursively) and rename all supported audio files, their .lrc files,
    and album folders based on consensus metadata.
    """
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    renamed: list[Path] = []
    all_audio_files = find_audio_files(dir_path, recursive=True)
    total_files_count = len(all_audio_files)
    folder_files = group_files_by_parent(all_audio_files)

    if report is None:
        report = RenameReport(total_files=total_files_count)
    else:
        report.total_files = total_files_count

    with create_progress() as progress:
        task = progress.add_task(
            "[cyan]Renaming audio files...", total=total_files_count
        )
        with (
            interactive_pause_listener(progress, task),
            ThreadPoolExecutor(max_workers=max_threads) as executor,
        ):
            try:
                for folder, files in folder_files.items():
                    album_consensus: Counter[tuple[str, str]] = Counter()
                    folder_renamed_paths: list[Path] = []

                    file_results = (
                        (
                            fut.result()
                            for fut in as_completed(
                                [
                                    executor.submit(_rename_single_worker, p, dry_run)
                                    for p in files
                                ]
                            )
                        )
                        if max_threads > 1 and len(files) > 1
                        else (_rename_single_worker(p, dry_run) for p in files)
                    )
                    for path, track_info, new_path, lrc_count in file_results:
                        wait_if_paused()
                        report.lrc_synced += lrc_count
                        if track_info is not None and new_path is not None:
                            search_artist = track_info.album_artist or track_info.artist
                            if (
                                search_artist != "Unknown Artist"
                                and track_info.album != "Unknown Album"
                            ):
                                album_consensus[(search_artist, track_info.album)] += 1
                            folder_renamed_paths.append(new_path)
                            if (
                                new_path.name != path.name
                                or new_path.parent != path.parent
                            ):
                                report.files_renamed += 1
                            else:
                                report.unchanged_files += 1
                        progress.advance(task)

                    final_folder = folder
                    if album_consensus:
                        top = album_consensus.most_common(1)
                        if top and top[0][1] >= len(files) / 2:
                            art_name, alb_name = top[0][0]
                        elif len({alb for _, alb in album_consensus}) == 1:
                            art_name, alb_name = (
                                "Various Artists",
                                next(iter(album_consensus))[1],
                            )
                        else:
                            art_name, alb_name = None, None

                        if art_name and alb_name:
                            final_folder = rename_album_folder(
                                folder, art_name, alb_name, dry_run=dry_run
                            )
                            if final_folder != folder:
                                report.folders_renamed += 1

                    renamed.extend(
                        (final_folder / p.name if final_folder != folder else p)
                        for p in folder_renamed_paths
                    )
            except (KeyboardInterrupt, RuntimeError) as exc:
                if not is_interruption(exc):
                    raise
                executor.shutdown(wait=True, cancel_futures=True)
                raise InterruptedOperationError(report) from None

    return renamed
