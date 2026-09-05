import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.markup import escape

from sonora.audio.metadata import read_track_metadata
from sonora.core.logger import (
    LOG,
    create_progress,
    interactive_pause_listener,
    wait_if_paused,
)
from sonora.core.models import RenameReport, TrackInfo
from sonora.core.utils import (
    deduplicate_title_features,
    find_audio_files,
    find_companion_lyrics,
    group_files_by_parent,
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

    except (OSError, ValueError, KeyError) as error:
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
) -> Path:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        if track_info is None:
            track_info = read_track_metadata(file_path)
    except (OSError, ValueError, RuntimeError) as error:
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
    if not companion_lyrics and track_info.track_number:
        track_clean = "".join(
            filter(str.isdigit, str(track_info.track_number).split("/")[0])
        )
        if track_clean:
            prefix = f"{int(track_clean):02d}"
            for candidate in folder.iterdir():
                if candidate.suffix.lower() == ".lrc" and (
                    candidate.name.startswith(prefix)
                    or candidate.name.startswith(str(int(track_clean)))
                ):
                    companion_lyrics.append(candidate)
                    break

    # Sync LRC metadata headers
    for companion in companion_lyrics:
        if companion.suffix.lower() == ".lrc" and not dry_run:
            sync_lrc_metadata(companion, track_info.artist, track_info.title)

    # Perform file rename
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
                relocate_companion_lyrics(file_path, new_path, dry_run=False)
            except (OSError, ValueError, RuntimeError) as error:
                LOG.warning(f"Failed to rename file {escape(file_path.name)}: {error}")
        else:
            LOG.info(
                f"[DRY-RUN] Would rename {escape(file_path.name)} -> {escape(new_name)}"
            )

    return new_path


def rename_album_folder(
    folder_path: Path, artist: str, album: str, dry_run: bool = False
) -> Path:
    if not album or album.lower() in ["singles", "unknown album", "unknown"]:
        return folder_path

    folder_now = folder_path.name
    is_in_singles = "singles" in (p.lower() for p in folder_path.parts)

    # Shield artist container folders from being renamed to album names
    if normalize_str(folder_now) == normalize_str(artist) and normalize_str(
        album
    ) != normalize_str(artist):
        return folder_path

    # Do not rename if this directory contains child directories other than CD/Disc folders
    try:
        if any(
            p.is_dir()
            and not re.match(r"^(?:cd|disc|side)\s*\d+$", p.name, re.IGNORECASE)
            for p in folder_path.iterdir()
        ):
            return folder_path
    except OSError:
        return folder_path

    expected_name = sanitize_name(f"{artist} - {album}")

    if normalize_str(folder_now) != normalize_str(expected_name):
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
            except (OSError, ValueError, RuntimeError) as error:
                LOG.warning(f"Failed to rename folder {escape(folder_now)}: {error}")
                return folder_path
        else:
            LOG.info(
                f"[DRY-RUN] Would rename album folder {escape(folder_now)} -> {escape(expected_name)}"
            )
    return folder_path


LAST_RENAME_REPORT: RenameReport = RenameReport()


def get_last_rename_report() -> RenameReport:
    return LAST_RENAME_REPORT


def rename_directory_files(
    dir_path: Path, dry_run: bool = False, max_threads: int = 4
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

    global LAST_RENAME_REPORT
    report = RenameReport(total_files=total_files_count)
    LAST_RENAME_REPORT = report

    with create_progress() as progress:
        task = progress.add_task(
            "[cyan]Renaming audio files...", total=total_files_count
        )
        with interactive_pause_listener(progress, task):
            executor = ThreadPoolExecutor(max_workers=max_threads)
            try:
                for folder, files in folder_files.items():
                    album_consensus: Counter[tuple[str, str]] = Counter()
                    folder_renamed_paths: list[Path] = []

                    def _process_file(
                        path: Path,
                    ) -> tuple[Path, TrackInfo | None, Path | None]:
                        try:
                            info = read_track_metadata(path)
                            new_path = rename_track_file(
                                path, track_info=info, dry_run=dry_run
                            )
                            return path, info, new_path
                        except (OSError, ValueError, RuntimeError) as error:
                            LOG.warning(
                                f"Failed to rename file {escape(str(path))}: {error}"
                            )
                            return path, None, None

                    if max_threads > 1 and len(files) > 1:
                        futures = [executor.submit(_process_file, p) for p in files]
                        for fut in as_completed(futures):
                            wait_if_paused()
                            path, info, new_path = fut.result()
                            if info is not None and new_path is not None:
                                search_artist = info.album_artist or info.artist
                                if (
                                    search_artist != "Unknown Artist"
                                    and info.album != "Unknown Album"
                                ):
                                    album_consensus[(search_artist, info.album)] += 1
                                folder_renamed_paths.append(new_path)
                                if (
                                    new_path.name != path.name
                                    or new_path.parent != path.parent
                                ):
                                    report.files_renamed += 1
                                else:
                                    report.unchanged_files += 1
                            progress.advance(task)
                    else:
                        for path in files:
                            wait_if_paused()
                            p, info, new_path = _process_file(path)
                            if info is not None and new_path is not None:
                                search_artist = info.album_artist or info.artist
                                if (
                                    search_artist != "Unknown Artist"
                                    and info.album != "Unknown Album"
                                ):
                                    album_consensus[(search_artist, info.album)] += 1
                                folder_renamed_paths.append(new_path)
                                if (
                                    new_path.name != path.name
                                    or new_path.parent != path.parent
                                ):
                                    report.files_renamed += 1
                                else:
                                    report.unchanged_files += 1
                            progress.advance(task)

                    # Rename album folder based on consensus for this folder
                    final_folder = folder
                    if album_consensus:
                        top = album_consensus.most_common(1)
                        if top and top[0][1] >= len(files) / 2:
                            top_artist, top_album = top[0][0]
                            final_folder = rename_album_folder(
                                folder, top_artist, top_album, dry_run=dry_run
                            )
                            if final_folder != folder:
                                report.folders_renamed += 1
                        else:
                            albums_found = {
                                album_title for (_, album_title) in album_consensus
                            }
                            if len(albums_found) == 1:
                                common_album = next(iter(albums_found))
                                final_folder = rename_album_folder(
                                    folder,
                                    "Various Artists",
                                    common_album,
                                    dry_run=dry_run,
                                )
                                if final_folder != folder:
                                    report.folders_renamed += 1

                    for p in folder_renamed_paths:
                        if final_folder != folder:
                            renamed.append(final_folder / p.name)
                        else:
                            renamed.append(p)
            finally:
                executor.shutdown(wait=False)

    return renamed
