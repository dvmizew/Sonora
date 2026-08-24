import re
from collections import Counter
from pathlib import Path

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from sonora.audio.metadata import read_track_metadata
from sonora.core.logger import CONSOLE, LOG
from sonora.core.models import TrackInfo
from sonora.core.utils import (
    find_audio_files,
    find_companion_lyrics,
    normalize_str,
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
        with open(lrc_path, "r", encoding="utf-8", errors="ignore") as file_handle:
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
        if not artist_header_found:
            headers.append(f"[ar:{artist}]\n")
        if not title_header_found:
            headers.append(f"[ti:{title}]\n")
        if headers:
            new_lines = headers + new_lines

        with open(lrc_path, "w", encoding="utf-8") as file_handle:
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

    clean_title = sanitize_name(title)
    track_number_str = str(track_number).split("/")[0] if track_number else ""
    track_digits = "".join(filter(str.isdigit, track_number_str))

    disc_prefix = ""
    if disc_number:
        disc_clean = "".join(filter(str.isdigit, str(disc_number).split("/")[0]))
        if disc_clean:
            discs_count = (
                int("".join(filter(str.isdigit, str(total_discs).split("/")[0])))
                if total_discs
                else 1
            )
            if int(disc_clean) > 1 or discs_count > 1:
                disc_prefix = f"{int(disc_clean)}-"

    if track_digits:
        return f"{disc_prefix}{int(track_digits):02d} - {clean_title}{extension}"
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
        title_clean = sanitize_name(track_info.title)
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
                if candidate.suffix.lower() in (".lrc", ".txt") and (
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
        if new_path.exists() and (
            file_path.parent != new_path.parent
            or file_path.name.lower() != new_path.name.lower()
        ):
            counter = 2
            while new_path.exists() and (
                file_path.parent != new_path.parent
                or file_path.name.lower() != new_path.name.lower()
            ):
                new_name = f"{new_path.stem} ({counter}){file_path.suffix}"
                new_path = folder / new_name
                counter += 1

        if not dry_run:
            try:
                # Handle case-only rename safely across all filesystems
                if (
                    file_path.parent == new_path.parent
                    and file_path.name.lower() == new_path.name.lower()
                    and file_path.name != new_path.name
                ):
                    temporary_path = folder / f".tmp_{file_path.name}"
                    file_path.rename(temporary_path)
                    temporary_path.rename(new_path)
                else:
                    file_path.rename(new_path)

                LOG.info(f"   ∟ 🎵 [dim]{file_path.name}[/] -> [white]{new_name}[/]")

                # Rename all companion lyric files (.lrc, .synced.lrc, .txt, etc.)
                for companion in companion_lyrics:
                    if companion.exists():
                        suffix = companion.name[len(file_path.stem) :]
                        new_companion = folder / f"{new_path.stem}{suffix}"
                        if not new_companion.exists():
                            companion.rename(new_companion)
                        else:
                            companion.unlink(missing_ok=True)
            except (OSError, ValueError, RuntimeError) as error:
                LOG.warning(f"Failed to rename file {file_path.name}: {error}")
        else:
            LOG.info(f"[DRY-RUN] Would rename {file_path.name} -> {new_name}")

    return new_path


def rename_album_folder(
    folder_path: Path, artist: str, album: str, dry_run: bool = False
) -> Path:
    if not album or album.lower() in ["singles", "unknown album", "unknown"]:
        return folder_path

    folder_now = folder_path.name
    is_in_singles = "singles" in str(folder_path).lower().replace("\\", "/").split("/")

    expected_name = sanitize_name(f"{artist} - {album}")

    if normalize_str(folder_now) != normalize_str(expected_name):
        if is_in_singles:
            base_album = album.split("(")[0].split("-")[0].strip()
            if normalize_str(artist) in normalize_str(folder_now) and normalize_str(
                base_album
            ) in normalize_str(folder_now):
                return folder_path

        new_folder = folder_path.with_name(expected_name)
        if new_folder.exists() and folder_path.resolve() != new_folder.resolve():
            return folder_path

        if not dry_run:
            try:
                folder_path.rename(new_folder)
                LOG.info(
                    f"   ∟ 📂 Album folder renamed: [dim]{folder_now}[/] -> [cyan]{expected_name}[/]"
                )
                return new_folder
            except (OSError, ValueError, RuntimeError) as error:
                LOG.warning(f"Failed to rename folder {folder_now}: {error}")
                return folder_path
        else:
            LOG.info(
                f"[DRY-RUN] Would rename album folder {folder_now} -> {expected_name}"
            )
    return folder_path


def rename_directory_files(dir_path: Path, dry_run: bool = False) -> list[Path]:
    """
    Scan a directory (recursively) and rename all supported audio files, their .lrc files,
    and album folders based on consensus metadata.
    """
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    renamed: list[Path] = []
    folder_files: dict[Path, list[Path]] = {}
    total_files_count = 0
    for path in find_audio_files(dir_path, recursive=True):
        folder_files.setdefault(path.parent, []).append(path)
        total_files_count += 1

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
        task = progress.add_task(
            "[cyan]Renaming audio files...", total=total_files_count
        )
        for folder, files in folder_files.items():
            album_consensus: Counter[tuple[str, str]] = Counter()
            for path in files:
                try:
                    info = read_track_metadata(path)
                    search_artist = info.album_artist or info.artist
                    if (
                        search_artist != "Unknown Artist"
                        and info.album != "Unknown Album"
                    ):
                        album_consensus[(search_artist, info.album)] += 1

                    new_path = rename_track_file(path, track_info=info, dry_run=dry_run)
                    renamed.append(new_path)
                except (OSError, ValueError, RuntimeError) as error:
                    LOG.warning(f"Failed to rename file {path}: {error}")
                progress.advance(task)

            if album_consensus:
                top = album_consensus.most_common(1)
                if top and top[0][1] >= len(files) / 2:
                    top_artist, top_album = top[0][0]
                    rename_album_folder(folder, top_artist, top_album, dry_run=dry_run)
                else:
                    albums_found = {album_title for (_, album_title) in album_consensus}
                    if len(albums_found) == 1:
                        common_album = next(iter(albums_found))
                        rename_album_folder(
                            folder, "Various Artists", common_album, dry_run=dry_run
                        )

    return renamed
