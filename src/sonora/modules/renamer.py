import re
from pathlib import Path

from sonora.audio.metadata import read_track_metadata
from sonora.core.constants import SUPPORTED_EXTS
from sonora.core.models import TrackInfo
from sonora.core.utils import normalize_str, sanitize_name


def sync_lrc_metadata(lrc_path: Path, artist: str, title: str) -> bool:
    """
    Update or insert [ar: Artist] and [ti: Title] metadata headers into an .lrc file.
    Preserves all synchronized timestamps [mm:ss.xx] intact.
    """
    if not lrc_path.exists() or not lrc_path.is_file():
        return False

    try:
        with open(lrc_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        new_lines: list[str] = []
        ar_found = False
        ti_found = False

        for line in lines:
            line_stripped = line.strip()
            if line_stripped.lower().startswith("[ar:"):
                new_lines.append(f"[ar:{artist}]\n")
                ar_found = True
            elif line_stripped.lower().startswith("[ti:"):
                new_lines.append(f"[ti:{title}]\n")
                ti_found = True
            else:
                new_lines.append(line)

        if not ar_found:
            new_lines.insert(0, f"[ar:{artist}]\n")
        if not ti_found:
            new_lines.insert(0, f"[ti:{title}]\n")

        with open(lrc_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        return True

    except (OSError, ValueError, KeyError) as e:
        from sonora.core.logger import LOG

        LOG.debug(f"Failed to sync LRC metadata for {lrc_path}: {e}")
        return False


def build_new_filename(
    track_num: int | str | None,
    title: str,
    ext: str,
    disc_num: int | str | None = None,
    total_discs: int | str | None = 1,
) -> str | None:
    """
    - '01 - Title.flac' for single disc albums
    - '1-01 - Title.flac' for multi-disc albums (when disc > 1 or total_discs > 1)
    """
    if not title:
        return None

    clean_title = sanitize_name(title)
    tn_str = str(track_num).split("/")[0] if track_num else ""
    tn_digits = "".join(filter(str.isdigit, tn_str))

    dn_str = ""
    if disc_num:
        dn_clean = "".join(filter(str.isdigit, str(disc_num).split("/")[0]))
        if dn_clean:
            t_discs = (
                int("".join(filter(str.isdigit, str(total_discs).split("/")[0])))
                if total_discs
                else 1
            )
            if int(dn_clean) > 1 or t_discs > 1:
                dn_str = f"{int(dn_clean)}-"

    if tn_digits:
        return f"{dn_str}{int(tn_digits):02d} - {clean_title}{ext}"
    return f"{dn_str}{clean_title}{ext}"


def rename_track_file(
    file_path: Path,
    format_pattern: str | None = None,
    track_info: TrackInfo | None = None,
    options: dict | None = None,
) -> Path:
    """
    Rename an audio file and its .lrc files based on metadata.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        if track_info is None:
            track_info = read_track_metadata(file_path)
    except (OSError, ValueError, RuntimeError) as e:
        raise RuntimeError(f"Cannot rename file without metadata: {e}") from e

    options = options or {}
    dry_run = options.get("dry_run", False)

    if format_pattern is None:
        new_name = build_new_filename(
            track_num=track_info.track_number,
            title=track_info.title,
            ext=file_path.suffix,
            disc_num=track_info.disc_number,
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

    LRC_EXTS = [".lrc", ".enhanced.lrc", ".synced.lrc", ".txt"]
    old_lrc_path: Path | None = None

    # Step 1: Look for exact stem match
    for lext in LRC_EXTS:
        cand = folder / f"{file_path.stem}{lext}"
        if cand.exists():
            old_lrc_path = cand
            break

    # Step 2: Fallback search by track number prefix
    if not old_lrc_path and track_info.track_number:
        tn_clean = "".join(filter(str.isdigit, str(track_info.track_number).split("/")[0]))
        if tn_clean:
            tn_p = f"{int(tn_clean):02d}"
            for cand in folder.iterdir():
                if any(cand.name.lower().endswith(e) for e in LRC_EXTS) and (
                    cand.name.startswith(tn_p) or cand.name.startswith(str(int(tn_clean)))
                ):
                    old_lrc_path = cand
                    break

    # Sync LRC metadata headers
    lrc_to_sync = old_lrc_path or (new_path.with_suffix(".lrc") if new_path.with_suffix(".lrc").exists() else None)
    if lrc_to_sync and lrc_to_sync.suffix.lower() == ".lrc" and not dry_run:
        sync_lrc_metadata(lrc_to_sync, track_info.artist, track_info.title)

    synced_lrc_path = folder / f"{file_path.stem}.synced.lrc"
    if synced_lrc_path.exists() and not dry_run:
        sync_lrc_metadata(synced_lrc_path, track_info.artist, track_info.title)

    # Perform file rename
    if file_path.name != new_name or file_path.parent != new_path.parent:
        if new_path.exists() and (file_path.parent != new_path.parent or file_path.name.lower() != new_path.name.lower()):
            counter = 2
            while new_path.exists() and (file_path.parent != new_path.parent or file_path.name.lower() != new_path.name.lower()):
                new_name = f"{new_path.stem} ({counter}){file_path.suffix}"
                new_path = folder / new_name
                counter += 1

        if not dry_run:
            try:
                # Handle case-only rename safely across all filesystems
                if file_path.parent == new_path.parent and file_path.name.lower() == new_path.name.lower() and file_path.name != new_path.name:
                    tmp_path = folder / f".tmp_{file_path.name}"
                    file_path.rename(tmp_path)
                    tmp_path.rename(new_path)
                else:
                    file_path.rename(new_path)
                from sonora.core.logger import LOG

                LOG.info(f"   ∟ 🎵 [dim]{file_path.name}[/] -> [white]{new_name}[/]")

                # Rename main .lrc
                if old_lrc_path and old_lrc_path.exists():
                    new_lrc = folder / f"{new_path.stem}.lrc"
                    if not new_lrc.exists():
                        old_lrc_path.rename(new_lrc)
                    else:
                        old_lrc_path.unlink(missing_ok=True)

                # Rename .synced.lrc
                old_synced_lrc = folder / f"{file_path.stem}.synced.lrc"
                if old_synced_lrc.exists():
                    new_synced_lrc = folder / f"{new_path.stem}.synced.lrc"
                    if not new_synced_lrc.exists():
                        old_synced_lrc.rename(new_synced_lrc)
                    else:
                        old_synced_lrc.unlink(missing_ok=True)
            except (OSError, ValueError, RuntimeError) as e:
                from sonora.core.logger import LOG

                LOG.warning(f"Failed to rename file {file_path.name}: {e}")
        else:
            from sonora.core.logger import LOG

            LOG.info(f"[DRY-RUN] Would rename {file_path.name} -> {new_name}")

    return new_path


def rename_album_folder(
    folder_path: Path, artist: str, album: str, options: dict | None = None
) -> Path:
    """
    Rename an album directory to 'Artist - Album'.
    """
    if not album or album.lower() in ["singles", "unknown album", "unknown"]:
        return folder_path

    options = options or {}
    dry_run = options.get("dry_run", False)

    folder_now = folder_path.name
    is_in_singles = "singles" in str(folder_path).lower().replace("\\", "/").split("/")

    expected_name = sanitize_name(f"{artist} - {album}")

    if normalize_str(folder_now) != normalize_str(expected_name):
        if is_in_singles:
            base_album = album.split("(")[0].split("-")[0].strip()
            if normalize_str(artist) in normalize_str(folder_now) and normalize_str(base_album) in normalize_str(folder_now):
                return folder_path

        new_folder = folder_path.with_name(expected_name)
        if new_folder.exists() and folder_path.resolve() != new_folder.resolve():
            return folder_path

        if not dry_run:
            try:
                folder_path.rename(new_folder)
                from sonora.core.logger import LOG

                LOG.info(f"   ∟ 📂 Album folder renamed: [dim]{folder_now}[/] -> [cyan]{expected_name}[/]")
                return new_folder
            except (OSError, ValueError, RuntimeError) as e:
                from sonora.core.logger import LOG

                LOG.warning(f"Failed to rename folder {folder_now}: {e}")
                return folder_path
        else:
            from sonora.core.logger import LOG

            LOG.info(f"[DRY-RUN] Would rename album folder {folder_now} -> {expected_name}")
    return folder_path


def rename_directory_files(dir_path: Path, options: dict | None = None) -> list[Path]:
    """
    Scan a directory (recursively) and rename all supported audio files, their .lrc files,
    and album folders based on consensus metadata.
    """
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    from collections import Counter

    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    from sonora.core.logger import CONSOLE, LOG

    renamed: list[Path] = []
    folder_files: dict[Path, list[Path]] = {}
    total_files_count = 0
    for path in sorted(dir_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
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
        task = progress.add_task("[cyan]Renaming audio files...", total=total_files_count)
        for folder, files in folder_files.items():
            album_consensus: Counter[tuple[str, str]] = Counter()
            for path in files:
                try:
                    info = read_track_metadata(path)
                    search_artist = info.album_artist or info.artist
                    if search_artist != "Unknown Artist" and info.album != "Unknown Album":
                        album_consensus[(search_artist, info.album)] += 1

                    new_p = rename_track_file(path, track_info=info, options=options)
                    renamed.append(new_p)
                except (OSError, ValueError, RuntimeError) as e:
                    LOG.warning(f"Failed to rename file {path}: {e}")
                progress.advance(task)

            if album_consensus:
                top = album_consensus.most_common(1)
                if top and top[0][1] >= len(files) / 2:
                    top_artist, top_album = top[0][0]
                    rename_album_folder(folder, top_artist, top_album, options=options)
                else:
                    albums_found = {a for (_, a) in album_consensus}
                    if len(albums_found) == 1:
                        common_album = next(iter(albums_found))
                        rename_album_folder(folder, "Various Artists", common_album, options=options)

    return renamed
