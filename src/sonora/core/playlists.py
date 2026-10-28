import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from sonora.core.artists import get_primary_artist
from sonora.core.files import atomic_write_bytes
from sonora.core.library_paths import parse_track_filename
from sonora.core.logger import LOG
from sonora.core.models import TrackInfo
from sonora.core.text import normalize_str
from sonora.core.titles import (
    clean_title,
    extract_title_features,
    score_artist_title_match,
)


def _title_key(title: str) -> str:
    return normalize_str(extract_title_features(clean_title(title))[0])


def _local_candidates(
    reference: Path,
    tracks: Sequence[TrackInfo],
    renames: Mapping[Path, Path],
    *,
    dry_run: bool,
) -> set[Path]:
    _, _, identity = parse_track_filename(reference.name)
    identity_key = _title_key(identity)
    _, reference_features = extract_title_features(identity)
    reference_identity = clean_title(identity)
    if reference_features:
        reference_identity += f" (feat. {', '.join(reference_features)})"
    matches: set[Path] = set()
    for track in tracks:
        if track.file_path.suffix.casefold() != reference.suffix.casefold():
            continue
        title_key = _title_key(track.title)
        if not title_key:
            continue
        identities = {track.title}
        for artist in (track.artist, track.album_artist):
            if artist:
                identities.add(f"{get_primary_artist(artist)} - {track.title}")
        feature_suffix = (
            f" (feat. {track.featured_artists})" if track.featured_artists else ""
        )
        if any(
            identity_key == _title_key(candidate)
            and score_artist_title_match(
                "", reference_identity, "", clean_title(candidate) + feature_suffix
            )
            > 0
            for candidate in identities
        ):
            destination = renames.get(track.file_path, track.file_path)
            if destination.is_file() or (dry_run and track.file_path.is_file()):
                matches.add(destination)
    return matches


def update_local_playlists(
    folder: Path,
    renames: Mapping[Path, Path],
    tracks: Sequence[TrackInfo] = (),
    *,
    dry_run: bool = False,
) -> list[Path]:
    """Preserve M3U text and resolve only explicit moves or unique local identities."""
    updated: list[Path] = []
    resolved_renames = {
        source.resolve(): target
        for source, target in renames.items()
        if dry_run or target.is_file()
    }
    for playlist in sorted(folder.iterdir()):
        if (
            playlist.suffix.casefold() not in {".m3u", ".m3u8"}
            or not playlist.is_file()
        ):
            continue
        original = playlist.read_bytes()
        try:
            text = original.decode("utf-8")
        except UnicodeDecodeError:
            LOG.warning(f"Playlist encoding requires review: {playlist}")
            continue
        result: list[str] = []
        for line in text.splitlines(keepends=True):
            content = line.rstrip("\r\n")
            reference_text = content.strip().lstrip("\ufeff")
            if (
                not reference_text
                or reference_text.startswith("#")
                or "://" in reference_text
            ):
                result.append(line)
                continue
            reference = Path(reference_text)
            source = playlist.parent / reference
            unresolved_reason = "Missing"
            replacement = resolved_renames.get(source.resolve())
            if (
                replacement is None
                and not source.exists()
                and reference.parent == Path(".")
            ):
                candidates = _local_candidates(
                    reference, tracks, renames, dry_run=dry_run
                )
                if len(candidates) == 1:
                    replacement = next(iter(candidates))
                elif candidates:
                    unresolved_reason = "Ambiguous"
            if replacement is None:
                if not source.exists():
                    LOG.warning(
                        f"{unresolved_reason} playlist entry retained for review: {playlist}: {reference_text}"
                    )
                result.append(line)
                continue
            rewritten = (
                str(replacement)
                if reference.is_absolute()
                else os.path.relpath(replacement, playlist.parent)
            )
            leading = content[: len(content) - len(content.lstrip())]
            trailing = content[len(content.rstrip()) :]
            marker = "\ufeff" if content.startswith("\ufeff") else ""
            result.append(
                marker + leading + rewritten + trailing + line[len(content) :]
            )
        payload = "".join(result).encode("utf-8")
        if payload != original:
            if dry_run:
                LOG.detail(f"[DRY-RUN] Would update playlist: {playlist}")
            else:
                atomic_write_bytes(playlist, payload)
            updated.append(playlist)
    return updated
