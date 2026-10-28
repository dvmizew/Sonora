import io
import os
from collections.abc import Callable
from pathlib import Path

import httpx
from PIL import Image
from rapidfuzz import fuzz
from rich.markup import escape

from sonora.core.config import get_config
from sonora.core.constants import (
    MIN_COVER_ART_DIMENSION,
)
from sonora.core.files import atomic_write_bytes
from sonora.core.http import SESSION
from sonora.core.library_paths import get_album_root_directory
from sonora.core.logger import LOG
from sonora.core.text import normalize_str
from sonora.services.deezer import fetch_deezer_cover_art_url
from sonora.services.fanart import (
    download_fanart_image_bytes,
    fetch_fanart_album,
    fetch_fanart_artist,
    fetch_fanart_label,
)
from sonora.services.itunes import fetch_itunes_cover_art_url
from sonora.services.musicbrainz import fetch_cover_art_archive_url
from sonora.services.theaudiodb import fetch_artist_images


def _valid_image(source: Path | bytes) -> bool:
    try:
        with Image.open(
            io.BytesIO(source) if isinstance(source, bytes) else source
        ) as image:
            image.load()
        return True
    except OSError:
        return False


def _save_image(destination: Path, payload: bytes | None) -> bool:
    if not payload or not _valid_image(payload):
        return False
    atomic_write_bytes(destination, payload)
    return True


def _repair_artist_image_copies(
    folder: Path, artist_dir: Path, *, dry_run: bool
) -> None:
    for filenames in (
        ("artist.jpg", "artist.png", "folder.jpg"),
        ("banner.jpg", "banner.png", "fanart.jpg"),
        ("logo.png",),
    ):
        replacement = next(
            (
                artist_dir / name
                for name in filenames
                if _valid_image(artist_dir / name)
            ),
            None,
        )
        if replacement is None:
            continue
        for name in filenames:
            # An album's folder.jpg is its cover, not the artist portrait.
            if name == "folder.jpg" and folder.resolve() != artist_dir.resolve():
                continue
            target = folder / name
            if target.is_file() and not _valid_image(target):
                if dry_run:
                    LOG.detail(f"[DRY-RUN] Would replace invalid artwork: {target}")
                else:
                    atomic_write_bytes(target, replacement.read_bytes())


def _fetch_candidate_artwork(
    artist: str,
    album: str,
    musicbrainz_album_id: str | None = None,
) -> tuple[bytes, tuple[int, int]] | None:
    """Try Cover Art Archive, iTunes and Deezer in order.

    Return the first image meeting the minimum dimensions, or the largest valid
    fallback when every source returns smaller artwork.
    """
    providers: list[tuple[str, Callable[[], str | None]]] = []
    if musicbrainz_album_id:
        providers.append(
            (
                "Cover Art Archive",
                lambda: fetch_cover_art_archive_url(musicbrainz_album_id),
            )
        )
    providers.append(("iTunes", lambda: fetch_itunes_cover_art_url(artist, album)))
    providers.append(("Deezer", lambda: fetch_deezer_cover_art_url(artist, album)))

    best_candidate: tuple[bytes, tuple[int, int]] | None = None

    for provider_name, url_fetcher in providers:
        try:
            candidate_url = url_fetcher()
            if not candidate_url:
                continue

            response = SESSION.get(candidate_url, timeout=15)
            response.raise_for_status()
            image_bytes = response.content

            try:
                with Image.open(io.BytesIO(image_bytes)) as img:
                    img.load()
                    candidate_dim = img.size
            except OSError:
                continue

            if (
                candidate_dim[0] >= MIN_COVER_ART_DIMENSION
                and candidate_dim[1] >= MIN_COVER_ART_DIMENSION
            ):
                return (image_bytes, candidate_dim)

            if not best_candidate:
                best_candidate = (image_bytes, candidate_dim)
            else:
                _, prev_dim = best_candidate
                if (candidate_dim[0] * candidate_dim[1]) > (prev_dim[0] * prev_dim[1]):
                    best_candidate = (image_bytes, candidate_dim)

        except (httpx.HTTPError, OSError) as error:
            LOG.debug(f"{provider_name} cover art fetch failed: {error}")
            continue

    return best_candidate


def process_album_cover_art(
    folder_path: Path,
    artist: str,
    album: str,
    musicbrainz_album_id: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    musicbrainz_releasegroup_id: str | None = None,
) -> Path | None:
    """
    Downloads and validates high-resolution album cover art (cover.jpg).
    Tries Cover Art Archive, iTunes API, and Deezer fallback.
    Returns Path to cover.jpg if present/downloaded, else None.
    """
    target_dir = get_album_root_directory(folder_path)
    cover_image_path = target_dir / "cover.jpg"
    is_low_res = False
    existing_dim: tuple[int, int] | None = None
    if cover_image_path.exists() and cover_image_path.stat().st_size > 0:
        try:
            with Image.open(cover_image_path) as cur_img:
                cur_img.load()
                existing_dim = cur_img.size
                if (
                    cur_img.width < MIN_COVER_ART_DIMENSION
                    or cur_img.height < MIN_COVER_ART_DIMENSION
                ):
                    is_low_res = True
        except OSError:
            is_low_res = True

    artwork_already_present = (
        cover_image_path.exists()
        and cover_image_path.stat().st_size > 0
        and not force
        and not is_low_res
    )

    if not artwork_already_present:
        if not dry_run and not os.access(target_dir, os.W_OK):
            LOG.debug(
                f"Target album directory is read-only, skipping cover download: {target_dir}"
            )
            return cover_image_path if _valid_image(cover_image_path) else None

        artwork_result = _fetch_candidate_artwork(
            artist=artist,
            album=album,
            musicbrainz_album_id=musicbrainz_album_id,
        )

        if artwork_result:
            new_artwork_bytes, new_dim = artwork_result
            if not dry_run:
                if existing_dim and any(
                    current > replacement
                    for current, replacement in zip(existing_dim, new_dim)
                ):
                    LOG.detail(
                        f"   ∟ 🛡️  [Quality Shield] Preserved cover: replacement reduces an image dimension ({existing_dim[0]}x{existing_dim[1]} -> {new_dim[0]}x{new_dim[1]})"
                    )
                else:
                    atomic_write_bytes(cover_image_path, new_artwork_bytes)
                    if existing_dim and (new_dim[0] * new_dim[1]) > (
                        existing_dim[0] * existing_dim[1]
                    ):
                        LOG.detail(
                            f"   ∟ 🖼️  Upgraded Cover Art ({existing_dim[0]}x{existing_dim[1]} -> {new_dim[0]}x{new_dim[1]})"
                        )
                    else:
                        LOG.detail("   ∟ 🖼️  Downloaded Cover Art")
            else:
                LOG.detail(
                    f"[DRY-RUN] Would download cover art to {cover_image_path.name}"
                )

    if musicbrainz_releasegroup_id and get_config().fanart_api_key:
        cdart_path = target_dir / "cdart.png"
        if not _valid_image(cdart_path) and not dry_run:
            try:
                fanart_album = fetch_fanart_album(musicbrainz_releasegroup_id)
                if fanart_album and fanart_album.cdart_urls:
                    cdart_bytes = download_fanart_image_bytes(
                        fanart_album.cdart_urls[0]
                    )
                    if _save_image(cdart_path, cdart_bytes):
                        LOG.detail("   ∟ 💿 Downloaded CD disc art -> cdart.png")
            except (httpx.HTTPError, OSError) as error:
                LOG.debug(f"CD art download failed: {error}")

    if _valid_image(cover_image_path):
        return cover_image_path
    return None


def _find_artist_directory(folder_path: Path, artist_name: str) -> Path:
    """Dynamically determine artist root folder by walking the directory hierarchy."""
    clean_artist = normalize_str(artist_name)
    if not clean_artist:
        return folder_path

    current = folder_path.resolve()
    candidates: list[Path] = []
    for _ in range(4):
        candidates.append(current)
        if current.parent == current:
            break
        current = current.parent

    for cand in candidates:
        if normalize_str(cand.name) == clean_artist:
            return cand

    for cand in candidates:
        if (
            not get_config().is_generic_container(cand.name)
            and not get_config().is_disc_folder(cand.name)
            and (
                clean_artist in normalize_str(cand.name)
                or fuzz.ratio(clean_artist, normalize_str(cand.name))
                >= get_config().artist_match_threshold
            )
        ):
            return cand

    return folder_path


def process_artist_artwork(
    folder_path: Path,
    artist_name: str,
    artist_mbid: str | None = None,
    dry_run: bool = False,
) -> None:
    """Ensure artist.jpg, banner.jpg, and optional fanart.tv logo.png exist in the artist root."""
    if not artist_name or artist_name.lower() in {
        "various artists",
        "unknown artist",
        "unknown",
    }:
        return

    artist_dir = _find_artist_directory(folder_path, artist_name)
    if not dry_run and not os.access(artist_dir, os.W_OK):
        LOG.debug(
            f"Artist directory is read-only, skipping artist artwork: {artist_dir}"
        )
        return

    has_artist_image = any(
        _valid_image(artist_dir / filename)
        for filename in ["artist.jpg", "artist.png", "folder.jpg"]
    )
    has_banner_image = any(
        _valid_image(artist_dir / filename)
        for filename in ["banner.jpg", "banner.png", "fanart.jpg"]
    )
    has_logo_image = _valid_image(artist_dir / "logo.png")

    if artist_mbid and get_config().fanart_api_key and not dry_run:
        try:
            fanart_artist = fetch_fanart_artist(artist_mbid)
            if fanart_artist:
                if not has_logo_image and fanart_artist.logo_urls:
                    logo_bytes = download_fanart_image_bytes(fanart_artist.logo_urls[0])
                    if _save_image(artist_dir / "logo.png", logo_bytes):
                        LOG.detail(
                            f"   ∟ 💎 Downloaded artist logo: {escape(artist_name)} -> logo.png"
                        )
                if not has_banner_image and fanart_artist.background_urls:
                    bg_bytes = download_fanart_image_bytes(
                        fanart_artist.background_urls[0]
                    )
                    if _save_image(artist_dir / "fanart.jpg", bg_bytes):
                        has_banner_image = True
                        LOG.detail(
                            f"   ∟ 🎨 Downloaded artist fanart: {escape(artist_name)} -> fanart.jpg"
                        )
                if not has_artist_image and fanart_artist.thumb_urls:
                    thumb_bytes = download_fanart_image_bytes(
                        fanart_artist.thumb_urls[0]
                    )
                    if _save_image(artist_dir / "artist.jpg", thumb_bytes):
                        has_artist_image = True
                        LOG.detail(
                            f"   ∟ 👤 Downloaded artist thumbnail: {escape(artist_name)} -> artist.jpg"
                        )
                if not has_banner_image and fanart_artist.banner_urls:
                    banner_bytes = download_fanart_image_bytes(
                        fanart_artist.banner_urls[0]
                    )
                    if _save_image(artist_dir / "banner.jpg", banner_bytes):
                        has_banner_image = True
                        LOG.detail(
                            f"   ∟ 🎨 Downloaded artist banner: {escape(artist_name)} -> banner.jpg"
                        )
        except (httpx.HTTPError, OSError) as error:
            LOG.debug(f"Fanart artist fetch failed: {error}")

    if has_artist_image and has_banner_image:
        _repair_artist_image_copies(folder_path, artist_dir, dry_run=dry_run)
        return

    try:
        thumbnail_bytes, banner_bytes = fetch_artist_images(
            artist_name, artist_mbid=artist_mbid
        )
    except (httpx.HTTPError, OSError) as error:
        LOG.debug(f"Failed to fetch artist artwork for {artist_name}: {error}")
        _repair_artist_image_copies(folder_path, artist_dir, dry_run=dry_run)
        return

    if thumbnail_bytes and not has_artist_image and not dry_run:
        try:
            if _save_image(artist_dir / "artist.jpg", thumbnail_bytes):
                LOG.detail(
                    f"   ∟ 👤 Downloaded artist avatar: {escape(artist_name)} -> artist.jpg"
                )
        except OSError as error:
            LOG.debug(f"Failed to write artist avatar image: {error}")

    if banner_bytes and not has_banner_image and not dry_run:
        try:
            if _save_image(artist_dir / "banner.jpg", banner_bytes):
                LOG.detail(
                    f"   ∟ 🎨 Downloaded artist banner: {escape(artist_name)} -> banner.jpg"
                )
        except OSError as error:
            LOG.debug(f"Failed to write artist banner image: {error}")
    _repair_artist_image_copies(folder_path, artist_dir, dry_run=dry_run)


def process_label_artwork(
    folder_path: Path,
    label_mbid: str | None = None,
    label_name: str | None = None,
    dry_run: bool = False,
) -> None:
    """Download transparent record label logo (label.png) from Fanart.tv if available."""
    if not label_mbid or not get_config().fanart_api_key or dry_run:
        return

    label_path = folder_path / "label.png"
    if _valid_image(label_path) or not os.access(folder_path, os.W_OK):
        return

    try:
        fanart_label = fetch_fanart_label(label_mbid)
        if fanart_label and fanart_label.label_urls:
            logo_bytes = download_fanart_image_bytes(fanart_label.label_urls[0])
            if _save_image(label_path, logo_bytes):
                name_display = escape(label_name) if label_name else "Record Label"
                LOG.detail(
                    f"   ∟ 🏷️  Downloaded label logo: {name_display} -> label.png"
                )
    except (httpx.HTTPError, OSError) as error:
        LOG.debug(f"Record label logo download failed for {label_mbid}: {error}")
