import io
import threading
from pathlib import Path

import httpx
import imagehash
from PIL import Image, ImageOps, UnidentifiedImageError

from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.services.deezer import fetch_deezer_cover_art_url
from sonora.services.itunes import fetch_itunes_cover_art_url
from sonora.services.musicbrainz import fetch_cover_art_archive_url
from sonora.services.theaudiodb import fetch_artist_images


def _load_normalized_image(image_bytes: bytes) -> Image.Image:
    """
    Decodes an image from bytes, forces full memory loading, applies EXIF transposition,
    and flattens transparency channels (RGBA, LA, P) onto an opaque white RGB canvas.
    """
    with Image.open(io.BytesIO(image_bytes)) as raw_image:
        raw_image.load()
        image: Image.Image = ImageOps.exif_transpose(raw_image) or raw_image.copy()

    if image.mode in ("RGBA", "LA", "P"):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")

    if image.mode != "RGB":
        return image.convert("RGB")

    return image


def check_image_similarity(
    first_image_bytes: bytes,
    second_image_bytes: bytes,
    max_distance: int = 12,
    threshold: float | None = None,
) -> bool:
    """
    Uses perceptual hashing (pHash) via ImageHash to check if two images represent the same artwork.
    Applies EXIF auto-rotation and alpha channel composite normalization.
    A Hamming distance <= max_distance (default: 12 out of 64 bits, ~81% visual similarity) indicates a match.
    """
    if not (first_image_bytes and second_image_bytes):
        return False
    if threshold is not None:
        max_distance = round((1.0 - threshold) * 64)

    try:
        first_image = _load_normalized_image(first_image_bytes)
        second_image = _load_normalized_image(second_image_bytes)

        first_hash = imagehash.phash(first_image)
        second_hash = imagehash.phash(second_image)

        return (first_hash - second_hash) <= max_distance
    except (OSError, ValueError, UnidentifiedImageError) as error:
        LOG.debug(f"Perceptual image comparison failed: {error}")
        return True  # Fallback to True to allow upgrade if something fails


_cover_locks: dict[str, threading.Lock] = {}
_cover_meta_lock = threading.Lock()


def _get_cover_lock(folder_path: Path) -> threading.Lock:
    key = str(folder_path.resolve())
    with _cover_meta_lock:
        if len(_cover_locks) > 1000:
            _cover_locks.clear()
        if key not in _cover_locks:
            _cover_locks[key] = threading.Lock()
        return _cover_locks[key]


def process_album_cover_art(
    folder_path: Path,
    artist: str,
    album: str,
    musicbrainz_album_id: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> Path | None:
    """
    Downloads and validates high-resolution album cover art (cover.jpg).
    Tries iTunes API first, then Cover Art Archive fallback.
    Returns Path to cover.jpg if present/downloaded, else None.
    """
    cover_image_path = folder_path / "cover.jpg"
    with _get_cover_lock(folder_path):
        artwork_downloaded = (
            cover_image_path.exists()
            and cover_image_path.stat().st_size > 0
            and not force
        )
        if not artwork_downloaded and not dry_run:
            cover_image_path.touch()

    if not artwork_downloaded:
        artwork_url = None
        if musicbrainz_album_id:
            artwork_url = fetch_cover_art_archive_url(musicbrainz_album_id)
        if not artwork_url:
            artwork_url = fetch_itunes_cover_art_url(artist, album)
        if not artwork_url:
            artwork_url = fetch_deezer_cover_art_url(artist, album)

        if artwork_url:
            try:
                response = SESSION.get(artwork_url, timeout=15)
                response.raise_for_status()
                new_artwork_bytes = response.content

                with _get_cover_lock(folder_path):
                    if not dry_run:
                        existing_bytes = (
                            cover_image_path.read_bytes()
                            if (
                                cover_image_path.exists()
                                and cover_image_path.stat().st_size > 0
                            )
                            else None
                        )
                        if (
                            existing_bytes
                            and not force
                            and not check_image_similarity(
                                existing_bytes, new_artwork_bytes
                            )
                        ):
                            LOG.info(
                                "   ∟ 🖼️  Skipped iTunes cover upgrade: visual mismatch"
                            )
                        else:
                            cover_image_path.write_bytes(new_artwork_bytes)
                            LOG.info("   ∟ 🖼️  Downloaded Cover Art")
                    else:
                        LOG.info(
                            f"[DRY-RUN] Would download cover art to {cover_image_path.name}"
                        )
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as error:
                LOG.debug(f"Cover art download failed: {error}")
                with _get_cover_lock(folder_path):
                    if (
                        not dry_run
                        and cover_image_path.exists()
                        and cover_image_path.stat().st_size == 0
                    ):
                        cover_image_path.unlink(missing_ok=True)
        else:
            with _get_cover_lock(folder_path):
                if (
                    not dry_run
                    and cover_image_path.exists()
                    and cover_image_path.stat().st_size == 0
                ):
                    cover_image_path.unlink(missing_ok=True)

    with _get_cover_lock(folder_path):
        if not dry_run and (
            not cover_image_path.exists() or cover_image_path.stat().st_size == 0
        ):
            return None
    return cover_image_path if cover_image_path.exists() else None


def process_artist_artwork(
    folder_path: Path, artist_name: str, dry_run: bool = False
) -> None:
    """Ensure artist.jpg and banner.jpg exist in the artist's root folder."""
    if not artist_name or artist_name in ["Various Artists", "Unknown Artist"]:
        return

    parent = folder_path.parent
    base_name = folder_path.name
    parent_base = parent.name

    artist_dir = (
        parent
        if parent_base not in ["FLAC", "Music", ""] and base_name != "Singles"
        else folder_path
    )
    if parent_base == "Singles":
        artist_dir = parent.parent

    has_artist_image = any(
        (artist_dir / filename).exists()
        for filename in ["artist.jpg", "artist.png", "folder.jpg"]
    )
    has_banner_image = any(
        (artist_dir / filename).exists()
        for filename in ["banner.jpg", "banner.png", "fanart.jpg"]
    )

    if has_artist_image and has_banner_image:
        return

    thumbnail_bytes, banner_bytes = fetch_artist_images(artist_name)
    if thumbnail_bytes and not has_artist_image and not dry_run:
        try:
            (artist_dir / "artist.jpg").write_bytes(thumbnail_bytes)
            LOG.info(f"   ∟ 👤 Downloaded artist avatar: {artist_name} -> artist.jpg")
        except OSError as error:
            LOG.debug(f"Failed to write artist avatar image: {error}")

    if banner_bytes and not has_banner_image and not dry_run:
        try:
            (artist_dir / "banner.jpg").write_bytes(banner_bytes)
            LOG.info(f"   ∟ 🎨 Downloaded artist banner: {artist_name} -> banner.jpg")
        except OSError as error:
            LOG.debug(f"Failed to write artist banner image: {error}")
