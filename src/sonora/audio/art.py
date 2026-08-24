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


def _load_normalized_image(data: bytes) -> Image.Image:
    """
    Decodes an image from bytes, forces full memory loading, applies EXIF transposition,
    and flattens transparency channels (RGBA, LA, P) onto an opaque white RGB canvas.
    """
    with Image.open(io.BytesIO(data)) as raw_img:
        raw_img.load()
        img: Image.Image = ImageOps.exif_transpose(raw_img) or raw_img.copy()

    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")

    if img.mode != "RGB":
        return img.convert("RGB")

    return img


def check_image_similarity(
    data1: bytes,
    data2: bytes,
    max_distance: int = 12,
    threshold: float | None = None,
) -> bool:
    """
    Uses perceptual hashing (pHash) via ImageHash to check if two images represent the same artwork.
    Applies EXIF auto-rotation and alpha channel composite normalization.
    A Hamming distance <= max_distance (default: 12 out of 64 bits, ~81% visual similarity) indicates a match.
    """
    if not (data1 and data2):
        return False
    if threshold is not None:
        max_distance = round((1.0 - threshold) * 64)

    try:
        img1 = _load_normalized_image(data1)
        img2 = _load_normalized_image(data2)

        hash1 = imagehash.phash(img1)
        hash2 = imagehash.phash(img2)

        return (hash1 - hash2) <= max_distance
    except (OSError, ValueError, UnidentifiedImageError) as e:
        LOG.debug(f"Perceptual image comparison failed: {e}")
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
    mb_album_id: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> Path | None:
    """
    Downloads and validates high-resolution album cover art (cover.jpg).
    Tries iTunes API first, then Cover Art Archive fallback.
    Returns Path to cover.jpg if present/downloaded, else None.
    """
    cover_jpg = folder_path / "cover.jpg"
    with _get_cover_lock(folder_path):
        art_downloaded = (
            cover_jpg.exists() and cover_jpg.stat().st_size > 0 and not force
        )
        if not art_downloaded and not dry_run:
            cover_jpg.touch()

    if not art_downloaded:
        art_url = None
        if mb_album_id:
            art_url = fetch_cover_art_archive_url(mb_album_id)
        if not art_url:
            art_url = fetch_itunes_cover_art_url(artist, album)
        if not art_url:
            art_url = fetch_deezer_cover_art_url(artist, album)

        if art_url:
            try:
                resp = SESSION.get(art_url, timeout=15)
                resp.raise_for_status()
                new_art_bytes = resp.content

                with _get_cover_lock(folder_path):
                    if not dry_run:
                        existing_bytes = (
                            cover_jpg.read_bytes()
                            if (cover_jpg.exists() and cover_jpg.stat().st_size > 0)
                            else None
                        )
                        if (
                            existing_bytes
                            and not force
                            and not check_image_similarity(
                                existing_bytes, new_art_bytes
                            )
                        ):
                            LOG.info(
                                "   ∟ 🖼️  Skipped iTunes cover upgrade: visual mismatch"
                            )
                        else:
                            cover_jpg.write_bytes(new_art_bytes)
                            LOG.info("   ∟ 🖼️  Downloaded Cover Art")
                    else:
                        LOG.info(
                            f"[DRY-RUN] Would download cover art to {cover_jpg.name}"
                        )
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as e:
                LOG.debug(f"Cover art download failed: {e}")
                with _get_cover_lock(folder_path):
                    if (
                        not dry_run
                        and cover_jpg.exists()
                        and cover_jpg.stat().st_size == 0
                    ):
                        cover_jpg.unlink(missing_ok=True)
        else:
            with _get_cover_lock(folder_path):
                if not dry_run and cover_jpg.exists() and cover_jpg.stat().st_size == 0:
                    cover_jpg.unlink(missing_ok=True)

    with _get_cover_lock(folder_path):
        if not dry_run and (not cover_jpg.exists() or cover_jpg.stat().st_size == 0):
            return None
    return cover_jpg if cover_jpg.exists() else None


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

    has_artist_img = any(
        (artist_dir / n).exists() for n in ["artist.jpg", "artist.png", "folder.jpg"]
    )
    has_banner_img = any(
        (artist_dir / n).exists() for n in ["banner.jpg", "banner.png", "fanart.jpg"]
    )

    if has_artist_img and has_banner_img:
        return

    thumb_bytes, banner_bytes = fetch_artist_images(artist_name)
    if thumb_bytes and not has_artist_img and not dry_run:
        try:
            (artist_dir / "artist.jpg").write_bytes(thumb_bytes)
            LOG.info(f"   ∟ 👤 Downloaded artist avatar: {artist_name} -> artist.jpg")
        except OSError as e:
            LOG.debug(f"Failed to write artist avatar image: {e}")

    if banner_bytes and not has_banner_img and not dry_run:
        try:
            (artist_dir / "banner.jpg").write_bytes(banner_bytes)
            LOG.info(f"   ∟ 🎨 Downloaded artist banner: {artist_name} -> banner.jpg")
        except OSError as e:
            LOG.debug(f"Failed to write artist banner image: {e}")
