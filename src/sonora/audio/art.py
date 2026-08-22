import io
import threading
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.services.itunes import fetch_itunes_cover_art_url
from sonora.services.musicbrainz import fetch_cover_art_archive_url
from sonora.services.theaudiodb import fetch_artist_images


def check_image_similarity(data1: bytes, data2: bytes, threshold: float = 0.82) -> bool:
    """
    Uses grayscale correlation via NumPy to check if two images are likely the same art.
    Returns True if correlation >= threshold, else False.
    """
    if not (data1 and data2):
        return False
    try:
        img1 = Image.open(io.BytesIO(data1)).convert('L').resize((64, 64), Image.Resampling.LANCZOS)
        img2 = Image.open(io.BytesIO(data2)).convert('L').resize((64, 64), Image.Resampling.LANCZOS)

        arr1 = np.asarray(img1, dtype=np.float64).ravel()
        arr2 = np.asarray(img2, dtype=np.float64).ravel()

        std1 = np.std(arr1)
        std2 = np.std(arr2)
        if std1 == 0 or std2 == 0:
            return True

        corr = float(np.corrcoef(arr1, arr2)[0, 1])
        return corr >= threshold
    except (OSError, ValueError, UnidentifiedImageError) as e:
        LOG.debug(f"Image comparison failed: {e}")
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
        art_downloaded = cover_jpg.exists() and cover_jpg.stat().st_size > 0 and not force
        if not art_downloaded and not dry_run:
            cover_jpg.touch()

    if not art_downloaded:
        art_url = fetch_itunes_cover_art_url(artist, album)
        if not art_url and mb_album_id:
            art_url = fetch_cover_art_archive_url(mb_album_id)

        if art_url:
            try:
                resp = SESSION.get(art_url, timeout=15)
                resp.raise_for_status()
                new_art_bytes = resp.content

                with _get_cover_lock(folder_path):
                    if not dry_run:
                        existing_bytes = cover_jpg.read_bytes() if (cover_jpg.exists() and cover_jpg.stat().st_size > 0) else None
                        if existing_bytes and not check_image_similarity(existing_bytes, new_art_bytes):
                            LOG.info("   ∟ 🖼️  Skipped iTunes cover upgrade: visual mismatch")
                        else:
                            cover_jpg.write_bytes(new_art_bytes)
                            LOG.info("   ∟ 🖼️  Downloaded Cover Art")
                    else:
                        LOG.info(f"[DRY-RUN] Would download cover art to {cover_jpg.name}")
            except (OSError, ValueError, RuntimeError) as e:
                LOG.debug(f"Cover art download failed: {e}")
                with _get_cover_lock(folder_path):
                    if not dry_run and cover_jpg.exists() and cover_jpg.stat().st_size == 0:
                        cover_jpg.unlink(missing_ok=True)
        else:
            with _get_cover_lock(folder_path):
                if not dry_run and cover_jpg.exists() and cover_jpg.stat().st_size == 0:
                    cover_jpg.unlink(missing_ok=True)

    with _get_cover_lock(folder_path):
        if not dry_run and (not cover_jpg.exists() or cover_jpg.stat().st_size == 0):
            return None
    return cover_jpg if cover_jpg.exists() else None


def process_artist_artwork(folder_path: Path, artist_name: str, dry_run: bool = False) -> None:
    """Ensure artist.jpg and banner.jpg exist in the artist's root folder."""
    if not artist_name or artist_name in ["Various Artists", "Unknown Artist"]:
        return

    parent = folder_path.parent
    base_name = folder_path.name
    parent_base = parent.name

    artist_dir = parent if parent_base not in ["FLAC", "Music", ""] and base_name != "Singles" else folder_path
    if parent_base == "Singles":
        artist_dir = parent.parent

    has_artist_img = any((artist_dir / n).exists() for n in ["artist.jpg", "artist.png", "folder.jpg"])
    has_banner_img = any((artist_dir / n).exists() for n in ["banner.jpg", "banner.png", "fanart.jpg"])

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
