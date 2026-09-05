import io
from pathlib import Path

import httpx
import imagehash
from PIL import Image, ImageOps, UnidentifiedImageError
from rapidfuzz import fuzz
from rich.markup import escape

from sonora.core.constants import ARTIST_MATCH_THRESHOLD
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.utils import normalize_str
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
        first_image.close()
        second_image.close()
        return bool((first_hash - second_hash) <= max_distance)
    except (OSError, ValueError, UnidentifiedImageError) as error:
        LOG.debug(f"Perceptual image comparison failed: {error}")
        return True


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
    artwork_already_present = (
        cover_image_path.exists() and cover_image_path.stat().st_size > 0 and not force
    )

    if not artwork_already_present:
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
                        temp_path = cover_image_path.with_suffix(".tmp")
                        temp_path.write_bytes(new_artwork_bytes)
                        temp_path.replace(cover_image_path)
                        LOG.info("   ∟ 🖼️  Downloaded Cover Art")
                else:
                    LOG.info(
                        f"[DRY-RUN] Would download cover art to {cover_image_path.name}"
                    )
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as error:
                LOG.debug(f"Cover art download failed: {error}")

    if cover_image_path.exists() and cover_image_path.stat().st_size > 0:
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

    # 1. Prioritize exact match anywhere in the ancestor hierarchy
    for cand in candidates:
        if normalize_str(cand.name) == clean_artist:
            return cand

    # 2. Check for close variation (excluding generic folder names like singles, flac, mp3)
    for cand in candidates:
        cand_norm = normalize_str(cand.name)
        if (
            cand_norm
            and cand_norm not in ("singles", "flac", "mp3", "music")
            and (
                fuzz.ratio(cand_norm, clean_artist) >= ARTIST_MATCH_THRESHOLD
                or fuzz.token_sort_ratio(cand_norm, clean_artist)
                >= ARTIST_MATCH_THRESHOLD
            )
        ):
            return cand

    return folder_path


def process_artist_artwork(
    folder_path: Path, artist_name: str, dry_run: bool = False
) -> None:
    """Ensure artist.jpg and banner.jpg exist in the artist's root folder."""
    if not artist_name or artist_name in ["Various Artists", "Unknown Artist"]:
        return

    artist_dir = _find_artist_directory(folder_path, artist_name)

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

    try:
        thumbnail_bytes, banner_bytes = fetch_artist_images(artist_name)
    except (httpx.HTTPError, OSError, ValueError, RuntimeError) as error:
        LOG.debug(f"Failed to fetch artist artwork for {artist_name}: {error}")
        return

    if thumbnail_bytes and not has_artist_image and not dry_run:
        try:
            (artist_dir / "artist.jpg").write_bytes(thumbnail_bytes)
            LOG.info(
                f"   ∟ 👤 Downloaded artist avatar: {escape(artist_name)} -> artist.jpg"
            )
        except OSError as error:
            LOG.debug(f"Failed to write artist avatar image: {error}")

    if banner_bytes and not has_banner_image and not dry_run:
        try:
            (artist_dir / "banner.jpg").write_bytes(banner_bytes)
            LOG.info(
                f"   ∟ 🎨 Downloaded artist banner: {escape(artist_name)} -> banner.jpg"
            )
        except OSError as error:
            LOG.debug(f"Failed to write artist banner image: {error}")
