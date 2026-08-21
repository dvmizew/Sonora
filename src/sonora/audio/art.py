import io

from sonora.core.logger import LOG

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    Image = None
    UnidentifiedImageError = OSError


def check_image_similarity(data1: bytes, data2: bytes, threshold: float = 0.82) -> bool:
    """
    Uses grayscale correlation to check if two images are likely the same art (e.g. Standard vs Deluxe).
    Returns True if correlation >= threshold, else False.
    """
    if not (data1 and data2):
        return False
    if Image is None:
        return True
    try:
        # Load and resize to 64x64 grayscale for fast comparison
        img1 = Image.open(io.BytesIO(data1)).convert('L').resize((64, 64), Image.Resampling.LANCZOS)
        img2 = Image.open(io.BytesIO(data2)).convert('L').resize((64, 64), Image.Resampling.LANCZOS)

        raw1 = [img1.getpixel((x, y)) for y in range(64) for x in range(64)]
        raw2 = [img2.getpixel((x, y)) for y in range(64) for x in range(64)]
        pixels1: list[float] = [float(p[0]) if isinstance(p, (tuple, list)) else float(p or 0) for p in raw1]
        pixels2: list[float] = [float(p[0]) if isinstance(p, (tuple, list)) else float(p or 0) for p in raw2]

        n = len(pixels1)
        mean1 = sum(pixels1) / n
        mean2 = sum(pixels2) / n

        var1 = sum((x - mean1) ** 2 for x in pixels1)
        var2 = sum((y - mean2) ** 2 for y in pixels2)

        if var1 == 0 or var2 == 0:
            return True

        covar = sum((x - mean1) * (y - mean2) for x, y in zip(pixels1, pixels2))
        corr = covar / ((var1 * var2) ** 0.5)

        return float(corr) >= threshold
    except (OSError, ValueError, UnidentifiedImageError) as e:
        LOG.debug(f"Image comparison failed: {e}")
        return True  # Fallback to True to allow upgrade if something fails
