from pathlib import Path

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.exceptions import APIServiceError

try:
    import acoustid
except ImportError:
    acoustid = None


def fingerprint_audio_file(file_path: Path) -> tuple[float, str]:
    """
    Generate Chromaprint acoustic fingerprint for an audio file.
    Returns (duration, fingerprint_string).
    """
    if not file_path.exists():
        raise APIServiceError(f"File not found: {file_path}")

    if acoustid is None:
        raise APIServiceError("pyacoustid library is not installed.")

    try:
        duration, fingerprint = acoustid.fingerprint_file(str(file_path))
        return float(duration), str(fingerprint)
    except Exception as e:
        raise APIServiceError(f"Chromaprint fingerprinting failed for {file_path}: {e}") from e


from sonora.core.utils import RateLimiter

_ACOUSTID_LIMITER = RateLimiter(interval_seconds=0.4)

def lookup_acoustid(file_path: Path, api_key: str) -> str | None:
    """
    Lookup track MBID on AcoustID service using Chromaprint fingerprint.
    """
    if not api_key or acoustid is None:
        return None

    try:
        duration, fingerprint = fingerprint_audio_file(file_path)
        cache_key = f"acoustid:{fingerprint}"
        cached = get_cached_api(cache_key)
        if cached is not None:
            return str(cached)

        _ACOUSTID_LIMITER.wait()
        results = acoustid.lookup(api_key, fingerprint, duration)
        for score, recording_id, _title, _artist in acoustid.parse_lookup_result(results):
            if score >= 0.8 and recording_id:
                rec_str = str(recording_id)
                set_cached_api(cache_key, rec_str)
                return rec_str
        return None
    except Exception as e:
        raise APIServiceError(f"AcoustID lookup failed for {file_path}: {e}") from e
