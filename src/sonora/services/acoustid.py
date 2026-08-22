from pathlib import Path

import acoustid

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.logger import LOG
from sonora.core.utils import RateLimiter


def fingerprint_audio_file(file_path: Path) -> tuple[float, str]:
    """
    Generate Chromaprint acoustic fingerprint for an audio file.
    Returns (duration, fingerprint_string).
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        duration, fingerprint = acoustid.fingerprint_file(str(file_path))
        return float(duration), str(fingerprint)
    except (acoustid.AcoustidError, acoustid.WebServiceError, OSError, ValueError, RuntimeError) as e:
        raise RuntimeError(f"Chromaprint fingerprinting failed for {file_path}: {e}") from e


_ACOUSTID_LIMITER = RateLimiter(interval_seconds=0.4)
_ACOUSTID_FAILURES = 0
_MAX_ACOUSTID_FAILURES = 3

def lookup_acoustid(file_path: Path, api_key: str | None = None) -> str | None:
    """
    Fingerprints an audio file and fetches MusicBrainz Recording ID from AcoustID.
    Returns the MBID string if found, otherwise None.
    """
    global _ACOUSTID_FAILURES
    if not api_key or _ACOUSTID_FAILURES >= _MAX_ACOUSTID_FAILURES:
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
                _ACOUSTID_FAILURES = 0
                return rec_str
        return None
    except (acoustid.AcoustidError, acoustid.WebServiceError, OSError, ValueError, KeyError, RuntimeError) as e:
        LOG.debug(f"AcoustID lookup failed for {file_path.name}: {e}")
        _ACOUSTID_FAILURES += 1
        return None
