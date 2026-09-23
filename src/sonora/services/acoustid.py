import subprocess
import threading
from pathlib import Path

import acoustid

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.constants import RATE_LIMIT_ACOUSTID
from sonora.core.logger import LOG
from sonora.core.utils import (
    RateLimiter,
    is_valid_uuid,
    match_score,
    normalize_str,
    safe_float,
)

_ACOUSTID_CACHE: dict[tuple[str, int, int], tuple[float, str]] = {}
_ACOUSTID_LOCK = threading.RLock()


def _fingerprint_file_with_timeout(
    file_path: Path, timeout: float = 20.0
) -> tuple[float, str]:
    """Execute fpcalc with an explicit timeout to prevent thread hangs on corrupt files."""
    command = ["fpcalc", "-length", "120", str(file_path.resolve())]
    try:
        process_result = subprocess.run(
            command, capture_output=True, check=True, timeout=timeout
        )
        duration: float | None = None
        fingerprint: str | None = None
        for line in process_result.stdout.splitlines():
            if line.startswith(b"DURATION="):
                duration = safe_float(
                    line.split(b"=", 1)[1].decode("ascii", errors="replace")
                )
            elif line.startswith(b"FINGERPRINT="):
                fingerprint = line.split(b"=", 1)[1].decode("ascii")
        if duration is not None and fingerprint is not None:
            return duration, str(fingerprint)
    except (subprocess.SubprocessError, OSError) as error:
        LOG.debug(f"Direct fpcalc execution with timeout failed: {error}")

    # Fallback to standard acoustid library call if direct invocation fails
    duration_val, fp_val = acoustid.fingerprint_file(str(file_path.resolve()))
    return float(duration_val), str(fp_val)


def fingerprint_audio_file(file_path: Path) -> tuple[float, str]:
    """
    Generate Chromaprint acoustic fingerprint for an audio file.
    Caches results in memory based on path, mtime, and size to avoid duplicate decoding.
    Returns (duration, fingerprint_string).
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        stat = file_path.stat()
        cache_key = (str(file_path.resolve()), stat.st_mtime_ns, stat.st_size)
        with _ACOUSTID_LOCK:
            if cache_key in _ACOUSTID_CACHE:
                return _ACOUSTID_CACHE[cache_key]

        duration, fingerprint = _fingerprint_file_with_timeout(file_path)
        result = (float(duration), str(fingerprint))
        with _ACOUSTID_LOCK:
            _ACOUSTID_CACHE[cache_key] = result
        return result
    except (
        acoustid.AcoustidError,
        acoustid.WebServiceError,
        OSError,
        subprocess.SubprocessError,
    ) as error:
        raise RuntimeError(
            f"Chromaprint fingerprinting failed for {file_path}: {error}"
        ) from error


_ACOUSTID_LIMITER = RateLimiter(interval_seconds=RATE_LIMIT_ACOUSTID)


def lookup_acoustid(
    file_path: Path,
    api_key: str | None = None,
    expected_artist: str | None = None,
    expected_title: str | None = None,
) -> str | None:
    """
    Fingerprints an audio file and fetches MusicBrainz Recording ID from AcoustID.
    Ranks candidate matches using a combination of acoustic score and title/artist match_score.
    Returns the MBID string if found, otherwise None.
    """
    if not api_key:
        return None

    try:
        duration, fingerprint = fingerprint_audio_file(file_path)
        cache_key = f"acoustid:{fingerprint}"
        if expected_artist and expected_title:
            cache_key += (
                f":{normalize_str(expected_artist)}:{normalize_str(expected_title)}"
            )

        cached = get_cached_api(cache_key)
        if cached is not None:
            return str(cached) if cached else None

        _ACOUSTID_LIMITER.wait()

        acoustid_lookup_payload = acoustid.lookup(api_key, fingerprint, duration)
        best_mbid = None
        best_combined_score = -1.0

        for (
            score,
            recording_id,
            candidate_title,
            candidate_artist,
        ) in acoustid.parse_lookup_result(acoustid_lookup_payload):
            if score >= 0.75 and recording_id and is_valid_uuid(str(recording_id)):
                combined_score = float(score) * 100.0
                if (
                    expected_artist
                    and expected_title
                    and candidate_title
                    and candidate_artist
                ):
                    text_score = match_score(
                        expected_artist,
                        expected_title,
                        str(candidate_artist),
                        str(candidate_title),
                    )
                    if text_score < 80.0:
                        continue
                    combined_score = (float(score) * 40.0) + (text_score * 0.6)

                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_mbid = str(recording_id)

        set_cached_api(cache_key, best_mbid)
        return best_mbid or None
    except (acoustid.AcoustidError, acoustid.WebServiceError, OSError) as error:
        LOG.debug(f"AcoustID lookup failed for {file_path.name}: {error}")
        return None
