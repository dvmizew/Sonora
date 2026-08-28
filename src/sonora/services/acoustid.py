from pathlib import Path

import acoustid

from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.constants import RATE_LIMIT_ACOUSTID
from sonora.core.logger import LOG
from sonora.core.utils import RateLimiter, is_valid_uuid, match_score, normalize_str

_ACOUSTID_CACHE: dict[tuple[str, int, int], tuple[float, str]] = {}


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
        if cache_key in _ACOUSTID_CACHE:
            return _ACOUSTID_CACHE[cache_key]

        duration, fingerprint = acoustid.fingerprint_file(str(file_path.resolve()))
        result = (float(duration), str(fingerprint))
        _ACOUSTID_CACHE[cache_key] = result
        return result
    except (
        acoustid.AcoustidError,
        acoustid.WebServiceError,
        OSError,
        ValueError,
        RuntimeError,
    ) as error:
        raise RuntimeError(
            f"Chromaprint fingerprinting failed for {file_path}: {error}"
        ) from error


_ACOUSTID_LIMITER = RateLimiter(interval_seconds=RATE_LIMIT_ACOUSTID)
_ACOUSTID_FAILURES = 0
_MAX_ACOUSTID_FAILURES = 3


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
    global _ACOUSTID_FAILURES
    if not api_key or _ACOUSTID_FAILURES >= _MAX_ACOUSTID_FAILURES:
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

        results = acoustid.lookup(api_key, fingerprint, duration)
        best_mbid = None
        best_combined_score = -1.0

        for (
            score,
            recording_id,
            candidate_title,
            candidate_artist,
        ) in acoustid.parse_lookup_result(results):
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
                    combined_score = (float(score) * 40.0) + (text_score * 0.6)

                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_mbid = str(recording_id)

        set_cached_api(cache_key, best_mbid)
        if best_mbid:
            _ACOUSTID_FAILURES = 0
            return best_mbid
        return None
    except (
        acoustid.AcoustidError,
        acoustid.WebServiceError,
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
    ) as error:
        LOG.debug(f"AcoustID lookup failed for {file_path.name}: {error}")
        _ACOUSTID_FAILURES += 1
        return None
