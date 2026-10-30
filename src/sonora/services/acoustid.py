import hashlib
import json
import math
import os
import subprocess
import threading
from pathlib import Path

import acoustid
from acoustid import AcoustidError, WebServiceError
from rich.markup import escape

from sonora.core.api_payloads import require_api_object, require_api_records
from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.config import get_config
from sonora.core.constants import RATE_LIMIT_ACOUSTID
from sonora.core.coordination import coalesce_lookup
from sonora.core.logger import LOG
from sonora.core.rate_limit import RateLimiter
from sonora.core.retry import retry_network_call
from sonora.core.text import is_valid_uuid, safe_float
from sonora.core.titles import recording_identity_matches

_ACOUSTID_CACHE: dict[tuple[str, int, int, int], tuple[float, str]] = {}
_ACOUSTID_LOCK = threading.RLock()


def _fingerprint_file_with_timeout(
    file_path: Path, timeout: float = 20.0
) -> tuple[float, str]:
    """Execute fpcalc with an explicit timeout to prevent thread hangs on corrupt files."""
    command = [
        os.environ.get("FPCALC", "fpcalc"),
        "-json",
        "-length",
        "120",
        str(file_path.resolve()),
    ]
    try:
        process_result = subprocess.run(
            command, capture_output=True, check=True, timeout=timeout
        )
    except (subprocess.SubprocessError, OSError) as error:
        LOG.debug(f"Direct fpcalc execution with timeout failed: {error}")

        raise OSError(f"Fingerprint decoding failed for {file_path}") from error
    try:
        fingerprint_payload = require_api_object(
            json.loads(process_result.stdout), "fpcalc output"
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise OSError(f"fpcalc returned invalid JSON for {file_path}") from error
    duration = safe_float(fingerprint_payload.get("duration"))
    fingerprint = fingerprint_payload.get("fingerprint")
    if (
        duration is not None
        and duration > 0
        and isinstance(fingerprint, str)
        and fingerprint.strip()
    ):
        return duration, fingerprint
    raise OSError(f"fpcalc returned incomplete fingerprint for {file_path}")


@coalesce_lookup
def fingerprint_audio_file(file_path: Path) -> tuple[float, str]:
    """
    Generate Chromaprint acoustic fingerprint for an audio file.
    Caches results in memory based on path, mtime, and size to avoid duplicate decoding.
    Returns (duration, fingerprint_string).
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    stat = file_path.stat()
    cache_key = (
        str(file_path.resolve()),
        stat.st_mtime_ns,
        stat.st_ctime_ns,
        stat.st_size,
    )
    with _ACOUSTID_LOCK:
        if cache_key in _ACOUSTID_CACHE:
            return _ACOUSTID_CACHE[cache_key]

    fingerprint_key = (
        "fingerprint:v2:"
        + hashlib.sha256(
            repr((cache_key, os.environ.get("FPCALC", "fpcalc"), 120)).encode("utf-8")
        ).hexdigest()
    )
    cached = get_cached_api(fingerprint_key)
    if isinstance(cached, tuple) and len(cached) == 2:
        cached_duration, cached_fingerprint = cached
        if isinstance(cached_duration, (float, int)) and isinstance(
            cached_fingerprint, str
        ):
            result = (float(cached_duration), cached_fingerprint)
            with _ACOUSTID_LOCK:
                if len(_ACOUSTID_CACHE) >= 1024:
                    _ACOUSTID_CACHE.clear()
                _ACOUSTID_CACHE[cache_key] = result
            return result

    duration, fingerprint = _fingerprint_file_with_timeout(file_path)
    current_stat = file_path.stat()
    if (current_stat.st_mtime_ns, current_stat.st_ctime_ns, current_stat.st_size) != (
        stat.st_mtime_ns,
        stat.st_ctime_ns,
        stat.st_size,
    ):
        raise OSError(f"Audio file changed during fingerprinting: {file_path}")
    result = (float(duration), str(fingerprint))
    set_cached_api(fingerprint_key, result)
    with _ACOUSTID_LOCK:
        if len(_ACOUSTID_CACHE) >= 1024:
            _ACOUSTID_CACHE.clear()
        _ACOUSTID_CACHE[cache_key] = result
    return result


_ACOUSTID_LIMITER = RateLimiter(
    interval_seconds=RATE_LIMIT_ACOUSTID, provider="api.acoustid.org"
)


@coalesce_lookup
def lookup_acoustid(
    file_path: Path,
    api_key: str | None = None,
    expected_artist: str | None = None,
    expected_title: str | None = None,
) -> str | None:
    """
    Fingerprints an audio file and fetches MusicBrainz Recording ID from AcoustID.
    Validates artist/title evidence, then ranks candidates by acoustic confidence.
    Returns the MBID string if found, otherwise None.
    """
    if not api_key:
        return None

    try:
        duration, fingerprint = fingerprint_audio_file(file_path)
        cache_key = f"acoustid:raw:v1:{duration}:{fingerprint}"
        acoustid_lookup_payload = get_cached_api(cache_key)
        if acoustid_lookup_payload is None:
            acoustid_lookup_payload = retry_network_call(
                lambda: acoustid.lookup(
                    api_key, fingerprint, duration, meta=["recordings", "isrcs"]
                ),
                max_retries=3,
                initial_delay=1.0,
                rate_limiter=_ACOUSTID_LIMITER,
                transient_exceptions=(
                    AcoustidError,
                    WebServiceError,
                    OSError,
                    TimeoutError,
                ),
            )
        payload = require_api_object(acoustid_lookup_payload, "AcoustID response")
        recording_durations: dict[tuple[str, float], set[float]] = {}
        for result in require_api_records(
            payload.get("results", []), "AcoustID results"
        ):
            result_score = safe_float(result.get("score"))
            for recording in require_api_records(
                result.get("recordings", []), "AcoustID recordings"
            ):
                recording_duration = safe_float(recording.get("duration"))
                if recording_duration is not None and result_score is not None:
                    recording_durations.setdefault(
                        (str(recording.get("id") or ""), result_score), set()
                    ).add(recording_duration)

        recording_scores: dict[str, float] = {}
        for (
            score,
            recording_id,
            candidate_title,
            candidate_artist,
        ) in acoustid.parse_lookup_result(payload):
            acoustic_score = safe_float(score)
            if (
                acoustic_score is None
                or not (
                    get_config().acoustid_confidence_threshold <= acoustic_score <= 1
                )
                or not is_valid_uuid(str(recording_id))
            ):
                continue
            # The same MBID can occur in several fingerprint results with different scores.
            candidate_durations = recording_durations.get(
                (str(recording_id), acoustic_score), set()
            )
            if any(
                candidate_duration <= 0
                or abs(candidate_duration - duration)
                > get_config().single_duration_tolerance_seconds
                for candidate_duration in candidate_durations
            ):
                continue
            if (expected_artist or expected_title) and not recording_identity_matches(
                expected_artist,
                expected_title or candidate_title,
                candidate_artist,
                candidate_title,
            ):
                continue
            recording_scores[str(recording_id)] = max(
                acoustic_score, recording_scores.get(str(recording_id), 0)
            )

        ranked_recordings = sorted(
            recording_scores, key=recording_scores.__getitem__, reverse=True
        )
        best_mbid = ranked_recordings[0] if ranked_recordings else None
        if len(ranked_recordings) > 1:
            score_gap = (
                recording_scores[ranked_recordings[0]]
                - recording_scores[ranked_recordings[1]]
            )
            margin = get_config().acoustid_match_margin
            if score_gap <= margin or math.isclose(score_gap, margin):
                LOG.warning(
                    f"Ambiguous acoustic match for {escape(file_path.name)}; recording ID left unresolved"
                )
                best_mbid = None

        set_cached_api(cache_key, payload)
        return best_mbid or None
    except (AcoustidError, WebServiceError, OSError) as error:
        LOG.warning(
            f"AcoustID identification unavailable for {escape(file_path.name)}: {escape(str(error))}"
        )
        return None
