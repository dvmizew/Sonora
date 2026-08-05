"""
AcoustID and Chromaprint acoustic fingerprint service client.
"""

from pathlib import Path

from sonora.core.exceptions import APIServiceError

try:
    import acoustid  # type: ignore
    HAS_ACOUSTID = True
except ImportError:
    acoustid = None
    HAS_ACOUSTID = False


def fingerprint_audio_file(file_path: Path) -> tuple[float, str]:
    """
    Generate Chromaprint acoustic fingerprint for an audio file.
    Returns (duration, fingerprint_string).
    """
    if not file_path.exists():
        raise APIServiceError(f"File not found: {file_path}")

    if not HAS_ACOUSTID:
        raise APIServiceError("pyacoustid library is not installed.")

    try:
        duration, fingerprint = acoustid.fingerprint_file(str(file_path))
        return float(duration), str(fingerprint)
    except Exception as e:
        raise APIServiceError(f"Chromaprint fingerprinting failed for {file_path}: {e}") from e


def lookup_acoustid(file_path: Path, api_key: str) -> str | None:
    """
    Lookup track MBID on AcoustID service using Chromaprint fingerprint.
    """
    if not api_key:
        return None

    try:
        duration, fingerprint = fingerprint_audio_file(file_path)
        results = acoustid.lookup(api_key, fingerprint, duration)
        for score, recording_id, title, artist in acoustid.parse_lookup_result(results):
            if score >= 0.8 and recording_id:
                return str(recording_id)
        return None
    except Exception as e:
        raise APIServiceError(f"AcoustID lookup failed for {file_path}: {e}") from e
