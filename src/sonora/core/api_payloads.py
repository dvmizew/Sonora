from typing import Any


def require_api_object(payload: object, context: str) -> dict[str, Any]:
    """Reject malformed provider envelopes instead of caching them as missing metadata."""
    if not isinstance(payload, dict):
        raise TypeError(f"{context} must be a JSON object")
    return payload


def require_api_records(payload: object, context: str) -> list[dict[str, Any]]:
    """Validate arrays before provider-specific metadata extraction."""
    if not isinstance(payload, list) or any(
        not isinstance(record, dict) for record in payload
    ):
        raise TypeError(f"{context} must be an array of JSON objects")
    return payload


def require_api_strings(payload: object, context: str) -> list[str]:
    """Validate textual arrays before exposing provider credits or lyric lines."""
    if not isinstance(payload, list) or any(
        not isinstance(value, str) for value in payload
    ):
        raise TypeError(f"{context} must be an array of strings")
    return payload
