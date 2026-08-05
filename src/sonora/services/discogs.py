"""
Discogs API service client.
"""

from typing import Any

from sonora.core.exceptions import APIServiceError

try:
    import discogs_client  # type: ignore
    HAS_DISCOGS = True
except ImportError:
    discogs_client = None
    HAS_DISCOGS = False


def search_discogs_release(artist: str, album: str, user_token: str | None = None) -> dict[str, Any] | None:
    """
    Search Discogs for album release metadata.
    Requires a Discogs user token.
    """
    if not user_token:
        return None

    if not HAS_DISCOGS or not discogs_client:
        raise APIServiceError("discogs-client library is not installed.")

    try:
        client = discogs_client.Client("Sonora/0.1.0", user_token=user_token)
        results = client.search(album, artist=artist, type="release")
        if results and len(results) > 0:
            first = results[0]
            return {
                "id": getattr(first, "id", None),
                "title": getattr(first, "title", None),
                "year": getattr(first, "year", None),
            }
        return None
    except Exception as e:
        raise APIServiceError(f"Discogs search failed for {artist} - {album}: {e}") from e
