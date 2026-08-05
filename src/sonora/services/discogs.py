"""
Discogs API service client.
"""

from types import ModuleType
from typing import Any

from sonora.core.exceptions import APIServiceError

discogs_client: ModuleType | None = None
try:
    import discogs_client  # type: ignore
except ImportError:
    pass


def search_discogs_release(artist: str, album: str, user_token: str | None = None) -> dict[str, Any] | None:
    """
    Search Discogs for album release metadata.
    Requires a Discogs user token.
    """
    if not user_token:
        return None

    if discogs_client is None:
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
