"""
Genius API service client for song descriptions and background stories.
"""

from sonora.core.http import SESSION
from sonora.core.exceptions import APIServiceError
from sonora.core.utils import RateLimiter

_GENIUS_LIMITER = RateLimiter(interval_seconds=0.5)


def fetch_genius_description(artist: str, title: str, api_token: str | None = None) -> str | None:
    """
    Fetch song description/story from Genius API.
    """
    if not api_token or not artist or not title:
        return None

    _GENIUS_LIMITER.wait()
    try:
        # Step 1: Search for the song
        from sonora.core.utils import normalize_str
        query = f"{normalize_str(artist)} {normalize_str(title)}"
        search_url = "https://api.genius.com/search"
        resp = SESSION.get(
            search_url,
            params={"q": query},
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=5
        )
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("response", {}).get("hits", [])
        if not hits:
            return None
        api_path = hits[0].get("result", {}).get("api_path")
        if not api_path:
            return None

        # Step 2: Fetch song details and plain text description
        _GENIUS_LIMITER.wait()
        song_url = f"https://api.genius.com{api_path}"
        resp_song = SESSION.get(
            song_url,
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=5
        )
        resp_song.raise_for_status()
        data_song = resp_song.json()
        song = data_song.get("response", {}).get("song", {})
        desc = song.get("description", {}).get("plain", "")
        if desc and desc.strip() != "?" and "lyrics for this song" not in desc.lower():
            return desc.strip()
        return None

    except Exception as e:
        raise APIServiceError(f"Genius description fetch failed for {artist} - {title}: {e}") from e
