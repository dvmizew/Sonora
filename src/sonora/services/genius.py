"""
Genius API service client for song descriptions and background stories.
Extracted from script.py.
"""

import json
import urllib.parse
import urllib.request

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
        query = f"{artist} {title}"
        search_url = f"https://api.genius.com/search?q={urllib.parse.quote(query)}"
        req = urllib.request.Request(
            search_url,
            headers={
                "Authorization": f"Bearer {api_token}",
                "User-Agent": "Sonora/0.1.0 (+https://github.com/dvmizew/Sonora)"
            }
        )

        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            hits = data.get("response", {}).get("hits", [])
            if not hits:
                return None
            api_path = hits[0].get("result", {}).get("api_path")
            if not api_path:
                return None

        # Step 2: Fetch song details and plain text description
        _GENIUS_LIMITER.wait()
        song_url = f"https://api.genius.com{api_path}"
        req_song = urllib.request.Request(
            song_url,
            headers={
                "Authorization": f"Bearer {api_token}",
                "User-Agent": "Sonora/0.1.0 (+https://github.com/dvmizew/Sonora)"
            }
        )

        with urllib.request.urlopen(req_song, timeout=5) as resp_song:
            data_song = json.loads(resp_song.read().decode("utf-8"))
            song = data_song.get("response", {}).get("song", {})
            desc = song.get("description", {}).get("plain", "")
            if desc and "?" not in desc[:20] and "lyrics for this song" not in desc.lower():
                return str(desc).strip()
            return None

    except Exception as e:
        raise APIServiceError(f"Genius description fetch failed for {artist} - {title}: {e}") from e
