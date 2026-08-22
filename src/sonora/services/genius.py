from sonora.core.http import SESSION
from sonora.core.utils import RateLimiter, normalize_str

_GENIUS_LIMITER = RateLimiter(interval_seconds=0.5)


def fetch_genius_description(artist: str, title: str, api_token: str | None = None) -> str | None:
    """
    Fetch song description/story from Genius API.
    """
    details = fetch_genius_song_details(artist, title, api_token)
    return str(details["description"]) if details and details.get("description") else None


def fetch_genius_song_details(artist: str, title: str, api_token: str | None = None) -> dict[str, object] | None:
    """
    Fetch extended song metadata (description, genius_song_id, featured_artists, producers) from Genius API.
    """
    if not api_token or not artist or not title:
        return None

    _GENIUS_LIMITER.wait()
    try:
        query = f"{normalize_str(artist)} {normalize_str(title)}"
        search_url = "https://api.genius.com/search"
        resp = SESSION.get(
            search_url,
            params={"q": query},
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("response", {}).get("hits", [])
        if not hits:
            return None
        result_hit = hits[0].get("result", {})
        api_path = result_hit.get("api_path")
        if not api_path:
            return None

        _GENIUS_LIMITER.wait()
        song_url = f"https://api.genius.com{api_path}"
        resp_song = SESSION.get(
            song_url,
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=5,
        )
        resp_song.raise_for_status()
        song = resp_song.json().get("response", {}).get("song", {})
        desc = song.get("description", {}).get("plain", "")
        clean_desc = desc.strip() if desc and desc.strip() != "?" and "lyrics for this song" not in desc.lower() else None

        raw_feats = song.get("featured_artists", [])
        feat_arts = [str(a.get("name")) for a in raw_feats if isinstance(a, dict) and a.get("name")]
        raw_prods = song.get("producer_artists", [])
        prods = [str(a.get("name")) for a in raw_prods if isinstance(a, dict) and a.get("name")]

        return {
            "genius_song_id": str(song.get("id")) if song.get("id") else None,
            "description": clean_desc,
            "featured_artists": ", ".join(feat_arts) if feat_arts else None,
            "producers": ", ".join(prods) if prods else None,
        }

    except (OSError, ValueError, KeyError) as e:
        raise RuntimeError(f"Genius details fetch failed for {artist} - {title}: {e}") from e
