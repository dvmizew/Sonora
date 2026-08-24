from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.utils import RateLimiter, clean_title, match_score

_GENIUS_LIMITER = RateLimiter(interval_seconds=0.5)


def fetch_genius_description(
    artist: str, title: str, api_token: str | None = None
) -> str | None:
    details = fetch_genius_song_details(artist, title, api_token)
    return (
        str(details["description"])
        if details and details.get("description")
        else None
    )


def fetch_genius_song_details(
    artist: str, title: str, api_token: str | None = None
) -> dict[str, object] | None:
    if not api_token or not artist or not title:
        return None

    _GENIUS_LIMITER.wait()
    try:
        c_title = clean_title(title)
        query = f"{artist} {c_title}"
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

        best_hit = None
        best_score = 0.0

        for hit in hits:
            res_item = hit.get("result", {})
            hit_artist = str(res_item.get("primary_artist", {}).get("name", ""))
            hit_title = str(res_item.get("title", ""))

            score = match_score(artist, c_title, hit_artist, hit_title)
            if score > best_score:
                best_score = score
                best_hit = res_item

        if not best_hit or best_score < 70.0:
            return None

        api_path = best_hit.get("api_path")
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
        song_data = resp_song.json().get("response", {}).get("song", {})

        desc_plain = song_data.get("description", {}).get("plain")
        if desc_plain and "Lyrics for this song are unavailable" in str(desc_plain):
            desc_plain = None

        genius_song_id = song_data.get("id")

        # Parse featured artists
        featured_list = song_data.get("featured_artists", [])
        featured_names = [
            str(f["name"]) for f in featured_list if isinstance(f, dict) and f.get("name")
        ]

        # Parse producers
        producer_list = song_data.get("producer_artists", [])
        producer_names = [
            str(p["name"]) for p in producer_list if isinstance(p, dict) and p.get("name")
        ]

        return {
            "genius_song_id": genius_song_id,
            "description": desc_plain,
            "featured_artists": ", ".join(featured_names) if featured_names else None,
            "producers": ", ".join(producer_names) if producer_names else None,
        }

    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        LOG.debug(f"Genius song details fetch failed for {artist} - {title}: {e}")
        return None
