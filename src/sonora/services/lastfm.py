import httpx

from sonora.core.api_payloads import require_api_object, require_api_records
from sonora.core.cache import get_cached_api, set_cached_api
from sonora.core.coordination import coalesce_lookup
from sonora.core.http import SESSION
from sonora.core.logger import LOG
from sonora.core.text import is_valid_uuid, normalize_str


@coalesce_lookup
def fetch_lastfm_tags(
    artist: str,
    title: str,
    api_key: str | None = None,
    mbid: str | None = None,
) -> list[str]:
    """Fetch up to five top tags, trying the recording ID before artist and title."""
    if not api_key:
        return []
    name_key = f"lastfm:{normalize_str(artist)}:{normalize_str(title)}:"
    cache_key = f"{name_key}{mbid or ''}"
    cached = get_cached_api(cache_key)
    if isinstance(cached, list):
        return cached

    queries: list[dict[str, str]] = []
    if mbid and is_valid_uuid(mbid):
        queries.append({"mbid": mbid})
    if artist and title:
        queries.append({"artist": artist, "track": title})
    for identity in queries:
        query_key = f"{name_key}{identity.get('mbid', '')}"
        if query_key != cache_key:
            cached = get_cached_api(query_key)
            if isinstance(cached, list):
                set_cached_api(cache_key, cached)
                return cached
        try:
            response = SESSION.get(
                "https://ws.audioscrobbler.com/2.0/",
                params={
                    "method": "track.getTopTags",
                    "api_key": api_key,
                    "format": "json",
                    **identity,
                },
                timeout=5,
            )
            response.raise_for_status()
            payload = require_api_object(response.json(), "Last.fm response")
            error_code = payload.get("error")
            if error_code:
                LOG.debug(f"Last.fm API error: {payload.get('message', '')}")
                if error_code in (6, 7) and "mbid" in identity:
                    continue
                return []
            top_tags = require_api_object(payload.get("toptags"), "Last.fm top tags")
            tags = require_api_records(top_tags.get("tag", []), "Last.fm tag records")
            names = [
                tag["name"].title()
                for tag in tags
                if isinstance(tag.get("name"), str) and tag["name"]
            ][:5]
            if not names and "mbid" in identity and len(queries) > 1:
                continue
            set_cached_api(query_key, names)
            if query_key != cache_key:
                set_cached_api(cache_key, names)
            return names
        except (httpx.HTTPError, OSError) as error:
            LOG.debug(f"Last.fm tag fetch failed for {artist} - {title}: {error}")
    return []
