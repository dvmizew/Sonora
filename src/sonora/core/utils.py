import re
import threading
import time
import unicodedata

from rapidfuzz import fuzz


def clean_title(title: str) -> str:
    """Clean track title by removing feat./ft./with brackets and noise."""
    if not title:
        return ""
    t = re.sub(r"\s*[\(\[\{](?:feat\.|ft\.|with).*?[\)\]\}]", "", title, flags=re.IGNORECASE)
    return t.strip()


def is_version_or_remix(s: str) -> bool:
    keywords = ["remix", "rework", "edit", "mix", "live", "acoustic", "instrumental", "version", "demo", "sped up", "slowed", "freestyle"]
    low = s.lower()
    return any(kw in low for kw in keywords)


def match_score(
    query_artist: str,
    query_title: str,
    candidate_artist: str,
    candidate_title: str,
) -> float:
    """
    Calculate a combined 0-100 similarity score between query (artist, title)
    and candidate (artist, title) using RapidFuzz WRatio and ratio with version penalties.
    """
    if not query_title or not candidate_title:
        return 0.0

    q_a = clean_title(query_artist).lower()
    c_a = clean_title(candidate_artist).lower()
    q_t = clean_title(query_title).lower()
    c_t = clean_title(candidate_title).lower()

    if q_t == c_t:
        title_score = 100.0
    else:
        title_wratio = fuzz.WRatio(q_t, c_t)
        title_ratio = fuzz.ratio(q_t, c_t)
        if len(q_t) <= 3:
            title_score = float(title_ratio)
        else:
            title_score = max(title_wratio, title_ratio)

    if not is_version_or_remix(q_t) and is_version_or_remix(c_t):
        title_score -= 35.0

    title_score = max(0.0, min(100.0, title_score))

    if q_a and c_a:
        artist_w = fuzz.WRatio(q_a, c_a)
        artist_token = fuzz.token_set_ratio(q_a, c_a)
        artist_score = max(artist_w, artist_token)
        return (title_score * 0.6) + (artist_score * 0.4)

    return float(title_score)


def normalize_str(s: str | None) -> str:
    """
    Converts to lowercase, normalizes NFD diacritics, replaces $ and !,
    replaces non-alphanumeric characters with space, and collapses spaces.
    """
    if not s:
        return ""
    s = str(s).replace('$', 's')
    s = s.replace('_', ' ')
    s = ''.join(
        c for c in unicodedata.normalize('NFD', s.lower())
        if unicodedata.category(c) != 'Mn'
    )
    s = re.sub(r'[^\w\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def normalize_date(d: str | None) -> str | None:
    """Ensure date is in YYYY-MM-DD format."""
    if not d:
        return None
    d_str = str(d).strip()
    match = re.search(r'(\d{4}-\d{2}-\d{2})', d_str)
    if match:
        return match.group(1)
    match = re.search(r'(\d{4})', d_str)
    if match:
        return match.group(1)
    return d_str if d_str else None


def normalize_genre(g: str | None) -> str | None:
    """Clean and standardize genre strings with strict keyword filtering."""
    if not g or not str(g).strip():
        return None
    from sonora.core.constants import BROAD_GENRE_KEYWORDS, GENRE_BLACKLIST, GENRE_MAP

    g_raw = str(g).strip()
    g_title = g_raw.title()
    g_lower = g_raw.lower()

    try:
        float(g_raw.replace(',', ''))
        return None
    except ValueError:
        pass

    if any(b.lower() in g_lower for b in GENRE_BLACKLIST) or g_raw.isdigit():
        return None

    if not any(kw.lower() in g_lower for kw in BROAD_GENRE_KEYWORDS):
        return None

    return GENRE_MAP.get(g_title, g_title)


def sanitize_name(name: str | None) -> str:
    """
    Clean string for safe filesystem paths.
    Replaces / and \\ with _, strips invalid Windows/Linux bad chars (<>:"|?*),
    and strips trailing dots/whitespace.
    """
    if not name:
        return "Unknown"
    s = str(name).replace('/', '_').replace('\\', '_')
    bad_chars = '<>:"|?*'
    for char in bad_chars:
        s = s.replace(char, '')
    s = re.sub(r'\s+', ' ', s).strip().rstrip('.')
    return s or "Unknown"


class RateLimiter:
    """Thread-safe rate limiter with precise target_time scheduling."""

    def __init__(self, interval_seconds: float) -> None:
        self.interval = interval_seconds
        self.lock = threading.Lock()
        self.last_call = 0.0

    def wait(self) -> float:
        with self.lock:
            now = time.time()
            target_time = max(now, self.last_call + self.interval)
            sleep_time = target_time - now
            self.last_call = target_time

        if sleep_time > 0:
            time.sleep(sleep_time)
        return sleep_time
