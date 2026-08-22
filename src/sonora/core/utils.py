import re
import threading
import time
import unicodedata


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
    """Clean and standardize genre strings with strict filtering."""
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
    
    words = g_title.split()
    if len(words) >= 2 and not any(kw.lower() in g_lower for kw in BROAD_GENRE_KEYWORDS):
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
