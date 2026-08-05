"""
Core text processing and filesystem path sanitization utilities.
"""

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
    """Thread-safe rate limiter for external API requests."""

    def __init__(self, interval_seconds: float) -> None:
        self.interval = interval_seconds
        self.lock = threading.Lock()
        self.last_call = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_call = time.time()
