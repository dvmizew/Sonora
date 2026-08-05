"""
Core text processing and filesystem path sanitization utilities.
"""

import re
import unicodedata
from typing import Optional


def normalize_str(s: Optional[str]) -> str:
    """
    Converts to lowercase, normalizes NFD diacritics, replaces $ and !,
    replaces non-alphanumeric characters with space, and collapses spaces.
    """
    if not s:
        return ""
    s = str(s).replace('$', 's').replace('!', 'i')
    s = s.replace('_', ' ')
    s = ''.join(
        c for c in unicodedata.normalize('NFD', s.lower())
        if unicodedata.category(c) != 'Mn'
    )
    s = re.sub(r'[^\w\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def sanitize_name(name: Optional[str]) -> str:
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
