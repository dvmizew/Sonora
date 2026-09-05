import sys

from platformdirs import PlatformDirs

from sonora import __version__

IS_WINDOWS = sys.platform == "win32"

FLAC_CMD = "flac.exe" if IS_WINDOWS else "flac"

DIRS: PlatformDirs = PlatformDirs(appname="sonora", appauthor=False)

SUPPORTED_EXTS = frozenset(
    {
        ".flac",
        ".mp3",
        ".m4a",
        ".mp4",
        ".alac",
        ".ogg",
        ".opus",
        ".wav",
        ".aiff",
        ".wma",
        ".ape",
        ".wv",
        ".mpc",
    }
)

COMPANION_LYRICS_EXTS = frozenset(
    {
        ".lrc",
    }
)

FEAT_KEYWORDS = r"\b(?:fea?t(?:uring)?|ft)(?:\.?(?!\w))|\u00d7"

USER_AGENT = f"Sonora/{__version__} (+https://github.com/dvmizew/Sonora)"

# Fuzzy similarity thresholds (0 - 100)
ARTIST_MATCH_THRESHOLD: float = 85.0
ALBUM_MATCH_THRESHOLD: float = 75.0
GENIUS_MATCH_THRESHOLD: float = 70.0

# API rate limits in seconds between requests (aligned with official documentation)
RATE_LIMIT_MUSICBRAINZ: float = (
    1.1  # MusicBrainz official NGS policy (max 1 req/s, 1.1s safety margin)
)
RATE_LIMIT_DISCOGS_AUTHENTICATED: float = (
    1.05  # Discogs API (60 req/min with token; 1.05s avoids window drift)
)
RATE_LIMIT_DISCOGS_UNAUTHENTICATED: float = (
    2.45  # Discogs API (25 req/min without token; 2.45s avoids window drift)
)
RATE_LIMIT_THEAUDIODB: float = 1.0  # TheAudioDB free tier (1 req/s)
RATE_LIMIT_LYRICS: float = 1.0  # SyncedLyrics upstream provider pool
RATE_LIMIT_LRCLIB: float = (
    0.25  # LRCLIB REST API policy (official doc recommends 200-500 ms)
)
RATE_LIMIT_ITUNES: float = (
    3.0  # Apple iTunes Search API (official policy approx 20 calls/min = 3.0s)
)
RATE_LIMIT_GENIUS: float = 0.5  # Genius API (max 2 req/s)
RATE_LIMIT_ACOUSTID: float = (
    0.4  # AcoustID API (official policy max 3 req/s = 0.33s; 0.4s safety buffer)
)
RATE_LIMIT_LASTFM: float = 0.25  # Last.fm API (recommended max 4-5 req/s)
RATE_LIMIT_DEEZER: float = (
    0.15  # Deezer API (official quota approx 50 req/5s = 10 req/s; 0.15s buffer)
)
RATE_LIMIT_FANART: float = 0.2  # Fanart.tv API (max 5 req/s)
RATE_LIMIT_SHAZAM: float = (
    3.0  # Shazam API (reverse-engineered endpoint triggers 429 at >20 req/min = 3.0s)
)
