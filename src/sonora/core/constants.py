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

ALBUM_COVER_NAMES: tuple[str, ...] = (
    "cover.jpg",
    "cover.png",
    "folder.jpg",
    "front.jpg",
)
MIN_COVER_ART_DIMENSION: int = 500

FEAT_KEYWORDS = (
    r"\b(?:fea?t(?:uring)?|ft|with|w/(?!\s*[oO](?:ut)?\b)|w\.)(?:\.?(?!\w))|\u00d7"
)

USER_AGENT = f"Sonora/{__version__} (+https://github.com/dvmizew/Sonora)"

RATE_LIMIT_MUSICBRAINZ: float = (
    1.1  # MusicBrainz official NGS policy (max 1 req/s, 1.1s safety margin)
)
RATE_LIMIT_DISCOGS_AUTHENTICATED: float = (
    1.05  # Discogs API (60 req/min with token; 1.05s avoids window drift)
)
RATE_LIMIT_DISCOGS_UNAUTHENTICATED: float = (
    2.45  # Discogs API (25 req/min without token; 2.45s avoids window drift)
)
RATE_LIMIT_THEAUDIODB: float = (
    2.05  # https://www.theaudiodb.com/free_music_api: 30 req/min.
)
RATE_LIMIT_LYRICS: float = 1.0  # SyncedLyrics upstream provider pool
RATE_LIMIT_LRCLIB: float = (
    1.0  # Below LRCLIB's approximate 90 requests/min for identified clients.
)
RATE_LIMIT_ITUNES: float = (
    3.0  # Apple iTunes Search API (official policy approx 20 calls/min = 3.0s)
)
RATE_LIMIT_GENIUS: float = 0.5
RATE_LIMIT_ACOUSTID: float = (
    0.4  # AcoustID API (official policy max 3 req/s = 0.33s; 0.4s safety buffer)
)
RATE_LIMIT_LASTFM: float = 0.25
RATE_LIMIT_DEEZER: float = (
    0.15  # Application pacing; server throttling is handled through Retry-After.
)
RATE_LIMIT_FANART: float = 0.2
RATE_LIMIT_SHAZAM: float = 3.0  # Application pacing for acoustic recognition.
