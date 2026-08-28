import sys

from sonora import __version__

IS_WINDOWS = sys.platform == "win32"

FLAC_CMD = "flac.exe" if IS_WINDOWS else "flac"

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

# API rate limits in seconds between requests
RATE_LIMIT_MUSICBRAINZ: float = 1.1  # MusicBrainz official NGS policy (~1 req/s)
RATE_LIMIT_DISCOGS_AUTHENTICATED: float = 1.0  # Discogs API (60 req/min with token)
RATE_LIMIT_DISCOGS_UNAUTHENTICATED: float = (
    2.4  # Discogs API (25 req/min without token)
)
RATE_LIMIT_THEAUDIODB: float = 1.0  # TheAudioDB API
RATE_LIMIT_LYRICS: float = 1.0  # SyncedLyrics upstream provider pool
RATE_LIMIT_ITUNES: float = 3.0  # Apple iTunes Search API (approx 20 req/min)
RATE_LIMIT_GENIUS: float = 0.5  # Genius API
RATE_LIMIT_ACOUSTID: float = 0.4  # AcoustID API (max 3 req/s)
RATE_LIMIT_LASTFM: float = 0.25  # Last.fm API (max 5 req/s)
RATE_LIMIT_DEEZER: float = 0.15  # Deezer API (max 10 req/s)
