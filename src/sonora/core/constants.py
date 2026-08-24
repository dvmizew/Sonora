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
        ".synced.lrc",
        ".enhanced.lrc",
        ".txt",
    }
)

FEAT_KEYWORDS = r"\b(?:fea?t(?:uring)?|ft)(?:\.?(?!\w))|×"

USER_AGENT = f"Sonora/{__version__} (+https://github.com/dvmizew/Sonora)"
