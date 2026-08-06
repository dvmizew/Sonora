import os

IS_WINDOWS = os.name == 'nt'

# Executable definitions
FLAC_CMD = "flac.exe" if IS_WINDOWS else "flac"
METAFLAC_CMD = "metaflac.exe" if IS_WINDOWS else "metaflac"
FFMPEG_CMD = "ffmpeg.exe" if IS_WINDOWS else "ffmpeg"
SOX_CMD = "sox.exe" if IS_WINDOWS else "sox"
BPM_TAG_CMD = "bpm-tag.exe" if IS_WINDOWS else "bpm-tag"

SUPPORTED_EXTS = frozenset({".flac", ".mp3", ".m4a", ".ogg", ".wav"})

GENRE_MAP = {
    "Hip-Hop": "Hip-Hop/Rap",
    "Hip Hop": "Hip-Hop/Rap",
    "Rap": "Hip-Hop/Rap",
    "Rnb": "R&B",
    "R&B": "R&B/Soul",
    "Electronic": "Electronic",
    "Dance": "Dance",
    "House": "House",
    "Pop/Rock": "Pop",
    "Drum And Bass": "Drum & Bass",
    "Synthpop": "Synth-pop",
    "Alternative Rock": "Alternative",
    "Indie Rock": "Indie",
}

GENRE_BLACKLIST = frozenset({
    "Billboard", "Hot 100", "Top 40", "Amazon", "Itunes",
    "Unknown", "Release", "Music", "Digital", "Various",
    "Produced By", "Written By"
})

from sonora import __version__

USER_AGENT = f"Sonora/{__version__} (+https://github.com/dvmizew/Sonora)"


