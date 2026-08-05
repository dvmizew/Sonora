import sys

IS_WINDOWS = sys.platform == "win32"
IS_LINUX = sys.platform == "linux"

FFMPEG_CMD = "ffmpeg.exe" if IS_WINDOWS else "ffmpeg"
FLAC_CMD = "flac.exe" if IS_WINDOWS else "flac"
METAFLAC_CMD = "metaflac.exe" if IS_WINDOWS else "metaflac"
SOX_CMD = "sox.exe" if IS_WINDOWS else "sox"

SUPPORTED_EXTS = [".flac", ".mp3", ".m4a", ".ogg", ".wav"]

FEAT_KEYWORDS = r"\b(?:fea?t(?:uring)?|ft)(?:\.?(?!\w))|×"
TECH_FEAT = r"\b(?:fea?t(?:uring)?|ft)(?:\.?(?!\w))"

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

GENRE_BLACKLIST = [
    "Billboard", "Hot 100", "Top 40", "Amazon", "Itunes",
    "Unknown", "Release", "Music", "Digital", "Various",
    "Produced By", "Written By"
]

PROTECTED_ARTISTS = [
    "Play & Win", "Play&Win", "Rauf & Faik", "Rauf&Faik",
    "Simon & Garfunkel", "Earth, Wind & Fire", "Belle & Sebastian",
    "Brooks & Dunn", "Hall & Oates", "Above & Beyond",
    "Cardi B & Megan Thee Stallion", "Mumford & Sons", "Kool & The Gang",
    "Sly & The Family Stone", "Blood, Sweat & Tears",
    "Emerson, Lake & Palmer", "Crosby, Stills, Nash & Young",
    "Huey Lewis & The News", "KC & The Sunshine Band"
]
