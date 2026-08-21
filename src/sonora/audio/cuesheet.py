import re
from pathlib import Path
from typing import Any


def parse_cuesheet(cue_path: Path) -> list[dict[str, Any]]:
    """
    Parse a CD .cue file into a list of track metadata dictionaries.
    """
    if not cue_path.exists():
        return []

    try:
        content = cue_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    tracks: list[dict[str, Any]] = []
    current_track: dict[str, Any] | None = None
    performer_global = "Unknown Artist"
    title_global = "Unknown Album"

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("PERFORMER") and current_track is None:
            match = re.search(r'PERFORMER\s+"?([^"]+)"?', line, re.IGNORECASE)
            if match:
                performer_global = match.group(1)
        elif line.startswith("TITLE") and current_track is None:
            match = re.search(r'TITLE\s+"?([^"]+)"?', line, re.IGNORECASE)
            if match:
                title_global = match.group(1)
        elif line.startswith("TRACK"):
            match = re.search(r'TRACK\s+(\d+)\s+AUDIO', line, re.IGNORECASE)
            if match:
                if current_track:
                    tracks.append(current_track)
                track_num = int(match.group(1))
                current_track = {
                    "track_number": track_num,
                    "artist": performer_global,
                    "title": f"Track {track_num}",
                    "album": title_global,
                }
        elif current_track is not None:
            if line.startswith("TITLE"):
                match = re.search(r'TITLE\s+"?([^"]+)"?', line, re.IGNORECASE)
                if match:
                    current_track["title"] = match.group(1)
            elif line.startswith("PERFORMER"):
                match = re.search(r'PERFORMER\s+"?([^"]+)"?', line, re.IGNORECASE)
                if match:
                    current_track["artist"] = match.group(1)
            elif line.startswith("INDEX 01"):
                match = re.search(r'INDEX 01\s+(\d+:\d+:\d+)', line, re.IGNORECASE)
                if match:
                    current_track["start_index"] = match.group(1)

    if current_track:
        tracks.append(current_track)

    return tracks


def read_cuesheet_content(cue_path: Path) -> str | None:
    """
    Read raw text content of a .cue file for CUESHEET Vorbis comment tag embedding.
    """
    if not cue_path.exists():
        return None
    try:
        return cue_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
