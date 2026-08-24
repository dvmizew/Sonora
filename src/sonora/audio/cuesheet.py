import shlex
from pathlib import Path

import ftfy


def read_cuesheet_content(cue_path: Path) -> str | None:
    """
    Read raw text content of a .cue file with multi-encoding fallback
    (UTF-8, UTF-8-BOM, CP1252, Latin-1) and ftfy Unicode sanitization.
    """
    if not cue_path.exists():
        return None
    try:
        raw_bytes = cue_path.read_bytes()
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                return ftfy.fix_text(raw_bytes.decode(encoding))
            except UnicodeDecodeError:
                continue
        return ftfy.fix_text(raw_bytes.decode("utf-8", errors="replace"))
    except OSError:
        return None


def parse_cuesheet(cue_path: Path) -> list[dict[str, str | int]]:
    """
    Parse a CD .cue file into a list of track metadata dictionaries.
    Supports REM fields (DATE, GENRE, DISCNUMBER, TOTALDISCS), ISRC, SONGWRITER/COMPOSER,
    INDEX 00/01, and global vs track-level PERFORMER/TITLE.
    """
    content = read_cuesheet_content(cue_path)
    if not content:
        return []

    tracks: list[dict[str, str | int]] = []
    current_track: dict[str, str | int] | None = None
    globals_meta: dict[str, str | int] = {
        "artist": "Unknown Artist",
        "album": "Unknown Album",
        "disc_number": 1,
    }

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        try:
            tokens = shlex.split(line)
        except ValueError:
            tokens = line.split()

        if not tokens:
            continue

        command = tokens[0].upper()

        # Handle REM metadata (DATE, GENRE, DISCNUMBER, TOTALDISCS)
        if command == "REM" and len(tokens) >= 3:
            key, val = tokens[1].upper(), " ".join(tokens[2:])
            if key in ("DATE", "YEAR"):
                if current_track:
                    current_track["date"] = val
                else:
                    globals_meta["date"] = val
            elif key == "GENRE":
                if current_track:
                    current_track["genre"] = val
                else:
                    globals_meta["genre"] = val
            elif key in ("DISCNUMBER", "DISC") and val.isdigit():
                globals_meta["disc_number"] = int(val)
            elif key in ("TOTALDISCS", "DISCTOTAL") and val.isdigit():
                globals_meta["total_discs"] = int(val)
            continue

        # Handle track start: TRACK 01 AUDIO
        if command == "TRACK" and len(tokens) >= 2:
            if current_track:
                tracks.append(current_track)
            track_num = int(tokens[1]) if tokens[1].isdigit() else len(tracks) + 1
            current_track = {
                "track_number": track_num,
                "title": f"Track {track_num}",
                **globals_meta,
            }
            continue

        # Handle track and global directives
        if len(tokens) >= 2:
            value = " ".join(tokens[1:])
            if command == "PERFORMER":
                if current_track:
                    current_track["artist"] = value
                else:
                    globals_meta["artist"] = value
            elif command == "TITLE":
                if current_track:
                    current_track["title"] = value
                else:
                    globals_meta["album"] = value
            elif current_track:
                if command == "ISRC":
                    current_track["isrc"] = tokens[1]
                elif command in ("SONGWRITER", "COMPOSER"):
                    current_track["composer"] = value
                elif command == "INDEX" and len(tokens) >= 3:
                    if tokens[1] == "01":
                        current_track["start_index"] = tokens[2]
                    elif tokens[1] == "00":
                        current_track["pregap_index"] = tokens[2]

    if current_track:
        tracks.append(current_track)

    return tracks
