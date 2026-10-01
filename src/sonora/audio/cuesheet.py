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
