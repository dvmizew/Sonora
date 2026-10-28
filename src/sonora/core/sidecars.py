import html
import re
import shutil
from collections.abc import Sequence
from pathlib import Path

import ftfy

from sonora.core.artists import get_primary_artist, normalize_artist_name
from sonora.core.constants import ALBUM_COVER_NAMES, COMPANION_LYRICS_EXTS
from sonora.core.library_paths import parse_track_filename, safe_case_rename
from sonora.core.logger import LOG
from sonora.core.text import normalize_str, safe_int
from sonora.core.titles import score_artist_title_match


def find_companion_lyrics(
    audio_file: Path,
    track_number: int | str | None = None,
    in_singles: bool = False,
) -> list[Path]:
    """Find all existing companion lyric files (.lrc) for a given audio file."""
    parent = audio_file.parent
    stem = audio_file.stem
    results: list[Path] = []
    for ext in COMPANION_LYRICS_EXTS:
        candidate = parent / f"{stem}{ext}"
        if candidate.is_file():
            results.append(candidate)

    if not results and track_number is not None and parent.is_dir():
        parsed_track = safe_int(track_number)
        if parsed_track is not None:
            try:
                for candidate in parent.iterdir():
                    _, lyric_number, _ = parse_track_filename(candidate.name)
                    if (
                        candidate.suffix.lower() in COMPANION_LYRICS_EXTS
                        and lyric_number == parsed_track
                    ):
                        results.append(candidate)
                        break
            except OSError:
                pass

    if not results and in_singles and parent.is_dir():
        try:
            lrc_candidates = [
                c
                for c in parent.iterdir()
                if c.suffix.lower() in COMPANION_LYRICS_EXTS and c.is_file()
            ]
            if len(lrc_candidates) == 1:
                results.append(lrc_candidates[0])
        except OSError:
            pass

    return results


def relocate_companion_artwork(
    source_dir: Path, target_dir: Path, dry_run: bool = False
) -> list[Path]:
    """
    Move companion album artwork ('cover.jpg', 'cover.png', etc.) from source_dir to target_dir.
    Returns list of relocated artwork files.
    """
    relocated: list[Path] = []
    if source_dir.resolve() == target_dir.resolve():
        return relocated

    for art_name in ALBUM_COVER_NAMES:
        source_art = source_dir / art_name
        target_art = target_dir / art_name
        if source_art.is_file() and not target_art.exists():
            if not dry_run:
                try:
                    shutil.move(str(source_art), str(target_art))
                    relocated.append(target_art)
                except OSError as error:
                    LOG.debug(
                        f"Failed to relocate companion artwork {source_art}: {error}"
                    )
            else:
                relocated.append(target_art)
    return relocated


def relocate_companion_lyrics(
    src_audio: Path,
    dst_audio: Path,
    dry_run: bool = False,
    sidecars: Sequence[Path] | None = None,
) -> list[Path]:
    """
    Move or rename all companion lyric files (.lrc) alongside an audio file to match the new audio location/stem.
    """
    moved_lyrics: list[Path] = []
    for companion in (
        sidecars if sidecars is not None else find_companion_lyrics(src_audio)
    ):
        if not companion.exists():
            continue
        suffix = companion.suffix
        target_companion = dst_audio.parent / f"{dst_audio.stem}{suffix}"
        if target_companion.exists() and target_companion != companion:
            continue

        if not dry_run:
            safe_case_rename(companion, target_companion)
        moved_lyrics.append(target_companion)
    return moved_lyrics


_LYRICS_TIMESTAMP_REGEX = re.compile(
    r"^(?:\[\d{1,2}:\d{2}(?:[\.:]\d{2,3})?(?:[+-]\d+)?\]|<\d{1,2}:\d{2}(?:[\.:]\d{2,3})?>)\s*(.*)$"
)
_LYRICS_EXTRACT_TS_REGEX = re.compile(
    r"[\[<](\d{1,3}):([0-5]\d)(?:[\.:](\d{2,3}))?[\]>]"
)
_LYRICS_JUNK_PATTERNS = [
    re.compile(r"^\d+\s*(?:Contributor|Translation|Embed)s?$", re.IGNORECASE),
    re.compile(r"^You might also like$", re.IGNORECASE),
    re.compile(r"^Read More$", re.IGNORECASE),
    re.compile(r"^See .* Live(?:Get tickets.*)?$", re.IGNORECASE),
    re.compile(
        r"^\[?(?:Produced|Written|Arranged|Composed|Mastered|Mixed|Recorded|Engineered)\b.*\]?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\[?(?:Producer|Writer|Composer|Arranger|Engineer|Mixer|Release Date|Recording Date|Studio|Label)s?\s*:.*\]?$",
        re.IGNORECASE,
    ),
    re.compile(r"^\[?(?:re|ve|by|length|offset)\s*:.*\]?$", re.IGNORECASE),
    re.compile(r".*megalobiz.*", re.IGNORECASE),
    re.compile(r"^(?:作词|作曲|编曲|制作人)\s*:.*$", re.IGNORECASE),
    re.compile(r"^This song is called\b.*$", re.IGNORECASE),
    re.compile(r"^\[?Instrumental\]?$", re.IGNORECASE),
    re.compile(r"^Lyrics (?:powered|licensed|provided) by.*$", re.IGNORECASE),
    re.compile(r"^Paroles de la chanson .* par .*$", re.IGNORECASE),
    re.compile(r"^Commercial use is strictly forbidden.*$", re.IGNORECASE),
    re.compile(r"^https?://\S+$", re.IGNORECASE),
    re.compile(r"^www\.\S+$", re.IGNORECASE),
    re.compile(
        r"^(?:Synced|Created|Uploaded|Encoded|Downloaded|LRC)\s*(?:by|from|using|with|via|:)\s+.*$",
        re.IGNORECASE,
    ),
    re.compile(r"<!--.*?-->", re.IGNORECASE),
]
_LYRICS_EMBED_END_REGEX = re.compile(r"\d*\s*Embed\s*$", re.IGNORECASE)
_LRC_METADATA_HEADER_PATTERN = re.compile(
    r"\[(ar|ti|al|au|by|re|ve|length|offset)\s*:([^\]\r\n]*)\]", re.IGNORECASE
)


def lyrics_identity_matches(
    lyrics_text: str | None, artist: str | None, title: str | None
) -> bool:
    """Reject conflicting available LRC headers; missing headers establish no identity."""
    if not lyrics_text:
        return True
    for line in lyrics_text.splitlines():
        header_line = line.lstrip("\ufeff \t")
        header_matches = list(_LRC_METADATA_HEADER_PATTERN.finditer(header_line))
        if not header_matches or header_matches[0].start() != 0:
            continue
        for header in header_matches:
            field_name, header_value = header.groups()
            field_name = field_name.lower()
            if field_name not in {"ar", "ti"}:
                continue
            expected_value = artist if field_name == "ar" else title
            # Placeholder labels provide no usable evidence against a local asset.
            if any(
                normalize_str(metadata_value)
                in {
                    "",
                    "unknown",
                    "unknown artist",
                    "unknown title",
                    "unknown track",
                    "untitled",
                    "artist",
                    "title",
                }
                or re.fullmatch(
                    r"(?:audio\s*)?track\s*\d*", normalize_str(metadata_value)
                )
                for metadata_value in (expected_value, header_value)
            ):
                continue
            if field_name == "ar":
                expected_artist = get_primary_artist(
                    normalize_artist_name(expected_value)
                )
                header_artist = get_primary_artist(normalize_artist_name(header_value))
                if normalize_str(expected_artist) != normalize_str(header_artist):
                    return False
            elif (
                score_artist_title_match("", expected_value or "", "", header_value)
                <= 0
            ):
                return False
    return True


def lyrics_word_body(lyrics_text: str) -> str:
    lyric_lines = []
    for line in lyrics_text.splitlines():
        header_line = line.lstrip("\ufeff \t")
        if _LRC_METADATA_HEADER_PATTERN.match(header_line):
            line = _LRC_METADATA_HEADER_PATTERN.sub("", header_line)
        lyric_lines.append(_LYRICS_EXTRACT_TS_REGEX.sub("", line))
    return normalize_str("\n".join(lyric_lines))


def repair_lyrics_headers(lyrics_text: str, artist: str, title: str) -> str:
    repaired_lines = []
    identity_headers = {"ar": artist, "ti": title}
    for line in lyrics_text.splitlines(keepends=True):
        if _LRC_METADATA_HEADER_PATTERN.match(line.lstrip("\ufeff \t")):
            for header in reversed(list(_LRC_METADATA_HEADER_PATTERN.finditer(line))):
                field_name = header.group(1).lower()
                if field_name in identity_headers:
                    line = (
                        line[: header.start()]
                        + f"[{field_name}:{identity_headers[field_name]}]"
                        + line[header.end() :]
                    )
        repaired_lines.append(line)
    return "".join(repaired_lines)


def clean_lyrics_text(text: str | None) -> str | None:
    """
    Strips web scraping artifacts (Genius, Musixmatch, AZLyrics, LRCLIB, Megalobiz junk)
    and unescapes HTML entities while preserving timestamped LRC lines ([mm:ss.xx] and <mm:ss.xx>).
    """
    if text is None:
        return None
    if not text.strip():
        return ""

    decoded = html.unescape(ftfy.fix_text(str(text)))
    lines = decoded.splitlines()
    cleaned: list[str] = []

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue

        timestamp_match = _LYRICS_TIMESTAMP_REGEX.match(stripped_line)
        content_to_check = (
            timestamp_match.group(1).strip() if timestamp_match else stripped_line
        )

        if timestamp_match and not content_to_check:
            continue

        if any(
            junk_pattern.search(content_to_check)
            for junk_pattern in _LYRICS_JUNK_PATTERNS
        ):
            continue

        cleaned.append(stripped_line)

    while cleaned and (
        _LYRICS_EMBED_END_REGEX.search(cleaned[-1]) or cleaned[-1] == ""
    ):
        if cleaned[-1] == "":
            cleaned.pop()
            continue
        cleaned[-1] = _LYRICS_EMBED_END_REGEX.sub("", cleaned[-1]).strip()
        if not cleaned[-1]:
            cleaned.pop()

    return "\n".join(cleaned).strip()


def extract_max_timestamp(lyrics_text: str | None) -> float | None:
    """Extract the maximum timestamp in seconds from LRC lyrics text."""
    if not lyrics_text:
        return None
    maximum: float | None = None
    for match in _LYRICS_EXTRACT_TS_REGEX.finditer(lyrics_text):
        minutes, seconds, fraction = match.groups()
        timestamp = int(minutes) * 60.0 + int(seconds)
        if fraction:
            timestamp += float(f"0.{fraction}")
        if maximum is None or timestamp > maximum:
            maximum = timestamp
    return maximum


def is_lyrics_duration_valid(
    lyrics_text: str | None,
    audio_duration: float | None,
    tolerance_seconds: float = 15.0,
) -> bool:
    """
    Validates that lyrics timestamps do not significantly exceed audio duration.
    Prevents cross-song collisions (e.g. short tracks matching longer songs).
    """
    if not lyrics_text or audio_duration is None or audio_duration <= 0:
        return True
    max_timestamp = extract_max_timestamp(lyrics_text)
    if max_timestamp is None:
        return True
    return max_timestamp <= (audio_duration + tolerance_seconds)


def get_lyrics_quality(lyrics_text: str | None) -> int:
    """
    Determines quality tier of LRC lyrics string:
      3: Enhanced / Word-level synced (<00:01.23> word timestamps)
      2: Synced / Line-level synced ([00:01.23] line timestamps)
      1: Plain text (no timestamps)
      0: None or empty / invalid
    """
    cleaned = clean_lyrics_text(lyrics_text)
    if not cleaned:
        return 0

    non_header_lines = [
        line.strip()
        for line in cleaned.splitlines()
        if line.strip()
        and not re.match(
            r"^\[(?:ar|ti|al|by|re|ve|length|offset):",
            line.strip(),
            re.IGNORECASE,
        )
    ]
    if not non_header_lines:
        return 0

    if re.search(r"<\d{1,2}:\d{2}[\.:]\d{2,3}>", cleaned):
        return 3
    if re.search(r"^\[\d{1,3}:[0-5]\d(?:[\.:]\d{2,3})?\]", cleaned, re.MULTILINE):
        return 2
    return 1


def strip_lyrics_headers(lyrics_text: str) -> str:
    return _LRC_METADATA_HEADER_PATTERN.sub(
        "", clean_lyrics_text(lyrics_text) or ""
    ).strip()
