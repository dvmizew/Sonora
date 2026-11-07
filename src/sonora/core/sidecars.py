import html
import math
import re
import unicodedata
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import anyascii
import ftfy
from rapidfuzz import fuzz

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
            try:
                if not dry_run:
                    safe_case_rename(source_art, target_art)
                relocated.append(target_art)
            except OSError as error:
                LOG.debug(f"Failed to relocate companion artwork {source_art}: {error}")
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
    r"^(?:\[\d{1,3}:[0-5]\d(?:[\.:]\d{1,6})?(?:[+-]\d+)?\]|<\d{1,3}:[0-5]\d(?:[\.:]\d{1,6})?>)+\s*(.*)$"
)
_LYRICS_EXTRACT_TS_REGEX = re.compile(
    r"(?:(\[)|<)(\d{1,3}):([0-5]\d)(?:[\.:](\d{1,6}))?(?(1)\]|>)"
)
_LYRICS_JUNK_PATTERNS = [
    re.compile(
        r"^(?:\d+\s*(?:Contributor|Translation|Embed)s?|Translations)$", re.IGNORECASE
    ),
    re.compile(r"^You might also like$", re.IGNORECASE),
    re.compile(r"^Read More$", re.IGNORECASE),
    re.compile(r"^See .* Live(?:Get tickets.*)?$", re.IGNORECASE),
    re.compile(
        r"^\[?(?:Produced|Written|Arranged|Composed|Mastered|Mixed|Recorded|Engineered)\s+by\b.*\]?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\[?(?:Producer|Writer|Composer|Arranger|Engineer|Mixer|Release Date|Recording Date|Studio|Label)s?\s*:.*\]?$",
        re.IGNORECASE,
    ),
    re.compile(r"^가사\s*제작\s*:\s*\S+.*$", re.IGNORECASE),
    re.compile(
        r"^(?:作词|作曲|编曲|制作人?|音频工程师|助理母带工程师|母带工程师|"
        r"混音师|混音工程师|编程|人声|封面|美声和声)\s*[:：].*$",
        re.IGNORECASE,
    ),
    re.compile(r"^Lyrics (?:powered|licensed|provided) by.*$", re.IGNORECASE),
    re.compile(r"^Paroles de la chanson .* par .*$", re.IGNORECASE),
    re.compile(r"^Commercial use is strictly forbidden.*$", re.IGNORECASE),
    re.compile(r"^https?://\S+$", re.IGNORECASE),
    re.compile(r"^www\.\S+$", re.IGNORECASE),
    re.compile(
        r"^(?:Synced|Created|Uploaded|Encoded|Downloaded|LRC)\s*(?:by|from|using|with|via|:)\s+.*$",
        re.IGNORECASE,
    ),
]
_LYRICS_EMBED_END_REGEX = re.compile(r"(?:\d+\s*Embed|^\s*Embed)\s*$", re.IGNORECASE)
_LRC_METADATA_HEADER_PATTERN = re.compile(
    r"\[(ar|ti|al|au|by|re|ve|length|offset)\s*:([^\]\r\n]*)\]", re.IGNORECASE
)
_LYRICS_TITLE_HEADER_REGEX = re.compile(
    r'(.+?) Lyrics|\[(?:Versuri(?:le piesei| pentru)?|Paroles de)\s+["“„](.+?)["”“](?: cu .+)?\]'
)
_CYRILLIC_TO_LATIN_HOMOGLYPHS = str.maketrans(
    {
        "\u0430": "a",
        "\u0410": "A",
        "\u0432": "b",
        "\u0412": "B",
        "\u0441": "c",
        "\u0421": "C",
        "\u0434": "d",
        "\u0435": "e",
        "\u0415": "E",
        "\u0451": "e",
        "\u0401": "E",
        "\u0456": "i",
        "\u0406": "I",
        "\u0458": "j",
        "\u0408": "J",
        "\u043a": "k",
        "\u041a": "K",
        "\u043c": "m",
        "\u041c": "M",
        "\u043d": "h",
        "\u041d": "H",
        "\u043e": "o",
        "\u041e": "O",
        "\u0440": "p",
        "\u0420": "P",
        "\u0455": "s",
        "\u0405": "S",
        "\u0442": "t",
        "\u0422": "T",
        "\u0443": "y",
        "\u0423": "Y",
        "\u0445": "x",
        "\u0425": "X",
    }
)


def _normalize_mixed_script_word(word: str) -> str:
    has_lat = any("a" <= c.lower() <= "z" for c in word)
    has_cyr = any("\u0400" <= c <= "\u04ff" for c in word)
    if (has_lat and has_cyr) or any(c in "\u0455\u0405" for c in word):
        return word.translate(_CYRILLIC_TO_LATIN_HOMOGLYPHS)
    return word


def normalize_lyrics_homoglyphs(text: str) -> str:
    """Normalize Cyrillic lookalike homoglyphs in mixed-script words."""
    return re.sub(
        r"[\w\u0400-\u04ff]+",
        lambda m: _normalize_mixed_script_word(m.group(0)),
        text,
    )


def lyrics_identity_matches(
    lyrics_text: str | None, artist: str | None, title: str | None
) -> bool:
    """Reject conflicting available LRC headers; missing headers establish no identity."""
    if not lyrics_text:
        return True
    matched_plain_title = False
    for line in lyrics_text.splitlines():
        header_line = line.lstrip("\ufeff \t")
        plain_title = (
            None
            if _LYRICS_TIMESTAMP_REGEX.match(header_line)
            else _LYRICS_TITLE_HEADER_REGEX.fullmatch(header_line)
        )
        if plain_title:
            extracted_title = plain_title[1] or plain_title[2]
            if title and score_artist_title_match("", title, "", extracted_title) > 0:
                matched_plain_title = True
            header_line = f"[ti:{extracted_title}]"
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
            elif field_name == "ti":
                clean_expected = re.sub(
                    r"\s*[\(\[\{].*?[\)\]\}]", "", expected_value or ""
                ).strip()
                clean_header = re.sub(
                    r"\s*[\(\[\{].*?[\)\]\}]", "", header_value or ""
                ).strip()
                match_ok = score_artist_title_match(
                    "", expected_value or "", "", header_value
                ) > 0 or (
                    clean_expected
                    and clean_header
                    and score_artist_title_match("", clean_expected, "", clean_header)
                    > 0
                )
                if not match_ok:
                    if matched_plain_title and plain_title and plain_title[2]:
                        continue
                    return False
    return True


@lru_cache(maxsize=512)
def lyrics_word_body(lyrics_text: str) -> str:
    lyric_lines = []
    for line in lyrics_text.splitlines():
        header_line = line.lstrip("\ufeff \t")
        if _LYRICS_TITLE_HEADER_REGEX.fullmatch(header_line):
            continue
        if _LRC_METADATA_HEADER_PATTERN.match(header_line):
            line = _LRC_METADATA_HEADER_PATTERN.sub("", header_line)
        if timestamp_match := _LYRICS_TIMESTAMP_REGEX.match(line):
            line = re.sub(r"^v\d{1,3}:\s*", "", timestamp_match[1])
        line = _LYRICS_EXTRACT_TS_REGEX.sub("", line)
        if re.fullmatch(
            r"\s*\[(?:intro|outro|verse|chorus|pre[- ]?chorus|post[- ]?chorus|bridge|hook|refrain|interlude|instrumental|solo|break|drop|vers|strof[aă]|refren|pre[- ]?refren|post[- ]?refren|punte)(?:\s+(?:x\s*)?\d+)?(?:\s*:\s*[^\]]+)?\]\s*",
            line,
            re.IGNORECASE,
        ):
            continue
        lyric_lines.append(line)
    text = unicodedata.normalize("NFC", "\n".join(lyric_lines)).casefold()
    text = re.sub(r"(?<=\w)['’ʼ`](?=\w)", "", text)
    return " ".join(re.sub(r"[^\w\s]|_", " ", text).split())


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


@lru_cache(maxsize=512)
def clean_lyrics_text(text: str | None) -> str | None:
    """Remove scrape artifacts while preserving lyric words and LRC timing metadata."""
    if text is None:
        return None
    if not text.strip():
        return ""

    decoded = normalize_lyrics_homoglyphs(html.unescape(ftfy.fix_text(str(text))))
    decoded = re.sub(
        r"<(script|style|noscript|head)\b[^>]*>.*?</\1\s*>",
        "",
        decoded,
        flags=re.IGNORECASE | re.DOTALL,
    )
    decoded = re.sub(
        r"<!--.*?-->|<!DOCTYPE\b[^>]*>",
        "",
        decoded,
        flags=re.IGNORECASE | re.DOTALL,
    )
    decoded = re.sub(
        r"<br\s*/?>|</?(?:p|div|li|h[1-6])\b[^>]*>",
        "\n",
        decoded,
        flags=re.IGNORECASE,
    )
    decoded = re.sub(
        r"</?(?:a|b|span|em|strong|i|u|html|body)\b[^>]*>",
        "",
        decoded,
        flags=re.IGNORECASE,
    )
    lines = decoded.splitlines()
    cleaned: list[str] = []
    has_lyric_content = False

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
        if timestamp_match:
            content_to_check = re.sub(r"^v\d{1,3}:\s*", "", content_to_check)
            stripped_line = stripped_line[: timestamp_match.start(1)] + content_to_check

        if timestamp_match and not content_to_check and not has_lyric_content:
            continue

        if any(
            junk_pattern.search(content_to_check)
            for junk_pattern in _LYRICS_JUNK_PATTERNS
        ):
            continue

        cleaned.append(stripped_line)
        if content_to_check and not _LRC_METADATA_HEADER_PATTERN.match(stripped_line):
            has_lyric_content = True

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
    for line in lyrics_text.splitlines():
        if _LRC_METADATA_HEADER_PATTERN.match(line.lstrip("\ufeff \t")):
            line = _LRC_METADATA_HEADER_PATTERN.sub("", line)
        for match in _LYRICS_EXTRACT_TS_REGEX.finditer(line):
            _, minutes, seconds, fraction = match.groups()
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
    if not lyrics_text:
        return True
    if any(
        _LYRICS_EXTRACT_TS_REGEX.fullmatch(match.group()) is None
        for match in re.finditer(
            r"[\[<]\d{1,3}:\d{2}(?:[\.:]\d{1,6})?[\]>]", lyrics_text
        )
    ):
        return False
    for line in lyrics_text.splitlines():
        # Repeated line timestamps are valid; only inline word marks must increase.
        if "<" not in line:
            continue
        timestamps = [
            int(match[2]) * 60 + int(match[3]) + float(f"0.{match[4] or '0'}")
            for match in _LYRICS_EXTRACT_TS_REGEX.finditer(line)
        ]
        if timestamps != sorted(timestamps):
            return False
    if audio_duration is None or audio_duration <= 0:
        return True
    if not math.isfinite(audio_duration):
        return False
    max_timestamp = extract_max_timestamp(lyrics_text)
    if max_timestamp is None:
        return True
    offsets = re.findall(r"(?im)^\s*\[offset\s*:\s*([+-]?\d+)\s*\]", lyrics_text)
    offset = int(offsets[-1]) / 1000.0 if offsets else 0.0
    return max_timestamp - offset <= audio_duration + tolerance_seconds


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

    body = lyrics_word_body(cleaned)
    if not body or body == "instrumental":
        return 0

    if re.search(r"<\d{1,3}:[0-5]\d(?:[\.:]\d{1,6})?>", cleaned):
        return 3
    if re.search(r"^\[\d{1,3}:[0-5]\d(?:[\.:]\d{1,6})?\]", cleaned, re.MULTILINE):
        return 2
    return 1


_NON_LATIN_SCRIPT_REGEX = re.compile(
    r"[\u0400-\u04ff\u0600-\u06ff\u4e00-\u9fff\uac00-\ud7af]"
)
_CYRILLIC_SCRIPT_REGEX = re.compile(r"[\u0400-\u04ff]")
_SLAVIC_MARKERS_REGEX = re.compile(
    r"\b(?:russia|russian|rossiya|raava|zhara)\b", re.IGNORECASE
)


def is_script_concordant(
    reference_text: str, lyrics_text: str, title: str | None = None
) -> bool:
    """Require lyrics to avoid predominantly foreign non-Latin scripts unless present in reference metadata."""
    if _NON_LATIN_SCRIPT_REGEX.search(reference_text) or _SLAVIC_MARKERS_REGEX.search(
        reference_text
    ):
        return True
    cleaned_lyrics = normalize_lyrics_homoglyphs(lyrics_text)
    letters = [char for char in cleaned_lyrics if char.isalpha()]
    if not letters:
        return True
    non_latin_count = sum(1 for char in letters if _NON_LATIN_SCRIPT_REGEX.search(char))
    if (non_latin_count / len(letters)) < 0.20:
        return True
    # Transliterated script concordance applies exclusively to Slavic Cyrillic releases
    cyrillic_count = sum(1 for char in letters if _CYRILLIC_SCRIPT_REGEX.search(char))
    if title and cyrillic_count > 0 and (cyrillic_count / non_latin_count) >= 0.80:
        body = lyrics_word_body(cleaned_lyrics)
        trans_body = normalize_str(anyascii.anyascii(body))
        norm_title = normalize_str(title)
        tokens = [tok for tok in norm_title.split() if len(tok) >= 3] or [
            tok for tok in norm_title.split() if tok
        ]
        if tokens:
            trans_words = trans_body.split()
            if any(
                tok in trans_body
                or any(
                    score_artist_title_match("", tok, "", w) >= 80 for w in trans_words
                )
                for tok in tokens
            ):
                return True
        digits = re.findall(r"\d+", norm_title)
        if digits and any(d in trans_body.split() or d in trans_body for d in digits):
            return True
    return False


def evaluate_lyrics_quality(
    lyrics_text: str | None,
    artist: str | None = None,
    title: str | None = None,
    audio_duration: float | None = None,
    album: str | None = None,
) -> float:
    """Score lyrics quality on a 0.0-100.0 scale; returns 0.0 for rejected or corrupt lyrics."""
    if not lyrics_text:
        return 0.0
    cleaned = clean_lyrics_text(lyrics_text)
    if not cleaned:
        return 0.0
    tier = get_lyrics_quality(cleaned)
    if tier <= 0:
        return 0.0
    if artist and title and not lyrics_identity_matches(cleaned, artist, title):
        return 0.0
    ref = f"{artist or ''} {album or ''} {title or ''}".strip()
    body = lyrics_word_body(cleaned)
    if (artist or title) and not is_script_concordant(ref, cleaned, title=title):
        return 0.0
    if not is_lyrics_duration_valid(cleaned, audio_duration):
        return 0.0

    score = 40.0 if tier == 3 else (30.0 if tier == 2 else 15.0)
    words = len(body.split())
    if words >= 80:
        score += 30.0
    elif words >= 40:
        score += 20.0
    elif words >= 15:
        score += 10.0

    if tier >= 2 and audio_duration and audio_duration > 0:
        max_ts = extract_max_timestamp(cleaned)
        if max_ts is not None:
            coverage = min(1.0, max_ts / audio_duration)
            if coverage >= 0.70:
                score += 30.0
            elif coverage >= 0.40:
                score += 20.0
            else:
                score += 10.0
    elif tier == 1:
        if words >= 80:
            score += 20.0
        elif words >= 40:
            score += 10.0

    return min(100.0, score)


def score_lyrics_consensus(candidate_text: str, candidate_pool: Sequence[str]) -> float:
    """Bonus score [0.0-25.0] when candidate matches independent lyric sources."""
    candidate_body = lyrics_word_body(candidate_text)
    if not candidate_body:
        return 0.0
    agreeing = sum(
        1
        for other in candidate_pool
        if other is not candidate_text
        and fuzz.token_set_ratio(candidate_body, lyrics_word_body(other)) >= 85
    )
    return min(25.0, agreeing * 12.5)


def strip_lyrics_headers(lyrics_text: str) -> str:
    return _LRC_METADATA_HEADER_PATTERN.sub(
        "", clean_lyrics_text(lyrics_text) or ""
    ).strip()
