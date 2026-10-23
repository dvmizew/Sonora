import re
from collections.abc import Callable
from functools import lru_cache

import ftfy
from music_metadata_filter.filter import MetadataFilter
from music_metadata_filter.functions import (
    fix_track_suffix,
    remove_clean_explicit,
    remove_feature,
    remove_parody,
    remove_reissue,
    remove_remastered,
    remove_zero_width,
    replace_nbsp,
    youtube,
)
from rapidfuzz import fuzz

from sonora.core.artists import (
    extract_artist_features,
    extract_balanced_features,
    extract_featured_artist_tokens,
    get_primary_artist,
)
from sonora.core.config import (
    get_bracket_feat_pattern,
    get_config,
    get_duplicate_feat_pattern,
)
from sonora.core.constants import FEAT_KEYWORDS
from sonora.core.text import (
    _COLLAPSE_SPACES_PATTERN,
    _NON_WORD_SPACES_PATTERN,
    clean_unicode_punct,
    normalize_str,
)

_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50}


def _parse_roman_numeral(token: str) -> int | None:
    total, previous_numeral = 0, 0
    for char in reversed(token):
        numeral_value = _ROMAN_VALUES.get(char)
        if numeral_value is None:
            return None
        total += numeral_value if numeral_value >= previous_numeral else -numeral_value
        previous_numeral = numeral_value
    return total if 1 <= total <= 50 else None


_WORD_NUMBERS = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
]

_WORD_NUMBER_MAP: dict[str, int] = {w: i for i, w in enumerate(_WORD_NUMBERS, 1)}


def extract_series_number(text: str | None) -> int | None:
    """
    Extract album or track series/volume number (e.g. 'Savage Mode II' -> 2, 'Pt. 2' -> 2, 'Vol. 3' -> 3).
    Returns integer series number or None if not part of a numbered series.
    """
    if not text:
        return None

    clean_text = ftfy.fix_text(str(text)).strip().lower()

    match = re.search(
        r"\b(?:vol(?:ume)?|pt|part|chapter|act|book)\.?\s*(\d{1,2}|[a-z]+)\b",
        clean_text,
        re.IGNORECASE,
    ) or re.search(r"\b(\d{1,2}|[a-z]+)\s*$", clean_text, re.IGNORECASE)

    if match:
        token = match.group(1).lower()
        if token.isdigit():
            return int(token)
        return _parse_roman_numeral(token) or _WORD_NUMBER_MAP.get(token)

    return None


_METADATA_FILTER = MetadataFilter(
    {
        "track": (
            remove_zero_width,
            replace_nbsp,
            remove_clean_explicit,
            remove_reissue,
            remove_remastered,
            remove_parody,
            remove_feature,
            fix_track_suffix,
        ),
    }
)

_TITLE_EDITION_PATTERN = re.compile(
    r"\s*[\(\[\{](?:\d{4}\s+)?(?:deluxe|bonus\s+track|mono|stereo|hq|hd|album\s+version|clean\s+version|explicit\s+version|parody|official\s+(?:music\s+)?video|official\s+audio).*?[\)\]\}]",
    re.IGNORECASE,
)

_UNBRACKETED_FEAT_PATTERN = re.compile(
    r"(?:[\(\[\{\s]+|\s+)(?:fea?t(?:uring)?|ft|w/(?!\s*[oO](?:ut)?\b)|w\.)\.?\s+([^\)\]\}\n]+)[\)\]\}\s]*",
    re.IGNORECASE,
)


def extract_title_features(
    title: str | None,
    primary_artist: str | None = None,
    *,
    group_classifier: Callable[[str | None], bool] | None = None,
) -> tuple[str, list[str]]:
    """
    Extract featuring artists from a track title and return (clean_title, list_of_featured_artists).
    Removes bracketed and unbracketed featuring phrases, keeping title clean.
    """
    if not title:
        return ("", [])
    fixed_title = clean_unicode_punct(ftfy.fix_text(str(title)))
    deduped_title = get_duplicate_feat_pattern().sub("", fixed_title)
    clean_base, bracket_feats, _, _ = extract_balanced_features(deduped_title)

    unbracketed_feats: list[str] = []
    feat_m = _UNBRACKETED_FEAT_PATTERN.search(clean_base)
    if feat_m:
        unbracketed_feats.append(feat_m.group(1).strip())
        clean_base = clean_base[: feat_m.start()].strip()

    cleaned = _METADATA_FILTER.filter_field("track", clean_base)
    cleaned = get_bracket_feat_pattern().sub("", cleaned)
    cleaned = _TITLE_EDITION_PATTERN.sub("", cleaned)
    cleaned_title = _COLLAPSE_SPACES_PATTERN.sub(" ", cleaned).strip()

    all_raw_feats = bracket_feats + unbracketed_feats
    featured_tokens = extract_featured_artist_tokens(
        all_raw_feats,
        primary_artist=primary_artist,
        group_classifier=group_classifier,
    )
    return cleaned_title, featured_tokens


@lru_cache(maxsize=8192)
def clean_title(title: str | None) -> str:
    """Clean track title by removing feat./ft./with markers, remaster suffixes, and mojibake text."""
    if not title:
        return ""
    clean_t, _ = extract_title_features(title)
    return clean_t


_VERSION_OR_REMIX_KEYWORDS = frozenset(
    {
        "remix",
        "extended",
        "rework",
        "edit",
        "mix",
        "live",
        "acoustic",
        "instrumental",
        "version",
        "demo",
        "sped up",
        "slowed",
        "freestyle",
    }
)

_ALBUM_VERSION_PATTERN = re.compile(
    r"[\(\[\{]\s*album\s+version\s*[\)\]\}]", re.IGNORECASE
)

_REMIX_MODIFIER_PATTERN = re.compile(
    r"[\(\[\{]([^\)\]\}]+?(?:remix|rework|edit|mix|version|dub|flip|vip|bootleg|demo|live|acoustic|instrumental))[\]\)\}]",
    re.IGNORECASE,
)


def extract_version_modifier(title: str) -> str | None:
    match = _REMIX_MODIFIER_PATTERN.search(title)
    if match:
        return match.group(1).lower().strip()
    return None


def is_version_or_remix(text: str) -> bool:
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in _VERSION_OR_REMIX_KEYWORDS)


_BRACKET_PATTERN = re.compile(r"[\(\[\{][^\(\)\[\]\{\}]+[\)\]\}]")


def extract_bracket_tokens(text: str) -> list[tuple[str, set[str]]]:
    """Extract all bracketed substrings and their constituent alphanumeric word tokens."""
    results: list[tuple[str, set[str]]] = []
    for match in _BRACKET_PATTERN.finditer(text):
        full_bracket = match.group(0)
        inner = full_bracket[1:-1].lower()
        tokens = set("".join(c if c.isalnum() else " " for c in inner).split())
        results.append((full_bracket, tokens))
    return results


def _is_corrupt_bracket(full_bracket: str, tokens: set[str]) -> bool:
    if is_version_or_remix(full_bracket) or re.search(
        FEAT_KEYWORDS, full_bracket, re.IGNORECASE
    ):
        return False

    inner_content = full_bracket.strip("()[]{}").strip()
    if inner_content.isdigit() and len(inner_content) == 4:
        return False

    dummy_title = f"Track {full_bracket}"
    if (
        youtube(dummy_title) == "Track"
        or remove_remastered(dummy_title) == "Track"
        or remove_clean_explicit(dummy_title) == "Track"
    ):
        return True

    return bool(tokens & get_config().codec_rip_keywords)


def strip_corrupt_brackets(text: str) -> str:
    """
    Remove unwanted or corrupt bracket metadata (e.g., [FLAC], (Official Video), [HQ], [Prod: ...])
    from a title or artist string, preserving legitimate features and version brackets.
    """
    if not text:
        return ""
    cleaned = text
    for full_bracket, tokens in extract_bracket_tokens(text):
        if _is_corrupt_bracket(full_bracket, tokens):
            cleaned = cleaned.replace(full_bracket, "")
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def _title_identity_score(query_title: str, candidate_title: str) -> float:
    config = get_config()
    query_title_clean = clean_title(query_title).lower()
    candidate_title_clean = clean_title(candidate_title).lower()

    query_series = extract_series_number(query_title_clean)
    candidate_series = extract_series_number(candidate_title_clean)
    if (
        query_series is not None
        and candidate_series is not None
        and query_series != candidate_series
    ):
        return 0.0
    if (query_series is not None) != (candidate_series is not None) and max(
        query_series or 0, candidate_series or 0
    ) > 1:
        return 0.0

    normalized_query_title = normalize_str(query_title_clean)
    normalized_candidate_title = normalize_str(candidate_title_clean)
    if query_title_clean == candidate_title_clean or (
        normalized_query_title and normalized_query_title == normalized_candidate_title
    ):
        title_score = 100.0
    else:
        title_ratio = float(fuzz.ratio(query_title_clean, candidate_title_clean))
        title_token_sort = float(
            fuzz.token_sort_ratio(query_title_clean, candidate_title_clean)
        )
        title_score = max(title_ratio, title_token_sort)

        query_word = _NON_WORD_SPACES_PATTERN.sub(" ", query_title_clean).strip()
        candidate_word = _NON_WORD_SPACES_PATTERN.sub(
            " ", candidate_title_clean
        ).strip()
        if (
            query_word
            and candidate_word
            and (
                query_word != query_title_clean
                or candidate_word != candidate_title_clean
            )
        ):
            title_score = max(
                title_score,
                float(fuzz.ratio(query_word, candidate_word)),
                float(fuzz.token_sort_ratio(query_word, candidate_word)),
            )
            query_tokens = query_word.split()
            candidate_tokens = candidate_word.split()
            if (
                min(len(query_tokens), len(candidate_tokens)) >= 3
                and min(len(query_word), len(candidate_word))
                / max(len(query_word), len(candidate_word))
                >= 0.5
            ):
                title_score = max(
                    title_score, float(fuzz.token_set_ratio(query_word, candidate_word))
                )

        query_no_articles = re.sub(r"^(?:the|a|an)\s+", "", query_title_clean).strip()
        candidate_no_articles = re.sub(
            r"^(?:the|a|an)\s+", "", candidate_title_clean
        ).strip()
        if (
            query_no_articles
            and candidate_no_articles
            and (
                query_no_articles != query_title_clean
                or candidate_no_articles != candidate_title_clean
            )
        ):
            art_ratio = float(fuzz.ratio(query_no_articles, candidate_no_articles))
            art_sort = float(
                fuzz.token_sort_ratio(query_no_articles, candidate_no_articles)
            )
            title_score = max(title_score, art_ratio, art_sort)

    query_version = is_version_or_remix(
        _ALBUM_VERSION_PATTERN.sub("", query_title)
    ) or is_version_or_remix(query_title_clean)
    candidate_version = is_version_or_remix(
        _ALBUM_VERSION_PATTERN.sub("", candidate_title)
    ) or is_version_or_remix(candidate_title_clean)
    if query_version != candidate_version:
        if query_title_clean == candidate_title_clean:
            title_score -= config.version_equivalence_penalty
        else:
            title_score -= config.version_mismatch_penalty
    elif query_version and candidate_version:
        query_modifier = extract_version_modifier(query_title)
        candidate_modifier = extract_version_modifier(candidate_title)
        if query_modifier and candidate_modifier:
            mod_ratio = float(fuzz.ratio(query_modifier, candidate_modifier))
            mod_sort = float(fuzz.token_sort_ratio(query_modifier, candidate_modifier))
            if max(mod_ratio, mod_sort) < config.remix_match_threshold:
                title_score -= config.remix_modifier_mismatch_penalty
        else:
            for kw in _VERSION_OR_REMIX_KEYWORDS:
                if (kw in query_title_clean) != (kw in candidate_title_clean):
                    title_score -= config.version_mismatch_penalty
                    break

    title_score = max(0.0, min(100.0, title_score))

    return title_score


@lru_cache(maxsize=8192)
def _credited_feature_artists(artist: str, title: str) -> frozenset[str]:
    primary_artist, artist_features = extract_artist_features(artist)
    _, title_features = extract_title_features(title, primary_artist=primary_artist)
    return frozenset(
        normalize_str(credit) for credit in (*artist_features, *title_features)
    )


def score_artist_title_match(
    query_artist: str,
    query_title: str,
    candidate_artist: str,
    candidate_title: str,
) -> float:
    """
    Calculate a combined 0-100 similarity score between query (artist, title)
    and candidate (artist, title) using RapidFuzz ratio with version, remix modifier, and series penalties.
    """
    if not query_title or not candidate_title:
        return 0.0
    query_features = _credited_feature_artists(query_artist, query_title)
    candidate_features = _credited_feature_artists(candidate_artist, candidate_title)
    # Providers may omit credits; two different explicit credit sets are conflicting evidence.
    if query_features and candidate_features and query_features != candidate_features:
        return 0.0

    config = get_config()
    query_artist_clean = clean_title(query_artist).lower()
    candidate_artist_clean = clean_title(candidate_artist).lower()
    query_title_clean = clean_title(query_title).lower()
    candidate_title_clean = clean_title(candidate_title).lower()
    normalized_query_artist = normalize_str(query_artist_clean)
    normalized_candidate_artist = normalize_str(candidate_artist_clean)

    title_score = _title_identity_score(query_title, candidate_title)

    if query_artist_clean:
        if not candidate_artist_clean:
            artist_score = 50.0
        elif query_artist_clean == candidate_artist_clean or (
            normalized_query_artist
            and normalized_query_artist == normalized_candidate_artist
        ):
            artist_score = 100.0
        else:
            query_primary = clean_title(get_primary_artist(query_artist_clean)).lower()
            candidate_primary = clean_title(
                get_primary_artist(candidate_artist_clean)
            ).lower()
            if query_primary == candidate_primary:
                artist_score = 100.0
            else:
                query_artist_core = re.sub(
                    r"^(?:the|a|an)\s+", "", query_primary
                ).strip()
                candidate_artist_core = re.sub(
                    r"^(?:the|a|an)\s+", "", candidate_primary
                ).strip()
                if query_artist_core == candidate_artist_core:
                    artist_score = 100.0
                else:
                    min_len = min(len(query_artist_core), len(candidate_artist_core))
                    if min_len <= 3:
                        artist_score = 0.0
                    else:
                        core_ratio = float(
                            fuzz.ratio(query_artist_core, candidate_artist_core)
                        )
                        core_sort = float(
                            fuzz.token_sort_ratio(
                                query_artist_core, candidate_artist_core
                            )
                        )
                        artist_score = max(core_ratio, core_sort)

        min_title_len = min(len(query_title_clean), len(candidate_title_clean))
        title_thresh = (
            config.short_title_match_threshold
            if min_title_len < config.short_title_length
            else config.title_match_threshold
        )
        if (
            title_score < 100
            and (
                title_score <= title_thresh
                or title_score < config.artist_match_threshold
            )
        ) or (
            artist_score < 100
            and (
                artist_score <= config.title_match_threshold
                or artist_score < config.artist_match_threshold
            )
        ):
            return 0.0

        return (title_score * config.title_score_weight) + (
            artist_score * (1 - config.title_score_weight)
        )

    min_title_len = min(len(query_title_clean), len(candidate_title_clean))
    title_thresh = (
        config.short_title_match_threshold
        if min_title_len < config.short_title_length
        else config.title_match_threshold
    )
    if title_score < 100 and (
        title_score <= title_thresh or title_score < config.artist_match_threshold
    ):
        return 0.0
    return float(title_score)


def recording_identity_matches(
    expected_artist: str | None,
    expected_title: str | None,
    candidate_artist: str | None,
    candidate_title: str | None,
    *,
    expected_duration: float | None = None,
    candidate_duration: float | None = None,
) -> bool:
    """Require recording text and available duration evidence to agree."""
    if (
        score_artist_title_match(
            expected_artist or "",
            expected_title or "",
            candidate_artist or "",
            candidate_title or "",
        )
        <= 0
    ):
        return False
    return not (
        expected_duration is not None
        and candidate_duration is not None
        and abs(expected_duration - candidate_duration)
        > get_config().single_duration_tolerance_seconds
    )


def clear_title_cache() -> None:
    """Invalidate title results derived from configured feature delimiters."""
    clean_title.cache_clear()
