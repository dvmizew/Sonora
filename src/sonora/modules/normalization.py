import re

from rapidfuzz import fuzz

from sonora.core.artists import (
    extract_artist_features,
    get_primary_artist,
    harmonize_artist_casing,
    normalize_featured_artists,
)
from sonora.core.config import get_config
from sonora.core.models import TrackInfo
from sonora.core.text import (
    clean_disambiguation,
    clean_unicode_punct,
    normalize_advisory,
    normalize_country_name,
    normalize_date,
    normalize_genre,
    normalize_language_name,
    normalize_script_name,
    normalize_str,
)
from sonora.core.titles import extract_title_features, strip_corrupt_brackets

_PROD_BRACKET_PATTERN = re.compile(
    r"\s*[\(\[\{]\s*(?:prod(?:\.|uced|uction)?\s*(?:by|:)?|produced\s+by)\s*([^()\[\]{}]+?)[\)\]\}]",
    re.IGNORECASE,
)


def normalize_track_metadata(
    track_info: TrackInfo,
) -> TrackInfo:
    """
    Apply canonical, deterministic normalization to all metadata fields of a TrackInfo object.
    Unifies offline tag normalization and online tag post-processing into a single source of truth.
    """
    if track_info.title:
        raw_t_lower = track_info.title.lower()
        if not track_info.advisory and (
            re.search(r"[\(\[\{]\s*explicit\s*[\)\]\}]", raw_t_lower)
            or "album version (explicit)" in raw_t_lower
        ):
            track_info.advisory = "Explicit"

        prod_matches = list(_PROD_BRACKET_PATTERN.finditer(track_info.title))
        if prod_matches:
            if not track_info.producers:
                prods = [m.group(1).strip() for m in prod_matches if m.group(1).strip()]
                if prods:
                    track_info.producers = ", ".join(prods)
            track_info.title = _PROD_BRACKET_PATTERN.sub("", track_info.title).strip()

        track_info.title = strip_corrupt_brackets(track_info.title)
        clean_base, title_feats = extract_title_features(
            track_info.title,
            primary_artist=track_info.artist,
        )
        if title_feats:
            track_info.featured_artists = normalize_featured_artists(
                [track_info.featured_artists, *title_feats],
                primary_artist=track_info.artist,
            )
        track_info.title = clean_unicode_punct(clean_base)

    if track_info.artist:
        track_info.artist = strip_corrupt_brackets(track_info.artist)
        base_artist, extracted_features = extract_artist_features(track_info.artist)
        if base_artist:
            track_info.artist = base_artist
        if extracted_features:
            track_info.featured_artists = normalize_featured_artists(
                [track_info.featured_artists, *extracted_features],
                primary_artist=track_info.artist,
            )
        track_info.artist = clean_disambiguation(track_info.artist)
        track_info.artist = clean_unicode_punct(track_info.artist)

    if track_info.featured_artists:
        track_info.featured_artists = normalize_featured_artists(
            track_info.featured_artists,
            primary_artist=track_info.artist,
        )

    if track_info.album:
        track_info.album = clean_unicode_punct(track_info.album)

    if track_info.album_artist:
        track_info.album_artist = clean_disambiguation(track_info.album_artist)
        track_info.album_artist = clean_unicode_punct(track_info.album_artist)

    if (
        track_info.artist
        and track_info.album_artist
        and normalize_str(track_info.artist) == normalize_str(track_info.album_artist)
        and track_info.artist != track_info.album_artist
    ):
        harmonized = harmonize_artist_casing(
            candidate_artist=track_info.artist,
            reference_artist=track_info.album_artist,
        )
        track_info.artist = harmonized
        track_info.album_artist = harmonized

    is_singles_track = bool(
        (track_info.total_tracks == 1 and (track_info.track_number or 1) > 1)
        or (
            track_info.release_type
            and track_info.release_type.lower() == "single"
            and (track_info.total_tracks or 1) <= 1
        )
        or "singles" in [p.lower() for p in track_info.file_path.parts]
        or get_config().is_generic_container(track_info.file_path.parent.name)
    )
    if is_singles_track and (track_info.total_tracks or 1) <= 1:
        track_info.track_number = 1
        track_info.total_tracks = 1

    for sort_field, artist in (
        ("artist_sort", track_info.artist),
        ("album_artist_sort", track_info.album_artist or track_info.artist or ""),
    ):
        sort_name = getattr(track_info, sort_field)
        if not sort_name or not artist:
            continue
        normalized_sort = normalize_str(sort_name)
        artist_candidates = [
            normalize_str(artist),
            normalize_str(get_primary_artist(artist)),
        ]
        sort_score = max(
            (
                max(
                    fuzz.ratio(normalized_sort, candidate),
                    fuzz.token_sort_ratio(normalized_sort, candidate),
                    fuzz.token_set_ratio(normalized_sort, candidate),
                )
                for candidate in artist_candidates
                if candidate
            ),
            default=0.0,
        )
        if sort_score < get_config().artist_similarity_threshold:
            setattr(track_info, sort_field, None)

    for field in ("mood", "style"):
        value = getattr(track_info, field)
        if value and (
            not any(character.isalnum() for character in value)
            or value.strip().casefold() in {"unknown", "n/a", "none", "null"}
        ):
            setattr(track_info, field, None)

    track_info.genre = normalize_genre(track_info.genre)
    track_info.date = normalize_date(track_info.date)
    track_info.original_date = normalize_date(track_info.original_date)
    track_info.release_country = normalize_country_name(track_info.release_country)
    track_info.language = normalize_language_name(track_info.language)
    track_info.script = normalize_script_name(track_info.script)
    track_info.advisory = normalize_advisory(track_info.advisory)

    return track_info
