import re
from collections.abc import Callable

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
    *,
    artist_resolver: Callable[[str | None], str] | None = None,
    group_classifier: Callable[[str | None], bool] | None = None,
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
            group_classifier=group_classifier,
        )
        if title_feats:
            track_info.featured_artists = normalize_featured_artists(
                [track_info.featured_artists, *title_feats],
                primary_artist=track_info.artist,
                group_classifier=group_classifier,
            )
        track_info.title = clean_unicode_punct(clean_base)

    if track_info.artist:
        track_info.artist = strip_corrupt_brackets(track_info.artist)
        base_artist, extracted_features = extract_artist_features(
            track_info.artist, group_classifier=group_classifier
        )
        if base_artist:
            track_info.artist = base_artist
        if extracted_features:
            track_info.featured_artists = normalize_featured_artists(
                [track_info.featured_artists, *extracted_features],
                primary_artist=track_info.artist,
                group_classifier=group_classifier,
            )
        track_info.artist = clean_disambiguation(track_info.artist)
        if artist_resolver is not None:
            track_info.artist = artist_resolver(track_info.artist)
        track_info.artist = clean_unicode_punct(track_info.artist)

    if track_info.featured_artists:
        track_info.featured_artists = normalize_featured_artists(
            track_info.featured_artists,
            primary_artist=track_info.artist,
            group_classifier=group_classifier,
        )

    if track_info.album:
        track_info.album = clean_unicode_punct(track_info.album)

    if track_info.album_artist:
        track_info.album_artist = clean_disambiguation(track_info.album_artist)
        if artist_resolver is not None:
            track_info.album_artist = artist_resolver(track_info.album_artist)
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

    if track_info.artist_sort and track_info.artist:
        sort_norm = normalize_str(track_info.artist_sort)
        art_candidates = [normalize_str(track_info.artist)]
        primary_artist_val = get_primary_artist(track_info.artist)
        art_candidates.append(normalize_str(primary_artist_val))
        sort_score = max(
            (
                max(
                    fuzz.ratio(sort_norm, cand),
                    fuzz.token_sort_ratio(sort_norm, cand),
                    fuzz.token_set_ratio(sort_norm, cand),
                )
                for cand in art_candidates
                if cand
            ),
            default=0.0,
        )
        if sort_score < get_config().artist_similarity_threshold:
            track_info.artist_sort = None

    cmp_sort_art = track_info.album_artist or track_info.artist or ""
    if track_info.album_artist_sort and cmp_sort_art:
        sort_norm = normalize_str(track_info.album_artist_sort)
        art_candidates = [normalize_str(cmp_sort_art)]
        primary_album_art = get_primary_artist(cmp_sort_art)
        art_candidates.append(normalize_str(primary_album_art))
        sort_score = max(
            (
                max(
                    fuzz.ratio(sort_norm, cand),
                    fuzz.token_sort_ratio(sort_norm, cand),
                    fuzz.token_set_ratio(sort_norm, cand),
                )
                for cand in art_candidates
                if cand
            ),
            default=0.0,
        )
        if sort_score < get_config().artist_similarity_threshold:
            track_info.album_artist_sort = None

    track_info.genre = normalize_genre(track_info.genre)
    track_info.date = normalize_date(track_info.date)
    track_info.original_date = normalize_date(track_info.original_date)
    track_info.release_country = normalize_country_name(track_info.release_country)
    track_info.language = normalize_language_name(track_info.language)
    track_info.script = normalize_script_name(track_info.script)
    track_info.advisory = normalize_advisory(track_info.advisory)

    return track_info
