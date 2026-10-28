import dataclasses
from collections.abc import Callable, Iterable
from typing import Any

from rich.markup import escape

from sonora.audio import key as key_module
from sonora.core.artists import normalize_featured_artists
from sonora.core.cache import api_cache_session
from sonora.core.logger import LOG
from sonora.core.models import MetadataSource, TrackInfo
from sonora.core.text import (
    clean_unicode_punct,
    normalize_advisory,
    normalize_country_name,
    normalize_date,
    normalize_genre,
    normalize_language_name,
    normalize_script_name,
    normalize_str,
    preserve_unicode_repertoire,
    safe_int,
)
from sonora.core.titles import extract_title_features
from sonora.modules.matching import (
    _is_generic,
    _is_generic_title,
    _recording_isrc_matches,
    provider_identity_matches,
)


@dataclasses.dataclass
class MetadataProposal:
    """Only provider changes and conflicts, never inherited audio or library state."""

    changes: dict[str, tuple[object, MetadataSource | None]]
    conflicts: dict[str, str]


_PROVIDER_FIELDS = tuple(
    attribute.name
    for attribute in dataclasses.fields(TrackInfo)
    if not attribute.name.startswith("_")
    and attribute.name
    not in {
        "file_path",
        "artist",
        "title",
        "duration",
        "metadata_sources",
        "metadata_conflicts",
        "sample_rate",
        "bitrate",
        "channels",
        "bits_per_sample",
        "is_lossless",
        "art_width",
        "art_height",
        "is_alien",
    }
)


@api_cache_session()
def collect_provider_proposal(
    recording: TrackInfo, enrich: Callable[[TrackInfo], None]
) -> MetadataProposal:
    candidate = recording.copy()
    enrich(candidate)
    return build_provider_proposal(recording, candidate)


def build_provider_proposal(
    recording: TrackInfo, candidate: TrackInfo
) -> MetadataProposal:
    if not provider_identity_matches(recording, candidate):
        return MetadataProposal(
            {},
            {
                "identity": (
                    "Provider proposed a conflicting recording identity; enrichment withheld."
                )
            },
        )
    return MetadataProposal(
        {
            name: (getattr(candidate, name), candidate.metadata_sources.get(name))
            for name in _PROVIDER_FIELDS
            if getattr(candidate, name) != getattr(recording, name)
            or candidate.metadata_sources.get(name)
            != recording.metadata_sources.get(name)
        },
        dict(candidate.metadata_conflicts),
    )


def merge_provider_proposals(
    recording: TrackInfo, proposals: Iterable[MetadataProposal]
) -> None:
    """Apply agreement without allowing provider order to determine disputed tags."""
    candidates_by_field: dict[str, list[tuple[object, MetadataSource | None]]] = {}
    for proposal in proposals:
        recording.metadata_conflicts.update(proposal.conflicts)
        for name, candidate in proposal.changes.items():
            candidates_by_field.setdefault(name, []).append(candidate)

    recording_isrcs = {
        normalize_str(str(value))
        for value, source in candidates_by_field.get("isrc", [])
        if value and source
    }
    if recording.isrc and recording.metadata_sources.get("isrc"):
        recording_isrcs.add(normalize_str(recording.isrc))
    if len(recording_isrcs) > 1:
        description = (
            "Providers disagree on ISRC; recording-specific enrichment withheld."
        )
        recording.metadata_conflicts["isrc"] = description
        LOG.detail(f"{escape(recording.file_path.name)}: {escape(description)}")
        return

    for name, candidates in candidates_by_field.items():
        if name not in recording.metadata_conflicts:
            _merge_provider_field(recording, candidates, name)


def _merge_provider_field(
    recording: TrackInfo,
    candidates: list[tuple[object, MetadataSource | None]],
    field_name: str,
) -> None:
    original = getattr(recording, field_name)
    original_source = recording.metadata_sources.get(field_name)
    if field_name == "featured_artists" and original:
        expanding_sources = {
            source
            for value, source in candidates
            if value != original and source is not None
        }
        if expanding_sources and len(expanding_sources) < 2:
            recording.metadata_conflicts[field_name] = (
                "Provider expansion of existing featured credits requires "
                "independent corroboration; existing credits preserved."
            )
            return
    if field_name == "advisory" and original == "Explicit":
        nonexplicit_sources = {
            source for value, source in candidates if value is None and source
        }
        # Multiple catalogs can describe the same clean edition. Their agreement
        # does not establish that the local audio belongs to that edition.
        if nonexplicit_sources:
            recording.metadata_conflicts["advisory"] = (
                "Provider rating does not independently verify the local edition; "
                "existing Explicit advisory preserved."
            )
            return
    if original_source in {
        "MusicBrainz",
        "AcoustID",
        "Shazam",
        "Tidal",
    }:
        candidates.append((original, original_source))
    distinct = {
        normalize_str(value) if isinstance(value, str) else value
        for value, _ in candidates
    }
    if len(distinct) > 1:
        alternatives = "; ".join(
            f"{source or 'provider'}={str(value)[:120]!r}"
            for value, source in candidates
        )
        description = f"Providers disagree on {field_name}: {alternatives}; existing value preserved."
        recording.metadata_conflicts[field_name] = description
        LOG.detail(f"{escape(recording.file_path.name)}: {escape(description)}")
        return
    value, source = candidates[0]
    setattr(recording, field_name, value)
    if source is not None:
        recording.metadata_sources[field_name] = source
    elif value is None:
        recording.metadata_sources.pop(field_name, None)


def apply_metadata_mapping(
    track_info: TrackInfo,
    metadata_dict: dict[str, Any],
    field_map: dict[str, str],
    force: bool = False,
    source: MetadataSource | None = None,
) -> None:
    for src_key, target_attr in field_map.items():
        provider_value = metadata_dict.get(src_key)
        if provider_value is None:
            continue
        normalized_value: str | None = str(provider_value).strip()
        if not normalized_value or normalized_value.lower() in ("none", "null"):
            continue
        if target_attr in {
            "mood",
            "style",
            "genre",
            "comment",
            "composer",
            "lyricist",
            "producers",
        } and (
            not any(character.isalnum() for character in normalized_value)
            or normalized_value.casefold() in {"unknown", "n/a"}
        ):
            continue

        if target_attr == "isrc" and _recording_isrc_matches(
            track_info.isrc, metadata_dict
        ):
            normalized_value = str(track_info.isrc).strip().upper()
        if target_attr == "genre":
            normalized_value = normalize_genre(normalized_value)
        elif target_attr in ("date", "original_date"):
            normalized_value = normalize_date(normalized_value)
        elif target_attr == "release_country":
            normalized_value = normalize_country_name(normalized_value)
        elif target_attr == "language":
            normalized_value = normalize_language_name(normalized_value)
        elif target_attr == "script":
            normalized_value = normalize_script_name(normalized_value)
        if not normalized_value:
            continue

        if target_attr == "title":
            normalized_value = clean_unicode_punct(normalized_value)
            clean_t, extracted_feats = extract_title_features(
                normalized_value, primary_artist=track_info.artist
            )
            if extracted_feats:
                track_info.featured_artists = normalize_featured_artists(
                    [track_info.featured_artists, *extracted_feats],
                    primary_artist=track_info.artist,
                )
            existing_title = getattr(track_info, "title", None)
            normalized_value = preserve_unicode_repertoire(
                str(existing_title) if existing_title else None,
                clean_t or normalized_value,
            )
        elif target_attr in ("artist", "album", "album_artist"):
            normalized_value = clean_unicode_punct(normalized_value)
            existing_val = getattr(track_info, target_attr, None)
            if (
                target_attr == "album"
                and existing_val
                and str(existing_val).casefold() == normalized_value.casefold()
            ):
                normalized_value = str(existing_val)
            normalized_value = preserve_unicode_repertoire(
                str(existing_val) if existing_val else None, normalized_value
            )
        elif target_attr == "advisory":
            normalized_value = normalize_advisory(normalized_value)
            if normalized_value is None:
                continue
        elif (
            target_attr == "initial_key"
            and key_module.key_to_camelot(normalized_value) is None
        ):
            continue
        elif target_attr == "featured_artists":
            existing_featured = getattr(track_info, "featured_artists", None)
            if existing_featured and not force:
                normalized_result = normalize_featured_artists(
                    [existing_featured, normalized_value],
                    primary_artist=track_info.artist,
                )
            else:
                normalized_result = normalize_featured_artists(
                    normalized_value,
                    primary_artist=track_info.artist,
                )
            if not normalized_result:
                continue
            normalized_value = normalized_result

        if target_attr in (
            "disc_number",
            "total_discs",
            "track_number",
            "total_tracks",
        ):
            val_int = safe_int(normalized_value)
            if val_int is None:
                continue
            if val_int < 0 or (target_attr != "track_number" and val_int == 0):
                continue
            if target_attr in ("total_tracks", "total_discs"):
                position_attribute = (
                    "track_number" if target_attr == "total_tracks" else "disc_number"
                )
                current_position = getattr(track_info, position_attribute)
                if current_position is not None and val_int < current_position:
                    track_info.metadata_conflicts[target_attr] = (
                        f"Provider {target_attr}={val_int} is below existing "
                        f"{position_attribute}={current_position}; existing value preserved."
                    )
                    continue
            cur_int = safe_int(getattr(track_info, target_attr, None))
            if target_attr in ("disc_number", "total_discs"):
                if cur_int is None or cur_int != val_int:
                    setattr(track_info, target_attr, val_int)
            elif cur_int is None or force:
                setattr(track_info, target_attr, val_int)
            if source is not None and getattr(track_info, target_attr) == val_int:
                if cur_int != val_int or force:
                    track_info.metadata_sources[target_attr] = source
                else:
                    track_info.metadata_sources.setdefault(target_attr, source)
            continue

        if target_attr in ("label", "copyright", "composer", "lyricist", "producers"):
            existing_credit = getattr(track_info, target_attr)
            if existing_credit and set(normalize_str(normalized_value).split()) < set(
                normalize_str(str(existing_credit)).split()
            ):
                continue

        current_value = getattr(track_info, target_attr)
        missing_identity = (
            target_attr == "title" and _is_generic_title(current_value)
        ) or (
            target_attr in ("artist", "album", "album_artist")
            and _is_generic(current_value, target_attr)
        )
        if (
            not current_value
            or missing_identity
            or target_attr == "featured_artists"
            or force
        ):
            setattr(track_info, target_attr, normalized_value)
            if source is not None:
                track_info.metadata_sources[target_attr] = source
        elif (
            getattr(track_info, target_attr) == normalized_value and source is not None
        ):
            track_info.metadata_sources.setdefault(target_attr, source)
