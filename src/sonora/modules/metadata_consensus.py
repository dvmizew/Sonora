import dataclasses
from collections.abc import Callable, Iterable

from rich.markup import escape

from sonora.core.cache import api_cache_session
from sonora.core.logger import LOG
from sonora.core.models import MetadataSource, TrackInfo
from sonora.core.text import normalize_str
from sonora.core.titles import recording_identity_matches


@api_cache_session()
def collect_provider_proposal(
    recording: TrackInfo, enrich: Callable[[TrackInfo], None]
) -> TrackInfo:
    proposal = dataclasses.replace(
        recording,
        metadata_sources=recording.metadata_sources.copy(),
        metadata_conflicts=recording.metadata_conflicts.copy(),
    )
    enrich(proposal)
    return proposal


def merge_provider_proposals(
    recording: TrackInfo, proposals: Iterable[TrackInfo]
) -> None:
    """Apply agreement without allowing provider order to determine disputed tags."""
    accepted: list[TrackInfo] = []
    for proposal in proposals:
        if (
            proposal.artist != recording.artist
            or proposal.title != recording.title
            or proposal.duration != recording.duration
        ) and not recording_identity_matches(
            recording.artist,
            recording.title,
            proposal.artist,
            proposal.title,
            expected_duration=recording.duration,
            candidate_duration=proposal.duration,
        ):
            description = "Provider proposed a conflicting recording identity; enrichment withheld."
            recording.metadata_conflicts["identity"] = description
            LOG.warning(f"{escape(recording.file_path.name)}: {escape(description)}")
            continue
        recording.metadata_conflicts.update(proposal.metadata_conflicts)
        accepted.append(proposal)

    recording_isrcs = {
        normalize_str(proposal.isrc)
        for proposal in accepted
        if proposal.isrc and proposal.metadata_sources.get("isrc")
    }
    if recording.isrc and recording.metadata_sources.get("isrc"):
        recording_isrcs.add(normalize_str(recording.isrc))
    if len(recording_isrcs) > 1:
        description = (
            "Providers disagree on ISRC; recording-specific enrichment withheld."
        )
        recording.metadata_conflicts["isrc"] = description
        LOG.warning(f"{escape(recording.file_path.name)}: {escape(description)}")
        return

    for attribute in dataclasses.fields(recording):
        field_name = attribute.name
        if (
            field_name.startswith("_")
            or field_name
            in {
                "file_path",
                "artist",
                "title",
                "duration",
                "metadata_sources",
                "metadata_conflicts",
            }
            or field_name in recording.metadata_conflicts
        ):
            continue
        original = getattr(recording, field_name)
        # Provenance distinguishes an independently accepted answer from inherited tags.
        candidates: list[tuple[object, MetadataSource | None]] = [
            (getattr(proposal, field_name), proposal.metadata_sources.get(field_name))
            for proposal in accepted
            if getattr(proposal, field_name) != original
            or proposal.metadata_sources.get(field_name)
            != recording.metadata_sources.get(field_name)
        ]
        if not candidates:
            continue
        if recording.metadata_sources.get(field_name) in {
            "MusicBrainz",
            "AcoustID",
            "Shazam",
        }:
            candidates.append((original, recording.metadata_sources[field_name]))
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
            LOG.warning(f"{escape(recording.file_path.name)}: {escape(description)}")
            continue
        value, source = candidates[0]
        setattr(recording, field_name, value)
        if source is not None:
            recording.metadata_sources[field_name] = source
        elif value is None:
            recording.metadata_sources.pop(field_name, None)
