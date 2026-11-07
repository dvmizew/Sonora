import datetime
from collections import Counter
from pathlib import Path
from typing import Any

import orjson
from rich.markup import escape

from sonora import __version__
from sonora.core.files import atomic_write_bytes
from sonora.core.logger import LOG
from sonora.core.models import TrackInfo
from sonora.core.text import safe_float


def _write_json_report(
    json_report: Path,
    report_data: dict[str, Any],
    label: str = "report",
) -> None:
    atomic_write_bytes(
        json_report,
        orjson.dumps(
            report_data,
            option=orjson.OPT_INDENT_2
            | orjson.OPT_NON_STR_KEYS
            | orjson.OPT_SERIALIZE_DATACLASS,
        ),
    )
    LOG.info(f"Saved {label} JSON report to [bold]{escape(str(json_report))}[/bold]")


def _tag_coverage_counts(tagged_tracks: list[TrackInfo]) -> dict[str, int]:
    presence_fields = {
        "mb": "musicbrainz_trackid",
        "genre": "genre",
        "isrc": "isrc",
        "rg": "replaygain_track_gain",
        "key": "initial_key",
        "composer": "composer",
        "producers": "producers",
        "advisory": "advisory",
        "discogs": "discogs_release_id",
    }
    counts = {
        metric: sum(getattr(track, field) is not None for track in tagged_tracks)
        for metric, field in presence_fields.items()
    }
    counts.update(
        bpm=sum((safe_float(track.bpm) or 0) > 0 for track in tagged_tracks),
        lyrics=sum(
            track.lyrics is not None or track.synced_lyrics is not None
            for track in tagged_tracks
        ),
        genius=sum(
            bool(track.genius_song_id) or "Genius" in track.metadata_sources.values()
            for track in tagged_tracks
        ),
        theaudiodb=sum(
            "TheAudioDB" in track.metadata_sources.values() for track in tagged_tracks
        ),
        lossless=sum(track.is_lossless for track in tagged_tracks),
    )
    return counts


def _emit_tag_summary_and_report(
    tagged_tracks: list[TrackInfo],
    failures: list[dict[str, str]],
    interrupted: bool,
    path: Path,
    threads: int,
    json_report: Path | None,
    dry_run: bool = False,
) -> None:
    total_tracks = len(tagged_tracks)
    counts = _tag_coverage_counts(tagged_tracks)
    lossy_count = total_tracks - counts["lossless"]
    percentages = {
        metric: (count / total_tracks * 100) if total_tracks > 0 else 0.0
        for metric, count in counts.items()
    }
    modified_tracks = sum(1 for track in tagged_tracks if track._diff_descriptions)
    total_tag_changes = sum(len(track._diff_descriptions) for track in tagged_tracks)
    unchanged_tracks = total_tracks - modified_tracks
    mod_pct = (modified_tracks / total_tracks * 100) if total_tracks > 0 else 0.0
    unchanged_pct = (unchanged_tracks / total_tracks * 100) if total_tracks > 0 else 0.0

    tracks_with_conflicts = [
        track for track in tagged_tracks if track.metadata_conflicts
    ]
    recording_identifier_fields = ("musicbrainz_trackid", "isrc", "itunes_trackid")
    tracks_with_recording_evidence = sum(
        any(
            getattr(track, field) and track.metadata_sources.get(field)
            for field in recording_identifier_fields
        )
        for track in tagged_tracks
    )

    coverage_labels = {
        "mb": "Tracks with MusicBrainz ID",
        "genre": "Tracks with Genre",
        "lyrics": "Tracks with Lyrics",
        "isrc": "Tracks with ISRC",
        "bpm": "Tracks with BPM",
        "rg": "Tracks with ReplayGain",
        "key": "Musical Key / Tonality",
        "composer": "Composers & Writers",
        "producers": "Tracks with Producer Credits",
        "genius": "Tracks with Genius ID",
        "advisory": "Parental Advisory",
    }
    tag_mod_counts: Counter[str] = Counter()
    for track in tagged_tracks:
        for change in track._diff_descriptions:
            tag_name = change.split(":")[0].strip()
            tag_mod_counts[tag_name] += 1

    tag_display_names = {
        "bpm": "BPM",
        "initial_key": "Key",
        "advisory": "Advisory",
        "musicbrainz_trackid": "MB TrackID",
        "musicbrainz_albumid": "MB AlbumID",
        "musicbrainz_artistid": "MB ArtistID",
        "musicbrainz_releasegroupid": "MB ReleaseGroupID",
        "genius_song_id": "Genius",
        "producers": "Producers",
        "lyricist": "Lyricist",
        "lyrics": "Lyrics",
        "cover_art": "Artwork",
        "genre": "Genre",
        "artist_sort": "Sort Artist",
        "release_status": "Release Status",
        "release_country": "Country",
        "catalog_number": "Catalog No",
        "media": "Media",
        "label": "Label",
    }
    top_mods_str = ", ".join(
        f"{tag_display_names.get(k, k)} (+{v})"
        for k, v in tag_mod_counts.most_common(6)
    )

    tag_summary_rows: list[tuple[str, str, str | None]] = [
        (
            "Total Tracks Previewed" if dry_run else "Total Tracks Processed",
            str(total_tracks),
            None,
        ),
        (
            "Tracks That Would Be Updated" if dry_run else "Tracks Updated / Tagged",
            f"{modified_tracks}/{total_tracks} ({mod_pct:.1f}%)"
            if total_tracks
            else "0",
            "green" if modified_tracks > 0 else None,
        ),
        (
            "Total Tag Changes Previewed"
            if dry_run
            else "Total Tags Written / Updated",
            f"{total_tag_changes:,}",
            "cyan" if total_tag_changes > 0 else None,
        ),
        *(
            [("Top Tags Modified", top_mods_str, "cyan")]
            if top_mods_str and total_tag_changes > 0
            else []
        ),
        (
            "Tracks Already Compliant",
            f"{unchanged_tracks}/{total_tracks} ({unchanged_pct:.1f}%)"
            if total_tracks
            else "0",
            "dim" if unchanged_tracks > 0 else None,
        ),
        *(
            (
                label,
                f"{counts[metric]}/{total_tracks} ({percentages[metric]:.2f}%)",
                None,
            )
            for metric, label in coverage_labels.items()
        ),
        (
            "Tracks with Provider Conflicts",
            str(len(tracks_with_conflicts)),
            "yellow" if tracks_with_conflicts else None,
        ),
        (
            "Recording IDs Matched This Run",
            f"{tracks_with_recording_evidence}/{total_tracks}",
            None,
        ),
        (
            "Audio Formats",
            f"{counts['lossless']} Lossless / {lossy_count} Lossy",
            None,
        ),
    ]
    if not interrupted:
        if dry_run:
            LOG.success(
                f"Tagging preview completed: {total_tracks} tracks evaluated, "
                f"{modified_tracks} tracks would be updated ({total_tag_changes:,} tags)."
            )
        else:
            LOG.success(
                f"Tagging completed: {total_tracks} tracks processed, "
                f"{modified_tracks} tracks updated ({total_tag_changes:,} tags written)."
            )
    LOG.summary_table("Tagging Summary", tag_summary_rows)

    if json_report:
        summary_text = (
            f"{'Previewed' if dry_run else 'Processed'} {total_tracks} tracks ({len(failures)} failures). "
            f"MusicBrainz {counts['mb']}/{total_tracks} ({percentages['mb']:.2f}%), "
            f"Genre {counts['genre']}/{total_tracks} ({percentages['genre']:.2f}%), "
            f"Lyrics {counts['lyrics']}/{total_tracks} ({percentages['lyrics']:.2f}%), "
            f"BPM tags present {counts['bpm']}/{total_tracks} ({percentages['bpm']:.2f}%), "
            f"ReplayGain {counts['rg']}/{total_tracks} ({percentages['rg']:.2f}%)."
        )
        coverage_keys = {
            "mb": ("musicbrainz_matched_count", "musicbrainz_percentage"),
            "genre": ("genre_tagged_count", "genre_percentage"),
            "lyrics": ("lyrics_tagged_count", "lyrics_percentage"),
            "isrc": ("isrc_tagged_count", "isrc_percentage"),
            "bpm": ("bpm_calculated_count", "bpm_percentage"),
            "rg": ("replaygain_calculated_count", "replaygain_percentage"),
            "key": ("initial_key_count", "initial_key_percentage"),
            "composer": ("composer_tagged_count", "composer_percentage"),
            "producers": ("producers_tagged_count", "producers_percentage"),
            "advisory": ("advisory_tagged_count", "advisory_percentage"),
            "discogs": ("discogs_matched_count", "discogs_percentage"),
            "genius": ("genius_matched_count", "genius_percentage"),
            "theaudiodb": ("theaudiodb_matched_count", "theaudiodb_percentage"),
        }
        coverage_statistics: dict[str, int | float] = {}
        for metric, (count_key, percentage_key) in coverage_keys.items():
            coverage_statistics[count_key] = counts[metric]
            coverage_statistics[percentage_key] = round(percentages[metric], 2)
        report_data = {
            "schema": "tag_report_v2",
            "generator": "Sonora",
            "version": __version__,
            "summary_text": summary_text,
            "aborted_by_user": interrupted,
            "execution": {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "target_path": str(path.resolve()),
                "threads_used": threads,
                "dry_run": dry_run,
            },
            "statistics": {
                "total_tracks": total_tracks,
                "tracks_modified": modified_tracks,
                "tags_modified": total_tag_changes,
                "tracks_unchanged": unchanged_tracks,
                "total_failures": len(failures),
                "metadata_coverage": coverage_statistics,
                "audio_formats": {
                    "lossless_tracks": counts["lossless"],
                    "lossy_tracks": lossy_count,
                },
            },
            "verification": {
                "meaning": "Tag presence is coverage, not independently established API accuracy",
                "tracks_with_recording_identifier_evidence": tracks_with_recording_evidence,
                "tracks_with_metadata_conflicts": len(tracks_with_conflicts),
                "metadata_conflicts": [
                    {
                        "file_path": str(track.file_path),
                        "conflicts": dict(track.metadata_conflicts),
                    }
                    for track in tracks_with_conflicts
                ],
                "recording_identifiers_without_provider_evidence": {
                    field: [
                        str(track.file_path)
                        for track in tagged_tracks
                        if getattr(track, field)
                        and not track.metadata_sources.get(field)
                    ]
                    for field in recording_identifier_fields
                },
                "tracks_with_provider_evidence": sum(
                    bool(track.metadata_sources) for track in tagged_tracks
                ),
                "tracks_without_provider_evidence": [
                    str(track.file_path)
                    for track in tagged_tracks
                    if not track.metadata_sources
                ],
            },
            "failures": failures,
            "tracks": [
                {**track.to_dict(), "changes": list(track._diff_descriptions)}
                for track in tagged_tracks
            ],
        }
        _write_json_report(json_report, report_data, "tagging")
