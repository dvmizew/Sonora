import datetime
from pathlib import Path
from typing import Any

import orjson
from rich.markup import escape

from sonora import __version__
from sonora.core.files import atomic_write_bytes
from sonora.core.logger import LOG
from sonora.core.models import TrackInfo


def _write_json_report(
    json_report: Path,
    report_data: dict[str, Any],
    label: str = "report",
) -> None:
    json_report.parent.mkdir(parents=True, exist_ok=True)
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
    counts = {
        "mb": sum(
            1
            for track_metadata in tagged_tracks
            if track_metadata.musicbrainz_trackid is not None
        ),
        "genre": sum(
            1 for track_metadata in tagged_tracks if track_metadata.genre is not None
        ),
        "lyrics": sum(
            1
            for track_metadata in tagged_tracks
            if track_metadata.lyrics is not None
            or track_metadata.synced_lyrics is not None
        ),
        "isrc": sum(
            1 for track_metadata in tagged_tracks if track_metadata.isrc is not None
        ),
        "bpm": sum(
            1 for track_metadata in tagged_tracks if track_metadata.bpm is not None
        ),
        "rg": sum(
            1
            for track_metadata in tagged_tracks
            if track_metadata.replaygain_track_gain is not None
        ),
        "key": sum(
            1
            for track_metadata in tagged_tracks
            if track_metadata.initial_key is not None
        ),
        "composer": sum(
            1 for track_metadata in tagged_tracks if track_metadata.composer is not None
        ),
        "producers": sum(
            1
            for track_metadata in tagged_tracks
            if track_metadata.producers is not None
        ),
        "advisory": sum(
            1 for track_metadata in tagged_tracks if track_metadata.advisory is not None
        ),
        "discogs": sum(
            1
            for track_metadata in tagged_tracks
            if track_metadata.discogs_release_id is not None
        ),
        "genius": sum(
            1
            for track_metadata in tagged_tracks
            if track_metadata.genius_song_id is not None
            or track_metadata.comment is not None
        ),
        "theaudiodb": sum(
            1
            for track in tagged_tracks
            if "TheAudioDB" in track.metadata_sources.values()
        ),
        "lossless": sum(
            1 for track_metadata in tagged_tracks if track_metadata.is_lossless
        ),
    }
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
    pcts = {
        k: (v / total_tracks * 100) if total_tracks > 0 else 0.0
        for k, v in counts.items()
    }
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

    tag_summary_rows = [
        ("Total Tracks Processed", str(total_tracks), None),
        (
            "Tracks with MusicBrainz ID",
            f"{counts['mb']}/{total_tracks} ({pcts['mb']:.2f}%)",
            None,
        ),
        (
            "Genre & Styles Tagged",
            f"{counts['genre']}/{total_tracks} ({pcts['genre']:.2f}%)",
            None,
        ),
        (
            "Tracks with Lyrics",
            f"{counts['lyrics']}/{total_tracks} ({pcts['lyrics']:.2f}%)",
            None,
        ),
        (
            "Tracks with ISRC",
            f"{counts['isrc']}/{total_tracks} ({pcts['isrc']:.2f}%)",
            None,
        ),
        (
            "Tracks with BPM",
            f"{counts['bpm']}/{total_tracks} ({pcts['bpm']:.2f}%)",
            None,
        ),
        (
            "Tracks with ReplayGain",
            f"{counts['rg']}/{total_tracks} ({pcts['rg']:.2f}%)",
            None,
        ),
        (
            "Musical Key / Tonality",
            f"{counts['key']}/{total_tracks} ({pcts['key']:.2f}%)",
            None,
        ),
        (
            "Composers & Writers",
            f"{counts['composer']}/{total_tracks} ({pcts['composer']:.2f}%)",
            None,
        ),
        (
            "Producers & Studio Credits",
            f"{counts['producers']}/{total_tracks} ({pcts['producers']:.2f}%)",
            None,
        ),
        (
            "Tracks with Genius Metadata",
            f"{counts['genius']}/{total_tracks} ({pcts['genius']:.2f}%)",
            None,
        ),
        (
            "Parental Advisory",
            f"{counts['advisory']}/{total_tracks} ({pcts['advisory']:.2f}%)",
            None,
        ),
        (
            "Tracks with Provider Conflicts",
            str(len(tracks_with_conflicts)),
            "yellow" if tracks_with_conflicts else None,
        ),
        (
            "Recording ID Evidence This Run",
            f"{tracks_with_recording_evidence}/{total_tracks}",
            None,
        ),
        (
            "Audio Quality Breakdown",
            f"{counts['lossless']} Lossless / {lossy_count} Lossy",
            None,
        ),
    ]
    LOG.summary_table("Tagging Summary", tag_summary_rows)

    if json_report:
        summary_text = (
            f"Processed {total_tracks} tracks ({len(failures)} failures). "
            f"MusicBrainz {counts['mb']}/{total_tracks} ({pcts['mb']:.2f}%), "
            f"Genre {counts['genre']}/{total_tracks} ({pcts['genre']:.2f}%), "
            f"Lyrics {counts['lyrics']}/{total_tracks} ({pcts['lyrics']:.2f}%), "
            f"BPM tags present {counts['bpm']}/{total_tracks} ({pcts['bpm']:.2f}%), "
            f"ReplayGain {counts['rg']}/{total_tracks} ({pcts['rg']:.2f}%)."
        )
        report_data = {
            "schema": "tag_report_v1",
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
                "total_failures": len(failures),
                "enrichment": {
                    "musicbrainz_matched_count": counts["mb"],
                    "musicbrainz_percentage": round(pcts["mb"], 2),
                    "genre_tagged_count": counts["genre"],
                    "genre_percentage": round(pcts["genre"], 2),
                    "lyrics_tagged_count": counts["lyrics"],
                    "lyrics_percentage": round(pcts["lyrics"], 2),
                    "isrc_tagged_count": counts["isrc"],
                    "isrc_percentage": round(pcts["isrc"], 2),
                    "bpm_calculated_count": counts["bpm"],
                    "bpm_percentage": round(pcts["bpm"], 2),
                    "replaygain_calculated_count": counts["rg"],
                    "replaygain_percentage": round(pcts["rg"], 2),
                    "initial_key_count": counts["key"],
                    "initial_key_percentage": round(pcts["key"], 2),
                    "composer_tagged_count": counts["composer"],
                    "composer_percentage": round(pcts["composer"], 2),
                    "producers_tagged_count": counts["producers"],
                    "producers_percentage": round(pcts["producers"], 2),
                    "advisory_tagged_count": counts["advisory"],
                    "advisory_percentage": round(pcts["advisory"], 2),
                    "discogs_matched_count": counts["discogs"],
                    "discogs_percentage": round(pcts["discogs"], 2),
                    "genius_matched_count": counts["genius"],
                    "genius_percentage": round(pcts["genius"], 2),
                    "theaudiodb_matched_count": counts["theaudiodb"],
                    "theaudiodb_percentage": round(pcts["theaudiodb"], 2),
                },
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
