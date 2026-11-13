import dataclasses
import gzip
import importlib
import json
import multiprocessing
import os
import runpy
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tracemalloc
import unittest
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import musicbrainzngs
from rich.progress import Progress, Task, TaskID

from sonora.audio import checksum
from sonora.cli import audio, library, reports
from sonora.cli import cache as cli_cache
from sonora.cli.main import main
from sonora.cli.reports import _emit_tag_summary_and_report, _tag_coverage_counts
from sonora.core import cache, config, coordination, logger
from sonora.core.cancellation import InterruptedOperationError
from sonora.core.config import (
    SonoraConfig,
    get_balanced_feat_pattern,
    get_bracket_feat_pattern,
    get_config,
)
from sonora.core.constants import USER_AGENT
from sonora.core.http import SESSION
from sonora.core.models import (
    CheckReport,
    MetadataSource,
    NormalizeReport,
    RenameReport,
    TrackInfo,
)
from sonora.core.rate_limit import RateLimiter
from sonora.core.retry import retry_network_call
from sonora.core.state import LibraryStateVault
from sonora.core.text import preserve_unicode_repertoire
from sonora.core.titles import score_artist_title_match
from sonora.modules import (
    backup,
    catalog_metadata,
    metadata_merge,
    renamer,
    tagger,
)
from sonora.modules.checker import check_file
from sonora.modules.metadata_merge import build_provider_proposal
from sonora.modules.tagger import process_single_track
from sonora.modules.track_processing import (
    _compute_tag_diffs,
    _describe_tag_diffs,
    _record_persisted_album,
    _render_tag_diffs,
    process_track_lyrics,
)
from sonora.services import (
    acoustid,
    deezer,
    discogs,
    fanart,
    genius,
    itunes,
    musicbrainz,
    shazam,
    theaudiodb,
)
from sonora.services.deezer import (
    fetch_deezer_album_details,
    fetch_deezer_cover_art_url,
    fetch_deezer_track_details,
)
from sonora.services.discogs import (
    fetch_discogs_release_details,
    search_discogs_release,
)
from sonora.services.genius import fetch_genius_song_details
from sonora.services.itunes import (
    fetch_itunes_cover_art_url,
    fetch_itunes_track_metadata,
    search_itunes,
)
from sonora.services.lastfm import fetch_lastfm_tags
from sonora.services.lyrics import fetch_synced_lyrics
from sonora.services.musicbrainz import (
    fetch_cover_art_archive_url,
    fetch_musicbrainz_recording_details,
    fetch_musicbrainz_release_details,
    fetch_track_mbid,
)
from sonora.services.theaudiodb import (
    _download_artwork_bytes,
    fetch_artist_images,
    fetch_theaudiodb_track_details,
)
from tests.support import IsolatedTestCase


class TestCLIInterface(IsolatedTestCase):
    def test_verbose_works_before_and_after_commands_without_leaking(self) -> None:
        from sonora.core.logger import LOG

        def tag_with_diagnostics(*args: object, **kwargs: object) -> list[TrackInfo]:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=1) as workers:
                workers.submit(LOG.detail, "Provider decision").result()
            return []

        for arguments in (
            ["--verbose", "tag", str(self.test_directory)],
            ["tag", str(self.test_directory), "--verbose"],
        ):
            with (
                self.subTest(arguments=arguments),
                patch("sonora.cli.library.tag_album_folder", return_value=[]) as tag,
                patch("sonora.core.logger.CONSOLE.print") as output,
            ):
                tag.side_effect = tag_with_diagnostics
                self.assertEqual(main(arguments), 0)
                self.assertTrue(
                    any(
                        "Provider decision" in str(call.args)
                        for call in output.call_args_list
                    )
                )
                self.assertFalse(LOG.verbose)
        with (
            patch("sonora.cli.library.tag_album_folder", return_value=[]) as tag,
            patch("sonora.core.logger.CONSOLE.print") as output,
        ):
            tag.side_effect = tag_with_diagnostics
            self.assertEqual(main(["tag", str(self.test_directory)]), 0)
            self.assertFalse(
                any(
                    "Provider decision" in str(call.args)
                    for call in output.call_args_list
                )
            )

    def test_verbose_service_list_renders_with_credentials(self) -> None:
        from io import StringIO

        from rich.console import Console

        output = StringIO()
        with (
            patch("sonora.cli.library.tag_album_folder", return_value=[]),
            patch("sonora.core.logger.CONSOLE", Console(file=output)),
        ):
            self.assertEqual(
                main(
                    [
                        "tag",
                        str(self.test_directory),
                        "--dry-run",
                        "--verbose",
                        "--lastfm-key",
                        "test-key",
                    ]
                ),
                0,
            )
        self.assertIn("Active services:", output.getvalue())
        self.assertIn("Last.fm", output.getvalue())

    def test_verbose_resets_after_command_failure(self) -> None:
        from sonora.core.logger import LOG

        with patch(
            "sonora.cli.library.tag_album_folder", side_effect=OSError("disk full")
        ):
            self.assertEqual(main(["tag", str(self.test_directory), "--verbose"]), 1)
        self.assertFalse(LOG.verbose)

    def test_default_check_limits_console_findings_without_truncating_json(
        self,
    ) -> None:
        output_path = self.test_directory / "findings.json"
        issues = {
            str(self.test_directory / f"recording-{index}.flac"): [
                "Missing artist",
                "Missing title",
            ]
            for index in range(8)
        }
        with (
            patch(
                "sonora.cli.library.check_library",
                return_value=CheckReport(total_files=8, issues=issues),
            ),
            patch("sonora.core.logger.LOG.warning") as warning,
        ):
            self.assertEqual(
                main(["check", str(self.test_directory), "--json", str(output_path)]), 0
            )
        findings = [
            call.args[0]
            for call in warning.call_args_list
            if "recording-" in call.args[0]
        ]
        self.assertEqual(len(findings), 5)
        self.assertIn("recording-0.flac", findings[0])
        report = json.loads(output_path.read_text())
        self.assertEqual(len(report["issues"]), 8)

    def test_main_no_args_returns_zero(self) -> None:
        exit_code = main([])
        self.assertEqual(exit_code, 0)

    def test_handle_tag_subcommand(self) -> None:
        with patch("sonora.cli.library.tag_album_folder") as mock_tag_album_folder:
            mock_tag_album_folder.return_value = [
                TrackInfo(file_path=Path("dummy.flac"))
            ]
            exit_code = main(["tag", str(self.test_directory), "-t", "2"])
            self.assertEqual(exit_code, 0)
            mock_tag_album_folder.assert_called_once()

    def test_handle_tag_subcommand_with_fanart_and_shazam_flags(self) -> None:
        with patch("sonora.cli.library.tag_album_folder") as mock_tag_album_folder:
            mock_tag_album_folder.return_value = [
                TrackInfo(file_path=Path("dummy.flac"))
            ]
            exit_code = main(
                [
                    "tag",
                    str(self.test_directory),
                    "--fanart-key",
                    "test_fanart_key",
                    "--no-shazam",
                ]
            )
            self.assertEqual(exit_code, 0)
            mock_tag_album_folder.assert_called_once()
            _, kwargs = mock_tag_album_folder.call_args
            self.assertEqual(kwargs.get("fanart_api_key"), "test_fanart_key")
            self.assertFalse(kwargs.get("enable_shazam"))

    @patch("sonora.cli.library.set_ignore_cache")
    def test_handle_tag_subcommand_with_no_dsp_and_ignore_cache(
        self, mock_set_ignore_cache: MagicMock
    ) -> None:
        with patch("sonora.cli.library.tag_album_folder") as mock_tag_album_folder:
            mock_tag_album_folder.return_value = [
                TrackInfo(file_path=Path("dummy.flac"))
            ]
            exit_code = main(
                ["tag", str(self.test_directory), "--no-dsp", "--ignore-cache"]
            )
            self.assertEqual(exit_code, 0)
            mock_tag_album_folder.assert_called_once()
            _, kwargs = mock_tag_album_folder.call_args
            self.assertFalse(kwargs.get("fetch_bpm"))
            self.assertFalse(kwargs.get("fetch_key"))
            self.assertFalse(kwargs.get("fetch_replaygain"))
            mock_set_ignore_cache.assert_called_once_with(True)

    def test_handle_tag_subcommand_with_json_report(self) -> None:
        with patch("sonora.cli.library.tag_album_folder") as mock_tag_album_folder:
            mock_tag_album_folder.return_value = [
                TrackInfo(
                    file_path=Path("dummy.flac"),
                    title="Song",
                    artist="Artist",
                    bpm=120.0,
                    genre="Pop",
                )
            ]
            json_output_path = self.test_directory / "tag_report.json"
            exit_code = main(
                ["tag", str(self.test_directory), "--json", str(json_output_path)]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(json_output_path.exists())
            content = json_output_path.read_text(encoding="utf-8")
            self.assertIn("summary_text", content)
            self.assertIn('"bpm_calculated_count": 1', content)

    def test_handle_check_subcommand(self) -> None:
        with patch("sonora.cli.library.check_library") as mock_check:
            mock_check.return_value = CheckReport(total_files=1, corrupt_files=0)
            exit_code = main(["check", str(self.test_directory)])
            self.assertEqual(exit_code, 0)
            mock_check.assert_called_once()

    def test_handle_rename_subcommand(self) -> None:
        with patch("sonora.cli.library.rename_directory_files") as mock_rename:
            mock_rename.return_value = [Path("01 - Artist - Title.flac")]
            exit_code = main(["rename", str(self.test_directory)])
            self.assertEqual(exit_code, 0)
            mock_rename.assert_called_once()

    def test_handle_organize_subcommand(self) -> None:
        with patch("sonora.cli.library.organize_library_singles") as mock_organize:
            mock_organize.return_value = 5
            target_directory = self.test_directory / "Singles"
            exit_code = main(
                [
                    "organize",
                    str(self.test_directory),
                    "--target-singles",
                    str(target_directory),
                ]
            )
            self.assertEqual(exit_code, 0)
            mock_organize.assert_called_once()

    def test_handle_backup_subcommand(self) -> None:
        with patch("sonora.cli.library.backup_library_tags") as mock_backup:
            mock_backup.return_value = self.test_directory / "backup.json"
            exit_code = main(["backup", str(self.test_directory)])
            self.assertEqual(exit_code, 0)
            mock_backup.assert_called_once()

    def test_handle_rename_subcommand_with_json(self) -> None:
        with patch("sonora.cli.library.rename_directory_files") as mock_rename:
            mock_rename.return_value = [Path("01 - Artist - Title.flac")]
            json_output = self.test_directory / "rename_report.json"
            exit_code = main(
                ["rename", str(self.test_directory), "--json", str(json_output)]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(json_output.exists())

    def test_rename_dry_run_report_labels_previewed_changes(self) -> None:
        json_output = self.test_directory / "rename_report.json"

        def preview_renames(
            _: Path,
            *,
            dry_run: bool = False,
            max_threads: int = 4,
            report: RenameReport | None = None,
        ) -> list[Path]:
            if report is None:
                raise AssertionError("rename command must pass its report")
            report.total_files = 3
            report.files_renamed = 2
            report.folders_renamed = 1
            return []

        with (
            patch(
                "sonora.cli.library.rename_directory_files", side_effect=preview_renames
            ),
            patch("sonora.cli.library.LOG.summary_table") as summary_table,
        ):
            self.assertEqual(
                main(
                    [
                        "rename",
                        str(self.test_directory),
                        "--dry-run",
                        "--json",
                        str(json_output),
                    ]
                ),
                0,
            )
        report = json.loads(json_output.read_text(encoding="utf-8"))
        self.assertTrue(report["dry_run"])
        labels = [row[0] for row in summary_table.call_args.args[1]]
        self.assertIn("Files Would Rename", labels)
        self.assertIn("Album Folders Would Rename", labels)

    def test_handle_organize_subcommand_with_json(self) -> None:
        with patch("sonora.cli.library.organize_library_singles") as mock_organize:
            mock_organize.return_value = 3
            json_output = self.test_directory / "organize_report.json"
            exit_code = main(
                ["organize", str(self.test_directory), "--json", str(json_output)]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(json_output.exists())

    def test_handle_restore_subcommand_with_json(self) -> None:
        with patch("sonora.cli.library.restore_library_tags") as mock_restore:
            mock_restore.return_value = 5
            dummy_backup = self.test_directory / "backup.json"
            dummy_backup.write_text("{}", encoding="utf-8")
            json_output = self.test_directory / "restore_report.json"
            exit_code = main(["restore", str(dummy_backup), "--json", str(json_output)])
            self.assertEqual(exit_code, 0)
            self.assertTrue(json_output.exists())

    def test_handle_normalize_subcommand(self) -> None:
        with patch("sonora.cli.library.normalize_library") as mock_normalize:
            mock_normalize.return_value = NormalizeReport(
                total_scanned=1,
                modified_files={"song.flac": ["artist: 'Old' -> 'New'"]},
            )
            json_output = self.test_directory / "normalize_report.json"
            exit_code = main(
                [
                    "normalize",
                    str(self.test_directory),
                    "--bpm",
                    "--replaygain",
                    "--json",
                    str(json_output),
                ]
            )
            self.assertEqual(exit_code, 0)
            mock_normalize.assert_called_once()
            self.assertTrue(json_output.exists())

    def test_handle_normalize_subcommand_no_dsp(self) -> None:
        with patch("sonora.cli.library.normalize_library") as mock_normalize:
            mock_normalize.return_value = NormalizeReport()
            exit_code = main(["normalize", str(self.test_directory), "--no-dsp"])
            self.assertEqual(exit_code, 0)
            mock_normalize.assert_called_once_with(
                self.test_directory,
                fetch_bpm=False,
                fetch_key=False,
                fetch_replaygain=False,
                force=False,
                dry_run=False,
                max_threads=4,
            )

    def test_handle_bpm_subcommand(self) -> None:
        song = self.test_directory / "song.flac"
        song.write_bytes(b"dummy")
        json_out = self.test_directory / "bpm.json"
        with (
            patch("sonora.cli.audio.calculate_bpm") as mock_calc_bpm,
            patch("sonora.cli.audio.read_track_metadata") as mock_read,
            patch("sonora.cli.audio.write_track_metadata") as mock_write,
        ):
            mock_read.return_value = TrackInfo(file_path=song, bpm=None)
            mock_calc_bpm.return_value = 128.0
            exit_code = main(["bpm", str(self.test_directory), "--json", str(json_out)])
            self.assertEqual(exit_code, 0)
            mock_write.assert_called_once()
            mock_read.assert_called_once()
            mock_calc_bpm.assert_called_once()
            self.assertTrue(json_out.exists())

    def test_handle_key_subcommand(self) -> None:
        song = self.test_directory / "song.flac"
        song.write_bytes(b"dummy")
        json_out = self.test_directory / "key.json"
        with (
            patch("sonora.cli.audio.detect_key_details") as mock_detect_key,
            patch("sonora.cli.audio.read_track_metadata") as mock_read,
            patch("sonora.cli.audio.write_track_metadata") as mock_write,
        ):
            mock_read.return_value = TrackInfo(file_path=song, initial_key=None)
            mock_detect_key.return_value = ("C#m", "12A", 0.85)
            exit_code = main(["key", str(self.test_directory), "--json", str(json_out)])
            self.assertEqual(exit_code, 0)
            mock_write.assert_called_once()
            mock_read.assert_called_once()
            mock_detect_key.assert_called_once()
            self.assertTrue(json_out.exists())
            mock_write.reset_mock()
            mock_read.reset_mock()
            mock_detect_key.reset_mock()
            mock_read.return_value = TrackInfo(file_path=song, initial_key="m")
            mock_detect_key.return_value = ("Am", "8A", 0.9)
            exit_code_invalid = main(["key", str(self.test_directory)])
            self.assertEqual(exit_code_invalid, 0)
            mock_detect_key.assert_called_once()
            mock_write.assert_called_once()

    def test_handle_replaygain_subcommand(self) -> None:
        song = self.test_directory / "song.flac"
        song.write_bytes(b"dummy")
        json_out = self.test_directory / "rg.json"
        with patch("sonora.cli.audio.calculate_album_replaygain") as mock_rg:
            mock_rg.return_value = True
            exit_code = main(
                ["replaygain", str(self.test_directory), "--json", str(json_out)]
            )
            self.assertEqual(exit_code, 0)
            mock_rg.assert_called_once()
            self.assertTrue(json_out.exists())

    def test_handle_lyrics_subcommand(self) -> None:
        song = self.test_directory / "song.flac"
        song.write_bytes(b"dummy")
        json_out = self.test_directory / "lyrics.json"
        with (
            patch("sonora.cli.audio.process_track_lyrics") as mock_lyrics,
            patch("sonora.cli.audio.read_track_metadata") as mock_read,
            patch("sonora.cli.audio.write_track_metadata") as mock_write,
        ):
            mock_read.return_value = TrackInfo(
                file_path=song, artist="Artist", title="Title", duration=60
            )
            mock_lyrics.return_value = ("[00:01.00] Line", "synced")
            exit_code = main(
                ["lyrics", str(self.test_directory), "--json", str(json_out)]
            )
            self.assertEqual(exit_code, 0)
            mock_read.assert_called_once()
            mock_lyrics.assert_called_once()
            self.assertEqual(mock_lyrics.call_args.kwargs["duration"], 60)
            mock_write.assert_called_once()
            self.assertTrue(json_out.exists())

    def test_handle_keyboard_interrupts(self) -> None:
        dummy_backup = self.test_directory / "backup.json"
        dummy_backup.write_text("{}", encoding="utf-8")
        song = self.test_directory / "song.flac"
        song.write_bytes(b"dummy")
        with (
            patch("sonora.cli.library.tag_album_folder", side_effect=KeyboardInterrupt),
            patch("sonora.cli.library.check_library", side_effect=KeyboardInterrupt),
            patch(
                "sonora.cli.library.rename_directory_files",
                side_effect=KeyboardInterrupt,
            ),
            patch(
                "sonora.cli.library.organize_library_singles",
                side_effect=KeyboardInterrupt,
            ),
            patch(
                "sonora.cli.library.backup_library_tags", side_effect=KeyboardInterrupt
            ),
            patch(
                "sonora.cli.library.restore_library_tags", side_effect=KeyboardInterrupt
            ),
            patch(
                "sonora.cli.library.normalize_library", side_effect=KeyboardInterrupt
            ),
            patch(
                "sonora.cli.audio.calculate_album_replaygain",
                side_effect=KeyboardInterrupt,
            ),
            patch("sonora.cli.audio.as_completed", side_effect=KeyboardInterrupt),
        ):
            for args in [
                ["tag", str(self.test_directory)],
                ["check", str(self.test_directory)],
                ["rename", str(self.test_directory)],
                ["organize", str(self.test_directory)],
                ["backup", str(self.test_directory)],
                ["restore", str(dummy_backup)],
                ["normalize", str(self.test_directory)],
                ["bpm", str(self.test_directory), "--force"],
                ["key", str(self.test_directory), "--force"],
                ["replaygain", str(self.test_directory), "--force"],
                ["lyrics", str(self.test_directory), "--force"],
            ]:
                with self.subTest(args=args):
                    exit_code = main(args)
                    self.assertEqual(exit_code, 130)


class TestTagReportVerification(IsolatedTestCase):
    def test_metadata_coverage_does_not_imply_recording_identity_evidence(self) -> None:
        recording_id = "11111111-2222-3333-4444-555555555555"
        tracks = [
            TrackInfo(
                Path("inherited.flac"),
                musicbrainz_trackid=recording_id,
                metadata_sources={"genre": "iTunes"},
            ),
            TrackInfo(
                Path("verified.flac"),
                musicbrainz_trackid=recording_id,
                metadata_sources={"musicbrainz_trackid": "MusicBrainz"},
                metadata_conflicts={"genre": "Providers disagree"},
            ),
            TrackInfo(Path("unmatched.flac")),
        ]
        output = self.test_directory / "report.json"
        with patch("sonora.cli.reports.LOG.summary_table"):
            _emit_tag_summary_and_report(
                tracks, [], False, self.test_directory, 1, output
            )
        report = json.loads(output.read_text())
        verification = report["verification"]
        self.assertEqual(verification["tracks_with_provider_evidence"], 2)
        self.assertEqual(verification["tracks_with_recording_identifier_evidence"], 1)
        self.assertEqual(verification["tracks_with_metadata_conflicts"], 1)
        self.assertEqual(
            verification["metadata_conflicts"],
            [
                {
                    "file_path": "verified.flac",
                    "conflicts": {"genre": "Providers disagree"},
                }
            ],
        )
        self.assertEqual(
            verification["recording_identifiers_without_provider_evidence"][
                "musicbrainz_trackid"
            ],
            ["inherited.flac"],
        )
        self.assertEqual(
            verification["tracks_without_provider_evidence"], ["unmatched.flac"]
        )

    def test_empty_report_has_no_identity_evidence_or_conflicts(self) -> None:
        output = self.test_directory / "report.json"
        with patch("sonora.cli.reports.LOG.summary_table"):
            _emit_tag_summary_and_report([], [], False, self.test_directory, 1, output)
        verification = json.loads(output.read_text())["verification"]
        self.assertEqual(verification["tracks_with_recording_identifier_evidence"], 0)
        self.assertEqual(verification["tracks_with_metadata_conflicts"], 0)
        self.assertEqual(verification["metadata_conflicts"], [])

    def test_incomplete_coverage_is_not_rounded_to_one_hundred_percent(self) -> None:
        tracks = [TrackInfo(Path("recording.flac"), genre="Pop") for _ in range(6507)]
        tracks.extend(
            [TrackInfo(Path("missing.flac")), TrackInfo(Path("also-missing.flac"))]
        )
        output = self.test_directory / "report.json"
        with patch("sonora.cli.reports.LOG.summary_table") as summary:
            _emit_tag_summary_and_report(
                tracks, [], False, self.test_directory, 1, output
            )
        report = json.loads(output.read_text())
        self.assertEqual(
            report["statistics"]["metadata_coverage"]["genre_percentage"], 99.97
        )
        rows = summary.call_args.args[1]
        self.assertIn(("Tracks with Genre", "6507/6509 (99.97%)", None), rows)


class TestReportProvenance(IsolatedTestCase):
    def test_unrelated_comments_do_not_count_as_genius_metadata(self) -> None:
        sources: tuple[MetadataSource | None, ...] = (None, "MusicBrainz", "TheAudioDB")
        for source in sources:
            with self.subTest(source=source):
                recording = TrackInfo(
                    Path("recording.flac"),
                    comment="An album description or a personal note",
                    metadata_sources={"comment": source} if source else {},
                )
                self.assertEqual(_tag_coverage_counts([recording])["genius"], 0)

    def test_genius_identifier_or_provenance_counts_each_track_once(self) -> None:
        recordings = [
            TrackInfo(Path("stored-id.flac"), genius_song_id="123"),
            TrackInfo(
                Path("description.flac"),
                comment="Verified song description",
                metadata_sources={"comment": "Genius"},
            ),
            TrackInfo(
                Path("credits.flac"),
                composer="Song Writer",
                metadata_sources={"composer": "Genius"},
            ),
            TrackInfo(
                Path("both.flac"),
                genius_song_id="456",
                metadata_sources={"genius_song_id": "Genius"},
            ),
            TrackInfo(Path("empty-id.flac"), genius_song_id=""),
        ]
        self.assertEqual(_tag_coverage_counts(recordings)["genius"], 4)

    def test_report_and_summary_share_corrected_counts_and_existing_schema(
        self,
    ) -> None:
        recordings = [
            TrackInfo(Path("stored-id.flac"), genius_song_id="123"),
            TrackInfo(
                Path("other-provider.flac"),
                comment="Album description",
                metadata_sources={"comment": "TheAudioDB"},
            ),
        ]
        output = self.test_directory / "report.json"
        with patch("sonora.cli.reports.LOG.summary_table") as summary:
            _emit_tag_summary_and_report(
                recordings, [], False, self.test_directory, 1, output, dry_run=True
            )
        report = json.loads(output.read_text())
        self.assertEqual(report["schema"], "tag_report_v2")
        self.assertTrue(report["execution"]["dry_run"])
        self.assertNotIn("enrichment", report["statistics"])
        mb_metadata = report["statistics"]["metadata_coverage"]
        self.assertEqual(mb_metadata["genius_matched_count"], 1)
        self.assertEqual(mb_metadata["genius_percentage"], 50.0)
        self.assertEqual(mb_metadata["theaudiodb_matched_count"], 1)
        self.assertEqual(
            set(mb_metadata),
            {
                "musicbrainz_matched_count",
                "musicbrainz_percentage",
                "genre_tagged_count",
                "genre_percentage",
                "lyrics_tagged_count",
                "lyrics_percentage",
                "isrc_tagged_count",
                "isrc_percentage",
                "bpm_calculated_count",
                "bpm_percentage",
                "replaygain_calculated_count",
                "replaygain_percentage",
                "initial_key_count",
                "initial_key_percentage",
                "composer_tagged_count",
                "composer_percentage",
                "producers_tagged_count",
                "producers_percentage",
                "advisory_tagged_count",
                "advisory_percentage",
                "discogs_matched_count",
                "discogs_percentage",
                "genius_matched_count",
                "genius_percentage",
                "theaudiodb_matched_count",
                "theaudiodb_percentage",
            },
        )
        rows = summary.call_args.args[1]
        self.assertIn(("Tracks with Genius ID", "1/2 (50.00%)", None), rows)
        self.assertEqual(rows[0], ("Total Tracks Previewed", "2", None))
        self.assertEqual(rows[-1][0], "Audio Formats")


if __name__ == "__main__":
    unittest.main()


entrypoint = importlib.import_module("sonora.cli.main")


class TestEntrypointContracts(IsolatedTestCase):
    def test_user_interrupt_returns_shell_cancellation_status(self) -> None:
        with (
            patch.object(entrypoint, "app") as app,
            patch("sonora.core.logger.LOG.warning") as warning,
        ):
            app.meta.side_effect = KeyboardInterrupt
            self.assertEqual(entrypoint.main(["tag"]), 130)
        self.assertIn("Aborted by user", warning.call_args.args[0])

    def test_python_module_import_and_execution_return_cli_status(self) -> None:
        assert entrypoint.__file__ is not None
        module_path = Path(entrypoint.__file__).parents[1] / "__main__.py"
        with patch.object(entrypoint, "main", return_value=7) as main:
            runpy.run_path(str(module_path), run_name="sonora.entrypoint_contract")
            main.assert_not_called()
            with self.assertRaises(SystemExit) as raised:
                runpy.run_path(str(module_path), run_name="__main__")
        self.assertEqual(raised.exception.code, 7)
        main.assert_called_once_with()

    def test_cli_script_supports_platform_without_sigcont(self) -> None:
        with (
            patch.dict(sys.modules, {"signal": ModuleType("signal")}),
            patch.object(sys, "argv", ["sonora", "--version"]),
            patch("socket.setdefaulttimeout"),
            self.assertRaises(SystemExit) as raised,
        ):
            runpy.run_path(str(entrypoint.__file__), run_name="__main__")
        self.assertEqual(raised.exception.code, 0)

    def test_sigcont_callback_and_registration_failure(self) -> None:
        for failure in (None, ValueError("not the main thread")):
            with (
                self.subTest(failure=failure),
                patch.object(signal, "SIGCONT", 25, create=True),
                patch.object(signal, "signal", side_effect=failure) as register,
                patch("socket.setdefaulttimeout"),
            ):
                runpy.run_path(
                    str(entrypoint.__file__), run_name="sonora.cli_script_contract"
                )
            if failure is None:
                with patch("sonora.core.logger.LOG.info") as log:
                    register.call_args.args[1]()
                self.assertIn("Resumed", log.call_args.args[0])

    def test_default_argv_and_noninteger_command_results(self) -> None:
        with (
            patch.object(sys, "argv", ["sonora", "--version"]),
            patch.object(entrypoint, "app") as app,
        ):
            app.meta.return_value = None
            self.assertEqual(entrypoint.main(), 0)
        app.meta.assert_called_once_with(["--version"], exit_on_error=False)

    def test_programming_failures_propagate_and_expected_io_errors_return_one(
        self,
    ) -> None:
        with patch.object(entrypoint, "app") as app:
            app.meta.side_effect = RuntimeError("command bug")
            with self.assertRaisesRegex(RuntimeError, "command bug"):
                entrypoint.main(["tag"])
        with (
            patch.object(entrypoint, "app") as app,
            patch("sonora.core.logger.LOG.error") as error,
        ):
            app.meta.side_effect = PermissionError("unwritable file")
            self.assertEqual(entrypoint.main(["tag"]), 1)
        self.assertIn("unwritable file", error.call_args.args[0])


class TestLibraryCommandContracts(IsolatedTestCase):
    def test_organize_json_identifies_dry_run(self) -> None:
        report_path = self.test_directory / "organize.json"
        with (
            patch.object(library, "organize_library_singles", return_value=2),
            patch("sonora.core.logger.LOG.summary_table"),
        ):
            self.assertEqual(
                entrypoint.main(
                    [
                        "organize",
                        str(self.test_directory),
                        "--dry-run",
                        "--json",
                        str(report_path),
                    ]
                ),
                0,
            )
        report = json.loads(report_path.read_text())
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["summary"]["single_tracks_organized"], 2)

    def test_partial_results_survive_interruptions_in_json_reports(self) -> None:
        recording = TrackInfo(self.test_directory / "recording.flac")
        cases = (
            ("tag", "tag_album_folder", [recording], "statistics", "total_tracks", 1),
            (
                "check",
                "check_library",
                CheckReport(total_files=4),
                "summary",
                "total_files",
                4,
            ),
            (
                "rename",
                "rename_directory_files",
                RenameReport(total_files=4, files_renamed=2),
                "summary",
                "files_renamed",
                2,
            ),
            (
                "organize",
                "organize_library_singles",
                2,
                "summary",
                "single_tracks_organized",
                2,
            ),
            ("restore", "restore_library_tags", 2, "summary", "tracks_restored", 2),
        )
        for command, boundary, partial_result, section, counter, expected in cases:
            report_path = self.test_directory / f"{command}.json"
            with (
                self.subTest(command=command),
                patch.object(
                    library,
                    boundary,
                    side_effect=InterruptedOperationError(partial_result),
                ),
                patch("sonora.core.logger.LOG.summary_table"),
            ):
                self.assertEqual(
                    entrypoint.main(
                        [command, str(self.test_directory), "--json", str(report_path)]
                    ),
                    130,
                )
            report = json.loads(report_path.read_text())
            self.assertTrue(report["aborted_by_user"])
            self.assertEqual(report[section][counter], expected)

    def test_interrupted_dry_runs_report_proposals_not_completed_changes(self) -> None:
        for command, boundary, partial_result, expected in (
            (
                "rename",
                "rename_directory_files",
                RenameReport(total_files=4, files_renamed=2),
                "Rename preview interrupted: 2/4 files would be renamed.",
            ),
            (
                "organize",
                "organize_library_singles",
                2,
                "Organization preview interrupted: 2 single tracks would be moved.",
            ),
        ):
            with (
                self.subTest(command=command),
                patch.object(
                    library,
                    boundary,
                    side_effect=InterruptedOperationError(partial_result),
                ),
                patch("sonora.core.logger.LOG.warning") as warning,
                patch("sonora.core.logger.LOG.summary_table"),
            ):
                self.assertEqual(
                    entrypoint.main([command, str(self.test_directory), "--dry-run"]),
                    130,
                )
            warning_text = "\n".join(
                str(call.args[0]) for call in warning.call_args_list
            )
            self.assertIn(expected, warning_text)
            self.assertNotIn("Partially renamed", warning_text)
            self.assertNotIn("Partially organized", warning_text)

    def test_unrelated_runtime_errors_are_not_reported_as_user_cancellation(
        self,
    ) -> None:
        for command, boundary in (
            ("tag", "tag_album_folder"),
            ("check", "check_library"),
            ("rename", "rename_directory_files"),
            ("organize", "organize_library_singles"),
            ("backup", "backup_library_tags"),
            ("restore", "restore_library_tags"),
            ("normalize", "normalize_library"),
        ):
            with (
                self.subTest(command=command),
                patch.object(
                    library, boundary, side_effect=RuntimeError("workflow bug")
                ),
                self.assertRaisesRegex(RuntimeError, "workflow bug"),
            ):
                entrypoint.main([command, str(self.test_directory)])

    def test_normalize_results_and_partial_reports_preserve_counts(self) -> None:
        for outcome in (
            NormalizeReport(total_scanned=1),
            NormalizeReport(total_scanned=1, failed_files=1),
            InterruptedOperationError(NormalizeReport(total_scanned=1)),
            InterruptedOperationError(None),
        ):
            with (
                self.subTest(outcome=outcome),
                patch.object(
                    library,
                    "normalize_library",
                    return_value=outcome,
                    side_effect=outcome if isinstance(outcome, BaseException) else None,
                ),
                patch("sonora.core.logger.LOG.summary_table"),
            ):
                report_path = self.test_directory / "normalize.json"
                self.assertEqual(
                    entrypoint.main(
                        [
                            "normalize",
                            str(self.test_directory),
                            "--json",
                            str(report_path),
                            "--no-dsp",
                        ]
                    ),
                    130
                    if isinstance(outcome, BaseException)
                    else int(bool(outcome.failed_files)),
                )
            report = json.loads(report_path.read_text())
            self.assertEqual(
                report["summary"]["total_scanned"],
                0
                if isinstance(outcome, InterruptedOperationError)
                and outcome.partial_result is None
                else 1,
            )

    def test_explicit_credentials_are_forwarded_to_tagging(self) -> None:
        flags = [
            "--discogs-token",
            "discogs",
            "--acoustid-key",
            "acoustid",
            "--genius-token",
            "genius",
            "--lastfm-key",
            "lastfm",
            "--musixmatch-token",
            "musixmatch",
            "--fanart-key",
            "fanart",
        ]
        with patch.object(library, "tag_album_folder", return_value=[]) as tag:
            self.assertEqual(
                entrypoint.main(["tag", str(self.test_directory), *flags, "--no-dsp"]),
                0,
            )
        self.assertEqual(tag.call_args.kwargs["acoustid_api_key"], "acoustid")
        self.assertEqual(tag.call_args.kwargs["lastfm_api_key"], "lastfm")
        self.assertEqual(tag.call_args.kwargs["genius_api_token"], "genius")
        self.assertEqual(tag.call_args.kwargs["musixmatch_token"], "musixmatch")


class TestAudioCommandFailureContracts(IsolatedTestCase):
    def test_lyrics_cleanup_continues_when_provider_fails(self) -> None:
        path = self.test_directory / "recording.flac"
        dirty = "Hello world\n42Embed"
        path.with_suffix(".lrc").write_text(dirty)
        track = TrackInfo(
            path, artist="Artist", title="Song", lyrics=dirty, duration=100
        )
        details: dict[Path, dict[str, object]] = {}
        with (
            patch.object(audio, "read_track_metadata", return_value=track),
            patch(
                "sonora.services.lyrics.fetch_synced_lyrics",
                side_effect=OSError("offline"),
            ),
            patch.object(audio, "write_track_metadata") as write,
        ):
            self.assertEqual(
                audio._process_lyrics_file(path, False, False, file_details=details),
                (path, "Hello world", "cleaned"),
            )
        self.assertEqual(path.with_suffix(".lrc").read_text(), "Hello world")
        self.assertEqual(write.call_args.args[0].lyrics, "Hello world")
        self.assertEqual(write.call_args.kwargs, {"lyrics_only": True})
        self.assertEqual(details[path]["lookup_status"], "failed")
        self.assertEqual(details[path]["action"], "cleanup")

    def test_deferred_lookup_is_reported_as_incomplete_not_unavailable(self) -> None:
        path = self.test_directory / "recording.flac"
        path.touch()
        report = self.test_directory / "lyrics.json"

        def deferred(
            *args: object, details: dict[str, object], **kwargs: object
        ) -> None:
            details.update(
                lookup_status="deferred", reason="Provider unavailable for this run."
            )

        with (
            patch.object(
                audio,
                "read_track_metadata",
                return_value=TrackInfo(
                    path, artist="Artist", title="Song", duration=100
                ),
            ),
            patch("sonora.services.lyrics.fetch_synced_lyrics", side_effect=deferred),
            patch.object(audio, "write_track_metadata") as write,
        ):
            self.assertEqual(
                audio.lyrics(self.test_directory, dry_run=True, json_report=report), 1
            )
        payload = json.loads(report.read_text())
        self.assertTrue(payload["incomplete"])
        self.assertEqual(payload["lookup_deferred"], 1)
        self.assertEqual(payload["unavailable"], 0)
        self.assertEqual(payload["details"][0]["file"], str(path.resolve()))
        self.assertEqual(payload["details"][0]["status"], "deferred")
        write.assert_not_called()
        self.assertFalse(path.with_suffix(".lrc").exists())

    def test_replaygain_skipped_albums_and_programming_errors_are_distinct(
        self,
    ) -> None:
        path = self.test_directory / "recording.wav"
        path.touch()
        with patch.object(
            audio, "calculate_album_replaygain", return_value=False
        ) as calculate:
            self.assertEqual(audio.replaygain(self.test_directory, dry_run=True), 0)
        calculate.assert_called_once()
        with (
            patch.object(
                audio,
                "calculate_album_replaygain",
                side_effect=RuntimeError("gain bug"),
            ),
            self.assertRaisesRegex(RuntimeError, "gain bug"),
        ):
            audio.replaygain(self.test_directory, dry_run=True)

    def test_lyrics_preview_and_missing_results_never_write_tags(self) -> None:
        path = self.test_directory / "recording.wav"
        for lyric_text in (None, "Plain lyrics"):
            with (
                self.subTest(lyric_text=lyric_text),
                patch.object(
                    audio, "read_track_metadata", return_value=TrackInfo(path)
                ),
                patch.object(
                    audio, "process_track_lyrics", return_value=(lyric_text, "plain")
                ),
                patch.object(audio, "write_track_metadata") as write,
            ):
                self.assertEqual(
                    audio._process_lyrics_file(path, True, True),
                    (path, lyric_text, "plain"),
                )
            write.assert_not_called()

    def test_tag_preview_retains_default_dsp_options(self) -> None:
        with patch.object(library, "tag_album_folder", return_value=[]) as tag:
            self.assertEqual(
                entrypoint.main(["tag", str(self.test_directory), "--dry-run"]), 0
            )
        self.assertTrue(tag.call_args.kwargs["fetch_bpm"])
        self.assertTrue(tag.call_args.kwargs["fetch_key"])
        self.assertTrue(tag.call_args.kwargs["fetch_replaygain"])
        self.assertIsNone(tag.call_args.kwargs["musixmatch_token"])

    def test_missing_and_empty_paths_never_start_workers(self) -> None:
        for command in (audio.bpm, audio.key, audio.lyrics, audio.replaygain):
            with (
                self.subTest(command=command.__name__),
                self.assertRaises(FileNotFoundError),
            ):
                command(self.test_directory / "absent", dry_run=True)
            with self.subTest(command=command.__name__, empty=True):
                self.assertEqual(command(self.test_directory, dry_run=True), 0)

    def test_audio_workers_report_expected_read_and_write_failures(self) -> None:
        path = self.test_directory / "recording.flac"
        for worker in (
            audio._process_bpm_file,
            audio._process_key_file,
            audio._process_lyrics_file,
        ):
            with (
                self.subTest(worker=worker.__name__),
                patch.object(
                    audio, "read_track_metadata", side_effect=OSError("locked file")
                ),
            ):
                result = worker(path, force=True, dry_run=False)
            self.assertEqual(result[:2], (path, None))
        with (
            patch.object(audio, "read_track_metadata", return_value=TrackInfo(path)),
            patch.object(
                audio, "process_track_lyrics", return_value=("Plain lyrics", "plain")
            ),
            patch.object(
                audio, "write_track_metadata", side_effect=OSError("locked file")
            ),
            patch("sonora.core.logger.LOG.warning") as warning,
        ):
            self.assertEqual(
                audio._process_lyrics_file(path, True, False), (path, None, "failed")
            )
        warning.assert_called_once()

    def test_no_key_detection_is_counted_as_skipped(self) -> None:
        path = self.test_directory / "recording.flac"
        with (
            patch.object(audio, "read_track_metadata", return_value=TrackInfo(path)),
            patch.object(audio, "detect_key_details", return_value=None),
        ):
            self.assertEqual(
                audio._process_key_file(path, True, True), (path, None, False)
            )

    def test_parallel_programming_failure_is_not_masked(self) -> None:
        path = self.test_directory / "recording.flac"
        path.write_bytes(b"audio fixture")
        worker = Mock(side_effect=RuntimeError("analysis bug"))
        with self.assertRaisesRegex(RuntimeError, "analysis bug"):
            audio._run_parallel_audio_task(
                self.test_directory, "Test analysis", worker, 1
            )


class TestCacheSizeFormatting(IsolatedTestCase):
    def test_file_size_conversion_includes_largest_supported_unit(self) -> None:
        for size, expected in (
            (0, "0 B"),
            (512, "512 B"),
            (1023, "1023 B"),
            (1024, "1.00 KB"),
            (1024**2, "1.00 MB"),
            (1024**3, "1.00 GB"),
            (1024**4, "1.00 TB"),
            (1024**5, "1024.00 TB"),
        ):
            with self.subTest(size=size):
                self.assertEqual(cli_cache._format_file_size(size), expected)


class TerminalScript:
    def __init__(
        self,
        keys: tuple[str | None, ...],
        stop_event: threading.Event,
        *,
        stop_on_ready: bool = False,
        drain_key: str | None = None,
    ) -> None:
        self.keys = deque(keys)
        self.stop_event = stop_event
        self.stop_on_ready = stop_on_ready
        self.drain_key = drain_key
        self.current_key = ""

    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return 17

    def read(self, size: int) -> str:
        return self.current_key

    def select(
        self,
        readers: list[object],
        writers: list[object],
        errors: list[object],
        timeout: float,
    ) -> tuple[list[object], list[object], list[object]]:
        if timeout == 0.05:
            if self.drain_key is not None:
                self.current_key, self.drain_key = (self.drain_key, None)
                return ([self], [], [])
            return ([], [], [])
        if not self.keys:
            self.stop_event.set()
            return ([], [], [])
        key = self.keys.popleft()
        if key is None:
            return ([], [], [])
        self.current_key = key
        if self.stop_on_ready:
            self.stop_event.set()
        return ([self], [], [])


class TestTerminalContracts(IsolatedTestCase):
    def setUp(self) -> None:
        self.enterContext(
            patch.dict(
                "sys.modules",
                termios=SimpleNamespace(
                    tcgetattr=Mock(), tcsetattr=Mock(), TCSADRAIN=1, error=OSError
                ),
                tty=SimpleNamespace(setcbreak=Mock()),
            )
        )

    def test_debug_messages_redact_credentials_and_escape_markup(self) -> None:
        instance = logger.SonoraLogger()
        instance.verbose = True
        with patch.object(logger.CONSOLE, "print") as output:
            instance.debug(
                "[red] token=secret&api_key=private client_key=hidden user_token=another"
            )
        message = output.call_args.args[0]
        self.assertNotIn("secret", message)
        self.assertNotIn("private", message)
        self.assertNotIn("hidden", message)
        self.assertNotIn("another", message)
        self.assertIn("<redacted>", message)
        self.assertIn("\\[red]", message)

    def test_error_and_buffer_lifecycle_preserve_message_order(self) -> None:
        instance = logger.SonoraLogger()
        with patch.object(logger.CONSOLE, "print") as output:
            instance.stop_buffering()
            output.assert_not_called()
            instance.start_buffering()
            instance.error("broken track")
            instance.info("next recording")
            output.assert_not_called()
            instance.stop_buffering()
        messages = [call.args[0] for call in output.call_args_list]
        self.assertIn("ERROR:", messages[0])
        self.assertIn("broken track", messages[0])
        self.assertIn("next recording", messages[1])

    def test_default_output_keeps_warnings_and_hides_diagnostics(self) -> None:
        instance = logger.SonoraLogger()
        with patch.object(logger.CONSOLE, "print") as output:
            instance.detail("Recording lookup and field changes")
            instance.debug("Diagnostic request")
            output.assert_not_called()
            instance.info("Starting tagging")
            instance.warning("Unable to save recording.flac")
            instance.error("Disk full")
        self.assertEqual(output.call_count, 3)
        self.assertIn("Unable to save", output.call_args_list[1].args[0])
        self.assertIn("Disk full", output.call_args_list[2].args[0])

    def test_verbose_detail_preserves_worker_buffer_order(self) -> None:
        instance = logger.SonoraLogger()
        instance.verbose = True
        with patch.object(logger.CONSOLE, "print") as output:
            instance.start_buffering()
            instance.detail("First decision")
            instance.detail("Second decision")
            output.assert_not_called()
            instance.stop_buffering()
        self.assertIn("First decision", output.call_args_list[0].args[0])
        self.assertIn("Second decision", output.call_args_list[1].args[0])

    def test_worker_pause_waits_until_resume(self) -> None:
        with patch.object(
            logger._PAUSE_EVENT, "wait", side_effect=[False, False, True]
        ) as wait:
            logger.wait_if_paused(0.01)
        self.assertEqual(wait.call_count, 3)
        self.assertEqual(wait.call_args.kwargs, {"timeout": 0.01})

    def test_default_pause_resume_corrects_task_clock(self) -> None:
        stop = threading.Event()
        terminal = TerminalScript((None, "x", "p", "P"), stop, drain_key="held key")
        task = Task(TaskID(3), "Tagging", 2, 0, time.monotonic)
        task.start_time = 20
        task.stop_time = 21
        progress = Mock(spec=Progress(auto_refresh=False))
        progress.tasks = [task]
        progress.get_time.side_effect = [100, 112]
        with (
            patch("sys.stdin", terminal),
            patch("threading.Event", return_value=stop),
            patch("threading.Thread") as thread,
            patch("select.select", side_effect=terminal.select),
            patch("termios.tcgetattr", return_value=[1, 2, 3]) as get_attributes,
            patch("termios.tcsetattr") as restore,
            patch("tty.setcbreak") as cbreak,
            patch.object(logger.LOG, "warning"),
            patch.object(logger.LOG, "info"),
        ):
            with logger.interactive_pause_listener(progress, TaskID(3)):
                thread.call_args.kwargs["target"]()
            get_attributes.assert_called_once_with(17)
            cbreak.assert_called_once_with(17)
            restore.assert_called_once()
            thread.return_value.join.assert_called_once_with(timeout=0.5)
        self.assertEqual(task.start_time, 32)
        self.assertIsNone(task.stop_time)
        progress.stop_task.assert_called_once_with(TaskID(3))
        self.assertEqual(progress.update.call_args.kwargs["description"], "Tagging")
        self.assertTrue(logger._PAUSE_EVENT.is_set())

    def test_missing_task_has_fallback_description_and_no_clock_update(self) -> None:
        task_variants: tuple[list[Task], ...] = (
            [],
            [Task(TaskID(3), "Tagging", 2, 0, time.monotonic)],
        )
        for tasks in task_variants:
            stop = threading.Event()
            terminal = TerminalScript((" ", " "), stop)
            progress = Mock(spec=Progress(auto_refresh=False))
            progress.tasks = tasks
            progress.get_time.side_effect = [5, 9]
            with (
                self.subTest(tasks=tasks),
                patch("sys.stdin", terminal),
                patch("threading.Event", return_value=stop),
                patch("threading.Thread") as thread,
                patch("select.select", side_effect=terminal.select),
                patch("termios.tcgetattr", return_value=[]),
                patch("termios.tcsetattr"),
                patch("tty.setcbreak"),
                patch.object(logger.LOG, "warning"),
                patch.object(logger.LOG, "info"),
                logger.interactive_pause_listener(progress, TaskID(3)),
            ):
                thread.call_args.kwargs["target"]()
            expected_description = "Tagging" if tasks else "[cyan]Processing..."
            self.assertEqual(
                progress.update.call_args.kwargs["description"], expected_description
            )
            if tasks:
                self.assertIsNone(tasks[0].start_time)

    def test_callbacks_handle_expected_errors_and_resume_without_default_pause(
        self,
    ) -> None:
        for pause_error, resume_error in (
            (None, None),
            (OSError("pause"), ValueError("resume")),
            (RuntimeError("pause"), RuntimeError("resume")),
        ):
            stop = threading.Event()
            terminal = TerminalScript(("p", "p"), stop)
            pause = Mock(side_effect=pause_error)
            resume = Mock(side_effect=resume_error)
            with (
                self.subTest(pause_error=pause_error),
                patch("sys.stdin", terminal),
                patch("threading.Event", return_value=stop),
                patch("threading.Thread") as thread,
                patch("select.select", side_effect=terminal.select),
                patch("termios.tcgetattr", return_value=[]),
                patch("termios.tcsetattr"),
                patch("tty.setcbreak"),
                patch.object(logger.LOG, "warning"),
                patch.object(logger.LOG, "info"),
                patch.object(logger.LOG, "debug") as debug,
            ):
                with logger.interactive_pause_listener(
                    on_pause=pause, on_resume=resume
                ):
                    thread.call_args.kwargs["target"]()
                pause.assert_called_once()
                resume.assert_called_once()
                self.assertEqual(debug.call_count, 0 if pause_error is None else 2)
            self.assertTrue(logger._PAUSE_EVENT.is_set())

    def test_custom_pause_with_default_resume_does_not_shift_task_time(self) -> None:
        stop = threading.Event()
        terminal = TerminalScript(("p", "p"), stop)
        progress = Mock(spec=Progress(auto_refresh=False))
        progress.tasks = []
        pause = Mock()
        with (
            patch("sys.stdin", terminal),
            patch("threading.Event", return_value=stop),
            patch("threading.Thread") as thread,
            patch("select.select", side_effect=terminal.select),
            patch("termios.tcgetattr", return_value=[]),
            patch("termios.tcsetattr"),
            patch("tty.setcbreak"),
            patch.object(logger.LOG, "warning"),
            patch.object(logger.LOG, "info"),
            logger.interactive_pause_listener(progress, TaskID(3), on_pause=pause),
        ):
            thread.call_args.kwargs["target"]()
        pause.assert_called_once()
        progress.get_time.assert_not_called()
        self.assertEqual(
            progress.update.call_args.kwargs["description"], "[cyan]Processing..."
        )

    def test_listener_ignores_key_ready_after_shutdown(self) -> None:
        stop = threading.Event()
        terminal = TerminalScript(("p",), stop, stop_on_ready=True)
        pause = Mock()
        with (
            patch("sys.stdin", terminal),
            patch("threading.Event", return_value=stop),
            patch("threading.Thread") as thread,
            patch("select.select", side_effect=terminal.select),
            patch("termios.tcgetattr", return_value=[]),
            patch("termios.tcsetattr", side_effect=OSError("terminal closed")),
            patch("tty.setcbreak"),
            logger.interactive_pause_listener(on_pause=pause),
        ):
            thread.call_args.kwargs["target"]()
        pause.assert_not_called()
        self.assertTrue(logger._PAUSE_EVENT.is_set())

    def test_terminal_setup_failure_preserves_worker_resume(self) -> None:
        stop = threading.Event()
        terminal = TerminalScript((), stop)
        with (
            patch("sys.stdin", terminal),
            patch("threading.Event", return_value=stop),
            patch("threading.Thread") as thread,
            patch("termios.tcgetattr", side_effect=OSError("no terminal")),
            logger.interactive_pause_listener(),
        ):
            thread.call_args.kwargs["target"]()
        self.assertTrue(stop.is_set())
        self.assertTrue(logger._PAUSE_EVENT.is_set())

    def test_default_callbacks_without_progress_only_toggle_pause(self) -> None:
        stop = threading.Event()
        terminal = TerminalScript(("p", "p"), stop)
        with (
            patch("sys.stdin", terminal),
            patch("threading.Event", return_value=stop),
            patch("threading.Thread") as thread,
            patch("select.select", side_effect=terminal.select),
            patch("termios.tcgetattr", return_value=[]),
            patch("termios.tcsetattr"),
            patch("tty.setcbreak"),
            patch.object(logger.LOG, "warning") as paused,
            patch.object(logger.LOG, "info") as resumed,
            logger.interactive_pause_listener(),
        ):
            thread.call_args.kwargs["target"]()
        paused.assert_called_once()
        resumed.assert_called_once()
        self.assertTrue(logger._PAUSE_EVENT.is_set())


class TestReadinessRepairs(IsolatedTestCase):
    def test_check_fails_for_audio_corruption_but_not_optional_metadata(self) -> None:
        for corrupt, expected in ((0, 0), (1, 1)):
            with (
                self.subTest(corrupt=corrupt),
                patch.object(
                    library,
                    "check_library",
                    return_value=CheckReport(
                        total_files=1, corrupt_files=corrupt, missing_metadata=1
                    ),
                ),
            ):
                self.assertEqual(library.check(Path("library")), expected)

    def test_bpm_coverage_counts_only_finite_positive_tempos(self) -> None:
        tracks = [
            TrackInfo(Path(f"{index}.flac"), bpm=value)
            for index, value in enumerate(
                (None, 0.0, -1.0, float("nan"), float("inf"), 120.0)
            )
        ]
        self.assertEqual(reports._tag_coverage_counts(tracks)["bpm"], 1)


MBID = "11111111-2222-3333-4444-555555555555"


class TestConfigurationAndCommandBoundaries(IsolatedTestCase):
    def test_equal_accent_counts_preserve_repertoire_and_artist_articles_match(
        self,
    ) -> None:
        self.assertEqual(preserve_unicode_repertoire("Cáfe", "Café"), "Cáfe")
        self.assertEqual(preserve_unicode_repertoire("CAFE", "Cafe"), "Cafe")
        self.assertEqual(
            score_artist_title_match(
                "The Ensemble", "Recording", "Ensemble", "Recording"
            ),
            100,
        )

    def test_boolean_file_configuration_defaults_and_rejects_wrong_types(self) -> None:
        for settings, expected in (
            ({}, True),
            ({"enable_shazam": False}, False),
            ({"enable_shazam": "false"}, None),
        ):
            config.clear_config_cache()
            with (
                self.subTest(settings=settings),
                patch.dict(os.environ, {}, clear=True),
                patch.object(config, "load_app_environment"),
                patch.object(
                    config, "_discover_config_file", return_value=Path("config.json")
                ),
                patch.object(config, "_parse_config_data", return_value=settings),
            ):
                if expected is None:
                    with self.assertRaises(TypeError):
                        config.get_config()
                else:
                    self.assertEqual(config.get_config().enable_shazam, expected)

    def test_cache_directory_change_replaces_the_open_client(self) -> None:
        original = cache.get_cache()
        assert original is not None
        with patch.dict(
            os.environ, {"XDG_CACHE_HOME": str(self.test_directory / "other")}
        ):
            replacement = cache.get_cache()
            assert replacement is not None
            self.assertIsNot(original, replacement)
            self.assertNotEqual(original.directory, replacement.directory)

    def test_invalid_command_has_usage_exit_status(self) -> None:
        self.assertEqual(main(["nonexistent-command"]), 2)

    def test_existing_audio_analysis_tags_skip_decoding(self) -> None:
        path = self.test_directory / "recording.flac"
        recording = TrackInfo(path, bpm=123, initial_key="Am")
        with (
            patch.object(audio, "read_track_metadata", return_value=recording),
            patch.object(audio, "calculate_bpm") as tempo,
            patch.object(audio, "detect_key_details") as key,
        ):
            self.assertEqual(
                audio._process_bpm_file(path, False, False), (path, 123, False)
            )
            self.assertEqual(
                audio._process_key_file(path, False, False), (path, "Am", False)
            )
        tempo.assert_not_called()
        key.assert_not_called()
        with self.assertRaises(FileNotFoundError):
            checksum.verify_flac_checksum(path)


class TestProviderBoundaryFailures(IsolatedTestCase):
    def test_deezer_album_transport_status_and_cover_fallback_failure(self) -> None:
        with (
            patch.object(deezer, "get_cached_api", return_value=None),
            patch.object(SESSION, "get", return_value=httpx.Response(503)),
        ):
            self.assertIsNone(deezer.fetch_deezer_album_details("Primary", "Record"))
        with (
            patch.object(deezer, "get_cached_api", return_value=None),
            patch.object(
                SESSION,
                "get",
                side_effect=[
                    httpx.Response(200, json={"data": []}),
                    httpx.Response(503),
                ],
            ),
        ):
            self.assertIsNone(deezer.fetch_deezer_cover_art_url("Primary", "Record"))

    def test_shazam_ignores_unrelated_metadata_and_actions_but_keeps_lyrics(
        self,
    ) -> None:
        path = self.test_directory / "recording.flac"
        path.write_bytes(b"audio fixture")
        payload = {
            "track": {
                "title": "Recording",
                "subtitle": "Primary",
                "sections": [
                    {"type": "SONG", "metadata": [{"title": "Length", "text": "100"}]},
                    {"type": "VIDEO"},
                    {"type": "LYRICS", "text": ["", "Line", " "]},
                ],
                "hub": {"actions": [{"type": "uri", "uri": "https://example.com"}]},
            }
        }
        with (
            patch.object(
                shazam, "get_config", return_value=SonoraConfig(enable_shazam=True)
            ),
            patch.object(shazam, "get_cached_api", return_value=None),
            patch.object(shazam._SHAZAM_LIMITER, "wait"),
            patch.object(
                shazam, "_recognize_async", new=AsyncMock(return_value=payload)
            ),
        ):
            result = shazam.recognize_audio_track(path)
        assert result is not None
        self.assertEqual(result.lyrics, "Line")
        self.assertIsNone(result.apple_music_id)

    def test_theaudiodb_non_success_artist_response_has_no_artwork(self) -> None:
        with (
            patch.object(theaudiodb, "get_cached_api", return_value=None),
            patch.object(SESSION, "get", return_value=httpx.Response(503)),
        ):
            self.assertEqual(theaudiodb.fetch_artist_images("Primary"), (None, None))

    def test_theaudiodb_track_transport_failure_preserves_a_missing_result(
        self,
    ) -> None:
        with (
            patch.object(theaudiodb, "get_cached_api", return_value=None),
            patch.object(SESSION, "get", side_effect=OSError("offline")),
        ):
            self.assertIsNone(
                theaudiodb.fetch_theaudiodb_track_details("Primary", "Recording")
            )

    def test_itunes_cover_fuzzy_editions_and_series_are_distinct(self) -> None:
        for album, candidates, expected in (
            ("Record", [{"artistName": "Wrong", "collectionName": "Record"}], None),
            (
                "Record",
                [
                    {
                        "artistName": "Primary",
                        "collectionName": "Record (Deluxe)",
                        "artworkUrl100": "https://example.com/100x100bb.jpg",
                    }
                ],
                "https://example.com/1200x1200bb.jpg",
            ),
            (
                "Record",
                [
                    {
                        "artistName": "Primary",
                        "collectionName": "Record II",
                        "artworkUrl100": "https://example.com/100x100bb.jpg",
                    }
                ],
                None,
            ),
            (
                "Record",
                [
                    {
                        "artistName": "Primary",
                        "collectionName": "Record",
                        "artworkUrl100": "https://example.com/200x200bb.jpg",
                    }
                ],
                "https://example.com/1200x1200bb.jpg",
            ),
            ("Record", [{"artistName": "Primary", "collectionName": "Record"}], None),
        ):
            with (
                self.subTest(album=album, candidates=candidates),
                patch.object(itunes, "search_itunes", return_value=candidates),
            ):
                self.assertEqual(
                    itunes.fetch_itunes_cover_art_url(
                        "Primary", album, resolution=1200
                    ),
                    expected,
                )

    def test_shazam_empty_results_and_invalid_lyric_arrays(self) -> None:
        path = self.test_directory / "recording.flac"
        path.write_bytes(b"audio fixture")
        for payload in (
            None,
            {
                "track": {
                    "title": "Recording",
                    "subtitle": "Primary",
                    "sections": [{"type": "LYRICS", "text": ["valid", 12]}],
                }
            },
        ):
            with (
                self.subTest(payload=payload),
                patch.object(
                    shazam, "get_config", return_value=SonoraConfig(enable_shazam=True)
                ),
                patch.object(shazam, "get_cached_api", return_value=None),
                patch.object(shazam._SHAZAM_LIMITER, "wait"),
                patch.object(
                    shazam, "_recognize_async", new=AsyncMock(return_value=payload)
                ),
            ):
                if payload is None:
                    self.assertIsNone(shazam.recognize_audio_track(path))
                else:
                    with self.assertRaisesRegex(TypeError, "array of strings"):
                        shazam.recognize_audio_track(path)
        with (
            patch.object(
                shazam, "get_config", return_value=SonoraConfig(enable_shazam=True)
            ),
            patch.object(
                shazam,
                "get_cached_api",
                return_value={"title": "Recording", "artist": "Primary"},
            ),
        ):
            result = shazam.recognize_audio_track(path)
        assert result is not None
        self.assertEqual((result.title, result.artist), ("Recording", "Primary"))

    def test_theaudiodb_empty_tracks_and_normalized_track_fields(self) -> None:
        payloads: tuple[dict[str, object], ...] = (
            {"track": None},
            {"track": []},
            {
                "track": [
                    {
                        "strArtist": "Primary",
                        "strTrack": "Recording",
                        "strMusicVid": "https://example.com",
                        "strMood": " Chill ",
                        "strOpenKey": "Am",
                        "intScore": "8.5",
                    }
                ]
            },
        )
        for payload in payloads:
            with (
                self.subTest(payload=payload),
                patch.object(theaudiodb, "get_cached_api", return_value=None),
                patch.object(
                    SESSION, "get", return_value=httpx.Response(200, json=payload)
                ),
            ):
                details = theaudiodb.fetch_theaudiodb_track_details(
                    "Primary", "Recording"
                )
            if payload["track"]:
                assert details is not None
                self.assertEqual(
                    (details["mood"], details["initial_key"], details["rating"]),
                    ("Chill", "Am", 8.5),
                )
            else:
                self.assertIsNone(details)
        self.assertIsNone(theaudiodb._clean_theaudiodb_str(" null "))

    def test_fanart_download_error_and_discogs_non_success_status(self) -> None:
        with patch.object(SESSION, "get", side_effect=OSError("offline")):
            self.assertIsNone(fanart.download_fanart_image_bytes("https://example.com"))
        with (
            patch.object(discogs, "get_cached_api", return_value=None),
            patch.object(SESSION, "get", return_value=httpx.Response(503)),
        ):
            self.assertIsNone(discogs.fetch_discogs_release_details(1))
            self.assertIsNone(
                discogs.search_discogs_release("Primary", "Record", "fixture")
            )
        self.assertIsNone(genius._normalize_genius_description(None))

    def test_musicbrainz_empty_results_cached_misses_and_discography_failure(
        self,
    ) -> None:
        with patch.object(musicbrainz, "get_cached_api", return_value=""):
            self.assertIsNone(musicbrainz.fetch_track_mbid("Primary", "Recording"))
        with (
            patch.object(musicbrainz, "get_cached_api", return_value=None),
            patch.object(
                musicbrainz, "_call_musicbrainz", return_value={"recording-list": []}
            ),
        ):
            self.assertIsNone(musicbrainz.fetch_track_mbid("Primary", "Recording"))
        with (
            patch.object(musicbrainz, "get_cached_api", return_value=None),
            patch.object(
                musicbrainz, "_call_musicbrainz", side_effect=OSError("offline")
            ),
        ):
            self.assertEqual(musicbrainz.fetch_artist_discography("Primary"), [])

    def test_itunes_track_metadata_and_unmatched_candidates(self) -> None:
        recording = {
            "trackName": "Recording",
            "artistName": "Primary",
            "trackId": 1,
            "artistId": 2,
            "collectionId": 3,
            "country": "US",
            "releaseDate": "2020-01-02T00:00:00Z",
            "trackExplicitness": "explicit",
            "primaryGenreName": "Rock",
        }
        with patch.object(
            itunes,
            "search_itunes",
            return_value=[{"trackName": "Other", "artistName": "Unrelated"}, recording],
        ):
            details = itunes.fetch_itunes_track_metadata("Primary", "Recording")
        assert details is not None
        self.assertEqual(
            (details["itunes_trackid"], details["advisory"], details["date"]),
            ("1", "Explicit", "2020-01-02"),
        )
        with patch.object(
            itunes,
            "search_itunes",
            return_value=[{"trackName": "Other", "artistName": "Unrelated"}],
        ):
            self.assertIsNone(
                itunes.fetch_itunes_track_metadata("Primary", "Recording")
            )


class TestCodeHygiene(IsolatedTestCase):
    def test_feature_patterns_accept_the_same_supported_tokens(self) -> None:
        for token in ("feat.", "ft", "w/", "w."):
            with self.subTest(token=token):
                self.assertIsNotNone(
                    get_balanced_feat_pattern().match(f"{token} Guest")
                )
                self.assertIsNotNone(
                    get_bracket_feat_pattern().search(f" ({token} Guest)")
                )

    def test_feature_patterns_preserve_non_feature_conjunctions(self) -> None:
        for token in ("and", "with", "vs", "w/o", "w/out"):
            with self.subTest(token=token):
                self.assertIsNone(get_balanced_feat_pattern().match(f"{token} Guest"))
                self.assertIsNone(
                    get_bracket_feat_pattern().search(f" ({token} Guest)")
                )

    def test_feature_patterns_share_custom_configuration(self) -> None:
        custom_config = dataclasses.replace(
            get_config(), featuring_conjunctions=("avec",)
        )
        with patch("sonora.core.config.get_config", return_value=custom_config):
            self.assertIsNotNone(get_balanced_feat_pattern().match("avec Guest"))
            self.assertIsNotNone(get_bracket_feat_pattern().search(" [avec Guest]"))
            self.assertIsNone(get_balanced_feat_pattern().match("feat. Guest"))
            self.assertIsNone(get_bracket_feat_pattern().search(" [feat. Guest]"))

    def test_itunes_exact_album_precedes_earlier_fuzzy_candidate(self) -> None:
        album_candidates = [
            {
                "artistName": "Artist Alpha",
                "collectionName": "Test Record Deluxe",
                "artworkUrl100": "https://example.com/fuzzy/100x100bb.jpg",
            },
            {
                "artistName": "Artist Alpha",
                "collectionName": "Test Record",
                "artworkUrl100": "https://example.com/exact/100x100bb.jpg",
            },
        ]
        with patch(
            "sonora.services.itunes.search_itunes", return_value=album_candidates
        ):
            self.assertEqual(
                fetch_itunes_cover_art_url("Artist Alpha", "Test Record"),
                "https://example.com/exact/1400x1400bb.jpg",
            )

    def test_retry_rejects_empty_attempt_budget_before_calling(self) -> None:
        for attempts in (0, -1):
            with self.subTest(attempts=attempts):
                provider_request = Mock(return_value="metadata")
                with self.assertRaisesRegex(ValueError, "max_retries must be positive"):
                    retry_network_call(provider_request, max_retries=attempts)
                provider_request.assert_not_called()

    def test_plain_and_rich_diffs_preserve_changes_and_hide_lyric_content(self) -> None:
        original = TrackInfo(Path("track.flac"), composer="Original composer")
        updated = dataclasses.replace(
            original, composer=None, label="[bold]", lyrics="private lyric content"
        )
        differences = _compute_tag_diffs(original, updated)
        descriptions = _describe_tag_diffs(differences)
        rendered = "".join(_render_tag_diffs(differences))
        self.assertIn("composer: removed 'Original composer'", descriptions)
        self.assertIn("label: added '[bold]'", descriptions)
        self.assertIn("[+] label: \\[bold]", rendered)
        self.assertIn("[-] composer: Original composer", rendered)
        self.assertNotIn("added ", rendered)
        self.assertNotIn("removed ", rendered)
        self.assertIn("21 characters", rendered)
        self.assertNotIn("private lyric content", rendered)
        self.assertNotIn("private lyric content", "".join(descriptions))
        self.assertEqual(_describe_tag_diffs([]), [])
        self.assertEqual(_render_tag_diffs([]), [])


class TestCheckerMetadataFailures(IsolatedTestCase):
    def setUp(self) -> None:
        self.audio_path = self.test_directory / "01 - Song.wav"
        self.audio_path.write_bytes(b"unreadable audio fixture")

    def test_metadata_failure_still_reports_missing_lyrics(self) -> None:
        with patch(
            "sonora.modules.checker.read_track_metadata",
            side_effect=OSError("unreadable metadata"),
        ):
            issues = check_file(self.audio_path)
        self.assertIn("Metadata read error: unreadable metadata", issues)
        self.assertIn("Missing synchronized lyrics (.lrc) file.", issues)

    def test_metadata_failure_still_validates_existing_lyrics(self) -> None:
        lyrics_path = self.audio_path.with_suffix(".lrc")
        with patch(
            "sonora.modules.checker.read_track_metadata",
            side_effect=OSError("unreadable metadata"),
        ):
            lyrics_path.write_text("[00:01.00] A lyric line", encoding="utf-8")
            issues = check_file(self.audio_path)
            self.assertIn("Metadata read error: unreadable metadata", issues)
            self.assertNotIn("Missing synchronized lyrics (.lrc) file.", issues)
            self.assertNotIn("Corrupt 0-byte synchronized lyrics (.lrc) file.", issues)
            lyrics_path.write_bytes(b"")
            issues = check_file(self.audio_path)
            self.assertIn("Corrupt 0-byte synchronized lyrics (.lrc) file.", issues)

    def test_supplied_metadata_avoids_reopening_unreadable_audio(self) -> None:
        supplied_metadata = TrackInfo(
            self.audio_path,
            artist="Artist",
            title="Song",
            album="Album",
            track_number=1,
        )
        self.audio_path.with_suffix(".lrc").write_text(
            "[00:01.00] A lyric line", encoding="utf-8"
        )
        with patch(
            "sonora.modules.checker.read_track_metadata",
            side_effect=OSError("unreadable metadata"),
        ) as metadata_reader:
            issues = check_file(self.audio_path, track_metadata=supplied_metadata)
        metadata_reader.assert_not_called()
        self.assertFalse(
            any(issue.startswith("Metadata read error:") for issue in issues)
        )
        self.assertNotIn("Missing synchronized lyrics (.lrc) file.", issues)


class TestImportBoundaries(unittest.TestCase):
    def test_cli_import_does_not_load_audio_providers(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json,sys; sys.path[:0] = json.loads(sys.argv[1]); import sonora.cli.main; print(json.dumps(sorted(sys.modules)))",
                json.dumps(sys.path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        imported_modules = set(json.loads(result.stdout))
        self.assertFalse(
            imported_modules.intersection({"scipy", "shazamio", "syncedlyrics"})
        )

    def test_small_modules_do_not_load_network_or_dsp_dependencies(self) -> None:
        for module_name in (
            "sonora.core",
            "sonora.audio",
            "sonora.services",
            "sonora.modules",
            "sonora.core.constants",
            "sonora.core.cancellation",
            "sonora.core.retry",
            "sonora.core.audio_files",
            "sonora.core.library_paths",
            "sonora.core.sidecars",
            "sonora.audio.checksum",
            "sonora.core.artists",
            "sonora.modules.normalization",
        ):
            with self.subTest(module=module_name):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import importlib,json,sys; sys.path[:0] = json.loads(sys.argv[2]); importlib.import_module(sys.argv[1]); print(json.dumps(sorted(sys.modules)))",
                        module_name,
                        json.dumps(sys.path),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                )
                imported_modules = set(json.loads(result.stdout))
                self.assertFalse(
                    imported_modules.intersection(
                        {"scipy", "shazamio", "httpx", "musicbrainzngs", "diskcache"}
                    ),
                    result.stdout,
                )


class TestNetwork(IsolatedTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        RateLimiter.set_disabled(True)

    @classmethod
    def tearDownClass(cls) -> None:
        RateLimiter.set_disabled(False)

    def setUp(self) -> None:
        lrclib_patch = patch("sonora.services.lyrics._query_lrclib", return_value=None)
        lrclib_patch.start()
        self.addCleanup(lrclib_patch.stop)
        artist_normalization = patch(
            "sonora.services.artist_resolution.resolve_artist_name",
            side_effect=lambda name: name,
        )
        artist_normalization.start()
        self.addCleanup(artist_normalization.stop)
        services = [
            "sonora.services.itunes",
            "sonora.services.lyrics",
            "sonora.services.musicbrainz",
            "sonora.services.discogs",
            "sonora.services.acoustid",
            "sonora.services.lastfm",
            "sonora.services.genius",
            "sonora.services.deezer",
            "sonora.services.theaudiodb",
        ]
        for service in services:
            patcher = patch(f"{service}.get_cached_api", return_value=None)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_session_sends_application_user_agent(self) -> None:
        with patch(
            "httpx.HTTPTransport.handle_request",
            return_value=httpx.Response(200, json={"ok": True}),
        ) as send:
            response = SESSION.get("https://api.deezer.com/test")
        self.assertEqual(response.json(), {"ok": True})
        self.assertEqual(send.call_args.args[0].headers["User-Agent"], USER_AGENT)

    @patch("sonora.services.theaudiodb.SESSION.get")
    def test_theaudiodb_server_disconnect(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = httpx.RemoteProtocolError("Server disconnected")
        thumb, banner = fetch_artist_images("Emil Lassaria")
        self.assertIsNone(thumb)
        self.assertIsNone(banner)

    @patch("sonora.services.theaudiodb.SESSION.get")
    def test_theaudiodb_image_download_disconnect(self, mock_get: MagicMock) -> None:
        mock_json_response = MagicMock()
        mock_json_response.status_code = 200
        mock_json_response.json.return_value = {
            "artists": [
                {
                    "strArtist": "Artist Name",
                    "strArtistThumb": "https://www.theaudiodb.com/images/thumb.jpg",
                    "strArtistBanner": "https://www.theaudiodb.com/images/banner.jpg",
                }
            ]
        }
        mock_get.side_effect = [
            mock_json_response,
            httpx.ConnectTimeout("Connection timed out"),
            httpx.RemoteProtocolError("Server disconnected"),
        ]
        thumb, banner = fetch_artist_images("Artist Name")
        self.assertIsNone(thumb)
        self.assertIsNone(banner)

    @patch("sonora.services.theaudiodb.SESSION.get")
    def test_theaudiodb_track_details_503_and_broken_json(
        self, mock_get: MagicMock
    ) -> None:
        mock_response_503 = MagicMock()
        mock_response_503.status_code = 503
        mock_get.return_value = mock_response_503
        self.assertIsNone(fetch_theaudiodb_track_details("Artist", "Title"))
        mock_broken_json = MagicMock()
        mock_broken_json.status_code = 200
        mock_broken_json.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_broken_json
        with self.assertRaisesRegex(ValueError, "Invalid JSON"):
            fetch_theaudiodb_track_details("Artist", "Title")

    def test_download_artwork_bytes_empty_url(self) -> None:
        self.assertIsNone(_download_artwork_bytes(None))
        self.assertIsNone(_download_artwork_bytes(""))

    @patch("sonora.services.musicbrainz.musicbrainzngs")
    def test_musicbrainz_recording_lookup_network_errors(
        self, mock_mb: MagicMock
    ) -> None:
        mock_mb.search_recordings.side_effect = musicbrainzngs.MusicBrainzError(
            "Service Unavailable 503"
        )
        self.assertIsNone(fetch_track_mbid("Artist", "Title"))
        mock_mb.get_recording_by_id.side_effect = OSError("Connection reset by peer")
        self.assertIsNone(
            fetch_musicbrainz_recording_details("c8b03190-306c-4125-9b32-3f9d86d60a12")
        )
        mock_mb.get_release_by_id.side_effect = musicbrainzngs.MusicBrainzError(
            "Gateway Timeout 504"
        )
        self.assertIsNone(
            fetch_musicbrainz_release_details("c8b03190-306c-4125-9b32-3f9d86d60a12")
        )

    @patch("sonora.services.musicbrainz.SESSION.head")
    def test_cover_art_archive_network_timeout(self, mock_head: MagicMock) -> None:
        mock_head.side_effect = httpx.ConnectTimeout("Connect timeout")
        self.assertIsNone(
            fetch_cover_art_archive_url("c8b03190-306c-4125-9b32-3f9d86d60a12")
        )

    @patch("sonora.services.itunes.SESSION.get")
    def test_itunes_rate_limit_and_timeout(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = httpx.HTTPStatusError(
            "429 Too Many Requests",
            request=MagicMock(),
            response=MagicMock(status_code=429),
        )
        self.assertEqual(search_itunes("Artist", "Title"), [])
        self.assertIsNone(fetch_itunes_cover_art_url("Artist", "Album"))
        self.assertIsNone(fetch_itunes_track_metadata("Artist", "Title"))
        mock_get.side_effect = httpx.ReadTimeout("Read timed out")
        self.assertEqual(search_itunes("Artist", "Title"), [])

    @patch("sonora.services.deezer.SESSION.get")
    def test_deezer_server_disconnect(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = httpx.RemoteProtocolError("Server disconnected")
        self.assertIsNone(fetch_deezer_cover_art_url("Artist", "Album"))
        self.assertIsNone(fetch_deezer_album_details("Artist", "Album"))
        self.assertIsNone(fetch_deezer_track_details("Artist", "Title"))

    @patch("sonora.services.discogs.SESSION.get")
    def test_discogs_network_errors(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = httpx.HTTPError("401 Unauthorized")
        self.assertIsNone(
            search_discogs_release("Artist", "Album", user_token="bad_token")
        )
        self.assertIsNone(
            fetch_discogs_release_details("12345", user_token="bad_token")
        )

    @patch("sonora.services.lastfm.SESSION.get")
    def test_lastfm_disconnect(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = httpx.RemoteProtocolError("Server disconnected")
        self.assertEqual(
            fetch_lastfm_tags("Artist", "Title", api_key="dummy_lastfm_key"), []
        )

    @patch("sonora.services.genius.SESSION.get")
    def test_genius_network_error(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = httpx.ConnectError("Connection refused")
        self.assertIsNone(
            fetch_genius_song_details("Artist", "Title", api_token="dummy_token")
        )

    @patch("syncedlyrics.search")
    def test_lyrics_upstream_exception(self, mock_search: MagicMock) -> None:
        mock_search.side_effect = httpx.RemoteProtocolError(
            "Musixmatch connection dropped"
        )
        with self.assertRaises(RuntimeError):
            fetch_synced_lyrics("Artist", "Title", providers=["Musixmatch"])
        lyrics_text, lyrics_type = process_track_lyrics(
            Path("/tmp/nonexistent.flac"), "Artist", "Title"
        )
        self.assertIsNone(lyrics_text)
        self.assertIsNone(lyrics_type)

    def test_rate_limiter_spacing(self) -> None:
        RateLimiter.set_disabled(False)
        try:
            limiter = RateLimiter(interval_seconds=0.05)
            timestamps: list[float] = []
            lock = threading.Lock()

            def worker() -> None:
                limiter.wait()
                with lock:
                    timestamps.append(time.monotonic())

            threads = [threading.Thread(target=worker) for _ in range(6)]
            start_time = time.monotonic()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            elapsed = time.monotonic() - start_time
            self.assertGreaterEqual(elapsed, 0.2)
            self.assertEqual(len(timestamps), 6)
            sorted_timestamps = sorted(timestamps)
            for i in range(1, len(sorted_timestamps)):
                diff = sorted_timestamps[i] - sorted_timestamps[i - 1]
                self.assertGreaterEqual(diff, 0.03)
        finally:
            RateLimiter.set_disabled(True)

    @patch("sonora.services.theaudiodb.fetch_theaudiodb_track_details")
    @patch("sonora.audio.metadata.read_track_metadata")
    def test_smart_gating_skips_complete_tracks(
        self, mock_read: MagicMock, mock_tadb: MagicMock
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            song_path = Path(tmp_dir) / "song.flac"
            song_path.write_bytes(b"dummy audio data")
            complete_track = TrackInfo(
                file_path=song_path,
                artist="Artist",
                title="Title",
                album="Album",
                mood="Energetic",
                style="Pop",
                initial_key="8A",
                rating=5.0,
                music_video_url="https://youtube.com/watch?v=123",
                comment="Existing comment",
            )
            mock_read.return_value = complete_track
            with (
                patch(
                    "sonora.services.genius.fetch_genius_song_details",
                    return_value=None,
                ),
                patch(
                    "sonora.services.deezer.fetch_deezer_track_details",
                    return_value=None,
                ),
                patch(
                    "sonora.services.deezer.fetch_deezer_album_details",
                    return_value=None,
                ),
                patch(
                    "sonora.services.musicbrainz.fetch_musicbrainz_recording_details",
                    return_value=None,
                ),
                patch(
                    "sonora.services.musicbrainz.fetch_track_mbid", return_value=None
                ),
                patch(
                    "sonora.services.musicbrainz.search_musicbrainz_release",
                    return_value=None,
                ),
                patch(
                    "sonora.services.itunes.fetch_itunes_track_metadata",
                    return_value=None,
                ),
                patch(
                    "sonora.services.artist_resolution.resolve_artist_name",
                    side_effect=lambda x: x,
                ),
                patch("sonora.audio.metadata.write_track_metadata"),
            ):
                result = process_single_track(
                    file_path=song_path,
                    fetch_bpm=False,
                    fetch_lyrics=False,
                    fetch_itunes_art=False,
                    force=False,
                )
            mock_tadb.assert_not_called()
            self.assertEqual(result.mood, "Energetic")
            self.assertEqual(result.initial_key, "8A")


if __name__ == "__main__":
    unittest.main()


@coordination.coalesce_lookup
def _counted_lookup(marker: Path) -> str | None:
    key = f"test-provider:{marker}"
    cached = cache.get_cached_api(key)
    if cached is not None:
        return str(cached) or None
    with marker.open("a") as output:
        output.write("request\n")
    cache.set_cached_api(key, None)
    return None


def _run_process_lookup(marker: str) -> None:
    _counted_lookup(Path(marker))


def _itunes_after_barrier(barrier: threading.Barrier) -> dict[str, object] | None:
    barrier.wait(timeout=5)
    return itunes.lookup_itunes_track("123")


def _propose_genre(
    track: TrackInfo,
    genre: str,
    source: MetadataSource,
    barrier: threading.Barrier | None = None,
    **keywords: object,
) -> None:
    if barrier:
        barrier.wait(timeout=5)
    track.genre = genre
    track.metadata_sources["genre"] = source


def _replace_during_fingerprint(path: Path) -> tuple[float, str]:
    path.write_bytes(b"changed audio")
    return (100, "fingerprint")


def _mark_identity_conflict(track_info: TrackInfo, **keywords: object) -> None:
    track_info.artist = "Wrong Artist"
    track_info.genre = "Wrong Genre"
    track_info.metadata_conflicts["identity"] = "Conflicting identity"


class TestPerformanceSafety(IsolatedTestCase):
    def test_proposals_retain_only_changes_and_cannot_modify_audio_properties(
        self,
    ) -> None:
        recording = TrackInfo(
            Path("recording.flac"),
            artist="Alpha",
            title="Original Song",
            genre="Rock",
            sample_rate=44100,
            is_lossless=True,
        )
        candidate = dataclasses.replace(
            recording,
            genre="Jazz",
            sample_rate=8000,
            is_lossless=False,
            metadata_sources={"genre": "Deezer"},
        )
        proposal = build_provider_proposal(recording, candidate)
        self.assertEqual(proposal.changes, {"genre": ("Jazz", "Deezer")})
        metadata_merge.merge_provider_proposals(recording, [proposal])
        self.assertEqual(recording.genre, "Jazz")
        self.assertEqual((recording.sample_rate, recording.is_lossless), (44100, True))
        self.assertEqual(build_provider_proposal(recording, recording).changes, {})

    def test_provider_confirmation_is_retained_without_changing_the_tag(self) -> None:
        recording = TrackInfo(Path("recording.flac"), genre="Rock")
        candidate = dataclasses.replace(recording, metadata_sources={"genre": "iTunes"})
        proposal = build_provider_proposal(recording, candidate)
        self.assertEqual(proposal.changes, {"genre": ("Rock", "iTunes")})
        metadata_merge.merge_provider_proposals(recording, [proposal])
        self.assertEqual(recording.metadata_sources, {"genre": "iTunes"})

    def test_proposals_agree_or_preserve_conflicts_independent_of_order(self) -> None:
        for genres, expected in (
            (("Rock", "Rock"), "Rock"),
            (("Rock", "Jazz"), "Original"),
        ):
            original = TrackInfo(
                Path("recording.flac"),
                artist="Alpha",
                title="Original Song",
                genre="Original",
            )
            sources: tuple[MetadataSource, ...] = ("iTunes", "Deezer")
            proposals = [
                metadata_merge.collect_provider_proposal(
                    original, partial(_propose_genre, genre=genre, source=source)
                )
                for genre, source in zip(genres, sources)
            ]
            self.assertEqual(original.genre, "Original")
            self.assertEqual(original.metadata_sources, {})
            for ordered in (proposals, proposals[::-1]):
                merged = dataclasses.replace(
                    original, metadata_sources={}, metadata_conflicts={}
                )
                metadata_merge.merge_provider_proposals(merged, ordered)
                self.assertEqual(merged.genre, expected)
                self.assertEqual(
                    bool(merged.metadata_conflicts), genres[0] != genres[1]
                )
        metadata_merge.merge_provider_proposals(original, [])
        self.assertEqual(original.genre, "Original")

    def test_foreign_identity_and_conflicting_isrc_withhold_metadata_updates(
        self,
    ) -> None:
        original = TrackInfo(
            Path("recording.flac"),
            artist="Alpha",
            title="Original Song",
            isrc="USAAA2400001",
            metadata_sources={"isrc": "MusicBrainz"},
        )
        for title, isrc in (
            ("Different Song", original.isrc),
            (original.title, "USAAA2400002"),
        ):
            with self.subTest(title=title, isrc=isrc):
                merged = dataclasses.replace(
                    original,
                    metadata_sources=original.metadata_sources.copy(),
                    metadata_conflicts={},
                )
                proposal = dataclasses.replace(
                    original,
                    title=title,
                    isrc=isrc,
                    genre="Rock",
                    metadata_sources={"isrc": "Deezer", "genre": "Deezer"},
                )
                metadata_merge.merge_provider_proposals(
                    merged, [build_provider_proposal(merged, proposal)]
                )
                self.assertIsNone(merged.genre)
                self.assertEqual(merged.isrc, original.isrc)
                self.assertTrue(merged.metadata_conflicts)
                self.assertTrue(merged.to_dict()["metadata_conflicts"])

    def test_parallel_provider_proposals_do_not_mutate_shared_identity(self) -> None:
        barrier = threading.Barrier(6)
        original = TrackInfo(
            Path("recording.flac"), artist="Alpha", title="Original Song"
        )
        with (
            patch.object(tagger, "_update_musicbrainz"),
            patch.object(tagger, "_update_acoustid"),
            patch.object(tagger, "_update_shazam"),
            patch.object(
                tagger,
                "_update_itunes",
                side_effect=partial(
                    _propose_genre, genre="Rock", source="iTunes", barrier=barrier
                ),
            ),
            patch.object(
                tagger,
                "_update_deezer",
                side_effect=partial(
                    _propose_genre, genre="Jazz", source="Deezer", barrier=barrier
                ),
            ),
            patch.object(
                tagger,
                "_update_lastfm",
                side_effect=partial(
                    _propose_genre, genre="Rock", source="Last.fm", barrier=barrier
                ),
            ),
            patch.object(
                tagger,
                "_update_discogs",
                side_effect=partial(
                    _propose_genre, genre="Rock", source="Discogs", barrier=barrier
                ),
            ),
            patch.object(
                tagger,
                "_update_genius",
                side_effect=partial(
                    _propose_genre, genre="Rock", source="Genius", barrier=barrier
                ),
            ),
            patch.object(
                tagger,
                "_update_theaudiodb",
                side_effect=partial(
                    _propose_genre, genre="Rock", source="TheAudioDB", barrier=barrier
                ),
            ),
            ThreadPoolExecutor(max_workers=6) as executor,
        ):
            tagger._update_metadata(
                original,
                original.file_path,
                None,
                None,
                None,
                None,
                True,
                provider_executor=executor,
                album_context=tagger._AlbumMetadataContext(),
            )
        self.assertIsNone(original.genre)
        self.assertEqual(original.artist, "Alpha")
        self.assertTrue(original.metadata_conflicts)

    def test_forced_metadata_update_does_not_repeat_shazam_preflight(self) -> None:
        track = TrackInfo(
            Path("recording.flac"), artist="Unknown Artist", title="Track 01"
        )
        with (
            patch.object(tagger, "_update_musicbrainz"),
            patch.object(tagger, "_update_shazam") as shazam,
            patch.object(tagger, "_update_acoustid"),
            patch.object(tagger, "_update_itunes"),
            patch.object(tagger, "_update_lastfm"),
            patch.object(tagger, "_update_discogs"),
            patch.object(tagger, "_update_deezer"),
            patch.object(tagger, "_update_genius"),
            patch.object(tagger, "_update_theaudiodb"),
        ):
            tagger._update_metadata(
                track,
                track.file_path,
                None,
                None,
                None,
                None,
                force=True,
                album_context=tagger._AlbumMetadataContext(),
                shazam_attempted=True,
            )
        shazam.assert_not_called()

    def test_identical_threaded_lookups_make_one_provider_request(self) -> None:
        barrier = threading.Barrier(6)
        response = httpx.Response(
            200,
            json={
                "results": [
                    {
                        "trackId": 123,
                        "trackName": "Original Song",
                        "artistName": "Alpha",
                    }
                ]
            },
            request=httpx.Request("GET", "https://itunes.apple.com/lookup"),
        )
        with (
            patch.object(SESSION, "get", return_value=response) as request,
            ThreadPoolExecutor(max_workers=6) as executor,
        ):
            results = list(executor.map(_itunes_after_barrier, [barrier] * 6))
        self.assertTrue(all(result == results[0] for result in results))
        self.assertIsNotNone(results[0])
        request.assert_called_once()

    def test_duplicate_processes_reuse_confirmed_negative_answer(self) -> None:
        marker = self.test_directory / "requests.txt"
        context = multiprocessing.get_context("spawn")
        processes = [
            context.Process(target=_run_process_lookup, args=(str(marker),))
            for _ in range(3)
        ]
        try:
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=10)
                self.assertFalse(process.is_alive(), "Lookup process did not finish")
                self.assertEqual(process.exitcode, 0)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                if process.pid is not None:
                    process.join()
                process.close()
        self.assertEqual(marker.read_text(), "request\n")

    def test_confirmed_misses_expire_earlier_than_successes(self) -> None:
        for key, value in (
            ("miss", None),
            ("empty", []),
            ("success", {"title": "Original Song"}),
        ):
            cache.set_cached_api(key, value)
        storage = cache.get_cache()
        assert storage is not None
        for key, expected in (("miss", 86400), ("empty", 86400), ("success", 2419200)):
            _, expiry = storage.get(key, expire_time=True)
            self.assertAlmostEqual(expiry - time.time(), expected, delta=2)
        self.assertIsNone(cache.get_cached_api("absent"))
        self.assertEqual(cache.get_cached_api("miss"), "")

    def test_provider_failures_are_not_negative_cached_but_not_found_is(self) -> None:
        for status, expected_calls in ((503, 2), (404, 1)):
            response = httpx.Response(
                status,
                request=httpx.Request("GET", "https://api.deezer.com/track/isrc:TEST"),
            )
            with (
                self.subTest(status=status),
                patch.object(SESSION, "get", return_value=response) as request,
            ):
                for _ in range(2):
                    self.assertIsNone(
                        deezer.fetch_deezer_track_by_isrc(f"TEST{status}")
                    )
            self.assertEqual(request.call_count, expected_calls)

    def test_persistent_fingerprint_survives_memory_clear_and_invalidates_changed_file(
        self,
    ) -> None:
        path = self.test_directory / "recording.flac"
        path.write_bytes(b"audio")
        with patch.object(
            acoustid,
            "_fingerprint_file_with_timeout",
            return_value=(100, "fingerprint"),
        ) as decode:
            first = acoustid.fingerprint_audio_file(path)
            acoustid._ACOUSTID_CACHE.clear()
            self.assertEqual(acoustid.fingerprint_audio_file(path), first)
            decode.assert_called_once()
            path.write_bytes(b"new audio")
            self.assertEqual(acoustid.fingerprint_audio_file(path), first)
            self.assertEqual(decode.call_count, 2)

    def test_file_changes_during_fingerprinting_are_rejected(self) -> None:
        path = self.test_directory / "recording.flac"
        path.write_bytes(b"audio")
        with (
            patch.object(
                acoustid,
                "_fingerprint_file_with_timeout",
                side_effect=_replace_during_fingerprint,
            ),
            self.assertRaisesRegex(OSError, "changed during fingerprinting"),
        ):
            acoustid.fingerprint_audio_file(path)

    def test_provider_quota_is_shared_and_other_hosts_are_independent(self) -> None:
        with (
            patch("sonora.core.coordination.time.time", return_value=100),
            patch("sonora.core.coordination.time.sleep") as sleep,
        ):
            self.assertEqual(coordination.wait_for_provider("provider-a", 1), 0)
            self.assertEqual(coordination.wait_for_provider("provider-b", 1), 0)
            self.assertEqual(coordination.wait_for_provider("provider-a", 1), 1)
        sleep.assert_called_once_with(1)

    def test_album_prefetch_is_bounded_and_skips_unchanged_albums(self) -> None:
        folders = {
            self.test_directory / str(index): [
                self.test_directory / str(index) / "recording.flac"
            ]
            for index in range(5)
        }
        changed = {path for paths in list(folders.values())[1:] for path in paths}
        with (
            patch.object(
                tagger,
                "_resolve_album_metadata_context",
                return_value=tagger._AlbumMetadataContext(),
            ) as resolve,
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            contexts = tagger._prefetch_album_contexts(
                folders,
                changed,
                executor,
                discogs_user_token=None,
                fetch_itunes_art=False,
                force=False,
                dry_run=True,
                window=2,
            )
            _, _, first = next(contexts)
            self.assertIsNone(first)
            _, _, second = next(contexts)
            assert second is not None
            second.result()
            self.assertLessEqual(resolve.call_count, 2)
            list(contexts)
        self.assertEqual(resolve.call_count, 4)

    def test_identity_conflict_prevents_tag_and_asset_writes(self) -> None:
        path = self.test_directory / "recording.flac"
        path.touch()
        recording = TrackInfo(path, artist="Alpha", title="Original Song")
        with (
            patch("sonora.audio.metadata.read_track_metadata", return_value=recording),
            patch(
                "sonora.services.artist_resolution.resolve_artist_name",
                return_value="Alpha",
            ),
            patch.object(
                tagger, "_update_metadata", side_effect=_mark_identity_conflict
            ),
            patch.object(tagger, "_process_audio_and_save_tags") as persist,
        ):
            result = tagger.process_single_track(path, force=True)
        self.assertEqual(result.artist, "Alpha")
        self.assertIsNone(result.genre)
        self.assertIn("identity", result.metadata_conflicts)
        persist.assert_not_called()

    def test_conflicts_survive_reporting_and_are_not_marked_complete(self) -> None:
        path = self.test_directory / "recording.flac"
        path.touch()
        recording = TrackInfo(
            path,
            artist="Alpha",
            title="Original Song",
            genre="Unknown",
            metadata_sources={"genre": "iTunes"},
            metadata_conflicts={"genre": "Disputed genre"},
        )
        vault = LibraryStateVault(self.test_directory / "state.db")
        self.addCleanup(vault.close)
        with patch(
            "sonora.audio.metadata.read_track_metadata",
            return_value=dataclasses.replace(recording, metadata_conflicts={}),
        ):
            persisted = _record_persisted_album(
                [recording],
                vault,
                "TAGGED_OK",
                fetch_bpm=False,
                fetch_key=False,
                fetch_replaygain=False,
                fetch_lyrics=False,
                fetch_itunes_art=False,
            )
        self.assertEqual(persisted[0].metadata_conflicts, recording.metadata_conflicts)
        self.assertEqual(
            vault.filter_outdated_tracks([path], expected_status="TAGGED_OK"), [path]
        )
        with patch("sonora.audio.metadata.write_track_metadata") as write:
            tagger._harmonize_album_genres(
                [
                    recording,
                    dataclasses.replace(recording, genre="Rock", metadata_conflicts={}),
                ],
                None,
                None,
            )
        self.assertEqual(recording.genre, "Unknown")
        write.assert_not_called()

    def test_standalone_itunes_cannot_replace_known_release_date_country_or_copyright(
        self,
    ) -> None:
        recording = TrackInfo(
            Path("recording.flac"),
            artist="Alpha",
            title="Original Song",
            date="2020",
            release_country="Romania",
            copyright="Original copyright",
        )
        with patch.object(
            itunes,
            "fetch_itunes_track_metadata",
            return_value={
                "date": "2025",
                "release_country": "United States",
                "copyright": "Reissue copyright",
            },
        ):
            catalog_metadata._update_itunes(recording, force=True)
        self.assertEqual(
            (recording.date, recording.release_country, recording.copyright),
            ("2020", "Romania", "Original copyright"),
        )
        with patch.object(
            itunes,
            "search_itunes",
            return_value=[
                {
                    "trackId": 123,
                    "trackName": "Original Song",
                    "artistName": "Alpha",
                    "country": "US",
                }
            ],
        ):
            payload = itunes.fetch_itunes_track_metadata("Alpha", "Original Song")
        assert payload is not None
        self.assertNotIn("release_country", payload)

    def test_musicbrainz_ties_abstain_and_cached_evidence_is_reranked(self) -> None:
        recordings = [
            {
                "id": identifier,
                "title": "Original Song",
                "artist-credit": [{"artist": {"name": "Alpha"}}],
            }
            for identifier in (
                "11111111-2222-3333-4444-555555555555",
                "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            )
        ]
        with patch.object(
            musicbrainz,
            "_call_musicbrainz",
            return_value={"recording-list": recordings},
        ) as request:
            self.assertIsNone(musicbrainz.fetch_track_mbid("Alpha", "Original Song"))
            self.assertIsNone(musicbrainz.fetch_track_mbid("Alpha", "Original Song"))
        request.assert_called_once()


def _accept_restored_tags(
    relative_path: str, backup_tags: dict[str, Any], restore_root: Path
) -> tuple[bool, bool]:
    return (True, False)


class TestRestoreMemoryAndPreflight(IsolatedTestCase):
    def test_large_artwork_restore_has_bounded_peak_memory(self) -> None:
        archive = self.test_directory / "backup.json.gz"
        with gzip.open(archive, "wb") as archive_stream:
            archive_stream.write(
                b'{"schema":"sonora_backup_v2","root":'
                + json.dumps(str(self.test_directory)).encode()
                + b',"tracks":{'
            )
            for index in range(48):
                if index:
                    archive_stream.write(b",")
                archive_stream.write(json.dumps(f"recording-{index}.flac").encode())
                archive_stream.write(b":")
                archive_stream.write(
                    json.dumps(
                        {
                            "tags": {"TITLE": [f"Recording {index}"]},
                            "complex_tags": {
                                "PICTURE": [{"data": {"base64": "A" * (512 * 1024)}}]
                            },
                        }
                    ).encode()
                )
            archive_stream.write(b"}}")
        tracemalloc.start()
        try:
            with patch.object(
                backup, "_restore_single_track", new=_accept_restored_tags
            ):
                self.assertEqual(
                    backup.restore_library_tags(archive, max_threads=2), 48
                )
            _, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak_bytes, 12 * 1024 * 1024)

    def test_invalid_final_entry_and_truncated_json_never_start_restoring(self) -> None:
        archive = self.test_directory / "backup.json"
        valid_tags = {"tags": {"TITLE": ["Recording"]}, "complex_tags": {}}
        manifest = {
            "schema": "sonora_backup_v2",
            "root": str(self.test_directory),
            "tracks": {"first.flac": valid_tags, "last.flac": {"tags": []}},
        }
        invalid_entry_payload = json.dumps(manifest)
        manifest["tracks"] = {"first.flac": valid_tags, "last.flac": valid_tags}
        for payload in (invalid_entry_payload, json.dumps(manifest)[:-1]):
            archive.write_text(payload)
            with (
                self.subTest(truncated=not payload.endswith("}")),
                patch.object(backup, "_restore_single_track") as restore,
                self.assertRaises((TypeError, ValueError)),
            ):
                backup.restore_library_tags(archive)
            restore.assert_not_called()

    def test_truncated_and_invalid_deflate_archives_fail_before_writes(self) -> None:
        archive = self.test_directory / "backup.json.gz"
        manifest = {
            "schema": "sonora_backup_v2",
            "root": str(self.test_directory),
            "tracks": {"recording.flac": {"tags": {}, "complex_tags": {}}},
        }
        compressed_archive = gzip.compress(json.dumps(manifest).encode())
        corrupted_archive = bytearray(compressed_archive)
        corrupted_archive[10] = 7
        for payload in (compressed_archive[:-8], bytes(corrupted_archive)):
            archive.write_bytes(payload)
            with (
                self.subTest(truncated=payload == compressed_archive[:-8]),
                patch.object(backup, "_restore_single_track") as restore,
                self.assertRaisesRegex(ValueError, "Invalid backup archive"),
            ):
                backup.restore_library_tags(archive)
            restore.assert_not_called()

    def test_empty_tracks_and_manifest_fields_after_tracks_are_supported(self) -> None:
        archive = self.test_directory / "backup.json"
        track_cases: tuple[dict[str, dict[str, Any]], ...] = (
            {},
            {"recording.flac": {"tags": {}, "complex_tags": {}}},
        )
        for tracks in track_cases:
            archive.write_text(
                json.dumps(
                    {
                        "tracks": tracks,
                        "root": str(self.test_directory),
                        "schema": "sonora_backup_v2",
                    }
                )
            )
            with (
                self.subTest(track_count=len(tracks)),
                patch.object(
                    backup, "_restore_single_track", side_effect=_accept_restored_tags
                ),
            ):
                self.assertEqual(backup.restore_library_tags(archive), len(tracks))

    def test_duplicate_manifest_keys_and_tracks_are_rejected_before_writes(
        self,
    ) -> None:
        archive = self.test_directory / "backup.json"
        prefix = '{"schema":"sonora_backup_v2","root":' + json.dumps(
            str(self.test_directory)
        )
        tags = '{"tags":{},"complex_tags":{}}'
        payloads = (
            prefix + ',"root":"other","tracks":{}}',
            prefix + ',"tracks":{"same.flac":' + tags + ',"same.flac":' + tags + "}}",
        )
        for payload in payloads:
            archive.write_text(payload)
            with (
                self.subTest(payload=payload),
                patch.object(backup, "_restore_single_track") as restore,
                self.assertRaisesRegex(ValueError, "Duplicate backup"),
            ):
                backup.restore_library_tags(archive)
            restore.assert_not_called()


class TestRenameAndLyrics(IsolatedTestCase):
    def test_invalid_utf8_lyrics_remain_unchanged(self) -> None:
        lyrics_path = self.test_directory / "recording.lrc"
        for payload in (b"[00:01.00]caf\xe9\n", b"[ar:Primary]\n[00:01.00]\xff\xfe\n"):
            with self.subTest(payload=payload):
                lyrics_path.write_bytes(payload)
                self.assertFalse(
                    renamer.sync_lrc_metadata(lyrics_path, "Artist", "Song")
                )
                self.assertEqual(lyrics_path.read_bytes(), payload)

    def test_utf8_lyrics_keep_timestamps_and_non_ascii_text(self) -> None:
        lyrics_path = self.test_directory / "recording.lrc"
        lyrics_path.write_text("[00:01.00]café\n", encoding="utf-8")
        self.assertTrue(renamer.sync_lrc_metadata(lyrics_path, "Artist", "Song"))
        self.assertIn("[00:01.00]café\n", lyrics_path.read_text(encoding="utf-8"))

    def test_matching_headers_are_a_noop_including_crlf_files(self) -> None:
        lyrics_path = self.test_directory / "recording.lrc"
        for newline in (b"\n", b"\r\n"):
            payload = newline.join((b"[ar:Artist]", b"[ti:Song]", b"[00:01.00] Lyrics"))
            lyrics_path.write_bytes(payload)
            with (
                self.subTest(newline=newline),
                patch.object(renamer, "atomic_write_bytes") as write,
            ):
                self.assertFalse(
                    renamer.sync_lrc_metadata(lyrics_path, "Artist", "Song")
                )
            write.assert_not_called()
            self.assertEqual(lyrics_path.read_bytes(), payload)

    def test_rename_preserves_conflicting_lyrics_without_audio_rename(self) -> None:
        audio_path = self.test_directory / "01 - Song.flac"
        audio_path.touch()
        lyrics_path = audio_path.with_suffix(".lrc")
        lyrics_path.write_text("[ar:Old]\n[ti:Old]\n[00:01.00] Lyrics\n")
        recording = TrackInfo(
            audio_path,
            artist="Artist",
            title="Song",
            track_number=1,
            album="Unknown Album",
        )
        with patch.object(renamer, "read_track_metadata", return_value=recording):
            first_report = RenameReport()
            renamer.rename_directory_files(self.test_directory, report=first_report)
            second_report = RenameReport()
            renamer.rename_directory_files(self.test_directory, report=second_report)
        self.assertEqual((first_report.files_renamed, first_report.lrc_synced), (0, 0))
        self.assertEqual(
            (second_report.files_renamed, second_report.lrc_synced), (0, 0)
        )
        self.assertEqual(
            lyrics_path.read_text(), "[ar:Old]\n[ti:Old]\n[00:01.00] Lyrics\n"
        )

    def test_rename_report_excludes_malformed_lyrics_and_dry_run_updates(self) -> None:
        audio_path = self.test_directory / "original.flac"
        audio_path.touch()
        lyrics_path = audio_path.with_suffix(".lrc")
        malformed_lyrics = b"[00:01.00]caf\xe9\n"
        lyrics_path.write_bytes(malformed_lyrics)
        recording = TrackInfo(
            audio_path,
            artist="Artist",
            title="Song",
            track_number=1,
            album="Unknown Album",
        )
        with patch.object(renamer, "read_track_metadata", return_value=recording):
            dry_report = RenameReport()
            renamer.rename_directory_files(
                self.test_directory, dry_run=True, report=dry_report
            )
            write_report = RenameReport()
            renamer.rename_directory_files(self.test_directory, report=write_report)
        self.assertEqual(dry_report.lrc_synced, 0)
        self.assertEqual((write_report.files_renamed, write_report.lrc_synced), (1, 0))
        self.assertEqual(
            (self.test_directory / "01 - Song.lrc").read_bytes(), malformed_lyrics
        )

    def test_successful_header_sync_is_reported_when_audio_rename_fails(self) -> None:
        audio_path = self.test_directory / "original.flac"
        audio_path.touch()
        audio_path.with_suffix(".lrc").write_text("[00:01.00] Lyrics\n")
        recording = TrackInfo(
            audio_path,
            artist="Artist",
            title="Song",
            track_number=1,
            album="Unknown Album",
        )
        report = RenameReport()
        with (
            patch.object(renamer, "read_track_metadata", return_value=recording),
            patch.object(
                renamer, "safe_case_rename", side_effect=PermissionError("locked")
            ),
        ):
            renamer.rename_directory_files(self.test_directory, report=report)
        self.assertEqual((report.failed_files, report.lrc_synced), (1, 1))

    def test_tied_distinct_albums_do_not_rename_folder_in_either_order(self) -> None:
        paths = [
            self.test_directory / "first.flac",
            self.test_directory / "second.flac",
        ]
        for path in paths:
            path.touch()
        tracks = {
            path: TrackInfo(
                path,
                artist=f"Artist {index}",
                album=f"Album {index}",
                title=f"Song {index}",
            )
            for index, path in enumerate(paths)
        }
        for ordered_paths in (paths, paths[::-1]):
            with (
                self.subTest(order=ordered_paths),
                patch.object(renamer, "find_audio_files", return_value=ordered_paths),
                patch.object(
                    renamer, "read_track_metadata", side_effect=tracks.__getitem__
                ),
                patch.object(renamer, "rename_album_folder") as rename_folder,
            ):
                renamer.rename_directory_files(
                    self.test_directory, dry_run=True, max_threads=1
                )
            rename_folder.assert_not_called()

    def test_tied_artists_on_same_album_use_compilation_credit(self) -> None:
        paths = [
            self.test_directory / "first.flac",
            self.test_directory / "second.flac",
        ]
        for path in paths:
            path.touch()
        tracks = {
            path: TrackInfo(
                path,
                artist=f"Artist {index}",
                album="Compilation",
                title=f"Song {index}",
            )
            for index, path in enumerate(paths)
        }
        with (
            patch.object(
                renamer, "read_track_metadata", side_effect=tracks.__getitem__
            ),
            patch.object(
                renamer, "rename_album_folder", return_value=self.test_directory
            ) as rename_folder,
        ):
            renamer.rename_directory_files(
                self.test_directory, dry_run=True, max_threads=1
            )
        rename_folder.assert_called_once_with(
            self.test_directory, "Various Artists", "Compilation", dry_run=True
        )

    def test_incomplete_album_evidence_does_not_become_compilation(self) -> None:
        paths = [
            self.test_directory / "first.flac",
            self.test_directory / "second.flac",
        ]
        for path in paths:
            path.touch()
        tracks = {
            paths[0]: TrackInfo(
                paths[0], artist="Primary", album="Record", title="Song"
            ),
            paths[1]: TrackInfo(
                paths[1],
                artist="Unknown Artist",
                album="Unknown Album",
                title="Unknown",
            ),
        }
        with (
            patch.object(
                renamer, "read_track_metadata", side_effect=tracks.__getitem__
            ),
            patch.object(renamer, "rename_album_folder") as rename_folder,
        ):
            renamer.rename_directory_files(
                self.test_directory, dry_run=True, max_threads=1
            )
        rename_folder.assert_not_called()
