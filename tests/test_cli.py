import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from sonora.cli.main import main
from sonora.cli.reports import _emit_tag_summary_and_report, _tag_coverage_counts
from sonora.core.models import CheckReport, MetadataSource, NormalizeReport, TrackInfo
from tests.support import IsolatedTestCase


class TestCLIInterface(IsolatedTestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.temporary_path = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_verbose_works_before_and_after_commands_without_leaking(self) -> None:
        from sonora.core.logger import LOG

        def tag_with_diagnostics(*args: object, **kwargs: object) -> list[TrackInfo]:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=1) as workers:
                workers.submit(LOG.detail, "Provider decision").result()
            return []

        for arguments in (
            ["--verbose", "tag", str(self.temporary_path)],
            ["tag", str(self.temporary_path), "--verbose"],
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
            self.assertEqual(main(["tag", str(self.temporary_path)]), 0)
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
                        str(self.temporary_path),
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
            self.assertEqual(main(["tag", str(self.temporary_path), "--verbose"]), 1)
        self.assertFalse(LOG.verbose)

    def test_default_check_limits_console_findings_without_truncating_json(
        self,
    ) -> None:
        output_path = self.temporary_path / "findings.json"
        issues = {
            str(self.temporary_path / f"recording-{index}.flac"): [
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
                main(["check", str(self.temporary_path), "--json", str(output_path)]), 0
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
            exit_code = main(["tag", str(self.temporary_path), "-t", "2"])
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
                    str(self.temporary_path),
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
                [
                    "tag",
                    str(self.temporary_path),
                    "--no-dsp",
                    "--ignore-cache",
                ]
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
            json_output_path = self.temporary_path / "tag_report.json"
            exit_code = main(
                ["tag", str(self.temporary_path), "--json", str(json_output_path)]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(json_output_path.exists())
            content = json_output_path.read_text(encoding="utf-8")
            self.assertIn("summary_text", content)
            self.assertIn('"bpm_calculated_count": 1', content)

    def test_handle_check_subcommand(self) -> None:
        with patch("sonora.cli.library.check_library") as mock_check:
            mock_check.return_value = CheckReport(total_files=1, corrupt_files=0)
            exit_code = main(["check", str(self.temporary_path)])
            self.assertEqual(exit_code, 0)
            mock_check.assert_called_once()

    def test_handle_rename_subcommand(self) -> None:
        with patch("sonora.cli.library.rename_directory_files") as mock_rename:
            mock_rename.return_value = [Path("01 - Artist - Title.flac")]
            exit_code = main(["rename", str(self.temporary_path)])
            self.assertEqual(exit_code, 0)
            mock_rename.assert_called_once()

    def test_handle_organize_subcommand(self) -> None:
        with patch("sonora.cli.library.organize_library_singles") as mock_organize:
            mock_organize.return_value = 5
            target_directory = self.temporary_path / "Singles"
            exit_code = main(
                [
                    "organize",
                    str(self.temporary_path),
                    "--target-singles",
                    str(target_directory),
                ]
            )
            self.assertEqual(exit_code, 0)
            mock_organize.assert_called_once()

    def test_handle_backup_subcommand(self) -> None:
        with patch("sonora.cli.library.backup_library_tags") as mock_backup:
            mock_backup.return_value = self.temporary_path / "backup.json"
            exit_code = main(["backup", str(self.temporary_path)])
            self.assertEqual(exit_code, 0)
            mock_backup.assert_called_once()

    def test_handle_rename_subcommand_with_json(self) -> None:
        with patch("sonora.cli.library.rename_directory_files") as mock_rename:
            mock_rename.return_value = [Path("01 - Artist - Title.flac")]
            json_output = self.temporary_path / "rename_report.json"
            exit_code = main(
                ["rename", str(self.temporary_path), "--json", str(json_output)]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(json_output.exists())

    def test_handle_organize_subcommand_with_json(self) -> None:
        with patch("sonora.cli.library.organize_library_singles") as mock_organize:
            mock_organize.return_value = 3
            json_output = self.temporary_path / "organize_report.json"
            exit_code = main(
                ["organize", str(self.temporary_path), "--json", str(json_output)]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(json_output.exists())

    def test_handle_restore_subcommand_with_json(self) -> None:
        with patch("sonora.cli.library.restore_library_tags") as mock_restore:
            mock_restore.return_value = 5
            dummy_backup = self.temporary_path / "backup.json"
            dummy_backup.write_text("{}", encoding="utf-8")
            json_output = self.temporary_path / "restore_report.json"
            exit_code = main(["restore", str(dummy_backup), "--json", str(json_output)])
            self.assertEqual(exit_code, 0)
            self.assertTrue(json_output.exists())

    def test_handle_normalize_subcommand(self) -> None:
        with patch("sonora.cli.library.normalize_library") as mock_normalize:
            mock_normalize.return_value = NormalizeReport(
                total_scanned=1,
                modified_files={"song.flac": ["artist: 'Old' -> 'New'"]},
            )
            json_output = self.temporary_path / "normalize_report.json"
            exit_code = main(
                [
                    "normalize",
                    str(self.temporary_path),
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
            exit_code = main(["normalize", str(self.temporary_path), "--no-dsp"])
            self.assertEqual(exit_code, 0)
            mock_normalize.assert_called_once_with(
                self.temporary_path,
                fetch_bpm=False,
                fetch_key=False,
                fetch_replaygain=False,
                force=False,
                dry_run=False,
                max_threads=4,
            )

    def test_handle_bpm_subcommand(self) -> None:
        song = self.temporary_path / "song.flac"
        song.write_bytes(b"dummy")
        json_out = self.temporary_path / "bpm.json"
        with (
            patch("sonora.cli.audio.calculate_bpm") as mock_calc_bpm,
            patch("sonora.cli.audio.read_track_metadata") as mock_read,
            patch("sonora.cli.audio.write_track_metadata") as mock_write,
        ):
            mock_read.return_value = TrackInfo(file_path=song, bpm=None)
            mock_calc_bpm.return_value = 128.0
            exit_code = main(["bpm", str(self.temporary_path), "--json", str(json_out)])
            self.assertEqual(exit_code, 0)
            mock_write.assert_called_once()
            mock_read.assert_called_once()
            mock_calc_bpm.assert_called_once()
            self.assertTrue(json_out.exists())

    def test_handle_key_subcommand(self) -> None:
        song = self.temporary_path / "song.flac"
        song.write_bytes(b"dummy")
        json_out = self.temporary_path / "key.json"
        with (
            patch("sonora.cli.audio.detect_key_details") as mock_detect_key,
            patch("sonora.cli.audio.read_track_metadata") as mock_read,
            patch("sonora.cli.audio.write_track_metadata") as mock_write,
        ):
            # Test key detection on missing key
            mock_read.return_value = TrackInfo(file_path=song, initial_key=None)
            mock_detect_key.return_value = ("C#m", "12A", 0.85)
            exit_code = main(["key", str(self.temporary_path), "--json", str(json_out)])
            self.assertEqual(exit_code, 0)
            mock_write.assert_called_once()
            mock_read.assert_called_once()
            mock_detect_key.assert_called_once()
            self.assertTrue(json_out.exists())

            # Test auto-healing on invalid key (e.g. 'm')
            mock_write.reset_mock()
            mock_read.reset_mock()
            mock_detect_key.reset_mock()
            mock_read.return_value = TrackInfo(file_path=song, initial_key="m")
            mock_detect_key.return_value = ("Am", "8A", 0.90)
            exit_code_invalid = main(["key", str(self.temporary_path)])
            self.assertEqual(exit_code_invalid, 0)
            mock_detect_key.assert_called_once()
            mock_write.assert_called_once()

    def test_handle_replaygain_subcommand(self) -> None:
        song = self.temporary_path / "song.flac"
        song.write_bytes(b"dummy")
        json_out = self.temporary_path / "rg.json"
        with patch("sonora.cli.audio.calculate_album_replaygain") as mock_rg:
            mock_rg.return_value = True
            exit_code = main(
                ["replaygain", str(self.temporary_path), "--json", str(json_out)]
            )
            self.assertEqual(exit_code, 0)
            mock_rg.assert_called_once()
            self.assertTrue(json_out.exists())

    def test_handle_lyrics_subcommand(self) -> None:
        song = self.temporary_path / "song.flac"
        song.write_bytes(b"dummy")
        json_out = self.temporary_path / "lyrics.json"
        with (
            patch("sonora.cli.audio.process_track_lyrics") as mock_lyrics,
            patch("sonora.cli.audio.read_track_metadata") as mock_read,
            patch("sonora.cli.audio.write_track_metadata") as mock_write,
        ):
            mock_read.return_value = TrackInfo(
                file_path=song, artist="Artist", title="Title"
            )
            mock_lyrics.return_value = ("[00:01.00] Line", "synced")
            exit_code = main(
                ["lyrics", str(self.temporary_path), "--json", str(json_out)]
            )
            self.assertEqual(exit_code, 0)
            mock_read.assert_called_once()
            mock_lyrics.assert_called_once()
            mock_write.assert_called_once()
            self.assertTrue(json_out.exists())

    def test_handle_keyboard_interrupts(self) -> None:
        dummy_backup = self.temporary_path / "backup.json"
        dummy_backup.write_text("{}", encoding="utf-8")
        song = self.temporary_path / "song.flac"
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
                "sonora.cli.library.backup_library_tags",
                side_effect=KeyboardInterrupt,
            ),
            patch(
                "sonora.cli.library.restore_library_tags",
                side_effect=KeyboardInterrupt,
            ),
            patch(
                "sonora.cli.library.normalize_library",
                side_effect=KeyboardInterrupt,
            ),
            patch(
                "sonora.cli.audio.calculate_album_replaygain",
                side_effect=KeyboardInterrupt,
            ),
            patch(
                "sonora.cli.audio.as_completed",
                side_effect=KeyboardInterrupt,
            ),
        ):
            for args in [
                ["tag", str(self.temporary_path)],
                ["check", str(self.temporary_path)],
                ["rename", str(self.temporary_path)],
                ["organize", str(self.temporary_path)],
                ["backup", str(self.temporary_path)],
                ["restore", str(dummy_backup)],
                ["normalize", str(self.temporary_path)],
                ["bpm", str(self.temporary_path), "--force"],
                ["key", str(self.temporary_path), "--force"],
                ["replaygain", str(self.temporary_path), "--force"],
                ["lyrics", str(self.temporary_path), "--force"],
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
        self.assertEqual(report["statistics"]["enrichment"]["genre_percentage"], 99.97)
        rows = summary.call_args.args[1]
        self.assertIn(("Genre & Styles Tagged", "6507/6509 (99.97%)", None), rows)


if __name__ == "__main__":
    unittest.main()


class TestReportProvenance(IsolatedTestCase):
    def test_unrelated_comments_do_not_count_as_genius_metadata(self) -> None:
        sources: tuple[MetadataSource | None, ...] = (
            None,
            "MusicBrainz",
            "TheAudioDB",
        )
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
        self.assertEqual(report["schema"], "tag_report_v1")
        self.assertTrue(report["execution"]["dry_run"])
        enrichment = report["statistics"]["enrichment"]
        self.assertEqual(enrichment["genius_matched_count"], 1)
        self.assertEqual(enrichment["genius_percentage"], 50.0)
        self.assertEqual(enrichment["theaudiodb_matched_count"], 1)
        self.assertEqual(
            set(enrichment),
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
        self.assertIn(("Tracks with Genius Metadata", "1/2 (50.00%)", None), rows)
        self.assertEqual(rows[0], ("Total Tracks Processed", "2", None))
        self.assertEqual(rows[-1][0], "Audio Quality Breakdown")
