import importlib
import json
import runpy
import signal
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

from sonora.cli import audio, cache, library
from sonora.core.cancellation import InterruptedOperationError
from sonora.core.models import CheckReport, NormalizeReport, RenameReport, TrackInfo
from tests.support import IsolatedTestCase

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

    def test_normalize_results_and_partial_reports_preserve_counts(
        self,
    ) -> None:
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
        with (
            patch.object(library, "tag_album_folder", return_value=[]) as tag,
        ):
            self.assertEqual(
                entrypoint.main(["tag", str(self.test_directory), *flags, "--no-dsp"]),
                0,
            )
        self.assertEqual(tag.call_args.kwargs["acoustid_api_key"], "acoustid")
        self.assertEqual(tag.call_args.kwargs["lastfm_api_key"], "lastfm")
        self.assertEqual(tag.call_args.kwargs["genius_api_token"], "genius")
        self.assertEqual(tag.call_args.kwargs["musixmatch_token"], "musixmatch")


class TestAudioCommandFailureContracts(IsolatedTestCase):
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
        with (
            patch.object(library, "tag_album_folder", return_value=[]) as tag,
        ):
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
                audio._process_lyrics_file(path, True, False),
                (path, "Plain lyrics", "plain"),
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
                self.assertEqual(cache._format_file_size(size), expected)
