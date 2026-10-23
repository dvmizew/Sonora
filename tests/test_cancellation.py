from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from unittest.mock import patch

from sonora.audio.replaygain import calculate_album_replaygain
from sonora.core.cancellation import InterruptedOperationError, is_interruption
from sonora.core.models import TrackInfo
from sonora.modules import tagger
from sonora.modules.backup import backup_library_tags, restore_library_tags
from sonora.modules.checker import (
    check_library,
)
from sonora.modules.organizer import (
    organize_library_singles,
)
from sonora.modules.renamer import (
    rename_directory_files,
)
from sonora.modules.tagger import (
    tag_album_folder,
)
from sonora.modules.track_processing import normalize_library
from tests.fixtures import write_silent_wav
from tests.module_support import OfflinePipelineTestCase
from tests.support import IsolatedTestCase


def _observe_shutdown(
    calls: list[tuple[ThreadPoolExecutor, bool, bool]],
    original: Callable[..., None],
    executor: ThreadPoolExecutor,
    wait: bool = True,
    *,
    cancel_futures: bool = False,
) -> None:
    calls.append((executor, wait, cancel_futures))
    original(executor, wait=wait, cancel_futures=cancel_futures)


class TestCancellation(OfflinePipelineTestCase):
    def test_backup_handles_keyboard_interrupt(self) -> None:
        wav = self.tmp_path / "test.wav"
        write_silent_wav(wav)
        with (
            patch(
                "sonora.modules.backup._read_track_for_backup",
                side_effect=KeyboardInterrupt,
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            backup_library_tags(self.tmp_path, max_threads=2)

    def test_restore_handles_keyboard_interrupt(self) -> None:
        backup_file = self.tmp_path / "backup.json"
        backup_file.write_text(
            '{"schema":"sonora_backup_v2","root":"/tmp","tracks":{"test.wav":{"tags":{},"complex_tags":{}}}}',
            encoding="utf-8",
        )
        with (
            patch(
                "sonora.modules.backup._restore_single_track",
                side_effect=KeyboardInterrupt,
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            restore_library_tags(backup_file, max_threads=2)

    def test_organize_handles_keyboard_interrupt(self) -> None:
        wav1 = self.tmp_path / "test1.wav"
        wav2 = self.tmp_path / "test2.wav"
        write_silent_wav(wav1)
        write_silent_wav(wav2)
        with (
            patch(
                "sonora.modules.organizer.read_track_metadata",
                side_effect=KeyboardInterrupt,
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            organize_library_singles(
                self.tmp_path, self.tmp_path / "Singles", max_threads=2
            )

    def test_renamer_handles_keyboard_interrupt(self) -> None:
        wav1 = self.tmp_path / "test1.wav"
        wav2 = self.tmp_path / "test2.wav"
        write_silent_wav(wav1)
        write_silent_wav(wav2)
        with (
            patch(
                "sonora.modules.renamer.read_track_metadata",
                side_effect=KeyboardInterrupt,
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            rename_directory_files(self.tmp_path, max_threads=2)

    def test_checker_handles_keyboard_interrupt(self) -> None:
        wav = self.tmp_path / "test.wav"
        write_silent_wav(wav)
        with (
            patch(
                "sonora.modules.checker._check_single_file",
                side_effect=KeyboardInterrupt,
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            check_library(self.tmp_path, max_threads=2)

    def test_replaygain_handles_keyboard_interrupt(self) -> None:
        wav1 = self.tmp_path / "test1.wav"
        wav2 = self.tmp_path / "test2.wav"
        write_silent_wav(wav1)
        write_silent_wav(wav2)
        with (
            patch(
                "sonora.audio.replaygain._measure_track_loudness",
                side_effect=KeyboardInterrupt,
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            calculate_album_replaygain([wav1, wav2], max_threads=2)

    def test_normalize_handles_keyboard_interrupt(self) -> None:
        wav = self.tmp_path / "test.wav"
        write_silent_wav(wav)
        with (
            patch(
                "sonora.modules.track_processing.normalize_single_track",
                side_effect=KeyboardInterrupt,
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            normalize_library(self.tmp_path, max_threads=2)

    def test_tagger_handles_keyboard_interrupt(self) -> None:
        wav = self.tmp_path / "test.wav"
        write_silent_wav(wav)
        with (
            patch(
                "sonora.modules.tagger.process_single_track",
                side_effect=KeyboardInterrupt,
            ),
            patch(
                "sonora.services.musicbrainz.search_musicbrainz_release",
                return_value=None,
            ),
            patch(
                "sonora.services.deezer.fetch_deezer_album_details", return_value=None
            ),
            patch(
                "sonora.services.itunes.fetch_itunes_album_details", return_value=None
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            tag_album_folder(
                self.tmp_path,
                max_threads=2,
                fetch_bpm=False,
                fetch_key=False,
                fetch_replaygain=False,
                fetch_lyrics=False,
                fetch_itunes_art=False,
            )


class TestCancellationChains(IsolatedTestCase):
    def test_keyboard_interrupt_and_ordinary_errors(self) -> None:
        self.assertTrue(is_interruption(KeyboardInterrupt()))
        self.assertFalse(is_interruption(RuntimeError("release unlocked lock")))
        self.assertFalse(is_interruption(RuntimeError("internal failure")))

    def test_nested_explicit_causes(self) -> None:
        worker_error = RuntimeError("worker cleanup")
        executor_error = RuntimeError("executor cleanup")
        worker_error.__cause__ = KeyboardInterrupt()
        executor_error.__cause__ = worker_error
        self.assertTrue(is_interruption(executor_error))

    def test_context_and_suppressed_context(self) -> None:
        worker_error = RuntimeError("worker cleanup")
        worker_error.__context__ = KeyboardInterrupt()
        self.assertTrue(is_interruption(worker_error))
        worker_error.__suppress_context__ = True
        self.assertFalse(is_interruption(worker_error))

    def test_cyclic_exception_chain(self) -> None:
        first_error = RuntimeError("first")
        second_error = RuntimeError("second")
        first_error.__cause__ = second_error
        second_error.__cause__ = first_error
        self.assertFalse(is_interruption(first_error))


class TestTaggingPoolCancellation(IsolatedTestCase):
    def test_interrupt_cancels_all_worker_queues_before_joining_any_pool(self) -> None:
        path = self.test_directory / "recording.flac"
        path.touch()
        recording = TrackInfo(path, artist="Alpha", title="Original Song")
        calls: list[tuple[ThreadPoolExecutor, bool, bool]] = []
        original = ThreadPoolExecutor.shutdown
        with (
            patch.object(
                tagger,
                "_resolve_album_metadata_context",
                return_value=tagger._AlbumMetadataContext(),
            ),
            patch.object(tagger, "process_single_track", return_value=recording),
            patch.object(tagger, "as_completed", side_effect=KeyboardInterrupt),
            patch.object(
                ThreadPoolExecutor,
                "shutdown",
                autospec=True,
                side_effect=partial(_observe_shutdown, calls, original),
            ),
            self.assertRaises(InterruptedOperationError),
        ):
            tagger.tag_album_folder(
                self.test_directory,
                force=True,
                dry_run=True,
                max_threads=1,
                fetch_itunes_art=False,
                fetch_replaygain=False,
            )
        self.assertEqual(
            [(wait, cancel) for _, wait, cancel in calls[:3]], [(False, True)] * 3
        )
        self.assertEqual(len({id(executor) for executor, _, _ in calls[:3]}), 3)
        self.assertTrue(all(wait and cancel for _, wait, cancel in calls[3:6]))
