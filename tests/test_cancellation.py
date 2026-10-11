from unittest.mock import patch

from sonora.audio.replaygain import calculate_album_replaygain
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
