import os
from pathlib import Path
from unittest.mock import patch

from sonora.core import utils
from sonora.core.config import SonoraConfig
from tests.support import IsolatedTestCase


class TestUtilityFileContracts(IsolatedTestCase):
    def test_configured_disc_folders_without_numbers_have_no_invented_index(
        self,
    ) -> None:
        with patch.object(
            utils,
            "get_config",
            return_value=SonoraConfig(disc_folder_patterns=("Bonus",)),
        ):
            self.assertIsNone(utils.extract_disc_number_from_folder("Bonus"))

    def test_hard_link_aliases_are_safe_rename_targets(self) -> None:
        source = self.test_directory / "recording.flac"
        source.write_bytes(b"recording")
        alternate = self.test_directory / "alternate.flac"
        alternate.hardlink_to(source)
        self.assertEqual(utils.safe_case_rename(source, alternate), alternate)
        self.assertEqual(alternate.read_bytes(), b"recording")
        other_directory = self.test_directory / "other"
        other_directory.mkdir()
        same_name = other_directory / source.name
        same_name.hardlink_to(source)
        self.assertEqual(utils.safe_case_rename(source, same_name), same_name)
        self.assertEqual(same_name.read_bytes(), b"recording")

    def test_companion_lyrics_can_be_relocated_in_a_preview(self) -> None:
        source = self.test_directory / "recording.flac"
        lyric = source.with_suffix(".lrc")
        lyric.write_text("lyrics")
        destination = self.test_directory / "renamed.flac"
        self.assertEqual(
            utils.relocate_companion_lyrics(source, destination, dry_run=True),
            [destination.with_suffix(".lrc")],
        )
        self.assertEqual(lyric.read_text(), "lyrics")

    def test_missing_directory_and_companion_lyric_search_fallbacks(self) -> None:
        self.assertEqual(utils.find_audio_files(self.test_directory / "absent"), [])
        audio = self.test_directory / "original.flac"
        lyric = self.test_directory / "02 - Different Name.lrc"
        lyric.touch()
        self.assertEqual(utils.find_companion_lyrics(audio, track_number=2), [lyric])
        self.assertEqual(
            utils.find_companion_lyrics(audio, track_number="invalid", in_singles=True),
            [lyric],
        )
        (self.test_directory / "other.lrc").touch()
        self.assertEqual(utils.find_companion_lyrics(audio, in_singles=True), [])
        with patch.object(
            Path, "iterdir", side_effect=PermissionError("locked directory")
        ):
            self.assertEqual(
                utils.find_companion_lyrics(audio, track_number=2, in_singles=True), []
            )

    def test_artwork_moves_preserve_existing_destinations_and_dry_run(self) -> None:
        source = self.test_directory / "source"
        target = self.test_directory / "target"
        source.mkdir()
        target.mkdir()
        cover = source / "cover.jpg"
        cover.write_bytes(b"artwork")
        self.assertEqual(utils.relocate_companion_artwork(source, source), [])
        self.assertEqual(
            utils.relocate_companion_artwork(source, target, dry_run=True),
            [target / "cover.jpg"],
        )
        self.assertTrue(cover.exists())
        with patch("shutil.move", side_effect=PermissionError("destination locked")):
            self.assertEqual(utils.relocate_companion_artwork(source, target), [])
        self.assertEqual(
            utils.relocate_companion_artwork(source, target), [target / "cover.jpg"]
        )
        self.assertEqual((target / "cover.jpg").read_bytes(), b"artwork")

    def test_safe_rename_handles_identity_permissions_and_collision(self) -> None:
        source = self.test_directory / "source.flac"
        source.write_bytes(b"recording")
        self.assertEqual(utils.safe_case_rename(source, source), source)
        with (
            patch.object(os, "access", return_value=False),
            self.assertRaises(PermissionError),
        ):
            utils.safe_case_rename(source, self.test_directory / "target.flac")
        destination = self.test_directory / "destination.flac"
        destination.touch()
        with self.assertRaises(FileExistsError):
            utils.safe_case_rename(source, destination)
        folder = self.test_directory / "folder"
        folder.mkdir()
        renamed = folder.with_name("renamed")
        self.assertEqual(utils.safe_case_rename(folder, renamed), renamed)
        self.assertTrue(renamed.is_dir())

    def test_missing_companions_and_existing_lyrics_do_not_overwrite(self) -> None:
        source = self.test_directory / "source.flac"
        destination = self.test_directory / "destination.flac"
        source_lrc = source.with_suffix(".lrc")
        destination_lrc = destination.with_suffix(".lrc")
        source_lrc.write_text("original")
        destination_lrc.write_text("destination")
        self.assertEqual(
            utils.relocate_companion_lyrics(
                source,
                destination,
                companions=[source_lrc, self.test_directory / "absent.lrc"],
            ),
            [],
        )
        self.assertEqual(destination_lrc.read_text(), "destination")

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
                self.assertEqual(utils.format_filesize(size), expected)
        self.assertFalse(
            utils.is_in_singles_hierarchy(
                Path("/outside/recording.flac"), self.test_directory
            )
        )
