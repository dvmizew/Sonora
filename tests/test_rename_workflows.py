from pathlib import Path
from typing import Any
from unittest.mock import patch

from sonora.core.models import TrackInfo
from sonora.modules.renamer import (
    rename_album_folder,
    rename_directory_files,
    rename_single_folder,
    rename_track_file,
    sync_lrc_metadata,
)
from sonora.modules.tagger import (
    _AlbumMetadataContext,
    process_single_track,
)
from tests.fixtures import write_silent_wav
from tests.module_support import ModuleTestCase


class TestRenameWorkflows(ModuleTestCase):
    def test_sync_lrc_metadata(self) -> None:
        lrc_file = self.tmp_path / "test.lrc"
        lrc_file.write_text("[00:10.00] Line 1\n", encoding="utf-8")

        success = sync_lrc_metadata(lrc_file, "Artist Name", "Track Title")
        self.assertTrue(success)

        content = lrc_file.read_text(encoding="utf-8")
        self.assertIn("[ar:Artist Name]", content)
        self.assertIn("[ti:Track Title]", content)
        self.assertIn("[00:10.00] Line 1", content)

    def test_sync_lrc_metadata_existing_headers(self) -> None:
        lrc_file = self.tmp_path / "existing.lrc"
        lrc_file.write_text(
            "[ar:Old Artist]\n[ti:Old Title]\n[00:15.00] Line 2\n", encoding="utf-8"
        )

        success = sync_lrc_metadata(lrc_file, "New Artist", "New Title")
        self.assertTrue(success)

        content = lrc_file.read_text(encoding="utf-8")
        self.assertIn("[ar:New Artist]", content)
        self.assertIn("[ti:New Title]", content)
        self.assertNotIn("[ar:Old Artist]", content)
        self.assertNotIn("[ti:Old Title]", content)
        self.assertIn("[00:15.00] Line 2", content)

    @patch("sonora.modules.renamer.read_track_metadata")
    def test_rename_track_file(self, mock_read: Any) -> None:
        wav_file = self.tmp_path / "old_name.wav"
        write_silent_wav(wav_file)

        mock_read.return_value = TrackInfo(
            file_path=wav_file, artist="Beyoncé", title="Halo", track_number=1
        )

        new_path = rename_track_file(wav_file)
        self.assertTrue(new_path.exists())
        self.assertEqual(new_path.name, "01 - Halo.wav")

    @patch("sonora.modules.renamer.read_track_metadata")
    def test_rename_directory_files(self, mock_read: Any) -> None:
        audio_file_1 = self.tmp_path / "song1.wav"
        audio_file_2 = self.tmp_path / "song2.wav"
        write_silent_wav(audio_file_1)
        write_silent_wav(audio_file_2)

        mock_read.side_effect = [
            TrackInfo(
                file_path=audio_file_1, artist="Artist", title="Title 1", track_number=1
            ),
            TrackInfo(
                file_path=audio_file_2, artist="Artist", title="Title 2", track_number=2
            ),
        ]

        renamed = rename_directory_files(self.tmp_path)
        self.assertEqual(len(renamed), 2)

    @patch("sonora.modules.renamer.read_track_metadata")
    def test_rename_multiple_album_directories(self, mock_read: Any) -> None:
        album1 = self.tmp_path / "folder_one"
        album2 = self.tmp_path / "folder_two"
        album1.mkdir()
        album2.mkdir()

        t1 = album1 / "track1.wav"
        t2 = album2 / "track2.wav"
        write_silent_wav(t1)
        write_silent_wav(t2)

        def mock_meta(path: Path) -> TrackInfo:
            if path == t1:
                return TrackInfo(
                    file_path=t1,
                    artist="Artist One",
                    album="Album One",
                    title="Song 1",
                    track_number=1,
                )
            return TrackInfo(
                file_path=t2,
                artist="Artist Two",
                album="Album Two",
                title="Song 2",
                track_number=1,
            )

        mock_read.side_effect = mock_meta
        renamed = rename_directory_files(self.tmp_path)
        self.assertEqual(len(renamed), 2)
        # Both album folders should be renamed
        self.assertTrue((self.tmp_path / "Artist One - Album One").exists())
        self.assertTrue((self.tmp_path / "Artist Two - Album Two").exists())

    def test_rename_album_folder(self) -> None:
        folder = self.tmp_path / "old_folder"
        folder.mkdir()
        new_folder = rename_album_folder(folder, "21 Savage", "Issa Album")
        self.assertTrue(new_folder.exists())
        self.assertEqual(new_folder.name, "21 Savage - Issa Album")

    def test_rename_album_folder_preserves_artist_container(self) -> None:
        artist_folder = self.tmp_path / "3 Doors Down"
        artist_folder.mkdir()
        # Case 1: Folder name matches artist, but album is different -> should not rename artist folder
        result = rename_album_folder(artist_folder, "3 Doors Down", "Away From The Sun")
        self.assertEqual(result, artist_folder)
        self.assertTrue(artist_folder.exists())

        # Case 2: Folder contains child directories that are not discs
        album_folder = self.tmp_path / "Album With Subdirs"
        album_folder.mkdir()
        (album_folder / "Singles").mkdir()
        result2 = rename_album_folder(album_folder, "Artist", "Album With Subdirs")
        self.assertEqual(result2, album_folder)

    def test_rename_single_folder(self) -> None:
        singles_dir = self.tmp_path / "Singles"
        singles_dir.mkdir()
        single_folder = singles_dir / "Alex Velea - Track 2"
        single_folder.mkdir()

        track_info = TrackInfo(
            file_path=single_folder / "01 - Minim Doi.wav",
            artist="Alex Velea",
            title="Minim Doi",
            album="Track 2",
        )
        renamed_folder = rename_single_folder(single_folder, track_info)
        self.assertTrue(renamed_folder.exists())
        self.assertEqual(renamed_folder.name, "Alex Velea - Minim Doi")

        # rename_album_folder must be preserved and not touch single folder in Singles/
        preserved = rename_album_folder(renamed_folder, "Alex Velea", "Track 2")
        self.assertEqual(preserved, renamed_folder)
        self.assertEqual(preserved.name, "Alex Velea - Minim Doi")

    @patch("sonora.modules.renamer.read_track_metadata")
    def test_rename_track_file_in_singles_forces_01(self, mock_read: Any) -> None:
        singles_folder = self.tmp_path / "Artist" / "Singles" / "Artist - Song"
        singles_folder.mkdir(parents=True)
        audio_file = singles_folder / "03 - Song.wav"
        write_silent_wav(audio_file)
        lrc_file = singles_folder / "03 - Song.lrc"
        lrc_file.write_text("[00:01.00] Test\n", encoding="utf-8")

        mock_read.return_value = TrackInfo(
            file_path=audio_file,
            artist="Artist",
            title="Song",
            track_number=3,
            disc_number=2,
            total_discs=2,
        )

        renamed_audio = rename_track_file(audio_file)
        self.assertTrue(renamed_audio.exists())
        self.assertEqual(renamed_audio.name, "01 - Song.wav")
        self.assertTrue((singles_folder / "01 - Song.lrc").exists())

    @patch("sonora.modules.renamer.read_track_metadata")
    def test_rename_directory_files_updates_corrupted_singles(
        self, mock_read: Any
    ) -> None:
        single_dir = (
            self.tmp_path
            / "Connect-R"
            / "Singles"
            / "Connect-R - Track 2 (feat. Joyce Meyer)"
        )
        single_dir.mkdir(parents=True)
        audio_file = single_dir / "03 - Tren De Noapte.wav"
        write_silent_wav(audio_file)
        lrc_file = single_dir / "03 - Tren De Noapte.lrc"
        lrc_file.write_text("[00:01.00] Lyrics\n", encoding="utf-8")

        mock_read.return_value = TrackInfo(
            file_path=audio_file,
            artist="Connect-R",
            title="Tren De Noapte",
            album="Track 2 (feat. Joyce Meyer)",
            track_number=3,
        )

        renamed = rename_directory_files(self.tmp_path)
        self.assertEqual(len(renamed), 1)

        updated_folder = (
            self.tmp_path / "Connect-R" / "Singles" / "Connect-R - Tren De Noapte"
        )
        self.assertTrue(updated_folder.exists())
        self.assertTrue((updated_folder / "01 - Tren De Noapte.wav").exists())
        self.assertTrue((updated_folder / "01 - Tren De Noapte.lrc").exists())

    @patch("sonora.audio.metadata.write_track_metadata")
    @patch("sonora.audio.metadata.read_track_metadata")
    def test_process_single_track_filename_alignment(
        self, mock_read: Any, mock_write: Any
    ) -> None:
        track_file = self.tmp_path / "02 - Track Title.wav"
        write_silent_wav(track_file)
        mock_read.return_value = TrackInfo(
            file_path=track_file,
            artist="Artist",
            title="Track Title",
            track_number=1,  # Corrupted track number from single release
            album="Great Album",
        )
        mbids = {(1, 1): "uuid-1", (1, 2): "uuid-2"}
        with (
            patch("sonora.modules.tagger._update_musicbrainz"),
            patch(
                "sonora.services.itunes.fetch_itunes_track_metadata", return_value=None
            ),
        ):
            track_metadata = process_single_track(
                track_file,
                album_context=_AlbumMetadataContext(track_mbids=mbids),
                force=True,
                fetch_bpm=False,
                fetch_itunes_art=False,
            )
            self.assertEqual(track_metadata.track_number, 2)

    @patch("sonora.modules.renamer.read_track_metadata")
    def test_rename_track_file_collision_handling(self, mock_read: Any) -> None:
        audio_file_1 = self.tmp_path / "song1.wav"
        audio_file_2 = self.tmp_path / "song2.wav"
        audio_file_3 = self.tmp_path / "song3.wav"
        write_silent_wav(audio_file_1)
        write_silent_wav(audio_file_2)
        write_silent_wav(audio_file_3)

        mock_read.side_effect = [
            TrackInfo(
                file_path=audio_file_1, artist="Artist", title="Title", track_number=1
            ),
            TrackInfo(
                file_path=audio_file_2, artist="Artist", title="Title", track_number=1
            ),
            TrackInfo(
                file_path=audio_file_3, artist="Artist", title="Title", track_number=1
            ),
        ]

        renamed_path_1 = rename_track_file(audio_file_1)
        renamed_path_2 = rename_track_file(audio_file_2)
        renamed_path_3 = rename_track_file(audio_file_3)

        self.assertEqual(renamed_path_1.name, "01 - Title.wav")
        self.assertEqual(renamed_path_2.name, "01 - Title (2).wav")
        self.assertEqual(renamed_path_3.name, "01 - Title (3).wav")
