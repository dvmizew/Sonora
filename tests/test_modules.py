import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image, ImageDraw

from sonora.audio.art import check_image_similarity, process_artist_artwork
from sonora.audio.cuesheet import parse_cuesheet, read_cuesheet_content
from sonora.audio.metadata import read_track_metadata, write_track_metadata
from sonora.core.models import TrackInfo
from sonora.core.utils import (
    RateLimiter,
    find_audio_files,
    is_valid_uuid,
    resolve_artist_name,
)
from sonora.modules.backup import backup_library_tags, restore_library_tags
from sonora.modules.checker import (
    check_brackets_corruption,
    check_file,
    check_library,
)
from sonora.modules.organizer import (
    is_single_folder,
    organize_library_singles,
)
from sonora.modules.renamer import (
    rename_album_folder,
    rename_directory_files,
    rename_track_file,
    sync_lrc_metadata,
)
from sonora.modules.tagger import (
    process_single_track,
    tag_album_folder,
)
from sonora.services.theaudiodb import fetch_artist_images


def create_dummy_wav(path: Path) -> None:
    """Create a minimal valid 44.1kHz mono WAV file."""
    sample_rate = 44100
    num_samples = 44100 // 10  # 0.1s
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)
    data_size = num_samples * block_align
    chunk_size = 36 + data_size

    header = bytearray()
    header.extend(b"RIFF")
    header.extend(chunk_size.to_bytes(4, "little"))
    header.extend(b"WAVEfmt ")
    header.extend((16).to_bytes(4, "little"))  # Subchunk1Size
    header.extend((1).to_bytes(2, "little"))  # AudioFormat (PCM)
    header.extend(num_channels.to_bytes(2, "little"))
    header.extend(sample_rate.to_bytes(4, "little"))
    header.extend(byte_rate.to_bytes(4, "little"))
    header.extend(block_align.to_bytes(2, "little"))
    header.extend(bits_per_sample.to_bytes(2, "little"))
    header.extend(b"data")
    header.extend(data_size.to_bytes(4, "little"))
    header.extend(b"\x00" * data_size)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file_handle:
        file_handle.write(header)


class TestCoreModules(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        RateLimiter.set_disabled(True)

    @classmethod
    def tearDownClass(cls) -> None:
        RateLimiter.set_disabled(False)

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        patcher = patch("sonora.services.theaudiodb.get_cached_api", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.tmp_dir.cleanup()

    @patch("sonora.core.utils.musicbrainzngs.search_artists")
    def test_resolve_artist_name(self, mock_search):
        resolve_artist_name.cache_clear()
        mock_search.return_value = {
            "artist-list": [
                {
                    "name": "M.G.L.",
                    "alias-list": [{"alias": "mgl"}],
                },
                {
                    "name": "Killa Fonic",
                    "alias-list": [],
                },
                {
                    "name": "Nane",
                    "alias-list": [],
                },
            ]
        }
        self.assertEqual(resolve_artist_name("mgl"), "M.G.L.")
        self.assertEqual(resolve_artist_name("killa fonic"), "Killa Fonic")
        self.assertEqual(resolve_artist_name("nane"), "Nane")

    def test_check_brackets_corruption(self):
        issues = check_brackets_corruption("Song Title [Official Video]")
        self.assertTrue(any("Corrupt bracket" in issue for issue in issues))

        no_issues = check_brackets_corruption("Song Title (Remix)")
        self.assertEqual(no_issues, [])

        # Ensure artist names containing keywords as substrings (e.g. Trippie -> rip, Claudio -> audio, MHD -> hd) are NOT flagged
        self.assertEqual(check_brackets_corruption("Candy (feat. Trippie Redd)"), [])
        self.assertEqual(
            check_brackets_corruption("Que Dieu me pardonne (feat. Claudio Capéo)"), []
        )
        self.assertEqual(check_brackets_corruption("Versus (feat. MHD)"), [])
        self.assertEqual(
            check_brackets_corruption("MODERN JAM (feat. Teezo Touchdown)"), []
        )

    def test_sync_lrc_metadata(self):
        lrc_file = self.tmp_path / "test.lrc"
        lrc_file.write_text("[00:10.00] Line 1\n", encoding="utf-8")

        success = sync_lrc_metadata(lrc_file, "Artist Name", "Track Title")
        self.assertTrue(success)

        content = lrc_file.read_text(encoding="utf-8")
        self.assertIn("[ar:Artist Name]", content)
        self.assertIn("[ti:Track Title]", content)
        self.assertIn("[00:10.00] Line 1", content)

    def test_sync_lrc_metadata_existing_headers(self):
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
    def test_rename_track_file(self, mock_read):
        wav_file = self.tmp_path / "old_name.wav"
        create_dummy_wav(wav_file)

        mock_read.return_value = TrackInfo(
            file_path=wav_file, artist="Beyoncé", title="Halo", track_number=1
        )

        new_path = rename_track_file(wav_file)
        self.assertTrue(new_path.exists())
        self.assertEqual(new_path.name, "01 - Halo.wav")

    @patch("sonora.modules.renamer.read_track_metadata")
    def test_rename_directory_files(self, mock_read):
        audio_file_1 = self.tmp_path / "song1.wav"
        audio_file_2 = self.tmp_path / "song2.wav"
        create_dummy_wav(audio_file_1)
        create_dummy_wav(audio_file_2)

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
    def test_rename_multiple_album_directories(self, mock_read):
        album1 = self.tmp_path / "folder_one"
        album2 = self.tmp_path / "folder_two"
        album1.mkdir()
        album2.mkdir()

        t1 = album1 / "track1.wav"
        t2 = album2 / "track2.wav"
        create_dummy_wav(t1)
        create_dummy_wav(t2)

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

    def test_rename_album_folder(self):
        folder = self.tmp_path / "old_folder"
        folder.mkdir()
        new_folder = rename_album_folder(folder, "21 Savage", "Issa Album")
        self.assertTrue(new_folder.exists())
        self.assertEqual(new_folder.name, "21 Savage - Issa Album")

    def test_is_single_folder(self):
        album_dir = self.tmp_path / "album"
        album_dir.mkdir()

        audio_file_1 = album_dir / "01.wav"
        audio_file_2 = album_dir / "02.wav"
        audio_file_3 = album_dir / "03.wav"
        create_dummy_wav(audio_file_1)
        create_dummy_wav(audio_file_2)
        create_dummy_wav(audio_file_3)

        with patch("sonora.modules.organizer.read_track_metadata") as mock_read:
            mock_read.side_effect = [
                TrackInfo(file_path=audio_file_1, album="Same Album"),
                TrackInfo(file_path=audio_file_2, album="Same Album"),
                TrackInfo(file_path=audio_file_3, album="Same Album"),
            ]
            self.assertFalse(is_single_folder(album_dir))

    @patch("sonora.modules.organizer.read_track_metadata")
    def test_organize_library_singles(self, mock_read):
        src_dir = self.tmp_path / "source"
        target_dir = self.tmp_path / "Singles"
        audio_file_1 = src_dir / "single.wav"
        create_dummy_wav(audio_file_1)

        mock_read.return_value = TrackInfo(
            file_path=audio_file_1, artist="Single Artist", title="Single Song"
        )

        moved = organize_library_singles(src_dir, target_dir)
        self.assertEqual(moved, 1)
        self.assertTrue(
            (target_dir / "Single Artist" / "Single Artist - Single Song.wav").exists()
        )

    @patch("sonora.modules.organizer.read_track_metadata")
    def test_organize_library_singles_with_lrc(self, mock_read):
        src_dir = self.tmp_path / "source_lrc"
        target_dir = self.tmp_path / "Singles"
        audio_file_1 = src_dir / "single.wav"
        lrc = src_dir / "single.lrc"
        create_dummy_wav(audio_file_1)
        lrc.write_text("[00:01.00] lyrics", encoding="utf-8")

        mock_read.return_value = TrackInfo(
            file_path=audio_file_1, artist="Single Artist", title="Single Song"
        )

        moved = organize_library_singles(src_dir, target_dir)
        self.assertEqual(moved, 1)
        self.assertTrue(
            (target_dir / "Single Artist" / "Single Artist - Single Song.wav").exists()
        )
        self.assertTrue(
            (target_dir / "Single Artist" / "Single Artist - Single Song.lrc").exists()
        )

    @patch("sonora.modules.organizer.get_primary_artist", side_effect=lambda a: a)
    @patch("sonora.modules.organizer.read_track_metadata")
    def test_organize_multiple_folders_and_albums(self, mock_read, mock_primary):
        src_dir = self.tmp_path / "multi_source"
        folder1 = src_dir / "folder1"
        folder2 = src_dir / "album_folder"
        folder3 = src_dir / "folder3"

        f1_track = folder1 / "f1.wav"
        f2_track1 = folder2 / "f2_1.wav"
        f2_track2 = folder2 / "f2_2.wav"
        f2_track3 = folder2 / "f2_3.wav"
        f3_track = folder3 / "f3.wav"

        for f in [f1_track, f2_track1, f2_track2, f2_track3, f3_track]:
            create_dummy_wav(f)

        def mock_read_metadata(path: Path) -> TrackInfo:
            if path == f1_track:
                return TrackInfo(
                    file_path=f1_track,
                    artist="Artist One",
                    title="Track One",
                    album="Single",
                )
            if path == f2_track1:
                return TrackInfo(
                    file_path=f2_track1,
                    artist="Album Artist",
                    title="Song A",
                    album="Full Album",
                )
            if path == f2_track2:
                return TrackInfo(
                    file_path=f2_track2,
                    artist="Album Artist",
                    title="Song B",
                    album="Full Album",
                )
            if path == f2_track3:
                return TrackInfo(
                    file_path=f2_track3,
                    artist="Album Artist",
                    title="Song C",
                    album="Full Album",
                )
            if path == f3_track:
                return TrackInfo(
                    file_path=f3_track,
                    artist="Artist Three",
                    title="Track Three",
                    album="Single",
                )
            return TrackInfo(file_path=path)

        mock_read.side_effect = mock_read_metadata
        target_dir = self.tmp_path / "Singles"

        moved = organize_library_singles(src_dir, target_dir)
        self.assertEqual(moved, 2)
        self.assertTrue(
            (target_dir / "Artist One" / "Artist One - Track One.wav").exists()
        )
        self.assertTrue(
            (target_dir / "Artist Three" / "Artist Three - Track Three.wav").exists()
        )
        # Album folder should remain intact
        self.assertTrue(f2_track1.exists())
        self.assertTrue(f2_track2.exists())
        self.assertTrue(f2_track3.exists())

    @patch("sonora.modules.checker.read_track_metadata")
    def test_check_library(self, mock_read):
        audio_file_1 = self.tmp_path / "song.wav"
        create_dummy_wav(audio_file_1)

        mock_read.return_value = TrackInfo(
            file_path=audio_file_1, artist="Artist [Official]", title="Title"
        )

        report = check_library(self.tmp_path, output_json=self.tmp_path / "report.json")
        self.assertEqual(report.total_files, 1)
        self.assertTrue((self.tmp_path / "report.json").exists())

    @patch("sonora.modules.checker.read_track_metadata")
    def test_check_library_multi_file_and_folder_issues(self, mock_read):
        album_dir = self.tmp_path / "AlbumFolder"
        album_dir.mkdir(parents=True, exist_ok=True)
        file1 = album_dir / "01 - Song 1.wav"
        file2 = album_dir / "02 - Song 2.wav"
        create_dummy_wav(file1)
        create_dummy_wav(file2)

        def _side_effect(path: Path) -> TrackInfo:
            if path.name == "01 - Song 1.wav":
                return TrackInfo(
                    file_path=path,
                    artist="Artist",
                    title="Song 1",
                    album="Album A",
                    album_artist="Artist",
                    track_number=1,
                    disc_number=1,
                )
            return TrackInfo(
                file_path=path,
                artist="Artist",
                title="Song 2",
                album="Album B",
                album_artist="Artist",
                track_number=1,
                disc_number=1,
            )

        mock_read.side_effect = _side_effect

        report = check_library(album_dir)
        self.assertEqual(report.total_files, 2)
        # Should detect inconsistent album name and duplicate track number 1 on Disc 1
        folder_issues = report.issues.get(str(album_dir), [])
        self.assertTrue(any("Inconsistent ALBUM" in issue for issue in folder_issues))
        self.assertTrue(
            any("Duplicate track number" in issue for issue in folder_issues)
        )

    @patch("sonora.modules.tagger.write_track_metadata")
    @patch("sonora.services.lyrics.fetch_synced_lyrics")
    @patch("sonora.modules.tagger.fetch_track_mbid")
    @patch("sonora.modules.tagger.read_track_metadata")
    def test_process_single_track(
        self,
        mock_read,
        mock_mbid,
        mock_lyrics,
        mock_write,
    ):
        wav_file = self.tmp_path / "song.wav"
        create_dummy_wav(wav_file)

        mock_read.return_value = TrackInfo(
            file_path=wav_file, artist="nane", title="Piesa"
        )
        mock_mbid.return_value = "c8b03190-306c-4125-9b32-3f9d86d60a12"
        mock_lyrics.return_value = "[00:01.00] Vers"

        with (
            patch("sonora.modules.tagger.process_album_cover_art", return_value=None),
            patch(
                "sonora.modules.tagger.fetch_theaudiodb_track_details",
                return_value=None,
            ),
            patch("sonora.modules.tagger.fetch_genius_song_details", return_value=None),
            patch(
                "sonora.modules.tagger.fetch_deezer_track_details", return_value=None
            ),
            patch(
                "sonora.modules.tagger.fetch_deezer_album_details", return_value=None
            ),
            patch(
                "sonora.modules.tagger.fetch_musicbrainz_recording_details",
                return_value={
                    "track_number": 3,
                    "date": "2021-04-30",
                    "isrc": "ROGRA2101575",
                    "advisory": "Explicit",
                },
            ),
            patch(
                "sonora.modules.tagger.search_musicbrainz_release", return_value=None
            ),
            patch("sonora.modules.tagger.resolve_artist_name", return_value="Nane"),
            patch(
                "sonora.modules.tagger.fetch_itunes_track_metadata", return_value=None
            ),
        ):
            info = process_single_track(wav_file, fetch_bpm=False)

        self.assertEqual(info.artist, "Nane")
        self.assertEqual(
            info.musicbrainz_trackid, "c8b03190-306c-4125-9b32-3f9d86d60a12"
        )
        mock_write.assert_called_once()

    @patch("sonora.modules.tagger.process_single_track")
    def test_tag_album_folder(self, mock_process):
        album_dir = self.tmp_path / "album"
        audio_file_1 = album_dir / "01.wav"
        audio_file_2 = album_dir / "02.wav"
        create_dummy_wav(audio_file_1)
        create_dummy_wav(audio_file_2)

        mock_process.side_effect = [
            TrackInfo(file_path=audio_file_1, artist="Artist", title="T1"),
            TrackInfo(file_path=audio_file_2, artist="Artist", title="T2"),
        ]

        with (
            patch(
                "sonora.modules.tagger.search_musicbrainz_release", return_value=None
            ),
            patch(
                "sonora.modules.tagger.fetch_deezer_album_details", return_value=None
            ),
        ):
            results = tag_album_folder(album_dir, max_threads=2)
        self.assertEqual(len(results), 2)

    def test_check_file_blacklisted_genre(self):
        wav = self.tmp_path / "song.wav"
        create_dummy_wav(wav)

        with patch("sonora.modules.checker.read_track_metadata") as mock_read:
            mock_read.return_value = TrackInfo(
                file_path=wav, artist="Artist", title="Title", genre="Top 40 Pop"
            )
            issues = check_file(wav)
            self.assertTrue(any("Blacklisted genre" in issue for issue in issues))

    def test_is_single_folder_empty_dir(self):
        empty_dir = self.tmp_path / "empty"
        empty_dir.mkdir()
        self.assertFalse(is_single_folder(empty_dir))

    def test_check_library_nonexistent_directory(self):
        with self.assertRaises(FileNotFoundError):
            check_library(self.tmp_path / "nonexistent_dir_999")

    def test_tag_album_folder_nonexistent_directory(self):
        with self.assertRaises(FileNotFoundError):
            tag_album_folder(self.tmp_path / "nonexistent_dir_999")

    @patch("sonora.modules.renamer.read_track_metadata")
    def test_rename_track_file_collision_handling(self, mock_read):
        audio_file_1 = self.tmp_path / "song1.wav"
        audio_file_2 = self.tmp_path / "song2.wav"
        audio_file_3 = self.tmp_path / "song3.wav"
        create_dummy_wav(audio_file_1)
        create_dummy_wav(audio_file_2)
        create_dummy_wav(audio_file_3)

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

    def test_organize_library_singles_skips_album_folders(self):
        album_dir = self.tmp_path / "AlbumFolder"
        album_dir.mkdir()
        audio_file_1 = album_dir / "01.wav"
        audio_file_2 = album_dir / "02.wav"
        audio_file_3 = album_dir / "03.wav"
        create_dummy_wav(audio_file_1)
        create_dummy_wav(audio_file_2)
        create_dummy_wav(audio_file_3)

        target_dir = self.tmp_path / "Singles"

        with patch("sonora.modules.organizer.read_track_metadata") as mock_read:
            mock_read.return_value = TrackInfo(
                file_path=audio_file_1, artist="Artist", album="Full Album"
            )
            moved = organize_library_singles(self.tmp_path, target_dir)
            self.assertEqual(moved, 0)
            self.assertTrue(audio_file_1.exists())
            self.assertTrue(audio_file_2.exists())
            self.assertTrue(audio_file_3.exists())

    @patch("sonora.modules.tagger.write_track_metadata")
    @patch("sonora.modules.tagger.search_discogs_release")
    @patch("sonora.modules.tagger.lookup_acoustid")
    @patch("sonora.modules.tagger.fetch_track_mbid")
    @patch("sonora.modules.tagger.read_track_metadata")
    def test_process_single_track_acoustid_discogs_fallback(
        self,
        mock_read,
        mock_mbid,
        mock_acoustid,
        mock_discogs,
        mock_write,
    ):
        wav_file = self.tmp_path / "song.wav"
        create_dummy_wav(wav_file)

        mock_read.return_value = TrackInfo(
            file_path=wav_file, artist="Artist", title="Title", genre=None
        )
        mock_mbid.return_value = None
        mock_acoustid.return_value = "c8b03190-306c-4125-9b32-3f9d86d60a12"
        mock_discogs.return_value = {"id": 123, "year": 2024}

        with (
            patch("sonora.modules.tagger.process_album_cover_art", return_value=None),
            patch(
                "sonora.modules.tagger.fetch_theaudiodb_track_details",
                return_value=None,
            ),
            patch("sonora.modules.tagger.fetch_genius_song_details", return_value=None),
            patch(
                "sonora.modules.tagger.fetch_deezer_track_details", return_value=None
            ),
            patch(
                "sonora.modules.tagger.fetch_deezer_album_details", return_value=None
            ),
            patch(
                "sonora.modules.tagger.fetch_musicbrainz_recording_details",
                return_value=None,
            ),
            patch(
                "sonora.modules.tagger.search_musicbrainz_release", return_value=None
            ),
            patch(
                "sonora.modules.tagger.resolve_artist_name",
                side_effect=lambda x: str(x).strip(),
            ),
            patch(
                "sonora.modules.tagger.fetch_itunes_track_metadata", return_value=None
            ),
        ):
            info = process_single_track(
                wav_file,
                fetch_bpm=False,
                fetch_lyrics=False,
                fetch_itunes_art=False,
                acoustid_api_key="acoustid_key",
                discogs_user_token="discogs_token",
            )

        self.assertEqual(
            info.musicbrainz_trackid, "c8b03190-306c-4125-9b32-3f9d86d60a12"
        )
        self.assertTrue(str(info.date).startswith("2024"))
        mock_acoustid.assert_called_once()
        mock_discogs.assert_called_once()
        mock_write.assert_called_once()

    @patch("sonora.modules.checker.detect_fake_lossless")
    @patch("sonora.modules.checker.verify_flac_checksum")
    @patch("sonora.modules.checker.read_track_metadata")
    def test_check_library_spectral_check_option(
        self, mock_read, mock_checksum, mock_spectral
    ):
        flac_file = self.tmp_path / "song.flac"
        flac_file.write_bytes(b"FLAC dummy content")

        mock_checksum.return_value = True
        mock_read.return_value = TrackInfo(
            file_path=flac_file, artist="Artist", title="Title"
        )
        mock_spectral.return_value = (
            True,
            0.0001,
            "Brickwall spectral cutoff detected at ~16-18kHz (likely upscaled 128-192kbps MP3 fake lossless)",
        )

        report = check_library(self.tmp_path, check_spectral=True)
        self.assertTrue(
            any(
                "fake lossless" in issue.lower()
                for issues in report.issues.values()
                for issue in issues
            )
        )
        mock_spectral.assert_called_once()

    def test_symfonium_extended_tags(self):
        wav_path = self.tmp_path / "extended.wav"
        create_dummy_wav(wav_path)

        info = read_track_metadata(wav_path)
        info.artist_sort = "Artist, Test"
        info.album_artist_sort = "Artist, Test Album"
        info.total_tracks = 12
        info.total_discs = 2
        info.release_type = "Album"
        info.release_status = "Official"
        info.release_country = "US"
        info.musicbrainz_trackid = "c8b03190-306c-4125-9b32-3f9d86d60a12"
        info.musicbrainz_albumid = "a1b2c3d4-e5f6-4a5b-8c9d-0e1f2a3b4c5d"
        info.label = "Test Label"
        info.barcode = "123456789012"

        write_track_metadata(info)

        reloaded = read_track_metadata(wav_path)
        self.assertEqual(
            reloaded.musicbrainz_trackid, "c8b03190-306c-4125-9b32-3f9d86d60a12"
        )
        self.assertEqual(
            reloaded.musicbrainz_albumid, "a1b2c3d4-e5f6-4a5b-8c9d-0e1f2a3b4c5d"
        )

    def test_image_similarity(self):
        # Image 1: White background with black square in top-left
        image_1 = Image.new("RGB", (64, 64), color="white")
        draw_1 = ImageDraw.Draw(image_1)
        draw_1.rectangle([5, 5, 25, 25], fill="black")

        # Image 2: Exact copy of image 1
        image_2 = Image.new("RGB", (64, 64), color="white")
        draw_2 = ImageDraw.Draw(image_2)
        draw_2.rectangle([5, 5, 25, 25], fill="black")

        # Image 3: Completely distinct pattern (large filled ellipse in bottom-right)
        image_3 = Image.new("RGB", (64, 64), color="white")
        draw_3 = ImageDraw.Draw(image_3)
        draw_3.ellipse([30, 30, 60, 60], fill="black")

        buffer_1, buffer_2, buffer_3 = io.BytesIO(), io.BytesIO(), io.BytesIO()

        image_1.save(buffer_1, format="JPEG")
        image_2.save(buffer_2, format="JPEG")
        image_3.save(buffer_3, format="JPEG")

        image_bytes_1 = buffer_1.getvalue()
        image_bytes_2 = buffer_2.getvalue()
        image_bytes_3 = buffer_3.getvalue()

        self.assertTrue(
            check_image_similarity(image_bytes_1, image_bytes_2, threshold=0.8)
        )
        self.assertFalse(
            check_image_similarity(image_bytes_1, image_bytes_3, max_distance=6)
        )
        self.assertFalse(check_image_similarity(b"", image_bytes_2))

    def test_cuesheet_parsing(self):
        cue_path = self.tmp_path / "test.cue"
        cue_path.write_text(
            'REM GENRE "Hip-Hop"\n'
            "REM DATE 2021\n"
            "REM DISCNUMBER 1\n"
            "REM TOTALDISCS 2\n"
            'PERFORMER "Album Artist"\n'
            'TITLE "Album Title"\n'
            "TRACK 01 AUDIO\n"
            '  TITLE "Track One"\n'
            '  PERFORMER "Track Artist"\n'
            '  SONGWRITER "Composer Name"\n'
            "  ISRC USUM71805166\n"
            "  INDEX 00 00:00:00\n"
            "  INDEX 01 00:02:00\n",
            encoding="latin-1",
        )

        tracks = parse_cuesheet(cue_path)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]["track_number"], 1)
        self.assertEqual(tracks[0]["title"], "Track One")
        self.assertEqual(tracks[0]["artist"], "Track Artist")
        self.assertEqual(tracks[0]["genre"], "Hip-Hop")
        self.assertEqual(tracks[0]["date"], "2021")
        self.assertEqual(tracks[0]["disc_number"], 1)
        self.assertEqual(tracks[0]["total_discs"], 2)
        self.assertEqual(tracks[0]["composer"], "Composer Name")
        self.assertEqual(tracks[0]["isrc"], "USUM71805166")
        self.assertEqual(tracks[0]["start_index"], "00:02:00")
        self.assertEqual(tracks[0]["pregap_index"], "00:00:00")

        content = read_cuesheet_content(cue_path)
        self.assertIsNotNone(content)
        self.assertIn('TITLE "Track One"', content or "")

    def test_backup_and_restore(self):
        wav_path = self.tmp_path / "song.wav"
        create_dummy_wav(wav_path)

        initial_info = read_track_metadata(wav_path)
        initial_info.artist = "Test Artist"
        write_track_metadata(initial_info)

        backup_file = self.tmp_path / "backup.json"
        backup_library_tags(self.tmp_path, output_file=backup_file)
        self.assertTrue(backup_file.exists())

        # Modify track info
        info = read_track_metadata(wav_path)
        info.artist = "Modified Artist"
        write_track_metadata(info)

        reloaded = read_track_metadata(wav_path)
        self.assertEqual(reloaded.artist, "Modified Artist")

        # Restore
        restored_count = restore_library_tags(backup_file)
        self.assertEqual(restored_count, 1)

        restored_info = read_track_metadata(wav_path)
        self.assertEqual(restored_info.artist, "Test Artist")

    def test_backup_and_restore_gzipped(self):
        wav_path = self.tmp_path / "song_gz.wav"
        create_dummy_wav(wav_path)

        initial_info = read_track_metadata(wav_path)
        initial_info.artist = "Gzip Artist"
        write_track_metadata(initial_info)

        gz_backup = self.tmp_path / "backup.json.gz"
        backup_library_tags(self.tmp_path, output_file=gz_backup)
        self.assertTrue(gz_backup.exists())

        # Modify
        initial_info.artist = "Changed Artist"
        write_track_metadata(initial_info)

        # Restore from gzip
        restored = restore_library_tags(gz_backup)
        self.assertGreaterEqual(restored, 1)

        reloaded = read_track_metadata(wav_path)
        self.assertEqual(reloaded.artist, "Gzip Artist")

    def test_backup_and_restore_portable_relocation(self):
        old_dir = self.tmp_path / "old_location"
        old_dir.mkdir()
        wav_path = old_dir / "portable.wav"
        create_dummy_wav(wav_path)

        info = read_track_metadata(wav_path)
        info.artist = "Original Artist"
        write_track_metadata(info)

        backup_file = self.tmp_path / "portable_backup.json"
        backup_library_tags(old_dir, output_file=backup_file)

        # Now move the audio file to a brand new folder (simulating disk move/NAS mount change)
        new_dir = self.tmp_path / "new_location"
        new_dir.mkdir()
        new_wav = new_dir / "portable.wav"
        wav_path.rename(new_wav)

        # Modify tags in new location
        mod_info = read_track_metadata(new_wav)
        mod_info.artist = "Altered Artist"
        write_track_metadata(mod_info)

        # Restore pointing target_directory to new_dir
        restored = restore_library_tags(backup_file, target_directory=new_dir)
        self.assertEqual(restored, 1)

        final_info = read_track_metadata(new_wav)
        self.assertEqual(final_info.artist, "Original Artist")

    @patch("sonora.services.theaudiodb.SESSION.get")
    def test_theaudiodb_service(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "artists": [
                {
                    "strArtistThumb": "http://example.com/thumb.jpg",
                    "strArtistBanner": "http://example.com/banner.jpg",
                }
            ]
        }
        mock_image_response = MagicMock()
        mock_image_response.status_code = 200
        mock_image_response.content = b"fakeimage"

        mock_get.side_effect = [
            mock_response,
            mock_image_response,
            mock_image_response,
            mock_response,
            mock_image_response,
            mock_image_response,
        ]

        thumbnail_bytes, banner_bytes = fetch_artist_images("21 Savage")
        self.assertEqual(thumbnail_bytes, b"fakeimage")
        self.assertEqual(banner_bytes, b"fakeimage")

        artist_folder = self.tmp_path / "21 Savage" / "Album"
        artist_folder.mkdir(parents=True, exist_ok=True)
        process_artist_artwork(artist_folder, "21 Savage")
        self.assertTrue((self.tmp_path / "21 Savage" / "artist.jpg").exists())
        self.assertTrue((self.tmp_path / "21 Savage" / "banner.jpg").exists())

    def test_find_audio_files_ignores_hidden_directories(self):
        normal_wav = self.tmp_path / "normal.wav"
        create_dummy_wav(normal_wav)

        hidden_dir = self.tmp_path / ".venv" / "lib"
        hidden_dir.mkdir(parents=True, exist_ok=True)
        hidden_wav = hidden_dir / "test.wav"
        create_dummy_wav(hidden_wav)

        dot_wav = self.tmp_path / "._hidden.flac"
        dot_wav.write_bytes(b"dummy")

        files = find_audio_files(self.tmp_path)
        self.assertEqual(files, [normal_wav])

        # When include_hidden=True, hidden files are included
        all_files = find_audio_files(self.tmp_path, include_hidden=True)
        self.assertIn(hidden_wav, all_files)
        self.assertIn(dot_wav, all_files)

    def test_is_valid_uuid_multivalue(self):
        uuid1 = "1c59ae05-207b-4fbd-9ee9-569489af6121"
        uuid2 = "bd81ebc9-d1c4-4dc3-b48e-718bdc5fde50"

        # Single UUID
        self.assertTrue(is_valid_uuid(uuid1))
        self.assertFalse(is_valid_uuid("invalid-uuid"))

        # Multiple UUIDs with allow_multivalue
        self.assertTrue(is_valid_uuid(f"{uuid1}; {uuid2}", allow_multivalue=True))
        self.assertTrue(is_valid_uuid(f"{uuid1} / {uuid2}", allow_multivalue=True))
        self.assertTrue(is_valid_uuid(f"{uuid1}, {uuid2}", allow_multivalue=True))
        self.assertFalse(is_valid_uuid(f"{uuid1}; invalid", allow_multivalue=True))


if __name__ == "__main__":
    unittest.main()
