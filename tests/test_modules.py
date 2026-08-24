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
from sonora.modules.backup import backup_library_tags, restore_library_tags
from sonora.modules.checker import (
    check_brackets_corruption,
    check_file,
    check_library,
)
from sonora.modules.organizer import (
    get_primary_artist,
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
    normalize_artist_alias,
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
    with open(path, "wb") as f:
        f.write(header)


class TestCoreModules(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_normalize_artist_alias(self):
        self.assertEqual(normalize_artist_alias("mgl"), "M.G.L.")
        self.assertEqual(normalize_artist_alias("killa fonic"), "Killa Fonic")
        self.assertEqual(normalize_artist_alias("nane"), "Nane")

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
        f1 = self.tmp_path / "song1.wav"
        f2 = self.tmp_path / "song2.wav"
        create_dummy_wav(f1)
        create_dummy_wav(f2)

        mock_read.side_effect = [
            TrackInfo(file_path=f1, artist="Artist", title="Title 1", track_number=1),
            TrackInfo(file_path=f2, artist="Artist", title="Title 2", track_number=2),
        ]

        renamed = rename_directory_files(self.tmp_path)
        self.assertEqual(len(renamed), 2)

    def test_get_primary_artist(self):
        self.assertEqual(
            get_primary_artist("21 Savage feat. Metro Boomin"), "21 Savage"
        )
        self.assertEqual(get_primary_artist("Drake & Future"), "Drake")
        self.assertEqual(get_primary_artist("Above & Beyond"), "Above & Beyond")

    def test_rename_album_folder(self):
        folder = self.tmp_path / "old_folder"
        folder.mkdir()
        new_f = rename_album_folder(folder, "21 Savage", "Issa Album")
        self.assertTrue(new_f.exists())
        self.assertEqual(new_f.name, "21 Savage - Issa Album")

    def test_is_single_folder(self):
        album_dir = self.tmp_path / "album"
        album_dir.mkdir()

        f1 = album_dir / "01.wav"
        f2 = album_dir / "02.wav"
        f3 = album_dir / "03.wav"
        create_dummy_wav(f1)
        create_dummy_wav(f2)
        create_dummy_wav(f3)

        with patch("sonora.modules.organizer.read_track_metadata") as mock_read:
            mock_read.side_effect = [
                TrackInfo(file_path=f1, album="Same Album"),
                TrackInfo(file_path=f2, album="Same Album"),
                TrackInfo(file_path=f3, album="Same Album"),
            ]
            self.assertFalse(is_single_folder(album_dir))

    @patch("sonora.modules.organizer.read_track_metadata")
    def test_organize_library_singles(self, mock_read):
        src_dir = self.tmp_path / "source"
        target_dir = self.tmp_path / "Singles"
        f1 = src_dir / "single.wav"
        create_dummy_wav(f1)

        mock_read.return_value = TrackInfo(
            file_path=f1, artist="Single Artist", title="Single Song"
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
        f1 = src_dir / "single.wav"
        lrc = src_dir / "single.lrc"
        create_dummy_wav(f1)
        lrc.write_text("[00:01.00] lyrics", encoding="utf-8")

        mock_read.return_value = TrackInfo(
            file_path=f1, artist="Single Artist", title="Single Song"
        )

        moved = organize_library_singles(src_dir, target_dir)
        self.assertEqual(moved, 1)
        self.assertTrue(
            (target_dir / "Single Artist" / "Single Artist - Single Song.wav").exists()
        )
        self.assertTrue(
            (target_dir / "Single Artist" / "Single Artist - Single Song.lrc").exists()
        )

    @patch("sonora.modules.checker.read_track_metadata")
    def test_check_library(self, mock_read):
        f1 = self.tmp_path / "song.wav"
        create_dummy_wav(f1)

        mock_read.return_value = TrackInfo(
            file_path=f1, artist="Artist [Official]", title="Title"
        )

        report = check_library(self.tmp_path, output_json=self.tmp_path / "report.json")
        self.assertEqual(report.total_files, 1)
        self.assertTrue((self.tmp_path / "report.json").exists())

    @patch("sonora.modules.tagger.write_track_metadata")
    @patch("sonora.services.lyrics.fetch_synced_lyrics")
    @patch("sonora.modules.tagger.fetch_track_mbid")
    @patch("sonora.modules.tagger.read_track_metadata")
    def test_process_single_track(self, mock_read, mock_mbid, mock_lyrics, mock_write):
        wav_file = self.tmp_path / "song.wav"
        create_dummy_wav(wav_file)

        # Return separate instances so orig_info snapshot differs from track_info
        mock_read.side_effect = [
            TrackInfo(file_path=wav_file, artist="nane", title="Piesa"),
            TrackInfo(file_path=wav_file, artist="nane", title="Piesa"),
        ]
        mock_mbid.return_value = "c8b03190-306c-4125-9b32-3f9d86d60a12"
        mock_lyrics.return_value = "[00:01.00] Vers"

        info = process_single_track(wav_file, fetch_bpm=False)
        self.assertEqual(info.artist, "Nane")
        self.assertEqual(
            info.musicbrainz_trackid, "c8b03190-306c-4125-9b32-3f9d86d60a12"
        )
        mock_write.assert_called_once()

    @patch("sonora.modules.tagger.process_single_track")
    def test_tag_album_folder(self, mock_process):
        album_dir = self.tmp_path / "album"
        f1 = album_dir / "01.wav"
        f2 = album_dir / "02.wav"
        create_dummy_wav(f1)
        create_dummy_wav(f2)

        mock_process.side_effect = [
            TrackInfo(file_path=f1, artist="Artist", title="T1"),
            TrackInfo(file_path=f2, artist="Artist", title="T2"),
        ]

        results = tag_album_folder(album_dir, max_workers=2)
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
        f1 = self.tmp_path / "song1.wav"
        f2 = self.tmp_path / "song2.wav"
        create_dummy_wav(f1)
        create_dummy_wav(f2)

        mock_read.side_effect = [
            TrackInfo(file_path=f1, artist="Artist", title="Title", track_number=1),
            TrackInfo(file_path=f2, artist="Artist", title="Title", track_number=1),
        ]

        p1 = rename_track_file(f1)
        p2 = rename_track_file(f2)

        self.assertEqual(p1.name, "01 - Title.wav")
        self.assertEqual(p2.name, "01 - Title (2).wav")

    def test_organize_library_singles_skips_album_folders(self):
        album_dir = self.tmp_path / "AlbumFolder"
        album_dir.mkdir()
        f1 = album_dir / "01.wav"
        f2 = album_dir / "02.wav"
        f3 = album_dir / "03.wav"
        create_dummy_wav(f1)
        create_dummy_wav(f2)
        create_dummy_wav(f3)

        target_dir = self.tmp_path / "Singles"

        with patch("sonora.modules.organizer.read_track_metadata") as mock_read:
            mock_read.return_value = TrackInfo(
                file_path=f1, artist="Artist", album="Full Album"
            )
            moved = organize_library_singles(self.tmp_path, target_dir)
            self.assertEqual(moved, 0)
            self.assertTrue(f1.exists())
            self.assertTrue(f2.exists())
            self.assertTrue(f3.exists())

    @patch("sonora.modules.tagger.write_track_metadata")
    @patch("sonora.modules.tagger.search_discogs_release")
    @patch("sonora.modules.tagger.lookup_acoustid")
    @patch("sonora.modules.tagger.fetch_track_mbid")
    @patch("sonora.modules.tagger.read_track_metadata")
    def test_process_single_track_acoustid_discogs_fallback(
        self, mock_read, mock_mbid, mock_acoustid, mock_discogs, mock_write
    ):
        wav_file = self.tmp_path / "song.wav"
        create_dummy_wav(wav_file)

        mock_read.return_value = TrackInfo(
            file_path=wav_file, artist="Artist", title="Title", genre=None
        )
        mock_mbid.return_value = None
        mock_acoustid.return_value = "c8b03190-306c-4125-9b32-3f9d86d60a12"
        mock_discogs.return_value = {"id": 123, "year": 2024}

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
        self.assertEqual(info.date, "2024")
        mock_acoustid.assert_called_once()
        mock_discogs.assert_called_once()

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
        img1 = Image.new("RGB", (64, 64), color="white")
        draw1 = ImageDraw.Draw(img1)
        draw1.rectangle([5, 5, 25, 25], fill="black")

        # Image 2: Exact copy of image 1
        img2 = Image.new("RGB", (64, 64), color="white")
        draw2 = ImageDraw.Draw(img2)
        draw2.rectangle([5, 5, 25, 25], fill="black")

        # Image 3: Completely distinct pattern (large filled ellipse in bottom-right)
        img3 = Image.new("RGB", (64, 64), color="white")
        draw3 = ImageDraw.Draw(img3)
        draw3.ellipse([30, 30, 60, 60], fill="black")

        buf1, buf2, buf3 = io.BytesIO(), io.BytesIO(), io.BytesIO()

        img1.save(buf1, format="JPEG")
        img2.save(buf2, format="JPEG")
        img3.save(buf3, format="JPEG")

        b1 = buf1.getvalue()
        b2 = buf2.getvalue()
        b3 = buf3.getvalue()

        self.assertTrue(check_image_similarity(b1, b2, threshold=0.8))
        self.assertFalse(check_image_similarity(b1, b3, max_distance=6))
        self.assertFalse(check_image_similarity(b"", b2))

    def test_cuesheet_parsing(self):
        cue_path = self.tmp_path / "test.cue"
        cue_path.write_text(
            'PERFORMER "Album Artist"\n'
            'TITLE "Album Title"\n'
            "TRACK 01 AUDIO\n"
            '  TITLE "Track One"\n'
            '  PERFORMER "Track Artist"\n'
            "  INDEX 01 00:00:00\n",
            encoding="utf-8",
        )

        tracks = parse_cuesheet(cue_path)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]["track_number"], 1)
        self.assertEqual(tracks[0]["title"], "Track One")
        self.assertEqual(tracks[0]["artist"], "Track Artist")

        content = read_cuesheet_content(cue_path)
        self.assertIsNotNone(content)
        self.assertIn('TITLE "Track One"', content or "")

    def test_backup_and_restore(self):
        wav_path = self.tmp_path / "song.wav"
        create_dummy_wav(wav_path)

        info_init = read_track_metadata(wav_path)
        info_init.artist = "Test Artist"
        write_track_metadata(info_init)

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
        restored_cnt = restore_library_tags(backup_file)
        self.assertEqual(restored_cnt, 1)

        restored_info = read_track_metadata(wav_path)
        self.assertEqual(restored_info.artist, "Test Artist")

    @patch("sonora.services.theaudiodb.get_cached_api", return_value=None)
    @patch("sonora.services.theaudiodb.SESSION.get")
    def test_theaudiodb_service(self, mock_get, _mock_cache):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "artists": [
                {
                    "strArtistThumb": "http://example.com/thumb.jpg",
                    "strArtistBanner": "http://example.com/banner.jpg",
                }
            ]
        }
        mock_img_resp = MagicMock()
        mock_img_resp.status_code = 200
        mock_img_resp.content = b"fakeimage"

        mock_get.side_effect = [
            mock_resp,
            mock_img_resp,
            mock_img_resp,
            mock_resp,
            mock_img_resp,
            mock_img_resp,
        ]

        thumb, banner = fetch_artist_images("21 Savage")
        self.assertEqual(thumb, b"fakeimage")
        self.assertEqual(banner, b"fakeimage")

        artist_folder = self.tmp_path / "21 Savage" / "Album"
        artist_folder.mkdir(parents=True, exist_ok=True)
        process_artist_artwork(artist_folder, "21 Savage")
        self.assertTrue((self.tmp_path / "21 Savage" / "artist.jpg").exists())
        self.assertTrue((self.tmp_path / "21 Savage" / "banner.jpg").exists())


if __name__ == "__main__":
    unittest.main()
