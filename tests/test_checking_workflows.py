from pathlib import Path
from typing import Any
from unittest.mock import patch

from sonora.core.models import TrackInfo
from sonora.core.text import is_valid_uuid
from sonora.modules.checker import (
    check_brackets_corruption,
    check_file,
    check_library,
)
from tests.fixtures import write_silent_wav
from tests.module_support import ModuleTestCase


class TestCheckingWorkflows(ModuleTestCase):
    def test_check_brackets_corruption(self) -> None:
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

    @patch("sonora.modules.checker.read_track_metadata")
    def test_check_library(self, mock_read: Any) -> None:
        audio_file_1 = self.tmp_path / "song.wav"
        write_silent_wav(audio_file_1)

        mock_read.return_value = TrackInfo(
            file_path=audio_file_1, artist="Artist [Official]", title="Title"
        )

        report = check_library(self.tmp_path, output_json=self.tmp_path / "report.json")
        self.assertEqual(report.total_files, 1)
        self.assertTrue((self.tmp_path / "report.json").exists())

    @patch("sonora.modules.checker.read_track_metadata")
    def test_check_library_multi_file_and_folder_issues(self, mock_read: Any) -> None:
        album_dir = self.tmp_path / "AlbumFolder"
        album_dir.mkdir(parents=True, exist_ok=True)
        file1 = album_dir / "01 - Song 1.wav"
        file2 = album_dir / "02 - Song 2.wav"
        write_silent_wav(file1)
        write_silent_wav(file2)

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

    @patch("sonora.modules.checker.read_track_metadata")
    def test_check_library_album_tag_mismatch(self, mock_read: Any) -> None:
        album_dir = self.tmp_path / "Artist - Album"
        album_dir.mkdir(parents=True, exist_ok=True)
        file1 = album_dir / "01 - Track 1.wav"
        file2 = album_dir / "02 - Track 2.wav"
        file3 = album_dir / "03 - Track 3.wav"
        write_silent_wav(file1)
        write_silent_wav(file2)
        write_silent_wav(file3)

        def _side_effect(path: Path) -> TrackInfo:
            if path.name == "03 - Track 3.wav":
                return TrackInfo(
                    file_path=path,
                    artist="Artist",
                    title="Track 3",
                    album="Album (Deluxe Edition)",
                    album_artist="Artist",
                    track_number=3,
                    disc_number=1,
                )
            num = 1 if "01" in path.name else 2
            return TrackInfo(
                file_path=path,
                artist="Artist",
                title=f"Track {num}",
                album="Album",
                album_artist="Artist",
                track_number=num,
                disc_number=1,
            )

        mock_read.side_effect = _side_effect
        report = check_library(album_dir)
        file3_issues = report.issues.get(str(file3), [])
        self.assertTrue(any("Mismatched ALBUM tag" in issue for issue in file3_issues))
        file1_issues = report.issues.get(str(file1), [])
        self.assertFalse(any("Mismatched ALBUM tag" in issue for issue in file1_issues))

    @patch("sonora.modules.checker.read_track_metadata")
    def test_check_library_alien_track_detection(self, mock_read: Any) -> None:
        album_dir = self.tmp_path / "Main Artist - Main Album"
        album_dir.mkdir(parents=True, exist_ok=True)
        file1 = album_dir / "01 - Track 1.wav"
        file2 = album_dir / "02 - Track 2.wav"
        alien_file = album_dir / "09 - Alien Song.wav"
        write_silent_wav(file1)
        write_silent_wav(file2)
        write_silent_wav(alien_file)

        def _side_effect(path: Path) -> TrackInfo:
            if path.name == "09 - Alien Song.wav":
                return TrackInfo(
                    file_path=path,
                    artist="Foreign Artist",
                    title="Alien Song",
                    album="Foreign Album",
                    album_artist="Foreign Artist",
                    track_number=9,
                    disc_number=1,
                )
            num = 1 if "01" in path.name else 2
            return TrackInfo(
                file_path=path,
                artist="Main Artist",
                title=f"Track {num}",
                album="Main Album",
                album_artist="Main Artist",
                track_number=num,
                disc_number=1,
            )

        mock_read.side_effect = _side_effect
        report = check_library(album_dir)
        alien_issues = report.issues.get(str(alien_file), [])
        self.assertTrue(
            any(
                "Alien track detected in album folder" in issue
                for issue in alien_issues
            )
        )

    @patch("sonora.modules.checker.read_track_metadata")
    def test_check_library_alien_single_duplicate_position(
        self, mock_read: Any
    ) -> None:
        album_dir = self.tmp_path / "Artist - Album"
        album_dir.mkdir(parents=True, exist_ok=True)
        file1 = album_dir / "01 - Album Track.wav"
        file2 = album_dir / "01 - Loose Single.wav"
        file3 = album_dir / "02 - Second Track.wav"
        write_silent_wav(file1)
        write_silent_wav(file2)
        write_silent_wav(file3)

        def _side_effect(path: Path) -> TrackInfo:
            if path.name == "01 - Loose Single.wav":
                return TrackInfo(
                    file_path=path,
                    artist="Artist",
                    title="Loose Single",
                    album="Loose Single",
                    album_artist="Artist",
                    track_number=1,
                    disc_number=1,
                )
            num = 1 if "01" in path.name else 2
            return TrackInfo(
                file_path=path,
                artist="Artist",
                title=f"Album Track {num}",
                album="Album",
                album_artist="Artist",
                track_number=num,
                disc_number=1,
            )

        mock_read.side_effect = _side_effect
        report = check_library(album_dir)
        single_issues = report.issues.get(str(file2), [])
        self.assertTrue(
            any(
                "Alien single detected in album folder" in issue
                for issue in single_issues
            )
        )

    @patch("sonora.modules.checker.read_track_metadata")
    def test_check_library_compilation_shielded_from_alien(
        self, mock_read: Any
    ) -> None:
        album_dir = self.tmp_path / "Various Artists - OST"
        album_dir.mkdir(parents=True, exist_ok=True)
        file1 = album_dir / "01 - Song by A.wav"
        file2 = album_dir / "02 - Song by B.wav"
        write_silent_wav(file1)
        write_silent_wav(file2)

        def _side_effect(path: Path) -> TrackInfo:
            if "01" in path.name:
                return TrackInfo(
                    file_path=path,
                    artist="Artist A",
                    title="Song A",
                    album="Movie Soundtrack",
                    album_artist="Various Artists",
                    track_number=1,
                    disc_number=1,
                )
            return TrackInfo(
                file_path=path,
                artist="Artist B",
                title="Song B",
                album="Movie Soundtrack",
                album_artist="Various Artists",
                track_number=2,
                disc_number=1,
            )

        mock_read.side_effect = _side_effect
        report = check_library(album_dir)
        for issues in report.issues.values():
            self.assertFalse(any("Alien track detected" in issue for issue in issues))

    def test_check_brackets_year_tolerance(self) -> None:
        self.assertEqual(check_brackets_corruption("Rebirth (2016)"), [])
        self.assertEqual(check_brackets_corruption("Marine Parade (2013)"), [])
        corrupt_issues = check_brackets_corruption("Song (2011 Remaster)")
        self.assertTrue(len(corrupt_issues) > 0)
        self.assertIn("Corrupt bracket metadata", corrupt_issues[0])

    @patch("sonora.modules.checker.verify_flac_checksum")
    @patch("sonora.modules.checker.read_track_metadata")
    def test_check_unsplit_collaborative_artist_subset(
        self, mock_read: Any, mock_checksum: Any
    ) -> None:
        mock_checksum.return_value = True
        wav_path = self.tmp_path / "03 - Ric Flair Drip.wav"
        write_silent_wav(wav_path)
        mock_read.return_value = TrackInfo(
            file_path=wav_path,
            artist="Offset & Metro Boomin",
            album_artist="21 Savage, Offset & Metro Boomin",
            title="Ric Flair Drip",
            album="Without Warning",
            track_number=3,
            disc_number=1,
            musicbrainz_trackid="11111111-2222-3333-4444-555555555555",
            musicbrainz_albumid="66666666-7777-8888-9999-000000000000",
            date="2017",
            bpm=134.0,
            replaygain_track_gain=-7.8,
            replaygain_track_peak=1.0,
        )
        issues = check_file(wav_path)
        self.assertFalse(any("ARTIST tag seems unsplit" in issue for issue in issues))

    @patch("sonora.modules.checker.verify_flac_checksum")
    @patch("sonora.modules.checker.read_track_metadata")
    def test_check_unsplit_collaborative_track_with_primary_artist(
        self, mock_read: Any, mock_checksum: Any
    ) -> None:
        mock_checksum.return_value = True
        wav_path = self.tmp_path / "03 - Stai.wav"
        write_silent_wav(wav_path)
        mock_read.return_value = TrackInfo(
            file_path=wav_path,
            artist="Killa Fonic & Nane",
            album_artist="Killa Fonic",
            title="Stai",
            album="III",
            track_number=3,
            disc_number=1,
            musicbrainz_trackid="11111111-2222-3333-4444-555555555555",
            musicbrainz_albumid="66666666-7777-8888-9999-000000000000",
            date="2019",
            bpm=120.0,
            replaygain_track_gain=-6.5,
            replaygain_track_peak=0.98,
        )
        issues = check_file(wav_path)
        self.assertFalse(any("ARTIST tag seems unsplit" in issue for issue in issues))

    @patch("sonora.modules.checker.verify_flac_checksum")
    @patch("sonora.modules.checker.read_track_metadata")
    def test_check_feat_in_artist_no_double_unsplit_warning(
        self, mock_read: Any, mock_checksum: Any
    ) -> None:
        mock_checksum.return_value = True
        wav_path = self.tmp_path / "04 - Mr. Right Now.wav"
        write_silent_wav(wav_path)
        mock_read.return_value = TrackInfo(
            file_path=wav_path,
            artist="21 Savage & Metro Boomin feat. Drake",
            album_artist="21 Savage & Metro Boomin",
            title="Mr. Right Now",
            album="Savage Mode II",
            track_number=4,
            disc_number=1,
            musicbrainz_trackid="11111111-2222-3333-4444-555555555555",
            musicbrainz_albumid="66666666-7777-8888-9999-000000000000",
            date="2020",
            bpm=130.0,
            replaygain_track_gain=-7.0,
            replaygain_track_peak=0.99,
        )
        issues = check_file(wav_path)
        self.assertTrue(any("contains 'feat' info" in issue for issue in issues))
        self.assertFalse(any("ARTIST tag seems unsplit" in issue for issue in issues))

    @patch("sonora.modules.checker.verify_flac_checksum")
    @patch("sonora.modules.checker.read_track_metadata")
    def test_check_unsplit_delimiter_true_positive(
        self, mock_read: Any, mock_checksum: Any
    ) -> None:
        mock_checksum.return_value = True
        wav_path = self.tmp_path / "01 - Intro.wav"
        write_silent_wav(wav_path)
        mock_read.return_value = TrackInfo(
            file_path=wav_path,
            artist="Artist 1 / Artist 2",
            album_artist="Various Artists",
            title="Intro",
            album="Compilation",
            track_number=1,
            disc_number=1,
            musicbrainz_trackid="11111111-2222-3333-4444-555555555555",
            musicbrainz_albumid="66666666-7777-8888-9999-000000000000",
            date="2024",
            bpm=120.0,
            replaygain_track_gain=-7.0,
            replaygain_track_peak=0.99,
        )
        issues = check_file(wav_path)
        self.assertTrue(any("ARTIST tag seems unsplit" in issue for issue in issues))

    @patch("sonora.modules.checker.verify_flac_checksum")
    @patch("sonora.modules.checker.read_track_metadata")
    def test_check_diacritic_collaborative_track_alignment(
        self, mock_read: Any, mock_checksum: Any
    ) -> None:
        mock_checksum.return_value = True
        wav_path = self.tmp_path / "01 - LVL UP.wav"
        write_silent_wav(wav_path)
        mock_read.return_value = TrackInfo(
            file_path=wav_path,
            artist="Nane & Amuly",
            album_artist="Nané",
            title="LVL UP",
            album="AVRAM",
            track_number=1,
            disc_number=1,
            musicbrainz_trackid="11111111-2222-3333-4444-555555555555",
            musicbrainz_albumid="66666666-7777-8888-9999-000000000000",
            date="2023",
            bpm=130.0,
            replaygain_track_gain=-6.0,
            replaygain_track_peak=0.98,
        )
        issues = check_file(wav_path)
        self.assertFalse(any("ARTIST tag seems unsplit" in issue for issue in issues))

    @patch("sonora.modules.checker.verify_flac_checksum")
    @patch("sonora.modules.checker.read_track_metadata")
    def test_check_collective_compilation_joint_billing_conjunction(
        self, mock_read: Any, mock_checksum: Any
    ) -> None:
        mock_checksum.return_value = True
        wav_path = self.tmp_path / "02 - CHAMPAIN & VACAY.wav"
        write_silent_wav(wav_path)
        mock_read.return_value = TrackInfo(
            file_path=wav_path,
            artist="Travis Scott & Don Toliver",
            album_artist="JACKBOYS",
            title="CHAMPAIN & VACAY",
            album="JACKBOYS 2",
            track_number=2,
            disc_number=1,
            musicbrainz_trackid="11111111-2222-3333-4444-555555555555",
            musicbrainz_albumid="66666666-7777-8888-9999-000000000000",
            date="2024",
            bpm=125.0,
            replaygain_track_gain=-6.8,
            replaygain_track_peak=0.97,
        )
        issues = check_file(wav_path)
        self.assertFalse(any("ARTIST tag seems unsplit" in issue for issue in issues))

    def test_check_file_blacklisted_genre(self) -> None:
        wav = self.tmp_path / "song.wav"
        write_silent_wav(wav)

        with patch("sonora.modules.checker.read_track_metadata") as mock_read:
            mock_read.return_value = TrackInfo(
                file_path=wav, artist="Artist", title="Title", genre="Top 40 Pop"
            )
            issues = check_file(wav)
            self.assertTrue(any("Blacklisted genre" in issue for issue in issues))

    def test_check_library_nonexistent_directory(self) -> None:
        with self.assertRaises(FileNotFoundError):
            check_library(self.tmp_path / "nonexistent_dir_999")

    @patch("sonora.modules.checker.detect_fake_lossless")
    @patch("sonora.modules.checker.verify_flac_checksum")
    @patch("sonora.modules.checker.read_track_metadata")
    def test_check_library_spectral_check_option(
        self, mock_read: Any, mock_checksum: Any, mock_spectral: Any
    ) -> None:
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

    def test_is_valid_uuid_multivalue(self) -> None:
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
