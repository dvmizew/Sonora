import dataclasses
from types import SimpleNamespace
from unittest.mock import patch

from sonora.core.models import CheckReport, TrackInfo
from sonora.modules import checker
from sonora.modules.checker import check_file
from tests.support import IsolatedTestCase


class TestCheckerContracts(IsolatedTestCase):
    def test_supplied_reports_count_worker_outcomes_without_false_issue_categories(
        self,
    ) -> None:
        for issues, recording, expected_corrupt in (
            ([], dataclasses.replace(self.recording, track_number=None), 0),
            (["Metadata read error: broken"], None, 0),
            (["Checksum check failed: corrupted stream"], None, 1),
        ):
            report = CheckReport()
            with (
                self.subTest(issues=issues),
                patch.object(
                    checker,
                    "_check_single_file",
                    return_value=(
                        self.path,
                        issues,
                        recording,
                        {checker._IssueCategory.CORRUPT} if expected_corrupt else set(),
                    ),
                ),
            ):
                self.assertIs(
                    checker.check_library(self.test_directory, report=report), report
                )
            self.assertEqual(report.total_files, 1)
            self.assertEqual(
                (report.corrupt_files, report.missing_metadata, report.missing_lrc),
                (expected_corrupt, 0, 0),
            )
            self.assertEqual(report.issues.get(str(self.path), []), issues)

    def test_valid_pictures_and_audio_do_not_report_corruption(
        self,
    ) -> None:
        path = self.path.with_suffix(".flac")
        path.touch()
        for is_lossless in (False, True):
            with (
                self.subTest(is_lossless=is_lossless),
                patch.object(checker, "verify_flac_checksum", return_value=True),
                patch.object(
                    checker,
                    "FLAC",
                    return_value=SimpleNamespace(
                        pictures=[SimpleNamespace(data=b"image")]
                    ),
                ),
                patch.object(
                    checker, "detect_fake_lossless", return_value=(False, 0, None)
                ) as spectral,
            ):
                issues = checker.check_file(
                    path,
                    check_spectral=True,
                    track_metadata=dataclasses.replace(
                        self.recording, is_lossless=is_lossless
                    ),
                )
            self.assertFalse(
                any(
                    "picture" in issue
                    or "checksum" in issue
                    or "fake lossless" in issue
                    for issue in issues
                )
            )
            self.assertEqual(spectral.call_count, int(is_lossless))

    def test_missing_track_numbers_do_not_count_as_extreme_positions(self) -> None:
        folder_issues: list[str] = []
        checker._check_folder_alien_tracks(
            self.test_directory,
            [
                (self.path, self.recording),
                (
                    self.path.with_name("missing.wav"),
                    dataclasses.replace(self.recording, track_number=None),
                ),
            ],
            "Record",
            "Primary",
            {},
            self.test_directory,
            CheckReport(),
            folder_issues,
        )
        self.assertEqual(folder_issues, [])

    def test_generic_containers_do_not_require_album_sequence(self) -> None:
        folder = self.test_directory / "Singles"
        folder.mkdir()
        path = folder / "recording.wav"
        path.touch()
        with patch.object(
            checker,
            "read_track_metadata",
            return_value=TrackInfo(
                path, title="Recording", artist="Primary", track_number=3
            ),
        ):
            report = checker.check_library(folder)
        self.assertEqual(report.total_files, 1)
        self.assertNotIn(str(folder), report.issues)

    def setUp(self) -> None:
        self.path = self.test_directory / "01 - Recording.wav"
        self.path.touch()
        self.recording = TrackInfo(
            self.path,
            artist="Primary",
            title="Recording",
            album="Record",
            album_artist="Primary",
            track_number=1,
            genre="Rock",
        )

    def test_placeholder_artist_does_not_become_a_folder_consensus(self) -> None:
        report = CheckReport()
        folder_issues: list[str] = []
        recording = dataclasses.replace(
            self.recording, artist="Unknown Artist", album_artist=None
        )
        self.assertEqual(
            checker._check_folder_album_consistency(
                self.test_directory,
                [(self.path, recording)],
                self.test_directory,
                report,
                folder_issues,
            ),
            ("Record", None),
        )

    def test_unreadable_worker_metadata_is_reported_without_a_false_track(self) -> None:
        with patch.object(
            checker, "read_track_metadata", side_effect=OSError("broken audio")
        ):
            path, issues, recording, _ = checker._check_single_file(self.path)
        self.assertEqual(path, self.path)
        self.assertIsNone(recording)
        self.assertTrue(issues)

    def test_displaced_tracks_and_position_collisions_are_reported_once(self) -> None:
        recordings = [
            dataclasses.replace(
                self.recording,
                file_path=self.path.with_name(f"{number}.wav"),
                track_number=number,
            )
            for number in (1, 2, 3)
        ]
        displaced = dataclasses.replace(
            self.recording,
            file_path=self.path.with_name("15.wav"),
            track_number=15,
            album="Other Record",
            artist="Unrelated",
        )
        recordings.append(displaced)
        report = CheckReport()
        folder_issues: list[str] = []
        tracks = [(recording.file_path, recording) for recording in recordings]
        for _ in range(2):
            checker._check_folder_alien_tracks(
                self.test_directory,
                tracks,
                "Record",
                "Primary",
                {(1, 15): ["15.wav", "other.wav"]},
                self.test_directory / "different-root",
                report,
                folder_issues,
            )
        issues = report.issues[str(displaced.file_path)]
        self.assertTrue(any("Displaced" in issue for issue in issues), issues)
        self.assertTrue(any("Alien single" in issue for issue in issues), issues)
        self.assertEqual(len(issues), len(set(issues)))
        self.assertEqual(len(folder_issues), len(issues))

    def test_bonus_titles_do_not_trigger_extreme_position_warnings(self) -> None:
        recordings = [
            dataclasses.replace(self.recording, track_number=number)
            for number in (1, 2, 3, 15)
        ]
        recordings[-1].title = "Recording (Bonus)"
        report = CheckReport()
        folder_issues: list[str] = []
        checker._check_folder_alien_tracks(
            self.test_directory,
            [(recording.file_path, recording) for recording in recordings],
            "Record",
            "Primary",
            {},
            self.test_directory,
            report,
            folder_issues,
        )
        self.assertEqual(folder_issues, [])

    def test_library_reports_missing_sequence_in_the_album_folder(
        self,
    ) -> None:
        recording = dataclasses.replace(self.recording, track_number=3)
        with patch.object(checker, "read_track_metadata", return_value=recording):
            report = checker.check_library(self.test_directory)
        self.assertEqual(report.total_files, 1)
        self.assertTrue(
            any(
                "Missing track numbers" in issue
                for issue in report.issues[str(self.test_directory)]
            )
        )

    def test_missing_and_unsupported_files_have_explicit_outcomes(self) -> None:
        self.assertEqual(
            checker.check_file(self.path.with_name("missing.wav")),
            [f"File not found: {self.path.with_name('missing.wav')}"],
        )
        unsupported = self.path.with_suffix(".txt")
        unsupported.touch()
        self.assertEqual(checker.check_file(unsupported), [])

    def test_checksum_picture_and_spectral_corruption_are_reported(self) -> None:
        path = self.path.with_suffix(".flac")
        path.touch()
        with (
            patch.object(checker, "verify_flac_checksum", return_value=False),
            patch.object(
                checker,
                "FLAC",
                return_value=SimpleNamespace(pictures=[SimpleNamespace(data=b"")]),
            ),
            patch.object(
                checker, "detect_fake_lossless", return_value=(True, 16, None)
            ),
        ):
            issues = checker.check_file(
                path, check_spectral=True, track_metadata=self.recording
            )
        self.assertIn(
            "FLAC audio stream MD5 checksum verification failed (corrupted FLAC).",
            issues,
        )
        self.assertIn("Corrupt 0-byte picture block at index 0.", issues)
        self.assertIn("Possible fake lossless (spectral cutoff below 16kHz).", issues)
        with (
            patch.object(
                checker, "verify_flac_checksum", side_effect=OSError("flac unavailable")
            ),
            patch.object(checker, "FLAC", side_effect=OSError("bad stream")),
            patch.object(
                checker, "detect_fake_lossless", side_effect=OSError("bad stream")
            ),
        ):
            issues = checker.check_file(
                path, check_spectral=True, track_metadata=self.recording
            )
        self.assertIn("Checksum check failed: flac unavailable", issues)

    def test_invalid_metadata_features_and_low_bitrate_are_actionable(self) -> None:
        path = self.test_directory / "01 - Recording (feat. Guest).mp3"
        path.touch()
        recording = dataclasses.replace(
            self.recording,
            file_path=path,
            artist="Primary / Guest",
            album_artist=None,
            title="Recording (feat. Guest)",
            genre="unknown",
            musicbrainz_trackid="invalid",
            initial_key="invalid",
            art_width=100,
            art_height=100,
            sample_rate=22050,
            bitrate=128,
            is_lossless=False,
        )
        issues = checker.check_file(path, track_metadata=recording)
        for fragment in (
            "Blacklisted genre",
            "Invalid INITIALKEY",
            "Invalid UUID",
            "Low resolution cover",
            "seems unsplit",
            "TITLE tag",
            "Filename contains 'feat'",
            "Sub-standard sample rate",
            "Sub-standard lossy bitrate",
        ):
            with self.subTest(fragment=fragment):
                self.assertTrue(any(fragment in issue for issue in issues), issues)

    def test_supplied_metadata_and_unreadable_companions_keep_issue_categories(
        self,
    ) -> None:
        self.path.with_suffix(".lrc").touch()
        issues = checker.check_file(self.path, track_metadata=self.recording)
        self.assertIn("Corrupt 0-byte synchronized lyrics (.lrc) file.", issues)
        self.assertNotIn("Missing synchronized lyrics (.lrc) file.", issues)
        artist_feature = dataclasses.replace(
            self.recording, artist="Primary feat. Guest"
        )
        issues = checker.check_file(self.path, track_metadata=artist_feature)
        self.assertTrue(any("ARTIST entry" in issue for issue in issues))
        self.assertFalse(any("seems unsplit" in issue for issue in issues))

    def test_folder_consensus_flags_specific_outliers_and_empty_input(self) -> None:
        report = CheckReport()
        folder_issues: list[str] = []
        self.assertEqual(
            checker._check_folder_album_consistency(
                self.test_directory, [], self.test_directory, report, folder_issues
            ),
            (None, None),
        )
        recordings = [
            self.recording,
            dataclasses.replace(
                self.recording, file_path=self.path.with_name("02 - Recording.wav")
            ),
            dataclasses.replace(
                self.recording,
                file_path=self.path.with_name("03 - Other.wav"),
                album="Other Record",
                album_artist="Other",
            ),
            dataclasses.replace(
                self.recording,
                file_path=self.path.with_name("04 - Missing.wav"),
                album="Unknown Album",
                album_artist=None,
            ),
        ]
        album, artist = checker._check_folder_album_consistency(
            self.test_directory,
            [(recording.file_path, recording) for recording in recordings],
            self.test_directory / "different-root",
            report,
            folder_issues,
        )
        self.assertEqual((album, artist), ("Record", "Primary"))
        outlier = report.issues[str(recordings[2].file_path)]
        self.assertTrue(any("Mismatched ALBUM tag" in issue for issue in outlier))
        self.assertTrue(any("Mismatched ALBUMARTIST" in issue for issue in outlier))
        self.assertNotIn(str(recordings[3].file_path), report.issues)

    def test_worker_programming_failures_are_not_masked_as_audio_issues(self) -> None:
        with (
            patch.object(
                checker, "_check_single_file", side_effect=RuntimeError("checker bug")
            ),
            self.assertRaisesRegex(RuntimeError, "checker bug"),
        ):
            checker.check_library(self.test_directory)


class TestCheckerMetadataValidation(IsolatedTestCase):
    def test_conflicting_lyrics_headers_and_timestamps_are_reported(self) -> None:
        audio_path = self.test_directory / "01 - Song.mp3"
        audio_path.touch()
        track = TrackInfo(
            audio_path,
            artist="Artist",
            title="Song",
            duration=60,
            lyrics="[ar:Artist]\n[ti:Unrelated Song]\n[00:01.00] words",
        )
        audio_path.with_suffix(".lrc").write_text(
            "[ar:Artist]\n[ti:Song]\n[02:00.00] words"
        )
        issues = check_file(audio_path, track_metadata=track)
        self.assertIn(
            "Embedded lyrics headers conflict with ARTIST or TITLE; verify recording identity.",
            issues,
        )
        self.assertIn(
            "Lyrics timestamps exceed audio duration: '01 - Song.lrc'.", issues
        )

    def test_matching_and_headerless_lyrics_are_not_reported_as_mismatches(
        self,
    ) -> None:
        audio_path = self.test_directory / "01 - Song.mp3"
        audio_path.touch()
        for lyrics in (
            "[ar:Artist]\n[ti:Song]\n[00:01.00] words",
            "[00:01.00] words",
            "",
        ):
            with self.subTest(lyrics=lyrics):
                track = TrackInfo(
                    audio_path,
                    artist="Artist",
                    title="Song",
                    duration=60,
                    lyrics=lyrics,
                )
                audio_path.with_suffix(".lrc").write_text(lyrics or "[00:01.00] words")
                issues = check_file(audio_path, track_metadata=track)
                self.assertFalse(
                    any(
                        "headers conflict" in issue or "timestamps exceed" in issue
                        for issue in issues
                    )
                )

    def test_invalid_utf8_lyrics_are_reported(self) -> None:
        audio_path = self.test_directory / "01 - Song.mp3"
        audio_path.touch()
        audio_path.with_suffix(".lrc").write_bytes(b"\xff\xfe")
        issues = check_file(audio_path, track_metadata=TrackInfo(audio_path))
        self.assertTrue(
            any("Cannot read synchronized lyrics" in issue for issue in issues)
        )

    def test_positions_exceeding_declared_release_totals_are_reported(self) -> None:
        audio_path = self.test_directory / "01 - Song.mp3"
        audio_path.touch()
        track = TrackInfo(
            audio_path, track_number=14, total_tracks=12, disc_number=2, total_discs=1
        )
        issues = check_file(audio_path, track_metadata=track)
        self.assertIn(
            "TRACKNUMBER 14 exceeds TRACKTOTAL 12; verify the release edition.", issues
        )
        self.assertIn(
            "DISCNUMBER 2 exceeds DISCTOTAL 1; verify the release edition.", issues
        )

    def test_consistent_missing_and_pregap_positions_are_not_out_of_bounds(
        self,
    ) -> None:
        audio_path = self.test_directory / "01 - Song.mp3"
        audio_path.touch()
        for position, total in ((None, None), (1, None), (None, 12), (12, 12), (0, 12)):
            with self.subTest(position=position, total=total):
                track = dataclasses.replace(
                    TrackInfo(audio_path),
                    track_number=position,
                    total_tracks=total,
                    disc_number=1,
                    total_discs=1,
                )
                issues = check_file(audio_path, track_metadata=track)
                self.assertFalse(any("exceeds" in issue for issue in issues))


class TestCheckerIssueCategories(IsolatedTestCase):
    def setUp(self) -> None:
        self.path = self.test_directory / "01 - Recording.wav"
        self.path.touch()
        self.recording = TrackInfo(
            self.path,
            artist="Primary",
            title="Recording",
            album="Record",
            album_artist="Primary",
            track_number=1,
        )

    def test_empty_artwork_and_lyrics_count_corrupt_files_without_parsing_messages(
        self,
    ) -> None:
        audio_path = self.test_directory / "01 - Recording.flac"
        audio_path.touch()
        lyrics_path = audio_path.with_suffix(".lrc")
        for picture_bytes, lyrics_text, expected_corrupt in (
            (b"", "[00:01.00]Words", 1),
            (b"picture", "", 1),
            (b"picture", "[00:01.00]Words", 0),
            (b"", "", 1),
        ):
            lyrics_path.write_text(lyrics_text)
            with (
                self.subTest(picture=picture_bytes, lyrics=lyrics_text),
                patch.object(checker, "find_audio_files", return_value=[audio_path]),
                patch.object(
                    checker, "read_track_metadata", return_value=self.recording
                ),
                patch.object(checker, "verify_flac_checksum", return_value=True),
                patch.object(
                    checker,
                    "FLAC",
                    return_value=SimpleNamespace(
                        pictures=[SimpleNamespace(data=picture_bytes)]
                    ),
                ),
            ):
                report = checker.check_library(self.test_directory)
            self.assertEqual(report.corrupt_files, expected_corrupt)
            self.assertEqual(report.missing_lrc, 0)

    def test_missing_lyrics_are_distinct_from_corruption(self) -> None:
        with patch.object(checker, "read_track_metadata", return_value=self.recording):
            report = checker.check_library(self.test_directory)
        self.assertEqual(report.missing_lrc, 1)
        self.assertEqual(report.corrupt_files, 0)
