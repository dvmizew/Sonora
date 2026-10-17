import dataclasses
from unittest.mock import patch

from sonora.core.cancellation import InterruptedOperationError
from sonora.core.models import TrackInfo
from sonora.modules import organizer
from tests.support import IsolatedTestCase


class TestSingleDeduplicationContracts(IsolatedTestCase):
    def setUp(self) -> None:
        self.source = self.test_directory / "incoming.flac"
        self.destination = self.test_directory / "existing.flac"
        self.source.write_bytes(b"incoming recording")
        self.destination.write_bytes(b"existing recording")
        self.quarantine = self.test_directory / ".duplicates"
        self.recording = TrackInfo(
            self.source,
            artist="Primary",
            title="Recording",
            album="Record",
            is_lossless=True,
            bits_per_sample=16,
            sample_rate=44100,
        )

    def test_distinct_musicbrainz_recordings_are_not_duplicates(self) -> None:
        registered = dataclasses.replace(
            self.recording,
            file_path=self.destination,
            musicbrainz_trackid="11111111-2222-3333-4444-555555555555",
        )
        deduplicator = organizer.SingleDeduplicator()
        deduplicator.register(registered)
        candidate = dataclasses.replace(
            self.recording, musicbrainz_trackid="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        )
        self.assertFalse(deduplicator.is_duplicate(candidate))

    def test_missing_registered_file_is_forgotten_before_comparing_quality(
        self,
    ) -> None:
        deduplicator = organizer.SingleDeduplicator()
        registered = dataclasses.replace(
            self.recording, file_path=self.test_directory / "removed.flac"
        )
        deduplicator.register(registered)
        target = self.test_directory / "new.flac"
        self.assertEqual(
            organizer._handle_single_deduplication(
                self.source,
                self.recording,
                target,
                "Primary - Recording",
                deduplicator,
                self.quarantine,
                False,
            ),
            (False, 0),
        )
        self.assertFalse(deduplicator.is_duplicate(self.recording))
        self.assertTrue(self.source.exists())

    def test_destination_metadata_failure_keeps_both_files(self) -> None:
        with patch.object(
            organizer, "_read_file_info", return_value=(self.destination, None)
        ):
            result = organizer._handle_single_deduplication(
                self.source,
                self.recording,
                self.destination,
                "Primary - Recording",
                organizer.SingleDeduplicator(),
                self.quarantine,
                False,
            )
        self.assertEqual(result, (False, 0))
        self.assertTrue(self.source.exists())
        self.assertTrue(self.destination.exists())

    def test_higher_quality_incoming_recording_quarantines_destination(self) -> None:
        incoming = dataclasses.replace(
            self.recording, bits_per_sample=24, sample_rate=96000
        )
        destination_metadata = dataclasses.replace(
            self.recording, file_path=self.destination
        )
        with patch.object(
            organizer,
            "_read_file_info",
            return_value=(self.destination, destination_metadata),
        ):
            result = organizer._handle_single_deduplication(
                self.source,
                incoming,
                self.destination,
                "Primary - Recording",
                organizer.SingleDeduplicator(),
                self.quarantine,
                False,
            )
        self.assertEqual(result, (False, 1))
        self.assertTrue(self.source.exists())
        self.assertFalse(self.destination.exists())
        self.assertEqual(
            (self.quarantine / self.destination.name).read_bytes(),
            b"existing recording",
        )

    def test_equal_quality_destination_quarantines_incoming_recording(self) -> None:
        destination_metadata = dataclasses.replace(
            self.recording, file_path=self.destination
        )
        with patch.object(
            organizer,
            "_read_file_info",
            return_value=(self.destination, destination_metadata),
        ):
            result = organizer._handle_single_deduplication(
                self.source,
                self.recording,
                self.destination,
                "Primary - Recording",
                organizer.SingleDeduplicator(),
                self.quarantine,
                False,
            )
        self.assertEqual(result, (True, 1))
        self.assertFalse(self.source.exists())
        self.assertTrue(self.destination.exists())
        self.assertEqual(
            (self.quarantine / self.source.name).read_bytes(), b"incoming recording"
        )

    def test_quarantine_failure_preserves_incoming_file_and_error(self) -> None:
        with (
            patch.object(
                organizer,
                "safe_case_rename",
                side_effect=PermissionError("directory locked"),
            ),
            self.assertRaises(PermissionError),
        ):
            organizer._quarantine_duplicate_single(
                self.source, "Primary - Recording", self.quarantine
            )
        self.assertEqual(self.source.read_bytes(), b"incoming recording")


class TestDirectoryConsolidationContracts(IsolatedTestCase):
    def test_unmatched_artist_tags_fall_back_to_existing_directory_branding(
        self,
    ) -> None:
        for name in ("Primary", "PRIMARY"):
            folder = self.test_directory / name
            folder.mkdir()
            (folder / "recording.flac").touch()
        with patch.object(
            organizer,
            "_read_file_info",
            side_effect=lambda path: (path, TrackInfo(path, artist="Other")),
        ):
            self.assertEqual(
                organizer._consolidate_directory_entries(
                    self.test_directory, dry_run=True
                ),
                1,
            )
        self.assertTrue((self.test_directory / "Primary").exists())
        self.assertTrue((self.test_directory / "PRIMARY").exists())

    def test_all_uppercase_variants_consolidate_and_empty_normalized_names_are_ignored(
        self,
    ) -> None:
        for name in ("PRIMARY", "PRIMARY!", "!!!"):
            (self.test_directory / name).mkdir()
        self.assertEqual(
            organizer._consolidate_directory_entries(self.test_directory, dry_run=True),
            1,
        )
        self.assertTrue((self.test_directory / "!!!").exists())

    def test_single_hierarchy_consolidates_nested_release_case_variants(self) -> None:
        singles = self.test_directory / "Primary" / "Singles"
        (singles / "Record").mkdir(parents=True)
        (singles / "RECORD").mkdir()
        self.assertEqual(
            organizer.consolidate_duplicate_artist_dirs(
                self.test_directory, dry_run=True
            ),
            1,
        )

    def test_album_evidence_and_disc_folders_are_never_reclassified_as_singles(
        self,
    ) -> None:
        paths = [self.test_directory / f"{number}.flac" for number in (1, 2, 3)]
        for name, albums, numbers, expected in (
            ("CD 2", ["Record"] * 3, [1, 2, 3], False),
            ("Record", ["(feat. Guest)"] * 3, [None] * 3, False),
            ("Record Alpha", ["Record Alpha", "Other", "Third"], [None] * 3, False),
            ("Mixed", ["First", "Second", "Third"], [None] * 3, True),
            ("Mixed", ["First", "Second", "Third"], [1, 2, 3], False),
        ):
            folder = self.test_directory / name
            recordings = [
                (path, TrackInfo(path, album=album, track_number=number))
                for path, album, number in zip(paths, albums, numbers)
            ]
            with self.subTest(name=name, albums=albums, numbers=numbers):
                self.assertEqual(
                    organizer._is_single_folder(
                        folder, paths, self.test_directory, recordings
                    ),
                    expected,
                )

    def test_quarantining_an_already_quarantined_file_is_an_identity_operation(
        self,
    ) -> None:
        quarantine = self.test_directory / ".duplicates"
        quarantine.mkdir()
        path = quarantine / "recording.flac"
        path.write_bytes(b"audio")
        path.with_suffix(".lrc").write_text("lyrics")
        self.assertEqual(organizer._quarantine_file(path, quarantine), path)
        self.assertEqual(path.read_bytes(), b"audio")
        self.assertEqual(path.with_suffix(".lrc").read_text(), "lyrics")

    def test_missing_root_and_case_variants_preserve_colliding_assets(self) -> None:
        self.assertEqual(
            organizer._consolidate_directory_entries(self.test_directory / "missing"), 0
        )
        canonical = self.test_directory / "Primary"
        duplicate = self.test_directory / "PRIMARY"
        canonical.mkdir()
        duplicate.mkdir()
        (canonical / "recording.flac").write_bytes(b"original")
        (duplicate / "recording.flac").write_bytes(b"incoming")
        (duplicate / "Nested").mkdir()
        (duplicate / "Nested" / "companion.lrc").write_text("lyrics")
        with patch.object(
            organizer, "_read_file_info", side_effect=lambda path: (path, None)
        ):
            self.assertEqual(
                organizer._consolidate_directory_entries(self.test_directory), 1
            )
        self.assertFalse(duplicate.exists())
        self.assertEqual(
            {path.read_bytes() for path in canonical.glob("*.flac")},
            {b"original", b"incoming"},
        )
        self.assertEqual((canonical / "Nested" / "companion.lrc").read_text(), "lyrics")

    def test_metadata_branding_selects_canonical_directory(self) -> None:
        first = self.test_directory / "Primary"
        second = self.test_directory / "PRIMARY"
        first.mkdir()
        second.mkdir()
        recording_path = first / "recording.flac"
        recording_path.touch()
        branded = TrackInfo(recording_path, artist="PRIMARY")
        with patch.object(
            organizer, "_read_file_info", return_value=(recording_path, branded)
        ):
            self.assertEqual(
                organizer._consolidate_directory_entries(
                    self.test_directory, dry_run=True
                ),
                1,
            )
        self.assertTrue(first.exists())
        self.assertTrue(second.exists())

    def test_consolidation_io_failure_does_not_remove_source_assets(self) -> None:
        first = self.test_directory / "Primary"
        second = self.test_directory / "PRIMARY"
        first.mkdir()
        second.mkdir()
        audio = second / "recording.flac"
        audio.write_bytes(b"recording")
        with (
            patch.object(
                organizer, "_read_file_info", side_effect=lambda path: (path, None)
            ),
            patch.object(
                organizer,
                "safe_case_rename",
                side_effect=PermissionError("directory locked"),
            ),
        ):
            self.assertEqual(
                organizer._consolidate_directory_entries(self.test_directory), 0
            )
        self.assertEqual(audio.read_bytes(), b"recording")

    def test_cleanup_preserves_editor_directories_and_nonempty_folders(self) -> None:
        for name in (".git", ".idea", ".vscode", "__pycache__"):
            (self.test_directory / name).mkdir()
        occupied = self.test_directory / "Record"
        occupied.mkdir()
        (occupied / "recording.flac").touch()
        self.assertEqual(organizer.cleanup_empty_dirs(self.test_directory), 0)
        self.assertTrue(occupied.exists())


class TestOrganizationFailureReporting(IsolatedTestCase):
    def test_existing_release_folder_only_needs_the_filename_normalized(self) -> None:
        self.path.unlink()
        folder = self.target / "Primary - Recording"
        folder.mkdir(parents=True)
        source = folder / "recording.flac"
        source.write_bytes(b"recording")
        recording = dataclasses.replace(self.recording, file_path=source)
        with patch.object(organizer, "read_track_metadata", return_value=recording):
            self.assertEqual(
                organizer.organize_library_singles(self.target, self.target), 1
            )
        self.assertEqual((folder / "01 - Recording.flac").read_bytes(), b"recording")
        self.assertFalse(source.exists())

    def test_move_phase_cancellation_carries_progress_and_programming_errors_propagate(
        self,
    ) -> None:
        for failure in (KeyboardInterrupt(), RuntimeError("target resolver bug")):
            with (
                self.subTest(failure=type(failure).__name__),
                patch.object(
                    organizer, "read_track_metadata", return_value=self.recording
                ),
                patch.object(
                    organizer, "_resolve_single_target_file", side_effect=failure
                ),
            ):
                if isinstance(failure, KeyboardInterrupt):
                    with self.assertRaises(InterruptedOperationError) as interrupted:
                        organizer.organize_library_singles(
                            self.test_directory, self.target
                        )
                    self.assertEqual(interrupted.exception.partial_result, 0)
                else:
                    with self.assertRaisesRegex(RuntimeError, "target resolver bug"):
                        organizer.organize_library_singles(
                            self.test_directory, self.target
                        )
        self.assertEqual(self.path.read_bytes(), b"recording")

    def test_duplicate_single_is_quarantined_without_overwriting_the_existing_edition(
        self,
    ) -> None:
        destination = self.target / "Primary - Recording" / "01 - Recording.flac"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"existing recording")
        recordings = {
            self.path: self.recording,
            destination: dataclasses.replace(self.recording, file_path=destination),
        }
        with patch.object(
            organizer, "read_track_metadata", side_effect=recordings.__getitem__
        ):
            self.assertEqual(
                organizer.organize_library_singles(self.test_directory, self.target), 0
            )
        self.assertEqual(destination.read_bytes(), b"existing recording")
        self.assertEqual(
            (self.target / ".duplicates" / self.path.name).read_bytes(), b"recording"
        )

    def test_optional_failure_collectors_are_not_required_for_io_recovery(self) -> None:
        with (
            patch.object(organizer, "read_track_metadata", return_value=self.recording),
            patch("sonora.modules.organizer.os.access", return_value=False),
        ):
            self.assertEqual(
                organizer.organize_library_singles(self.test_directory, self.target), 0
            )
        with (
            patch.object(organizer, "read_track_metadata", return_value=self.recording),
            patch.object(
                organizer, "safe_case_rename", side_effect=OSError("move failed")
            ),
        ):
            self.assertEqual(
                organizer.organize_library_singles(self.test_directory, self.target), 0
            )
        self.assertTrue(self.path.exists())

    def setUp(self) -> None:
        self.path = self.test_directory / "recording.flac"
        self.path.write_bytes(b"recording")
        self.target = self.test_directory / "Singles"
        self.recording = TrackInfo(
            self.path,
            artist="Primary",
            title="Recording",
            album="Record",
            total_tracks=1,
        )

    def test_missing_and_empty_libraries_have_explicit_results(self) -> None:
        with self.assertRaises(FileNotFoundError):
            organizer.organize_library_singles(
                self.test_directory / "absent", self.target
            )
        self.path.unlink()
        self.assertEqual(
            organizer.organize_library_singles(self.test_directory, self.target), 0
        )

    def test_distinct_editions_at_the_same_target_are_kept_in_separate_folders(
        self,
    ) -> None:
        destination = self.target / "Primary - Recording" / "01 - Recording.flac"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"existing edition")
        existing = dataclasses.replace(
            self.recording, file_path=destination, isrc="USAAA2100001"
        )
        incoming = dataclasses.replace(self.recording, isrc="USAAA2100002")
        recordings = {self.path: incoming, destination: existing}
        with patch.object(
            organizer, "read_track_metadata", side_effect=recordings.__getitem__
        ):
            self.assertEqual(
                organizer.organize_library_singles(self.test_directory, self.target), 1
            )
        edition = self.target / "Primary - Recording [USAAA2100002]" / destination.name
        self.assertEqual(edition.read_bytes(), b"recording")
        self.assertEqual(destination.read_bytes(), b"existing edition")

    def test_unreadable_audio_is_reported_without_moving_file(self) -> None:
        failures: list[str] = []
        with patch.object(
            organizer, "read_track_metadata", side_effect=OSError("bad audio")
        ):
            self.assertEqual(
                organizer.organize_library_singles(
                    self.test_directory, self.target, failures=failures
                ),
                0,
            )
            self.assertEqual(
                organizer.organize_library_singles(self.test_directory, self.target), 0
            )
        self.assertEqual(failures, [str(self.path)])
        self.assertTrue(self.path.exists())

    def test_permission_failure_and_move_failure_are_counted(self) -> None:
        for permitted in (True, False):
            failures: list[str] = []
            with (
                self.subTest(permitted=permitted),
                patch.object(
                    organizer, "read_track_metadata", return_value=self.recording
                ),
                patch("sonora.modules.organizer.os.access", return_value=permitted),
                patch.object(
                    organizer, "safe_case_rename", side_effect=OSError("move failed")
                ),
            ):
                self.assertEqual(
                    organizer.organize_library_singles(
                        self.test_directory, self.target, failures=failures
                    ),
                    0,
                )
            self.assertEqual(failures, [str(self.path)])
            self.assertTrue(self.path.exists())

    def test_worker_programming_failure_is_not_masked(self) -> None:
        with (
            patch.object(
                organizer, "_read_file_info", side_effect=RuntimeError("worker bug")
            ),
            self.assertRaisesRegex(RuntimeError, "worker bug"),
        ):
            organizer.organize_library_singles(self.test_directory, self.target)
