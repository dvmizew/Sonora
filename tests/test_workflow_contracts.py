import dataclasses
from pathlib import Path
from unittest.mock import patch

from sonora.audio import art, bpm, key, metadata, replaygain
from sonora.core.models import RenameReport, TrackInfo
from sonora.modules import (
    metadata_consensus,
    normalization,
    organizer,
    renamer,
    track_processing,
)
from tests.support import IsolatedTestCase


def _change_proposal_annotations(recording: TrackInfo) -> None:
    recording.metadata_sources["title"] = "Deezer"
    recording.metadata_conflicts.clear()
    recording._diff_descriptions.append("Provider change")


class TestTrackProcessingContracts(IsolatedTestCase):
    def test_provider_proposals_do_not_share_mutable_track_state(self) -> None:
        self.recording.metadata_sources["title"] = "MusicBrainz"
        self.recording.metadata_conflicts["album"] = "Review edition"
        self.recording._diff_descriptions.append("Existing change")
        proposal = metadata_consensus.collect_provider_proposal(
            self.recording, _change_proposal_annotations
        )
        self.assertEqual(self.recording.metadata_sources, {"title": "MusicBrainz"})
        self.assertEqual(self.recording.metadata_conflicts, {"album": "Review edition"})
        self.assertEqual(self.recording._diff_descriptions, ["Existing change"])
        self.assertEqual(proposal.changes, {})
        self.assertEqual(proposal.conflicts, {})

    def test_blank_artist_and_album_fields_normalize_without_inventing_identity(
        self,
    ) -> None:
        recording = TrackInfo(
            self.path, artist=" ", title="", album="", album_artist=""
        )
        normalized = normalization.normalize_track_metadata(recording)
        self.assertEqual((normalized.artist, normalized.album), ("", ""))

    def test_normalization_preview_reports_changes_without_writing_tags(self) -> None:
        with (
            patch.object(
                metadata,
                "read_track_metadata",
                return_value=dataclasses.replace(self.recording, genre="rock"),
            ),
            patch.object(metadata, "write_track_metadata") as write,
        ):
            normalized = track_processing.normalize_single_track(
                self.path, dry_run=True
            )
        assert normalized is not None
        self.assertEqual(normalized.genre, "Rock")
        self.assertTrue(normalized._diff_descriptions)
        write.assert_not_called()

    def setUp(self) -> None:
        self.path = self.test_directory / "recording.flac"
        self.path.touch()
        self.recording = TrackInfo(
            self.path, artist="Primary", title="Recording", album="Record"
        )

    def test_cuesheets_use_explicit_content_then_companion_without_replacing_tags(
        self,
    ) -> None:
        companion = self.test_directory / "record.cue"
        companion.write_text('TITLE "Record"')
        track_processing._enrich_cuesheet(self.recording, self.path, None)
        self.assertEqual(self.recording.cuesheet, 'TITLE "Record"')
        companion.write_text('TITLE "Changed"')
        track_processing._enrich_cuesheet(self.recording, self.path, None)
        self.assertEqual(self.recording.cuesheet, 'TITLE "Record"')
        track_processing._enrich_cuesheet(self.recording, self.path, 'TITLE "Explicit"')
        self.assertEqual(self.recording.cuesheet, 'TITLE "Explicit"')

    def test_artwork_enrichment_can_be_disabled_and_recovers_from_io_errors(
        self,
    ) -> None:
        with patch.object(
            art,
            "process_album_cover_art",
            side_effect=OSError("offline"),
        ) as fetch:
            self.assertIsNone(
                track_processing._enrich_artwork(self.recording, self.path, False)
            )
            fetch.assert_not_called()
            self.assertIsNone(
                track_processing._enrich_artwork(self.recording, self.path, True)
            )
        fetch.assert_called_once()

    def test_unchanged_normalization_is_successful_and_replaygain_is_forwarded(
        self,
    ) -> None:
        with (
            patch.object(metadata, "read_track_metadata", return_value=self.recording),
            patch.object(metadata, "write_track_metadata") as write,
            patch.object(replaygain, "calculate_album_replaygain") as gain,
        ):
            result = track_processing.normalize_single_track(self.path)
            assert result is not None
            self.assertEqual(result._diff_descriptions, [])
            report = track_processing.normalize_library(
                self.test_directory, fetch_replaygain=True, dry_run=True
            )
        self.assertEqual(report.total_scanned, 1)
        write.assert_not_called()
        gain.assert_called_once_with(
            [self.path], force=False, dry_run=True, max_threads=4
        )

    def test_normalization_handles_empty_identity_and_existing_sort_names(self) -> None:
        for artist, album_artist, title in (
            ("", "", ""),
            ("!!!", "!!!", "Recording (prod.   )"),
            ("Primary", "Primary", "Recording feat. Guest"),
        ):
            recording = TrackInfo(
                self.path,
                artist=artist,
                album_artist=album_artist,
                title=title,
                artist_sort=artist,
                album_artist_sort=album_artist,
            )
            with (
                self.subTest(artist=artist, title=title),
                patch(
                    "sonora.services.artist_resolution.resolve_artist_name",
                    side_effect=lambda name: name,
                ) as resolve,
            ):
                normalized = normalization.normalize_track_metadata(recording)
            self.assertIs(normalized, recording)
            self.assertEqual(normalized.artist, artist)
            resolve.assert_not_called()
            self.assertIsNone(normalized.producers)
        self.assertEqual(normalized.featured_artists, "Guest")

    def test_dsp_enrichment_accepts_results_and_handles_decode_failures(self) -> None:
        with patch.object(bpm, "calculate_bpm", return_value=123.4):
            track_processing._enrich_bpm(self.recording, self.path, True)
        self.assertEqual(self.recording.bpm, 123.4)
        with patch.object(bpm, "calculate_bpm") as calculate:
            track_processing._enrich_bpm(self.recording, self.path, True)
        calculate.assert_not_called()
        with patch.object(key, "detect_musical_key", return_value="Am"):
            track_processing._enrich_key(self.recording, self.path, True)
        self.assertEqual(self.recording.initial_key, "Am")
        for result in (None, OSError("decode failed")):
            with (
                self.subTest(result=result),
                patch.object(
                    key, "detect_musical_key", return_value=None, side_effect=result
                ),
                patch.object(
                    bpm, "calculate_bpm", return_value=None, side_effect=result
                ),
            ):
                track_processing._enrich_key(
                    self.recording, self.path, True, force=True
                )
                track_processing._enrich_bpm(
                    self.recording, self.path, True, force=True
                )
            self.assertEqual(self.recording.initial_key, "Am")
            self.assertEqual(self.recording.bpm, 123.4)

    def test_lyric_enrichment_logs_existing_forced_and_missing_assets(self) -> None:
        for existing, force in ((False, False), (True, False), (True, True)):
            if existing:
                self.path.with_suffix(".lrc").write_text("[00:01] Existing")
            with (
                self.subTest(existing=existing, force=force),
                patch.object(metadata, "get_audio_duration", return_value=100),
                patch.object(
                    track_processing,
                    "process_track_lyrics",
                    return_value=("[00:01] Lyrics", "synced"),
                ),
            ):
                track_processing._enrich_lyrics(
                    self.recording, self.path, True, force=force
                )
            self.assertEqual(self.recording.lyrics, "[00:01] Lyrics")
        with patch.object(
            track_processing, "process_track_lyrics", side_effect=OSError("offline")
        ):
            track_processing._enrich_lyrics(self.recording, self.path, True)
        self.assertEqual(self.recording.lyrics, "[00:01] Lyrics")

    def test_normalization_read_failure_and_failed_write_produce_no_success(
        self,
    ) -> None:
        self.assertIsNone(
            track_processing.normalize_single_track(self.path.with_name("missing.flac"))
        )
        with patch.object(
            metadata, "read_track_metadata", side_effect=OSError("bad metadata")
        ):
            self.assertIsNone(track_processing.normalize_single_track(self.path))
        with (
            patch.object(
                metadata,
                "read_track_metadata",
                return_value=dataclasses.replace(self.recording, genre="rock"),
            ),
            patch.object(
                metadata, "write_track_metadata", side_effect=OSError("disk full")
            ),
        ):
            self.assertIsNone(track_processing.normalize_single_track(self.path))

    def test_empty_library_and_worker_failures_preserve_domain_errors(self) -> None:
        with self.assertRaises(ValueError):
            track_processing.normalize_library(self.path)
        empty_directory = self.test_directory / "empty"
        empty_directory.mkdir()
        self.assertEqual(
            track_processing.normalize_library(empty_directory).total_scanned, 0
        )
        with patch.object(
            track_processing, "normalize_single_track", return_value=None
        ):
            report = track_processing.normalize_library(self.test_directory)
            self.assertEqual(
                (report.total_scanned, report.failed_files, report.total_unchanged),
                (1, 1, 0),
            )
        with (
            patch.object(
                track_processing,
                "normalize_single_track",
                side_effect=RuntimeError("worker bug"),
            ),
            self.assertRaisesRegex(RuntimeError, "worker bug"),
        ):
            track_processing.normalize_library(self.test_directory)

    def test_canonical_normalization_preserves_known_credits_and_drops_wrong_sorts(
        self,
    ) -> None:
        recording = TrackInfo(
            self.path,
            artist="Primary feat. Guest",
            title="Recording (prod. Producer) [Explicit]",
            album="Record",
            album_artist="Primary",
            artist_sort="Unrelated",
            album_artist_sort="Unrelated",
            producers="Verified",
            release_type="single",
            total_tracks=1,
            track_number=4,
        )
        normalized = normalization.normalize_track_metadata(recording)
        self.assertEqual(normalized.title, "Recording")
        self.assertEqual(
            (normalized.artist, normalized.featured_artists, normalized.advisory),
            ("Primary", "Guest", "Explicit"),
        )
        self.assertEqual(normalized.producers, "Verified")
        self.assertEqual((normalized.track_number, normalized.total_tracks), (1, 1))
        self.assertIsNone(normalized.artist_sort)
        self.assertIsNone(normalized.album_artist_sort)
        before = dataclasses.replace(normalized)
        self.assertEqual(normalization.normalize_track_metadata(normalized), before)


class TestRenameFailureContracts(IsolatedTestCase):
    def test_preview_returns_audio_destination_and_preserves_companion_content(
        self,
    ) -> None:
        path = self.test_directory / "recording.flac"
        path.touch()
        companion = path.with_suffix(".lrc")
        companion.write_text("[00:01] Verified lyrics")
        relocated: list[Path] = []
        renamed = renamer.rename_track_file(
            path,
            track_info=TrackInfo(path, title="Recording", track_number=1),
            dry_run=True,
            relocated_lrc_collector=relocated,
        )
        self.assertEqual(renamed.name, "01 - Recording.flac")
        self.assertEqual(relocated, [])
        self.assertFalse(renamed.with_suffix(".lrc").exists())
        self.assertEqual(companion.read_text(), "[00:01] Verified lyrics")

    def test_single_hierarchy_failures_and_canonical_folder_identity_are_reported(
        self,
    ) -> None:
        folder = self.test_directory / "Singles" / "Primary - Recording"
        folder.mkdir(parents=True)
        path = folder / "01 - Recording.flac"
        path.touch()
        failed = RenameReport()
        with patch.object(
            renamer, "read_track_metadata", side_effect=OSError("broken tags")
        ):
            self.assertEqual(
                renamer.rename_directory_files(self.test_directory, report=failed), []
            )
        self.assertEqual((failed.failed_files, failed.folders_renamed), (1, 0))
        report = RenameReport()
        with patch.object(
            renamer,
            "read_track_metadata",
            return_value=TrackInfo(
                path, artist="Primary", title="Recording", track_number=1
            ),
        ):
            self.assertEqual(
                renamer.rename_directory_files(self.test_directory, report=report),
                [path],
            )
        self.assertEqual((report.unchanged_files, report.folders_renamed), (1, 0))

    def test_generic_containers_and_disc_folders_are_not_renamed_as_single_releases(
        self,
    ) -> None:
        for name in ("Singles", "CD 2", "Downloads"):
            folder = self.test_directory / name
            folder.mkdir()
            recording = TrackInfo(
                folder / "recording.flac", artist="Primary", title="Recording"
            )
            with self.subTest(name=name):
                self.assertEqual(
                    renamer.rename_single_folder(folder, recording), folder
                )

    def test_single_folder_preview_returns_canonical_path_without_moving_assets(
        self,
    ) -> None:
        folder = self.test_directory / "Record"
        folder.mkdir()
        recording = TrackInfo(
            folder / "recording.flac", artist="Primary", title="Recording"
        )
        self.assertEqual(
            renamer.rename_single_folder(folder, recording, dry_run=True),
            self.test_directory / "Primary - Recording",
        )
        self.assertTrue(folder.exists())

    def test_unreadable_single_directory_is_preserved(self) -> None:
        folder = self.test_directory / "Record"
        folder.mkdir()
        with patch.object(Path, "iterdir", side_effect=OSError("locked directory")):
            self.assertEqual(
                renamer.rename_single_folder(
                    folder,
                    TrackInfo(
                        folder / "recording.flac", artist="Primary", title="Recording"
                    ),
                ),
                folder,
            )

    def test_nonexistent_directory_and_programming_failure_preserve_domain_errors(
        self,
    ) -> None:
        with self.assertRaises(FileNotFoundError):
            renamer.rename_directory_files(self.test_directory / "absent")
        path = self.test_directory / "recording.flac"
        path.touch()
        with (
            patch.object(
                renamer,
                "read_track_metadata",
                side_effect=RuntimeError("tag reader bug"),
            ),
            self.assertRaisesRegex(RuntimeError, "tag reader bug"),
        ):
            renamer.rename_directory_files(self.test_directory)

    def test_single_folders_containing_subdirectories_are_not_renamed(self) -> None:
        folder = self.test_directory / "Record"
        (folder / "Nested").mkdir(parents=True)
        self.assertEqual(
            renamer.rename_single_folder(
                folder,
                TrackInfo(
                    folder / "recording.flac", artist="Primary", title="Recording"
                ),
            ),
            folder,
        )

    def test_single_folder_is_renamed_after_track_success(self) -> None:
        folder = self.test_directory / "Singles" / "Record"
        folder.mkdir(parents=True)
        path = folder / "recording.flac"
        path.touch()
        recording = TrackInfo(path, artist="Primary", title="Recording", total_tracks=1)
        with patch.object(renamer, "read_track_metadata", return_value=recording):
            renamed = renamer.rename_directory_files(self.test_directory)
        self.assertEqual(
            renamed,
            [
                self.test_directory
                / "Singles"
                / "Primary - Recording"
                / "01 - Recording.flac"
            ],
        )
        self.assertTrue(renamed[0].exists())

    def test_multidisc_and_numberless_filenames_follow_declared_structure(self) -> None:
        self.assertEqual(
            renamer.build_new_filename(
                1, "Recording", ".flac", disc_number=1, total_discs=2
            ),
            "1-01 - Recording.flac",
        )
        self.assertEqual(
            renamer.build_new_filename(None, "Recording", ".flac", disc_number=2),
            "2-Recording.flac",
        )
        self.assertEqual(
            renamer.build_new_filename(3, "Recording", ".flac"), "03 - Recording.flac"
        )

    def test_multiple_singles_preserve_their_individual_track_numbers(self) -> None:
        folder = self.test_directory / "Singles"
        folder.mkdir()
        path = folder / "recording.flac"
        path.touch()
        (folder / "other.flac").touch()
        recording = TrackInfo(path, artist="Primary", title="Recording", track_number=2)
        self.assertEqual(
            renamer.rename_track_file(path, track_info=recording, dry_run=True),
            folder / "02 - Recording.flac",
        )

    def test_canonical_and_colliding_single_folders_are_preserved(self) -> None:
        canonical = self.test_directory / "Primary - Recording"
        canonical.mkdir()
        recording = TrackInfo(
            canonical / "recording.flac",
            artist="Primary",
            title="Recording",
            featured_artists="Guest",
        )
        self.assertEqual(renamer.rename_single_folder(canonical, recording), canonical)
        (self.test_directory / "Primary - Recording (feat. Guest)").mkdir()
        source = self.test_directory / "Original"
        source.mkdir()
        self.assertEqual(
            renamer.rename_single_folder(
                source,
                dataclasses.replace(recording, file_path=source / "recording.flac"),
            ),
            source,
        )

    def test_album_folder_identity_collision_and_preview_are_explicit(self) -> None:
        canonical = self.test_directory / "Primary - Record"
        canonical.mkdir()
        self.assertEqual(
            renamer.rename_album_folder(canonical, "Primary", "Record"), canonical
        )
        source = self.test_directory / "Original"
        source.mkdir()
        self.assertEqual(
            renamer.rename_album_folder(source, "Primary", "Record"), source
        )
        self.assertEqual(
            renamer.rename_album_folder(source, "Primary", "Other", dry_run=True),
            self.test_directory / "Primary - Other",
        )
        self.assertTrue(source.exists())

    def test_directory_report_records_failed_and_unchanged_files(self) -> None:
        path = self.test_directory / "01 - Recording.flac"
        path.touch()
        report = RenameReport()
        with patch.object(
            renamer, "read_track_metadata", side_effect=OSError("unreadable")
        ):
            self.assertEqual(
                renamer.rename_directory_files(self.test_directory, report=report), []
            )
        self.assertEqual((report.total_files, report.failed_files), (1, 1))
        recording = TrackInfo(
            path,
            artist="Unknown Artist",
            album="Unknown Album",
            title="Recording",
            track_number=1,
        )
        report = RenameReport()
        with patch.object(renamer, "read_track_metadata", return_value=recording):
            self.assertEqual(
                renamer.rename_directory_files(self.test_directory, report=report),
                [path],
            )
        self.assertEqual(
            (report.unchanged_files, report.files_renamed, report.failed_files),
            (1, 0, 0),
        )

    def test_collaborative_album_consensus_and_mixed_release_directory(self) -> None:
        paths = [
            self.test_directory / f"{position:02} - Recording {position}.flac"
            for position in range(1, 4)
        ]
        for path in paths:
            path.touch()
        metadata_by_path = {
            path: TrackInfo(
                path,
                artist=f"Performer {position}",
                album="Record",
                title=f"Recording {position}",
                track_number=position,
            )
            for position, path in enumerate(paths, 1)
        }
        with (
            patch.object(
                renamer, "read_track_metadata", side_effect=metadata_by_path.__getitem__
            ),
            patch.object(
                renamer, "rename_album_folder", return_value=self.test_directory
            ) as rename_folder,
        ):
            self.assertEqual(
                len(
                    renamer.rename_directory_files(
                        self.test_directory, dry_run=True, max_threads=2
                    )
                ),
                3,
            )
        rename_folder.assert_called_once_with(
            self.test_directory, "Various Artists", "Record", dry_run=True
        )
        for position, recording in enumerate(metadata_by_path.values(), 1):
            recording.album = f"Record {position}"
        with (
            patch.object(
                renamer, "read_track_metadata", side_effect=metadata_by_path.__getitem__
            ),
            patch.object(renamer, "rename_album_folder") as rename_folder,
        ):
            self.assertEqual(
                len(
                    renamer.rename_directory_files(
                        self.test_directory, dry_run=True, max_threads=2
                    )
                ),
                3,
            )
        rename_folder.assert_not_called()

    def test_metadata_read_failure_preserves_the_original_io_error(self) -> None:
        path = self.test_directory / "recording.flac"
        path.touch()
        failure = OSError("unreadable tags")
        with (
            patch.object(renamer, "read_track_metadata", side_effect=failure),
            self.assertRaises(OSError) as raised,
        ):
            renamer.rename_track_file(path)
        self.assertIs(raised.exception, failure)
        self.assertTrue(path.exists())

    def test_empty_title_and_missing_audio_have_explicit_outcomes(self) -> None:
        self.assertIsNone(renamer.build_new_filename(1, "", ".flac"))
        with self.assertRaises(FileNotFoundError):
            renamer.rename_track_file(self.test_directory / "absent.flac")
        path = self.test_directory / "recording.flac"
        path.touch()
        self.assertEqual(
            renamer.rename_track_file(path, track_info=TrackInfo(path, title="")), path
        )
        self.assertEqual(
            renamer.rename_track_file(
                path,
                format_pattern="{artist} - {title}",
                track_info=TrackInfo(path, artist="Primary", title="Recording"),
                dry_run=True,
            ),
            self.test_directory / "Primary - Recording.flac",
        )
        self.assertTrue(path.exists())

    def test_lyric_header_sync_missing_file_and_write_failure(self) -> None:
        path = self.test_directory / "recording.lrc"
        self.assertFalse(renamer.sync_lrc_metadata(path, "Primary", "Recording"))
        path.write_text("[ar:Old]\n[ti:Old]\n[00:01] Lyrics")
        self.assertTrue(renamer.sync_lrc_metadata(path, "Primary", "Recording"))
        self.assertEqual(
            path.read_text(), "[ar:Primary]\n[ti:Recording]\n[00:01] Lyrics"
        )
        with patch.object(
            renamer, "atomic_write_bytes", side_effect=OSError("disk full")
        ):
            self.assertFalse(renamer.sync_lrc_metadata(path, "Changed", "Changed"))
        self.assertIn("[ar:Primary]", path.read_text())

    def test_rename_failure_and_collision_retry_preserve_source_recording(self) -> None:
        path = self.test_directory / "recording.flac"
        path.write_bytes(b"audio")
        recording = TrackInfo(path, artist="Primary", title="Recording", track_number=1)
        with (
            patch.object(
                renamer,
                "safe_case_rename",
                side_effect=PermissionError("read-only directory"),
            ),
            self.assertRaises(PermissionError),
        ):
            renamer.rename_track_file(path, track_info=recording)
        self.assertEqual(path.read_bytes(), b"audio")
        with patch.object(
            renamer,
            "safe_case_rename",
            side_effect=[FileExistsError("racing creator"), None],
        ) as rename:
            result = renamer.rename_track_file(path, track_info=recording)
        self.assertEqual(result.name, "01 - Recording.flac")
        self.assertEqual(rename.call_count, 2)

    def test_folder_rename_protects_containers_and_handles_io_failure(self) -> None:
        folder = self.test_directory / "Original"
        folder.mkdir()
        recording = TrackInfo(
            folder / "recording.flac", artist="Primary", title="Recording"
        )
        with patch.object(
            Path, "iterdir", side_effect=PermissionError("directory locked")
        ):
            self.assertEqual(renamer.rename_single_folder(folder, recording), folder)
            self.assertEqual(
                renamer.rename_album_folder(folder, "Primary", "Record"), folder
            )
        with patch.object(
            renamer, "safe_case_rename", side_effect=PermissionError("directory locked")
        ):
            self.assertEqual(renamer.rename_single_folder(folder, recording), folder)
            self.assertEqual(
                renamer.rename_album_folder(folder, "Primary", "Record"), folder
            )
        (folder / "Nested").mkdir()
        self.assertEqual(renamer.rename_single_folder(folder, recording), folder)
        self.assertEqual(
            renamer.rename_album_folder(folder, "Primary", "Record"), folder
        )


class TestOrganizationContracts(IsolatedTestCase):
    def test_single_folder_classification_respects_album_sequence_and_consensus(
        self,
    ) -> None:
        folder = self.test_directory / "Collection"
        folder.mkdir()
        paths = [folder / f"recording-{index}.flac" for index in range(3)]
        cases = (
            (
                [
                    TrackInfo(path, album=f"Record {index}", track_number=1)
                    for index, path in enumerate(paths)
                ],
                True,
            ),
            (
                [
                    TrackInfo(path, album="", track_number=index + 1)
                    for index, path in enumerate(paths)
                ],
                False,
            ),
            ([TrackInfo(path, album="", total_tracks=10) for path in paths], False),
            ([TrackInfo(path, album="Shared Record") for path in paths], False),
            ([TrackInfo(paths[0], album="", track_number=3)], False),
            ([TrackInfo(paths[0], album="Collection", total_tracks=2)], False),
        )
        for recordings, expected in cases:
            with self.subTest(recordings=recordings):
                self.assertEqual(
                    organizer._is_single_folder(
                        folder,
                        [recording.file_path for recording in recordings],
                        self.test_directory,
                        [(recording.file_path, recording) for recording in recordings],
                    ),
                    expected,
                )
        self.assertFalse(
            organizer._is_single_folder(folder, paths, self.test_directory, [])
        )

    def test_cleanup_protects_target_hierarchy_and_removes_known_junk(self) -> None:
        protected = self.test_directory / "Primary" / "Singles"
        protected.mkdir(parents=True)
        junk_directory = self.test_directory / "Empty"
        junk_directory.mkdir()
        (junk_directory / ".DS_Store").touch()
        self.assertEqual(
            organizer.cleanup_empty_dirs(
                self.test_directory, target_singles_dir=protected
            ),
            1,
        )
        self.assertTrue(protected.exists())
        self.assertFalse(junk_directory.exists())
        self.assertEqual(
            organizer.cleanup_empty_dirs(self.test_directory / "missing"), 0
        )
        self.assertEqual(
            organizer.consolidate_duplicate_artist_dirs(
                self.test_directory / "missing"
            ),
            0,
        )
