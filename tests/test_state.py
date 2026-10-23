import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sonora.core import state
from sonora.core.state import LibraryStateVault
from tests.support import IsolatedTestCase


class TestLibraryState(IsolatedTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_state.db"
        self.state_mgr = LibraryStateVault(db_path=self.db_path)

        self.test_file1 = Path(self.temp_dir.name) / "song1.flac"
        self.test_file2 = Path(self.temp_dir.name) / "song2.flac"
        self.test_file1.write_bytes(b"dummy audio 1")
        self.test_file2.write_bytes(b"dummy audio 2")

    def tearDown(self) -> None:
        self.state_mgr.close()
        self.temp_dir.cleanup()

    def test_storage_stats_do_not_create_a_database_for_an_unused_library(self) -> None:
        state_path = state.get_default_state_db_path()
        self.assertFalse(state_path.exists())
        self.assertEqual(state.get_state_storage_stats(), (0, 0))
        self.assertFalse(state_path.exists())

    def test_storage_stats_include_active_state_and_reopen_existing_database(
        self,
    ) -> None:
        library_state = state.get_library_state()
        library_state.record_track_state(self.test_file1, status="TAGGED_OK")
        for reopen in (False, True):
            with self.subTest(reopen=reopen):
                if reopen:
                    state.reset_library_state()
                entry_count, storage_bytes = state.get_state_storage_stats()
                self.assertEqual(entry_count, 1)
                self.assertGreater(storage_bytes, 0)

    def test_storage_stats_preserve_vault_io_failure_handling(self) -> None:
        with patch.object(Path, "exists", side_effect=PermissionError("locked file")):
            self.assertEqual(state.get_state_storage_stats(), (0, 0))
        library_state = state.get_library_state()
        for method in ("stat", "exists"):
            with (
                self.subTest(method=method),
                patch.object(Path, method, side_effect=PermissionError("locked file")),
            ):
                self.assertEqual(state.get_state_storage_stats(), (0, 0))
        self.assertIs(state.get_library_state(), library_state)

    def test_state_lifecycle(self) -> None:
        # Initially not up to date
        self.assertFalse(self.state_mgr.is_track_up_to_date(self.test_file1))
        outdated = self.state_mgr.filter_outdated_tracks(
            [self.test_file1, self.test_file2]
        )
        self.assertEqual(len(outdated), 2)

        # Record state for file1
        self.state_mgr.record_track_state(self.test_file1, status="TAGGED_OK")
        self.assertTrue(self.state_mgr.is_track_up_to_date(self.test_file1))
        self.assertFalse(self.state_mgr.is_track_up_to_date(self.test_file2))

        outdated = self.state_mgr.filter_outdated_tracks(
            [self.test_file1, self.test_file2]
        )
        self.assertEqual(outdated, [self.test_file2])

        # Batch record for file2
        self.state_mgr.record_tracks_state_batch([self.test_file2], status="TAGGED_OK")
        self.assertTrue(self.state_mgr.is_track_up_to_date(self.test_file2))

        outdated_empty = self.state_mgr.filter_outdated_tracks(
            [self.test_file1, self.test_file2]
        )
        self.assertEqual(len(outdated_empty), 0)

        # Modify file1 content and verify it is detected as outdated
        self.test_file1.write_bytes(b"modified audio data")
        self.assertFalse(self.state_mgr.is_track_up_to_date(self.test_file1))
        outdated_modified = self.state_mgr.filter_outdated_tracks(
            [self.test_file1, self.test_file2]
        )
        self.assertEqual(outdated_modified, [self.test_file1])


if __name__ == "__main__":
    unittest.main()
