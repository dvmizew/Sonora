import tempfile
import unittest
from pathlib import Path

from sonora.core.state import LibraryStateManager


class TestLibraryState(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_state.db"
        self.state_mgr = LibraryStateManager(db_path=self.db_path)

        self.test_file1 = Path(self.temp_dir.name) / "song1.flac"
        self.test_file2 = Path(self.temp_dir.name) / "song2.flac"
        self.test_file1.write_bytes(b"dummy audio 1")
        self.test_file2.write_bytes(b"dummy audio 2")

    def tearDown(self) -> None:
        self.state_mgr.close()
        self.temp_dir.cleanup()

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
