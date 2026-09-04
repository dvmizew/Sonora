"""
Unit tests for Sonora caching layer, XDG Base Directory compliance, and CLI cache management.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import unittest

from sonora.audio.metadata import clear_metadata_cache, get_metadata_cache_size
from sonora.cli.main import main
from sonora.core.cache import (
    CacheStats,
    ClearResult,
    clear_cache,
    close_cache,
    get_cache_dir,
    get_cache_stats,
    get_cached_api,
    set_cached_api,
    set_ignore_cache,
)
from sonora.core.state import LibraryStateManager, reset_library_state
from sonora.core.utils import clear_utils_cache, format_filesize


class TestDiskCache(unittest.TestCase):
    def tearDown(self) -> None:
        set_ignore_cache(False)

    @patch("sonora.core.cache.get_cache")
    def test_get_cached_api_hit(self, mock_get_cache: MagicMock) -> None:
        mock_cache_inst = MagicMock()
        mock_cache_inst.get.return_value = {"title": "Test Album"}
        mock_get_cache.return_value = mock_cache_inst

        val = get_cached_api("test_key")
        self.assertEqual(val, {"title": "Test Album"})
        mock_cache_inst.get.assert_called_once_with("test_key")

    @patch("sonora.core.cache.get_cache")
    def test_get_cached_api_miss(self, mock_get_cache: MagicMock) -> None:
        mock_cache_inst = MagicMock()
        mock_cache_inst.get.return_value = None
        mock_get_cache.return_value = mock_cache_inst

        val = get_cached_api("missing_key")
        self.assertIsNone(val)
        mock_cache_inst.get.assert_called_once_with("missing_key")

    @patch("sonora.core.cache.get_cache")
    def test_get_cached_api_ignore_cache(self, mock_get_cache: MagicMock) -> None:
        mock_cache_inst = MagicMock()
        mock_cache_inst.get.return_value = {"title": "Test Album"}
        mock_get_cache.return_value = mock_cache_inst

        set_ignore_cache(True)
        val = get_cached_api("test_key")
        self.assertIsNone(val)
        mock_cache_inst.get.assert_not_called()

    @patch("sonora.core.cache.get_cache")
    def test_set_cached_api(self, mock_get_cache: MagicMock) -> None:
        mock_cache_inst = MagicMock()
        mock_get_cache.return_value = mock_cache_inst

        set_cached_api("test_key", {"title": "Test Album"}, expire_seconds=86400)
        mock_cache_inst.set.assert_called_once_with(
            "test_key", {"title": "Test Album"}, expire=86400
        )


class TestCacheArchitecture(unittest.TestCase):
    def test_get_cache_dir_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            expected = Path.home() / ".cache" / "sonora"
            self.assertEqual(get_cache_dir(), expected)

    def test_get_cache_dir_xdg_override(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"XDG_CACHE_HOME": tmpdir}),
        ):
            self.assertEqual(get_cache_dir(), Path(tmpdir) / "sonora")

    def test_format_filesize(self) -> None:
        self.assertEqual(format_filesize(0), "0 B")
        self.assertEqual(format_filesize(512), "512 B")
        self.assertEqual(format_filesize(1024), "1.00 KB")
        self.assertEqual(format_filesize(1024 * 1024), "1.00 MB")
        self.assertEqual(format_filesize(1024 * 1024 * 1024), "1.00 GB")

    def test_clear_utils_cache(self) -> None:
        # Should execute cleanly and clear all 9 utility LRU caches
        clear_utils_cache()

    def test_metadata_cache_lifecycle(self) -> None:
        self.assertIsInstance(get_metadata_cache_size(), int)
        cleared = clear_metadata_cache()
        self.assertIsInstance(cleared, int)
        self.assertEqual(get_metadata_cache_size(), 0)

    def test_library_state_manager_methods(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "test_state.db"
            state_mgr = LibraryStateManager(db_file)

            self.assertEqual(state_mgr.get_state_count(), 0)
            self.assertGreater(state_mgr.get_state_size(), 0)

            # Record a dummy track
            dummy_track = Path(tmpdir) / "song.mp3"
            dummy_track.write_text("dummy audio")
            state_mgr.record_track_state(dummy_track, status="TAGGED_OK")

            self.assertEqual(state_mgr.get_state_count(), 1)
            self.assertTrue(state_mgr.is_track_up_to_date(dummy_track))

            # Filter outdated tracks
            outdated = state_mgr.filter_outdated_tracks([dummy_track])
            self.assertEqual(len(outdated), 0)

            # Clear state (truncate rows)
            cleared_count = state_mgr.clear_state(purge=False)
            self.assertEqual(cleared_count, 1)
            self.assertEqual(state_mgr.get_state_count(), 0)
            self.assertTrue(db_file.exists())

            # Re-record and purge
            state_mgr.record_track_state(dummy_track, status="TAGGED_OK")
            purged_count = state_mgr.clear_state(purge=True)
            self.assertEqual(purged_count, 1)
            self.assertFalse(db_file.exists())

            # Re-record after purge on the same instance (verifying auto-schema healing)
            state_mgr.record_track_state(dummy_track, status="TAGGED_OK")
            self.assertEqual(state_mgr.get_state_count(), 1)
            self.assertTrue(state_mgr.is_track_up_to_date(dummy_track))

    def test_clear_cache_with_isolated_env(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"XDG_CACHE_HOME": tmpdir}),
        ):
            close_cache()
            reset_library_state()

            # Add an item to cache
            set_cached_api("unit_test_key", {"data": 123})
            stats_before = get_cache_stats()
            self.assertIsInstance(stats_before, CacheStats)
            self.assertEqual(stats_before.api_entries, 1)

            # Clear default (api + memory, keeping state)
            res = clear_cache(
                clear_api=True, clear_state=False, clear_memory=True, purge=False
            )
            self.assertIsInstance(res, ClearResult)
            self.assertTrue(res.api_cleared)
            self.assertFalse(res.state_cleared)
            self.assertEqual(res.api_entries_cleared, 1)

            stats_after = get_cache_stats()
            self.assertEqual(stats_after.api_entries, 0)

            close_cache()
            reset_library_state()

    def test_cli_cache_commands(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"XDG_CACHE_HOME": tmpdir}),
        ):
            close_cache()
            reset_library_state()

            # Test cache stats
            ret_stats = main(["cache", "stats"])
            self.assertEqual(ret_stats, 0)

            # Test cache stats --json
            ret_stats_json = main(["cache", "stats", "--json"])
            self.assertEqual(ret_stats_json, 0)

            # Test cache default entrypoint
            ret_default = main(["cache"])
            self.assertEqual(ret_default, 0)

            # Test cache clear --dry-run
            ret_dry_run = main(["cache", "clear", "--dry-run"])
            self.assertEqual(ret_dry_run, 0)

            # Test cache clear --all --dry-run --json
            ret_dry_run_json = main(["cache", "clear", "--all", "--dry-run", "--json"])
            self.assertEqual(ret_dry_run_json, 0)

            # Test clear-cache shortcut alias
            ret_shortcut = main(["clear-cache", "--dry-run"])
            self.assertEqual(ret_shortcut, 0)

            # Test cache clear (default live)
            ret_clear = main(["cache", "clear"])
            self.assertEqual(ret_clear, 0)

            # Test cache clear --all (live)
            ret_clear_all = main(["cache", "clear", "--all"])
            self.assertEqual(ret_clear_all, 0)

            # Test cache clear --all --purge (live)
            ret_clear_purge = main(["cache", "clear", "--all", "--purge"])
            self.assertEqual(ret_clear_purge, 0)

            close_cache()
            reset_library_state()


if __name__ == "__main__":
    unittest.main()
