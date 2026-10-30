"""
Unit tests for Sonora caching layer, XDG Base Directory compliance, and CLI cache management.
"""

import os
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import diskcache

from sonora.cli.main import main
from sonora.core.cache import (
    CacheStats,
    ClearResult,
    clear_cache,
    close_cache,
    get_api_cache_dir,
    get_cache,
    get_cache_dir,
    get_cache_stats,
    get_cached_api,
    set_cached_api,
    set_ignore_cache,
)
from sonora.core.constants import DIRS
from sonora.core.state import LibraryStateVault, reset_library_state
from tests.support import IsolatedTestCase


class TestDiskCache(IsolatedTestCase):
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

    @patch("sonora.core.cache.get_cache")
    def test_set_cached_api_none_value(self, mock_get_cache: MagicMock) -> None:
        set_cached_api("key_none", None)
        mock_get_cache.return_value.set.assert_called_once_with(
            "key_none", "", expire=1800
        )

    @patch("sonora.core.cache.get_cache")
    def test_get_cached_api_exception(self, mock_get_cache: MagicMock) -> None:
        mock_cache_inst = MagicMock()
        mock_cache_inst.get.side_effect = diskcache.Timeout("Lock timeout")
        mock_get_cache.return_value = mock_cache_inst

        result = get_cached_api("timeout_key")
        self.assertIsNone(result)

    @patch("sonora.core.cache.get_cache")
    def test_set_cached_api_exception(self, mock_get_cache: MagicMock) -> None:
        mock_cache_inst = MagicMock()
        mock_cache_inst.set.side_effect = diskcache.Timeout("Lock timeout")
        mock_get_cache.return_value = mock_cache_inst

        with patch("sonora.core.cache.LOG.debug") as debug:
            set_cached_api("timeout_key", {"title": "Recording"})
        mock_cache_inst.set.assert_called_once_with(
            "timeout_key", {"title": "Recording"}, expire=2419200
        )
        debug.assert_called_once()
        self.assertIn("Lock timeout", debug.call_args.args[0])

    def test_real_diskcache_roundtrip(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"XDG_CACHE_HOME": tmpdir}),
        ):
            close_cache()
            try:
                self.assertIsNone(get_cached_api("real_key"))
                set_cached_api("real_key", {"artist": "Radiohead", "album": "Kid A"})
                self.assertEqual(
                    get_cached_api("real_key"),
                    {"artist": "Radiohead", "album": "Kid A"},
                )

                set_ignore_cache(True)
                self.assertIsNone(get_cached_api("real_key"))
                set_ignore_cache(False)
                self.assertEqual(
                    get_cached_api("real_key"),
                    {"artist": "Radiohead", "album": "Kid A"},
                )
            finally:
                close_cache()


class TestCacheArchitecture(IsolatedTestCase):
    def test_get_cache_dir_default(self) -> None:
        with patch.dict(os.environ, {"HOME": str(Path.home())}, clear=True):
            expected = Path.home() / ".cache" / "sonora"
            self.assertEqual(get_cache_dir(), expected)

    def test_get_cache_dir_xdg_override(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"XDG_CACHE_HOME": tmpdir}),
        ):
            self.assertEqual(get_cache_dir(), Path(tmpdir) / "sonora")

    def test_dirs_cache_path(self) -> None:
        self.assertEqual(get_cache_dir(), DIRS.user_cache_path)

    def test_dirs_config_dir_default(self) -> None:
        with patch.dict(os.environ, {"HOME": str(Path.home())}, clear=True):
            expected = Path.home() / ".config" / "sonora"
            self.assertEqual(DIRS.user_config_path, expected)

    def test_dirs_config_dir_xdg_override(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"XDG_CONFIG_HOME": tmpdir}),
        ):
            self.assertEqual(DIRS.user_config_path, Path(tmpdir) / "sonora")

    def test_library_state_survives_clear_purge_and_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "test_state.db"
            state_vault = LibraryStateVault(db_file)
            try:
                self.assertEqual(state_vault.get_state_count(), 0)
                self.assertGreater(state_vault.get_state_size(), 0)

                # Record a dummy track
                dummy_track = Path(tmpdir) / "song.mp3"
                dummy_track.write_text("dummy audio")
                state_vault.record_track_state(dummy_track, status="TAGGED_OK")

                self.assertEqual(state_vault.get_state_count(), 1)
                self.assertTrue(state_vault.is_track_up_to_date(dummy_track))

                # Filter outdated tracks
                outdated = state_vault.filter_outdated_tracks([dummy_track])
                self.assertEqual(len(outdated), 0)

                # Clear state (truncate rows)
                cleared_count = state_vault.clear_state(purge=False)
                self.assertEqual(cleared_count, 1)
                self.assertEqual(state_vault.get_state_count(), 0)
                self.assertTrue(db_file.exists())

                # Re-record and purge
                state_vault.record_track_state(dummy_track, status="TAGGED_OK")
                purged_count = state_vault.clear_state(purge=True)
                self.assertEqual(purged_count, 1)
                self.assertFalse(db_file.exists())

                # Re-record after purge on the same instance (verifying schema recreation)
                state_vault.record_track_state(dummy_track, status="TAGGED_OK")
                self.assertEqual(state_vault.get_state_count(), 1)
                self.assertTrue(state_vault.is_track_up_to_date(dummy_track))
            finally:
                state_vault.close()

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
            cache_clear_result = clear_cache(
                clear_api=True, clear_state=False, clear_memory=True, purge=False
            )
            self.assertIsInstance(cache_clear_result, ClearResult)
            self.assertTrue(cache_clear_result.api_cleared)
            self.assertFalse(cache_clear_result.state_cleared)
            self.assertEqual(cache_clear_result.api_entries_cleared, 1)

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

    def test_multithreaded_cache_access_and_close(self) -> None:
        """Worker threads can concurrently read/write cached API responses and close cleanly."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"XDG_CACHE_HOME": tmpdir}),
        ):
            close_cache()

            def worker(k: str, v: str) -> None:
                try:
                    set_cached_api(k, v)
                    self.assertEqual(get_cached_api(k), v)
                finally:
                    cache = get_cache()
                    if cache is not None:
                        cache.close()

            threads = [
                threading.Thread(target=worker, args=(f"k{i}", f"v{i}"))
                for i in range(4)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            close_cache()


class TestCacheStatsAndResult(IsolatedTestCase):
    def test_cache_stats_dataclass_and_to_dict(self) -> None:
        stats_path = Path("/tmp/sonora_test_cache")
        stats = CacheStats(
            cache_dir=stats_path,
            api_entries=42,
            api_size_bytes=1048576,
            state_entries=100,
            state_size_bytes=524288,
            memory_metadata_entries=15,
            total_size_bytes=1572864,
        )
        self.assertEqual(stats.cache_dir, stats_path)
        self.assertEqual(stats.api_entries, 42)
        self.assertEqual(stats.api_size_bytes, 1048576)
        self.assertEqual(stats.state_entries, 100)
        self.assertEqual(stats.state_size_bytes, 524288)
        self.assertEqual(stats.memory_metadata_entries, 15)
        self.assertEqual(stats.total_size_bytes, 1572864)

        stats_dict = stats.to_dict()
        expected_dict = {
            "cache_dir": str(stats_path),
            "api_entries": 42,
            "api_size_bytes": 1048576,
            "state_entries": 100,
            "state_size_bytes": 524288,
            "memory_metadata_entries": 15,
            "total_size_bytes": 1572864,
        }
        self.assertEqual(stats_dict, expected_dict)

    def test_clear_result_dataclass_and_to_dict(self) -> None:
        result_path = Path("/tmp/sonora_test_cache")
        clear_res = ClearResult(
            cache_dir=result_path,
            api_cleared=True,
            api_entries_cleared=10,
            api_bytes_freed=2048,
            state_cleared=True,
            state_entries_cleared=5,
            state_bytes_freed=4096,
            memory_cleared=True,
            memory_metadata_cleared=3,
            purged=False,
            dry_run=True,
        )
        self.assertEqual(clear_res.total_bytes_freed, 6144)
        result_dict = clear_res.to_dict()
        self.assertEqual(
            result_dict,
            {
                "cache_dir": str(result_path),
                "dry_run": True,
                "api_cleared": True,
                "api_entries_cleared": 10,
                "api_bytes_freed": 2048,
                "state_cleared": True,
                "state_entries_cleared": 5,
                "state_bytes_freed": 4096,
                "memory_cleared": True,
                "memory_metadata_cleared": 3,
                "purged": False,
                "total_bytes_freed": 6144,
            },
        )

    def test_cache_stats_reports_read_errors(self) -> None:
        unreadable_cache = MagicMock()
        unreadable_cache.__len__.side_effect = OSError("database is locked")
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory)
            (cache_path / "cache.db").touch()
            for cache in (None, unreadable_cache):
                with (
                    self.subTest(cache=cache),
                    patch(
                        "sonora.core.cache.get_api_cache_dir", return_value=cache_path
                    ),
                    patch("sonora.core.cache.get_cache", return_value=cache),
                    self.assertRaisesRegex(OSError, "API cache"),
                ):
                    get_cache_stats()

    def test_get_cache_stats_populated(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"XDG_CACHE_HOME": tmpdir}),
        ):
            close_cache()
            reset_library_state()
            try:
                set_cached_api("stat_key_1", {"album": "OK Computer"})
                set_cached_api("stat_key_2", {"album": "In Rainbows"})

                from sonora.core.state import get_library_state

                state_vault = get_library_state()
                track_path = Path(tmpdir) / "track1.flac"
                track_path.write_bytes(b"FLACDATA")
                state_vault.record_track_state(track_path, status="TAGGED_OK")

                stats = get_cache_stats()
                self.assertIsInstance(stats, CacheStats)
                self.assertEqual(stats.api_entries, 2)
                self.assertGreater(stats.api_size_bytes, 0)
                self.assertEqual(stats.state_entries, 1)
                self.assertGreater(stats.state_size_bytes, 0)
                self.assertEqual(
                    stats.total_size_bytes,
                    stats.api_size_bytes + stats.state_size_bytes,
                )
                self.assertIn("api_entries", stats.to_dict())
            finally:
                close_cache()
                reset_library_state()


class TestClearCachePermutations(IsolatedTestCase):
    def test_clear_cache_dry_run_permutations(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"XDG_CACHE_HOME": tmpdir}),
        ):
            close_cache()
            reset_library_state()
            try:
                set_cached_api("dry_run_key", {"data": "test"})
                from sonora.core.state import get_library_state

                state_vault = get_library_state()
                dummy_track = Path(tmpdir) / "dummy.flac"
                dummy_track.write_bytes(b"audio")
                state_vault.record_track_state(dummy_track, status="TAGGED_OK")

                # Dry run with all flags True
                res_all = clear_cache(
                    clear_api=True,
                    clear_state=True,
                    clear_memory=True,
                    purge=False,
                    dry_run=True,
                )
                self.assertTrue(res_all.dry_run)
                self.assertTrue(res_all.api_cleared)
                self.assertTrue(res_all.state_cleared)
                self.assertTrue(res_all.memory_cleared)
                self.assertFalse(res_all.purged)
                self.assertEqual(res_all.api_entries_cleared, 1)
                self.assertEqual(res_all.state_entries_cleared, 1)
                self.assertGreater(res_all.total_bytes_freed, 0)

                stats = get_cache_stats()
                self.assertEqual(stats.api_entries, 1)
                self.assertEqual(stats.state_entries, 1)

                # Dry run with all flags False
                res_none = clear_cache(
                    clear_api=False,
                    clear_state=False,
                    clear_memory=False,
                    purge=False,
                    dry_run=True,
                )
                self.assertTrue(res_none.dry_run)
                self.assertFalse(res_none.api_cleared)
                self.assertFalse(res_none.state_cleared)
                self.assertFalse(res_none.memory_cleared)
                self.assertEqual(res_none.api_entries_cleared, 0)
                self.assertEqual(res_none.state_entries_cleared, 0)
                self.assertEqual(res_none.total_bytes_freed, 0)
            finally:
                close_cache()
                reset_library_state()

    def test_clear_cache_state_only(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"XDG_CACHE_HOME": tmpdir}),
        ):
            close_cache()
            reset_library_state()
            try:
                set_cached_api("preserve_api_key", {"data": "keep_me"})
                from sonora.core.state import get_library_state

                state_vault = get_library_state()
                dummy_track = Path(tmpdir) / "dummy.flac"
                dummy_track.write_bytes(b"audio")
                state_vault.record_track_state(dummy_track, status="TAGGED_OK")

                cache_clear_result = clear_cache(
                    clear_api=False,
                    clear_state=True,
                    clear_memory=False,
                    purge=False,
                    dry_run=False,
                )
                self.assertFalse(cache_clear_result.api_cleared)
                self.assertTrue(cache_clear_result.state_cleared)
                self.assertEqual(cache_clear_result.state_entries_cleared, 1)

                stats = get_cache_stats()
                self.assertEqual(stats.api_entries, 1)
                self.assertEqual(stats.state_entries, 0)
                self.assertEqual(
                    get_cached_api("preserve_api_key"), {"data": "keep_me"}
                )
            finally:
                close_cache()
                reset_library_state()

    def test_clear_cache_memory_only(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"XDG_CACHE_HOME": tmpdir}),
        ):
            close_cache()
            reset_library_state()
            try:
                set_cached_api("api_key", {"data": "val"})
                cache_clear_result = clear_cache(
                    clear_api=False,
                    clear_state=False,
                    clear_memory=True,
                    purge=False,
                    dry_run=False,
                )
                self.assertFalse(cache_clear_result.api_cleared)
                self.assertFalse(cache_clear_result.state_cleared)
                self.assertTrue(cache_clear_result.memory_cleared)
                self.assertEqual(get_cache_stats().api_entries, 1)
            finally:
                close_cache()
                reset_library_state()

    def test_clear_cache_all_live(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"XDG_CACHE_HOME": tmpdir}),
        ):
            close_cache()
            reset_library_state()
            try:
                set_cached_api("all_key", {"data": "val"})
                from sonora.core.state import get_library_state

                state_vault = get_library_state()
                dummy_track = Path(tmpdir) / "dummy.flac"
                dummy_track.write_bytes(b"audio")
                state_vault.record_track_state(dummy_track, status="TAGGED_OK")

                cache_clear_result = clear_cache(
                    clear_api=True,
                    clear_state=True,
                    clear_memory=True,
                    purge=False,
                    dry_run=False,
                )
                self.assertTrue(cache_clear_result.api_cleared)
                self.assertTrue(cache_clear_result.state_cleared)
                self.assertTrue(cache_clear_result.memory_cleared)
                self.assertFalse(cache_clear_result.purged)

                stats = get_cache_stats()
                self.assertEqual(stats.api_entries, 0)
                self.assertEqual(stats.state_entries, 0)
                # Databases should still exist
                self.assertTrue((get_api_cache_dir() / "cache.db").exists())
                self.assertTrue(state_vault.db_path.exists())
            finally:
                close_cache()
                reset_library_state()

    def test_clear_cache_purge_mode(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"XDG_CACHE_HOME": tmpdir}),
        ):
            close_cache()
            reset_library_state()
            try:
                set_cached_api("purge_key", {"data": "to_be_purged"})
                from sonora.core.state import get_library_state

                state_vault = get_library_state()
                dummy_track = Path(tmpdir) / "dummy.flac"
                dummy_track.write_bytes(b"audio")
                state_vault.record_track_state(dummy_track, status="TAGGED_OK")

                api_dir = get_api_cache_dir()
                self.assertTrue(api_dir.exists())

                cache_clear_result = clear_cache(
                    clear_api=True,
                    clear_state=True,
                    clear_memory=True,
                    purge=True,
                    dry_run=False,
                )
                self.assertTrue(cache_clear_result.purged)
                self.assertTrue(cache_clear_result.api_cleared)
                self.assertTrue(cache_clear_result.state_cleared)
                self.assertFalse(api_dir.exists())
                self.assertFalse(state_vault.db_path.exists())
            finally:
                close_cache()
                reset_library_state()
