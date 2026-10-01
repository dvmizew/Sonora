"""
Unit tests for Sonora caching layer, XDG Base Directory compliance, and CLI cache management.
"""

import os
import sqlite3
import sys
import tempfile
import threading
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import diskcache

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import unittest

from sonora.audio.metadata import clear_metadata_cache, get_metadata_cache_size
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
from sonora.core.utils import clear_utils_cache, format_filesize


class _SpyingConnection:
    """Wrapper around sqlite3.Connection to inspect statement execution and transaction isolation."""

    def __init__(
        self,
        real_conn: sqlite3.Connection,
        statements: list[str],
        isolation_levels: list[str | None],
    ) -> None:
        self._conn = real_conn
        self._statements = statements
        self._isolation_levels = isolation_levels

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

    @property
    def isolation_level(self) -> Any:
        return self._conn.isolation_level

    @isolation_level.setter
    def isolation_level(self, level: Any) -> None:
        self._conn.isolation_level = level

    def execute(self, sql: str, *args: Any) -> sqlite3.Cursor:
        self._statements.append(sql.strip())
        if "VACUUM" in sql.upper() or "WAL_CHECKPOINT" in sql.upper():
            self._isolation_levels.append(self._conn.isolation_level)
        return self._conn.execute(sql, *args)

    def close(self) -> None:
        self._conn.close()


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

    @patch("sonora.core.cache.get_cache")
    def test_set_cached_api_none_value(self, mock_get_cache: MagicMock) -> None:
        set_cached_api("key_none", None)
        mock_get_cache.assert_not_called()

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

        # Should log and handle gracefully without raising
        set_cached_api("timeout_key", {"data": 123})

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

    def test_dirs_cache_path(self) -> None:
        self.assertEqual(get_cache_dir(), DIRS.user_cache_path)

    def test_dirs_config_dir_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            expected = Path.home() / ".config" / "sonora"
            self.assertEqual(DIRS.user_config_path, expected)

    def test_dirs_config_dir_xdg_override(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"XDG_CONFIG_HOME": tmpdir}),
        ):
            self.assertEqual(DIRS.user_config_path, Path(tmpdir) / "sonora")

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
            state_mgr = LibraryStateVault(db_file)
            try:
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
            finally:
                state_mgr.close()

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


class TestCacheStatsAndResult(unittest.TestCase):
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


class TestClearCachePermutations(unittest.TestCase):
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

                # Ensure disk data was NOT modified
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

                res = clear_cache(
                    clear_api=False,
                    clear_state=True,
                    clear_memory=False,
                    purge=False,
                    dry_run=False,
                )
                self.assertFalse(res.api_cleared)
                self.assertTrue(res.state_cleared)
                self.assertEqual(res.state_entries_cleared, 1)

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
                res = clear_cache(
                    clear_api=False,
                    clear_state=False,
                    clear_memory=True,
                    purge=False,
                    dry_run=False,
                )
                self.assertFalse(res.api_cleared)
                self.assertFalse(res.state_cleared)
                self.assertTrue(res.memory_cleared)
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

                res = clear_cache(
                    clear_api=True,
                    clear_state=True,
                    clear_memory=True,
                    purge=False,
                    dry_run=False,
                )
                self.assertTrue(res.api_cleared)
                self.assertTrue(res.state_cleared)
                self.assertTrue(res.memory_cleared)
                self.assertFalse(res.purged)

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

                res = clear_cache(
                    clear_api=True,
                    clear_state=True,
                    clear_memory=True,
                    purge=True,
                    dry_run=False,
                )
                self.assertTrue(res.purged)
                self.assertTrue(res.api_cleared)
                self.assertTrue(res.state_cleared)
                self.assertFalse(api_dir.exists())
                self.assertFalse(state_vault.db_path.exists())
            finally:
                close_cache()
                reset_library_state()


class TestSqliteVacuumAndCheckpoint(unittest.TestCase):
    def test_clear_cache_vacuum_outside_transaction(self) -> None:
        """Verify VACUUM and PRAGMA wal_checkpoint(TRUNCATE) execute with isolation_level = None."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"XDG_CACHE_HOME": tmpdir}),
        ):
            close_cache()
            reset_library_state()
            try:
                set_cached_api("vacuum_test_key", {"title": "Test Title"})
                api_db_path = get_api_cache_dir() / "cache.db"
                self.assertTrue(api_db_path.exists())

                real_connect = sqlite3.connect
                observed_isolation_levels: list[str | None] = []
                executed_statements: list[str] = []

                def mock_connect(*args: Any, **kwargs: Any) -> _SpyingConnection:
                    real_conn = real_connect(*args, **kwargs)
                    return _SpyingConnection(
                        real_conn, executed_statements, observed_isolation_levels
                    )

                with patch("sqlite3.connect", side_effect=mock_connect):
                    res = clear_cache(
                        clear_api=True,
                        clear_state=False,
                        clear_memory=False,
                        purge=False,
                    )
                    self.assertTrue(res.api_cleared)

                self.assertIn("VACUUM;", executed_statements)
                self.assertIn("PRAGMA wal_checkpoint(TRUNCATE);", executed_statements)
                self.assertTrue(
                    all(level is None for level in observed_isolation_levels),
                    f"Expected isolation_level to be None, got {observed_isolation_levels}",
                )
            finally:
                close_cache()
                reset_library_state()

    def test_library_state_vacuum_outside_transaction(self) -> None:
        """Verify LibraryStateVault.clear_state executes VACUUM with isolation_level = None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "state_test.db"
            vault = LibraryStateVault(db_path)
            try:
                track_path = Path(tmpdir) / "test.flac"
                track_path.write_bytes(b"DATA")
                vault.record_track_state(track_path, status="TAGGED_OK")

                real_connect = sqlite3.connect
                observed_isolation_levels: list[str | None] = []
                executed_statements: list[str] = []

                def mock_connect(*args: Any, **kwargs: Any) -> _SpyingConnection:
                    real_conn = real_connect(*args, **kwargs)
                    return _SpyingConnection(
                        real_conn, executed_statements, observed_isolation_levels
                    )

                with patch("sqlite3.connect", side_effect=mock_connect):
                    cleared_count = vault.clear_state(purge=False)
                    self.assertEqual(cleared_count, 1)

                self.assertIn("VACUUM;", executed_statements)
                self.assertIn("PRAGMA wal_checkpoint(TRUNCATE);", executed_statements)
                self.assertTrue(
                    all(level is None for level in observed_isolation_levels),
                    f"Expected isolation_level to be None, got {observed_isolation_levels}",
                )
            finally:
                vault.close()

    def test_clear_cache_vacuum_error_handled(self) -> None:
        """Verify sqlite3 error during VACUUM in clear_cache is safely handled."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"XDG_CACHE_HOME": tmpdir}),
        ):
            close_cache()
            reset_library_state()
            try:
                set_cached_api("vacuum_err_key", {"title": "Test Title"})
                with patch(
                    "sqlite3.connect",
                    side_effect=sqlite3.OperationalError("disk I/O error"),
                ):
                    res = clear_cache(
                        clear_api=True,
                        clear_state=False,
                        clear_memory=False,
                        purge=False,
                    )
                    self.assertTrue(res.api_cleared)
            finally:
                close_cache()
                reset_library_state()


class TestXDGCacheCompliance(unittest.TestCase):
    def test_api_cache_dir_xdg_isolation(self) -> None:
        """Verify DiskCache storage is isolated in api/ subdirectory under XDG cache dir (Rule 8)."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"XDG_CACHE_HOME": tmpdir}),
        ):
            close_cache()
            reset_library_state()
            try:
                base_cache_dir = get_cache_dir()
                api_dir = get_api_cache_dir()
                self.assertEqual(api_dir, base_cache_dir / "api")

                cache_instance = get_cache()
                self.assertIsNotNone(cache_instance)
                self.assertEqual(Path(cache_instance.directory), api_dir)

                from sonora.core.state import get_library_state

                state_vault = get_library_state()
                self.assertEqual(
                    state_vault.db_path, base_cache_dir / "library_state.db"
                )
                self.assertNotEqual(state_vault.db_path.parent, api_dir)
            finally:
                close_cache()
                reset_library_state()


class TestCacheConcurrencyAndStorageInvariants(unittest.TestCase):
    """
    Test suite verifying internal storage invariants of Sonora's cache according to GEMINI.md Rule 8:
    1. API cache is isolated in api/ subdirectory and does not conflict with library_state.db.
    2. Thread safety: 16 concurrent threads performing reads/writes without deadlocks or locked errors.
    3. Negative caching: empty string '' and empty dict {} are preserved and distinguished from cache misses.
    4. Cache recovery: corrupted temp files / shards are cleanly recovered via check(fix=True) without exceptions.
    5. clear_cache(clear_api=True, clear_state=False) leaves library_state.db completely intact in normal and purge modes.
    """

    def test_api_cache_isolation_and_library_state_independence(self) -> None:
        """API cache is confined to api/ and check(fix=True) never touches library_state.db."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"XDG_CACHE_HOME": tmpdir}),
        ):
            close_cache()
            reset_library_state()
            try:
                base_dir = get_cache_dir()
                api_dir = get_api_cache_dir()
                self.assertEqual(api_dir, base_dir / "api")

                set_cached_api(
                    "test_song", {"artist": "Radiohead", "track": "Karma Police"}
                )
                self.assertEqual(
                    get_cached_api("test_song"),
                    {"artist": "Radiohead", "track": "Karma Police"},
                )

                from sonora.core.state import get_library_state

                state_vault = get_library_state()
                dummy_track = Path(tmpdir) / "track_01.flac"
                dummy_track.write_bytes(b"FLAC_AUDIO_CONTENT")
                state_vault.record_track_state(dummy_track, status="TAGGED_OK")

                state_db = base_dir / "library_state.db"
                api_db = api_dir / "cache.db"
                self.assertTrue(state_db.exists())
                self.assertTrue(api_db.exists())

                cache_inst = get_cache()
                self.assertIsNotNone(cache_inst)
                with warnings.catch_warnings(action="always"):
                    check_warnings = cache_inst.check(fix=True)
                self.assertIsInstance(check_warnings, list)

                self.assertTrue(state_db.exists())
                self.assertTrue(state_vault.is_track_up_to_date(dummy_track))
            finally:
                close_cache()
                reset_library_state()

    def test_concurrent_16_threads_reads_and_writes(self) -> None:
        """16 concurrent threads performing reads/writes execute with zero errors, locked databases, or deadlocks."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"XDG_CACHE_HOME": tmpdir}),
        ):
            close_cache()
            reset_library_state()
            try:
                num_threads = 16
                ops_per_thread = 50
                exceptions: list[Exception] = []
                locked_errors: list[sqlite3.OperationalError] = []
                mismatches: list[str] = []

                barrier = threading.Barrier(num_threads)

                def worker(thread_idx: int) -> None:
                    try:
                        barrier.wait(timeout=10.0)
                        for op_idx in range(ops_per_thread):
                            key = f"thread_{thread_idx}_key_{op_idx}"
                            payload = {
                                "thread": thread_idx,
                                "op": op_idx,
                                "data": f"content_{thread_idx}_{op_idx}" * 20,
                            }
                            set_cached_api(key, payload)
                            retrieved = get_cached_api(key)
                            if retrieved != payload:
                                mismatches.append(
                                    f"Key {key} mismatch: expected {payload}, got {retrieved}"
                                )

                            peer_thread = (thread_idx + 1) % num_threads
                            peer_key = f"thread_{peer_thread}_key_{op_idx}"
                            peer_val = get_cached_api(peer_key)
                            if peer_val is not None and (
                                peer_val.get("thread") != peer_thread
                                or peer_val.get("op") != op_idx
                            ):
                                mismatches.append(f"Cross-thread mismatch: {peer_val}")
                    except sqlite3.OperationalError as error:
                        locked_errors.append(error)
                        exceptions.append(error)
                    except (
                        diskcache.Timeout,
                        OSError,
                        ValueError,
                        RuntimeError,
                    ) as error:
                        exceptions.append(error)

                threads = [
                    threading.Thread(target=worker, args=(t_idx,))
                    for t_idx in range(num_threads)
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=30.0)
                    if thread.is_alive():
                        exceptions.append(
                            TimeoutError(f"Thread {thread} timed out / deadlocked.")
                        )

                self.assertEqual(
                    len(exceptions), 0, f"Exceptions occurred: {exceptions}"
                )
                self.assertEqual(
                    len(locked_errors), 0, f"Database locked errors: {locked_errors}"
                )
                self.assertEqual(len(mismatches), 0, f"Data mismatches: {mismatches}")
            finally:
                close_cache()
                reset_library_state()

    def test_negative_caching_empty_string_and_empty_dict(self) -> None:
        """Empty string '' and empty dict {} are stored, preserved, and distinguished from None (miss)."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"XDG_CACHE_HOME": tmpdir}),
        ):
            close_cache()
            reset_library_state()
            try:
                set_cached_api("neg_empty_str", "")
                result_str = get_cached_api("neg_empty_str")
                self.assertIsNotNone(result_str)
                self.assertEqual(result_str, "")

                set_cached_api("neg_empty_dict", {})
                result_dict = get_cached_api("neg_empty_dict")
                self.assertIsNotNone(result_dict)
                self.assertEqual(result_dict, {})

                set_cached_api("neg_empty_list", [])
                result_list = get_cached_api("neg_empty_list")
                self.assertIsNotNone(result_list)
                self.assertEqual(result_list, [])

                self.assertIsNone(get_cached_api("non_existent_key"))

                set_cached_api("neg_none_val", None)
                self.assertIsNone(get_cached_api("neg_none_val"))
            finally:
                close_cache()
                reset_library_state()

    def test_cache_recovery_check_fix_corrupted_files(self) -> None:
        """Corrupted temp files and shard directories are safely cleaned up without uncaught exceptions."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"XDG_CACHE_HOME": tmpdir}),
        ):
            close_cache()
            reset_library_state()
            try:
                for idx in range(10):
                    set_cached_api(f"valid_{idx}", {"id": idx, "name": f"Track {idx}"})

                api_dir = get_api_cache_dir()
                self.assertTrue(api_dir.exists())

                corrupt_tmp = api_dir / "orphaned_upload.tmp"
                corrupt_tmp.write_bytes(b"\xde\xad\xbe\xefcorrupted_bytes")

                corrupt_shard = api_dir / "099"
                corrupt_shard.mkdir(exist_ok=True)
                (corrupt_shard / "bad_shard.dat").write_bytes(b"garbage")

                cache_inst = get_cache()
                self.assertIsNotNone(cache_inst)

                with warnings.catch_warnings(record=True) as captured_warnings:
                    warnings.simplefilter("always")
                    check_warnings = cache_inst.check(fix=True)
                self.assertIsInstance(check_warnings, list)
                self.assertGreater(len(check_warnings) + len(captured_warnings), 0)

                self.assertFalse(corrupt_tmp.exists())

                for idx in range(10):
                    val = get_cached_api(f"valid_{idx}")
                    self.assertIsInstance(val, dict)
                    assert isinstance(val, dict)
                    self.assertEqual(val["id"], idx)

                set_cached_api("new_post_fix_key", {"status": "ok"})
                self.assertEqual(get_cached_api("new_post_fix_key"), {"status": "ok"})
            finally:
                close_cache()
                reset_library_state()

    def test_clear_cache_preserves_library_state_in_all_modes(self) -> None:
        """clear_cache with clear_state=False preserves library_state.db in both normal and purge modes."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"XDG_CACHE_HOME": tmpdir}),
        ):
            close_cache()
            reset_library_state()
            try:
                base_dir = get_cache_dir()
                state_db = base_dir / "library_state.db"

                from sonora.core.state import get_library_state

                state_vault = get_library_state()
                track_paths: list[Path] = []
                for idx in range(5):
                    track = Path(tmpdir) / f"song_{idx}.flac"
                    track.write_bytes(f"audio_{idx}".encode())
                    state_vault.record_track_state(track, status="TAGGED_OK")
                    track_paths.append(track)

                self.assertEqual(state_vault.get_state_count(), 5)
                self.assertTrue(state_db.exists())

                set_cached_api("api_entry_1", {"data": 1})
                self.assertIsNotNone(get_cached_api("api_entry_1"))

                res_normal = clear_cache(
                    clear_api=True, clear_state=False, clear_memory=True, purge=False
                )
                self.assertTrue(res_normal.api_cleared)
                self.assertFalse(res_normal.state_cleared)

                self.assertIsNone(get_cached_api("api_entry_1"))
                self.assertTrue(state_db.exists())
                self.assertEqual(state_vault.get_state_count(), 5)
                for track in track_paths:
                    self.assertTrue(state_vault.is_track_up_to_date(track))

                set_cached_api("api_entry_2", {"data": 2})
                self.assertIsNotNone(get_cached_api("api_entry_2"))

                res_purge = clear_cache(
                    clear_api=True, clear_state=False, clear_memory=True, purge=True
                )
                self.assertTrue(res_purge.api_cleared)
                self.assertFalse(res_purge.state_cleared)
                self.assertTrue(res_purge.purged)

                self.assertIsNone(get_cached_api("api_entry_2"))
                self.assertTrue(state_db.exists())
                self.assertEqual(state_vault.get_state_count(), 5)
                for track in track_paths:
                    self.assertTrue(state_vault.is_track_up_to_date(track))
            finally:
                close_cache()
                reset_library_state()


if __name__ == "__main__":
    unittest.main()
