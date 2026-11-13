"""
Unit tests for Sonora caching layer, XDG Base Directory compliance,
MetadataDisk msgpack serialization, SQLite state vault, and atomic storage contracts.
"""

import errno
import io
import os
import shutil
import sqlite3
import stat
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import diskcache
import msgpack
from filelock import FileLock

from sonora.cli.main import main
from sonora.core import cache as cache_module
from sonora.core import coordination, state
from sonora.core.cache import (
    CacheStats,
    ClearResult,
    api_cache_session,
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
from sonora.core.cache_storage import CacheDecodeError, MetadataDisk
from sonora.core.constants import DIRS
from sonora.core.coordination import coalesce_lookup
from sonora.core.files import (
    atomic_audio_edit,
    atomic_write_bytes,
    move_file_exclusively,
)
from sonora.core.state import (
    LibraryStateVault,
    get_default_state_db_path,
    get_library_state,
    get_state_storage_stats,
    reset_library_state,
)
from tests.support import IsolatedTestCase


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


def _worker_cache_session(
    fail: bool = False,
) -> tuple[diskcache.Cache, sqlite3.Connection]:
    with api_cache_session():
        owned_cache = get_cache()
        assert owned_cache is not None
        owned_connection = owned_cache._con
        with api_cache_session():
            assert get_cache() is owned_cache
            set_cached_api("worker-session", {"title": "Recording"})
        if fail:
            raise OSError("worker failure")
    assert get_cache() is None
    return owned_cache, owned_connection


def _close_and_remove_api_directory(api_directory: Path) -> None:
    close_cache()
    shutil.rmtree(api_directory)


def _cache_without_storage() -> object:
    with cache_module.api_cache_session():
        return cache_module.get_cached_api("uncached recording")


def _synchronize_initialization(
    barrier: threading.Barrier, lock: threading.RLock
) -> None:
    barrier.wait(timeout=3)
    lock.acquire()


class TestDiskCache(IsolatedTestCase):
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
    def test_confirmed_misses_last_a_day_without_extending_explicit_expiry(
        self, mock_get_cache: MagicMock
    ) -> None:
        values: tuple[Any, ...] = (None, "", {}, [], {"results": []})
        for value in values:
            with self.subTest(value=value):
                mock_get_cache.return_value.set.reset_mock()
                set_cached_api("missing", value)
                mock_get_cache.return_value.set.assert_called_once_with(
                    "missing", "" if value is None else value, expire=86400
                )
        mock_get_cache.return_value.set.reset_mock()
        set_cached_api("missing", None, expire_seconds=60)
        mock_get_cache.return_value.set.assert_called_once_with(
            "missing", "", expire=60
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

    def test_clear_cache_with_isolated_env(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"XDG_CACHE_HOME": tmpdir}),
        ):
            close_cache()
            reset_library_state()

            set_cached_api("unit_test_key", {"data": 123})
            stats_before = get_cache_stats()
            self.assertIsInstance(stats_before, CacheStats)
            self.assertEqual(stats_before.api_entries, 1)

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

            self.assertEqual(main(["cache", "stats"]), 0)
            self.assertEqual(main(["cache", "stats", "--json"]), 0)
            self.assertEqual(main(["cache"]), 0)
            self.assertEqual(main(["cache", "clear", "--dry-run"]), 0)
            self.assertEqual(
                main(["cache", "clear", "--all", "--dry-run", "--json"]), 0
            )
            self.assertEqual(main(["clear-cache", "--dry-run"]), 0)
            self.assertEqual(main(["cache", "clear"]), 0)
            self.assertEqual(main(["cache", "clear", "--all"]), 0)
            self.assertEqual(main(["cache", "clear", "--all", "--purge"]), 0)

            close_cache()
            reset_library_state()

    def test_multithreaded_cache_access_and_close(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"XDG_CACHE_HOME": tmpdir}),
        ):
            close_cache()

            def worker(index: int) -> None:
                k, v = f"k{index}", f"v{index}"
                try:
                    set_cached_api(k, v)
                    self.assertEqual(get_cached_api(k), v)
                finally:
                    c = get_cache()
                    if c is not None:
                        c.close()

            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(worker, range(4)))

            close_cache()


class TestCacheStatsAndResult(IsolatedTestCase):
    def test_cache_stats_dataclass_and_to_dict(self) -> None:
        stats_path = Path("/tmp/sonora_test_cache")
        stats_obj = CacheStats(
            cache_dir=stats_path,
            api_entries=42,
            api_size_bytes=1048576,
            state_entries=100,
            state_size_bytes=524288,
            memory_metadata_entries=15,
            total_size_bytes=1572864,
        )
        self.assertEqual(stats_obj.cache_dir, stats_path)
        self.assertEqual(stats_obj.api_entries, 42)
        self.assertEqual(stats_obj.api_size_bytes, 1048576)
        self.assertEqual(stats_obj.state_entries, 100)
        self.assertEqual(stats_obj.state_size_bytes, 524288)
        self.assertEqual(stats_obj.memory_metadata_entries, 15)
        self.assertEqual(stats_obj.total_size_bytes, 1572864)

        stats_dict = stats_obj.to_dict()
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
            for c in (None, unreadable_cache):
                with (
                    self.subTest(cache=c),
                    patch(
                        "sonora.core.cache.get_api_cache_dir", return_value=cache_path
                    ),
                    patch("sonora.core.cache.get_cache", return_value=c),
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

                vault = get_library_state()
                track_path = Path(tmpdir) / "track1.flac"
                track_path.write_bytes(b"FLACDATA")
                vault.record_track_state(track_path, status="TAGGED_OK")

                stats_obj = get_cache_stats()
                self.assertIsInstance(stats_obj, CacheStats)
                self.assertEqual(stats_obj.api_entries, 2)
                self.assertGreater(stats_obj.api_size_bytes, 0)
                self.assertEqual(stats_obj.state_entries, 1)
                self.assertGreater(stats_obj.state_size_bytes, 0)
                self.assertEqual(
                    stats_obj.total_size_bytes,
                    stats_obj.api_size_bytes + stats_obj.state_size_bytes,
                )
                self.assertIn("api_entries", stats_obj.to_dict())
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
                vault = get_library_state()
                dummy_track = Path(tmpdir) / "dummy.flac"
                dummy_track.write_bytes(b"audio")
                vault.record_track_state(dummy_track, status="TAGGED_OK")

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

                stats_obj = get_cache_stats()
                self.assertEqual(stats_obj.api_entries, 1)
                self.assertEqual(stats_obj.state_entries, 1)

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
                vault = get_library_state()
                dummy_track = Path(tmpdir) / "dummy.flac"
                dummy_track.write_bytes(b"audio")
                vault.record_track_state(dummy_track, status="TAGGED_OK")

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

                stats_obj = get_cache_stats()
                self.assertEqual(stats_obj.api_entries, 1)
                self.assertEqual(stats_obj.state_entries, 0)
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
                vault = get_library_state()
                dummy_track = Path(tmpdir) / "dummy.flac"
                dummy_track.write_bytes(b"audio")
                vault.record_track_state(dummy_track, status="TAGGED_OK")

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

                stats_obj = get_cache_stats()
                self.assertEqual(stats_obj.api_entries, 0)
                self.assertEqual(stats_obj.state_entries, 0)
                self.assertTrue((get_api_cache_dir() / "cache.db").exists())
                self.assertTrue(vault.db_path.exists())
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
                vault = get_library_state()
                dummy_track = Path(tmpdir) / "dummy.flac"
                dummy_track.write_bytes(b"audio")
                vault.record_track_state(dummy_track, status="TAGGED_OK")

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
                self.assertFalse(vault.db_path.exists())
            finally:
                close_cache()
                reset_library_state()


class TestMetadataCacheStorage(IsolatedTestCase):
    def test_catalog_indexes_and_artwork_types_survive_reopening(self) -> None:
        metadata = {
            "artist": "Beyoncé / 宇多田ヒカル",
            "tracks": [{"title": "Halo", "position": 1}],
            "by_position": {1: {"title": "Halo"}, (2, 1): {"title": "第二曲"}},
            "artwork": (b"\x00\xffcover", None),
            "credits": ("Composer", ("Producer", "Remixer")),
            "bpm": 123.456789,
            "explicit": False,
        }
        set_cached_api("catalog", metadata)
        close_cache()
        restored = get_cached_api("catalog")
        self.assertEqual(restored, metadata)
        assert isinstance(restored, dict)
        self.assertIsInstance(restored["tracks"], list)
        self.assertIsInstance(restored["artwork"], tuple)
        self.assertIn((2, 1), restored["by_position"])

    def test_binary_assets_use_external_cache_files(self) -> None:
        asset = bytes(range(256)) * 1024
        set_cached_api("artwork", (asset, asset[:12]))
        self.assertEqual(get_cached_api("artwork"), (asset, asset[:12]))
        self.assertTrue(any(get_api_cache_dir().rglob("*.val")))

    def test_negative_catalog_results_remain_distinct(self) -> None:
        for index, empty_result in enumerate((False, "", [], {}, ())):
            with self.subTest(empty_result=empty_result):
                key = f"empty:{index}"
                set_cached_api(key, empty_result)
                restored = get_cached_api(key)
                self.assertEqual(restored, empty_result)
                self.assertIs(type(restored), type(empty_result))
        self.assertIsNone(get_cached_api("missing"))

    def test_tuple_keys_roundtrip_through_cache_iteration(self) -> None:
        c = get_cache()
        assert c is not None
        c.set(("album", 2, 1), {"title": "Second disc"})
        self.assertEqual(list(c.iterkeys()), [("album", 2, 1)])
        self.assertEqual(c.get(("album", 2, 1)), {"title": "Second disc"})

    def test_corrupted_values_are_skipped_without_base_deserialization(self) -> None:
        set_cached_api("damaged", {"title": "Title"})
        with (
            closing(sqlite3.connect(get_api_cache_dir() / "cache.db")) as database,
            database,
        ):
            database.execute(
                "UPDATE Cache SET value = ?, mode = 4, filename = NULL",
                (b"cos\nsystem\n.",),
            )
        with patch(
            "diskcache.Disk.fetch",
            side_effect=AssertionError("Base deserialization must not run"),
        ):
            self.assertIsNone(get_cached_api("damaged"))
        set_cached_api("damaged", {"title": "Recovered"})
        self.assertEqual(get_cached_api("damaged"), {"title": "Recovered"})

    def test_pickle_payload_cannot_execute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "executed"
            payload = b"cos\nsystem\n(S'touch " + str(sentinel).encode() + b"'\ntR."
            disk = MetadataDisk(directory)
            with self.assertRaises(CacheDecodeError):
                disk.fetch(4, None, payload, False)
            with self.assertRaises(CacheDecodeError):
                disk.get(payload, False)
            self.assertFalse(sentinel.exists())

    def test_unknown_extensions_and_invalid_tuple_payloads_are_rejected(self) -> None:
        payloads = (
            msgpack.packb(msgpack.ExtType(2, b"unknown")),
            msgpack.packb(msgpack.ExtType(1, msgpack.packb("not an array"))),
            b"\xc1",
            b"\x81",
            b"\x00\x00",
        )
        with tempfile.TemporaryDirectory() as directory:
            disk = MetadataDisk(directory)
            for payload in payloads:
                with self.subTest(payload=payload), self.assertRaises(CacheDecodeError):
                    disk.fetch(0, None, payload, False)

    def test_cache_files_cannot_escape_the_cache_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_directory = Path(directory) / "cache"
            cache_directory.mkdir()
            external = Path(directory) / "external.val"
            encoded_metadata = msgpack.packb("private metadata")
            assert encoded_metadata is not None
            external.write_bytes(encoded_metadata)
            disk = MetadataDisk(str(cache_directory))
            for filename in ("../external.val", str(external)):
                with (
                    self.subTest(filename=filename),
                    self.assertRaises(CacheDecodeError),
                ):
                    disk.fetch(2, filename, None, False)
            (cache_directory / "linked.val").symlink_to(external)
            with self.assertRaises(CacheDecodeError):
                disk.fetch(2, "linked.val", None, False)

    def test_unsupported_objects_surface_as_programming_errors(self) -> None:
        with self.assertRaisesRegex(TypeError, "Unsupported metadata cache type"):
            set_cached_api("unsupported", object())

    def test_missing_binary_cache_file_is_an_operational_error(self) -> None:
        set_cached_api("artwork", b"x" * 100_000)
        for cache_file in get_api_cache_dir().rglob("*.val"):
            cache_file.unlink()
        self.assertIsNone(get_cached_api("artwork"))

    def test_metadata_cache_rejects_file_stream_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            disk = MetadataDisk(directory)
            with self.assertRaises(TypeError):
                disk.store(io.BytesIO(b"asset"), True)
            with self.assertRaises(TypeError):
                disk.fetch(0, None, b"asset", True)

    def test_cache_uses_only_its_current_serialization_directory(self) -> None:
        previous_cache = get_api_cache_dir().parent / "api"
        previous_cache.mkdir(parents=True)
        old_database = previous_cache / "cache.db"
        old_database.write_bytes(b"unsupported previous format")
        set_cached_api("current", {"title": "Title"})
        self.assertEqual(get_cached_api("current"), {"title": "Title"})
        self.assertEqual(old_database.read_bytes(), b"unsupported previous format")
        self.assertEqual(get_api_cache_dir().name, "api-msgpack-v1")

    def test_duplicate_encoded_map_keys_are_rejected(self) -> None:
        packed = msgpack.Packer().pack_map_pairs(
            [("title", "first"), ("title", "second")]
        )
        disk = MetadataDisk(str(self.test_directory))
        with self.assertRaises(CacheDecodeError):
            disk.fetch(0, None, packed, False)

    def test_cache_directory_cannot_be_used_as_a_value_file(self) -> None:
        disk = MetadataDisk(str(self.test_directory))
        with self.assertRaises(CacheDecodeError):
            disk.fetch(0, ".", None, False)
        with self.assertRaises(CacheDecodeError):
            disk.fetch(0, None, "not binary", False)
        with self.assertRaises(CacheDecodeError):
            disk.get("not binary", True)
        disk.remove(None)
        self.assertTrue(self.test_directory.is_dir())


class TestSqliteVacuumAndCheckpoint(IsolatedTestCase):
    def test_clear_cache_vacuum_outside_transaction(self) -> None:
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
                    cache_clear_result = clear_cache(
                        clear_api=True,
                        clear_state=False,
                        clear_memory=False,
                        purge=False,
                    )
                    self.assertTrue(cache_clear_result.api_cleared)

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

    def test_clear_cache_vacuum_error_propagates(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"XDG_CACHE_HOME": tmpdir}),
        ):
            close_cache()
            reset_library_state()
            try:
                set_cached_api("vacuum_err_key", {"title": "Test Title"})
                with (
                    patch(
                        "sqlite3.connect",
                        side_effect=sqlite3.OperationalError("disk I/O error"),
                    ),
                    self.assertRaisesRegex(sqlite3.OperationalError, "disk I/O error"),
                ):
                    clear_cache(
                        clear_api=True,
                        clear_state=False,
                        clear_memory=False,
                        purge=False,
                    )
            finally:
                close_cache()
                reset_library_state()


class TestCacheMaintenanceAndSafety(IsolatedTestCase):
    def test_unhashable_map_keys_are_cache_misses(self) -> None:
        invalid_keys = (
            msgpack.packb([1]),
            msgpack.packb(msgpack.ExtType(1, msgpack.packb([[]]))),
        )
        for encoded_key in invalid_keys:
            assert encoded_key is not None
            encoded_value = msgpack.packb("value")
            assert encoded_value is not None
            set_cached_api("damaged", {"title": "Title"})
            with (
                closing(sqlite3.connect(get_api_cache_dir() / "cache.db")) as database,
                database,
            ):
                database.execute(
                    "UPDATE Cache SET value = ?, filename = NULL",
                    (b"\x81" + encoded_key + encoded_value,),
                )
            self.assertIsNone(get_cached_api("damaged"))

    def test_cache_removal_rejects_external_paths(self) -> None:
        c = get_cache()
        assert c is not None
        external = get_api_cache_dir().parent / "outside.val"
        external.write_bytes(b"keep")
        for operation in ("clear", "delete", "cull"):
            for filename in ("../outside.val", str(external)):
                with self.subTest(operation=operation, filename=filename):
                    c.set("artwork", b"x" * 100_000)
                    with (
                        closing(
                            sqlite3.connect(get_api_cache_dir() / "cache.db")
                        ) as database,
                        database,
                    ):
                        database.execute("UPDATE Cache SET filename = ?", (filename,))
                    with self.assertRaises(CacheDecodeError):
                        if operation == "clear":
                            c.clear()
                        elif operation == "delete":
                            del c["artwork"]
                        else:
                            c.reset("size_limit", 0)
                            try:
                                c.cull()
                            finally:
                                c.reset("size_limit", 1 << 30)
                    self.assertEqual(external.read_bytes(), b"keep")
                    with (
                        closing(
                            sqlite3.connect(get_api_cache_dir() / "cache.db")
                        ) as database,
                        database,
                    ):
                        database.execute(
                            "UPDATE Cache SET filename = NULL, value = ?",
                            (msgpack.packb(None),),
                        )

    def test_cache_removal_rejects_parent_symlink_escape(self) -> None:
        cache_directory = get_api_cache_dir()
        cache_directory.mkdir(parents=True)
        external = cache_directory.parent / "outside.val"
        external.write_bytes(b"keep")
        (cache_directory / "linked").symlink_to(
            cache_directory.parent, target_is_directory=True
        )
        with self.assertRaises(CacheDecodeError):
            MetadataDisk(str(cache_directory)).remove("linked/outside.val")
        self.assertEqual(external.read_bytes(), b"keep")

    def test_normal_shard_removal_preserves_cache_root(self) -> None:
        directory = get_api_cache_dir()
        shard = directory / "aa" / "bb" / "record.val"
        shard.parent.mkdir(parents=True)
        shard.write_bytes(b"record")
        MetadataDisk(str(directory)).remove("aa/bb/record.val")
        self.assertFalse(shard.exists())
        self.assertTrue(directory.is_dir())

    def test_purge_failure_surfaces_without_success_summary(self) -> None:
        set_cached_api("recording", {"title": "Title"})
        with (
            patch(
                "sonora.core.cache.shutil.rmtree",
                side_effect=PermissionError("locked cache"),
            ),
            patch("sonora.cli.cache.LOG.success") as success,
        ):
            self.assertEqual(main(["cache", "clear", "--api", "--purge"]), 1)
            success.assert_not_called()
        self.assertEqual(get_cached_api("recording"), {"title": "Title"})

    def test_clear_database_failure_surfaces_without_success_summary(self) -> None:
        c = get_cache()
        assert c is not None
        with (
            patch.object(
                c, "clear", side_effect=sqlite3.OperationalError("locked cache")
            ),
            patch("sonora.cli.cache.LOG.success") as success,
        ):
            self.assertEqual(main(["cache", "clear", "--api"]), 1)
            success.assert_not_called()

    def test_successful_clear_uses_actual_removed_count(self) -> None:
        set_cached_api("first", "First")
        set_cached_api("second", "Second")
        result = cache_module.clear_cache(clear_memory=False)
        self.assertEqual(result.api_entries_cleared, 2)
        self.assertIsNone(get_cached_api("first"))

    def test_state_purge_failure_propagates(self) -> None:
        vault = get_library_state()
        track = vault.db_path.parent / "track.flac"
        track.write_bytes(b"audio")
        vault.record_track_state(track)
        with (
            patch.object(Path, "unlink", side_effect=PermissionError("locked state")),
            self.assertRaises(PermissionError),
        ):
            vault.clear_state(purge=True)
        self.assertEqual(vault.get_state_count(), 1)

    def test_state_primary_key_has_no_redundant_index(self) -> None:
        vault = LibraryStateVault(get_api_cache_dir().parent / "state.db")
        self.addCleanup(vault.close)
        with closing(sqlite3.connect(vault.db_path)) as database:
            indexes = list(database.execute("PRAGMA index_list('track_state')"))
        self.assertEqual(len(indexes), 1)

    def test_shard_cleanup_tolerates_concurrent_or_nonempty_directories(self) -> None:
        disk = MetadataDisk(str(self.test_directory))
        for error_number in (errno.ENOENT, errno.EEXIST, errno.ENOTEMPTY):
            shard = self.test_directory / "aa" / "record.val"
            shard.parent.mkdir(exist_ok=True)
            shard.write_bytes(b"recording")
            with (
                self.subTest(error_number=error_number),
                patch(
                    "pathlib.Path.rmdir",
                    side_effect=OSError(error_number, "directory changed"),
                ),
            ):
                disk.remove("aa/record.val")
            self.assertFalse(shard.exists())
            self.assertTrue(self.test_directory.exists())

    def test_shard_cleanup_surfaces_permission_errors(self) -> None:
        disk = MetadataDisk(str(self.test_directory))
        shard = self.test_directory / "aa" / "record.val"
        shard.parent.mkdir()
        shard.write_bytes(b"recording")
        with (
            patch(
                "pathlib.Path.rmdir",
                side_effect=PermissionError(errno.EACCES, "locked directory"),
            ),
            self.assertRaises(PermissionError),
        ):
            disk.remove("aa/record.val")
        self.assertFalse(shard.exists())
        self.assertTrue(self.test_directory.exists())


class TestCacheWorkerCoordination(IsolatedTestCase):
    def test_lookup_lock_files_use_a_bounded_set_of_shards(self) -> None:
        @coordination.coalesce_lookup
        def lookup(key: str) -> str:
            return key

        with patch.object(coordination, "_LOOKUP_LOCK_SHARDS", 4):
            for index in range(32):
                self.assertEqual(lookup(str(index)), str(index))

        lock_files = list((get_cache_dir() / "coordination" / "lookups").glob("*.lock"))
        self.assertLessEqual(len(lock_files), 4)

    def test_coalesced_worker_lookup_reuses_one_cache_session(self) -> None:
        @coalesce_lookup
        def lookup(key: str) -> str:
            value = get_cached_api(key)
            if value is None:
                set_cached_api(key, "cached")
            return str(get_cached_api(key))

        with (
            patch.object(
                cache_module,
                "_open_api_cache",
                wraps=cache_module._open_api_cache,
            ) as open_cache,
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            result = executor.submit(lookup, "worker-session").result()

        self.assertEqual(result, "cached")
        open_cache.assert_called_once_with(get_api_cache_dir())

    def test_nested_lookups_do_not_reacquire_the_same_shard_lock(self) -> None:
        @coalesce_lookup
        def inner_lookup() -> str:
            return "complete"

        @coalesce_lookup
        def outer_lookup() -> str:
            return inner_lookup()

        def short_lock(path: str, timeout: float = 180) -> object:
            return FileLock(path, timeout=0.05)

        with (
            patch.object(coordination, "_LOOKUP_LOCK_SHARDS", 1),
            patch("sonora.core.coordination.FileLock", side_effect=short_lock),
        ):
            self.assertEqual(outer_lookup(), "complete")

    def test_worker_closes_owned_connection_and_preserves_main_connection(self) -> None:
        shared_cache = get_cache()
        assert shared_cache is not None
        main_connection = shared_cache._con
        original_close = diskcache.Cache.close
        with (
            patch.object(
                diskcache.Cache, "close", autospec=True, side_effect=original_close
            ) as close_call,
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            owned_cache, owned_connection = executor.submit(
                _worker_cache_session
            ).result()
            self.assertIs(owned_cache, shared_cache)
            self.assertIsNot(owned_connection, main_connection)
            with self.assertRaisesRegex(sqlite3.ProgrammingError, "closed database"):
                _ = owned_connection.in_transaction
            self.assertTrue(
                any(call.args[0] is owned_cache for call in close_call.call_args_list)
            )
            self.assertIs(get_cache(), shared_cache)
            self.assertIs(shared_cache._con, main_connection)
            self.assertEqual(main_connection.execute("SELECT 1").fetchone(), (1,))
            self.assertEqual(get_cached_api("worker-session"), {"title": "Recording"})

    def test_worker_exception_closes_session_and_allows_next_task(self) -> None:
        original_close = diskcache.Cache.close
        with (
            patch.object(
                diskcache.Cache, "close", autospec=True, side_effect=original_close
            ) as close_call,
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            with self.assertRaisesRegex(OSError, "worker failure"):
                executor.submit(_worker_cache_session, True).result()
            self.assertTrue(close_call.called)
            executor.submit(_worker_cache_session).result()
        self.assertEqual(get_cached_api("worker-session"), {"title": "Recording"})

    def test_main_session_reuses_shared_cache(self) -> None:
        shared_cache = get_cache()
        with api_cache_session():
            self.assertIs(get_cache(), shared_cache)
        self.assertIs(get_cache(), shared_cache)

    def test_worker_cache_failure_is_a_miss_and_does_not_close_shared_cache(
        self,
    ) -> None:
        shared = cache_module.get_cache()
        with (
            patch("sonora.core.cache._open_api_cache", return_value=None),
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            self.assertIsNone(pool.submit(_cache_without_storage).result())
        self.assertIs(cache_module.get_cache(), shared)


class TestAtomicFileContracts(IsolatedTestCase):
    def test_atomic_replacement_preserves_existing_permissions(self) -> None:
        destination = self.test_directory / "lyrics.lrc"
        destination.write_bytes(b"old")
        destination.chmod(0o640)
        atomic_write_bytes(destination, b"new")
        self.assertEqual(destination.read_bytes(), b"new")
        self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o640)

    def test_audio_replacement_with_same_size_and_timestamp_is_not_overwritten(
        self,
    ) -> None:
        destination = self.test_directory / "recording.flac"
        destination.write_bytes(b"old audio")
        before = destination.stat()
        replacement = self.test_directory / "replacement.flac"
        replacement.write_bytes(b"new audio")
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        with (
            self.assertRaisesRegex(OSError, "changed during tagging"),
            atomic_audio_edit(destination) as staging,
        ):
            staging.write_bytes(b"tagged audio")
            replacement.replace(destination)
        self.assertEqual(destination.read_bytes(), b"new audio")
        self.assertEqual(list(self.test_directory.glob(".recording.*")), [])

    def test_sync_failure_keeps_original_and_removes_staging_file(self) -> None:
        destination = self.test_directory / "lyrics.lrc"
        destination.write_bytes(b"original lyrics")
        with (
            patch("sonora.core.files.os.fsync", side_effect=OSError("disk full")),
            self.assertRaisesRegex(OSError, "disk full"),
        ):
            atomic_write_bytes(destination, b"replacement lyrics")
        self.assertEqual(destination.read_bytes(), b"original lyrics")
        self.assertEqual(list(self.test_directory.glob(".lyrics.lrc.*")), [])

    def test_cross_device_move_preserves_contents_and_timestamps(self) -> None:
        source = self.test_directory / "source.flac"
        source.write_bytes(b"complete recording")
        original_mtime = source.stat().st_mtime_ns
        destination = self.test_directory / "destination.flac"
        with patch(
            "sonora.core.files.os.link",
            side_effect=OSError(errno.EXDEV, "cross device"),
        ):
            move_file_exclusively(source, destination)
        self.assertFalse(source.exists())
        self.assertEqual(destination.read_bytes(), b"complete recording")
        self.assertEqual(destination.stat().st_mtime_ns, original_mtime)

    def test_move_falls_back_when_filesystem_rejects_hard_links(self) -> None:
        source = self.test_directory / "source.flac"
        source.write_bytes(b"complete recording")
        destination = self.test_directory / "destination.flac"
        with patch(
            "sonora.core.files.os.link",
            side_effect=OSError(errno.EPERM, "hard links unsupported"),
        ):
            move_file_exclusively(source, destination)
        self.assertFalse(source.exists())
        self.assertEqual(destination.read_bytes(), b"complete recording")

    def test_hard_link_fallback_refuses_existing_destination(self) -> None:
        source = self.test_directory / "source.flac"
        destination = self.test_directory / "destination.flac"
        source.write_bytes(b"source recording")
        destination.write_bytes(b"existing recording")
        with (
            patch(
                "sonora.core.files.os.link",
                side_effect=OSError(errno.EPERM, "hard links unsupported"),
            ),
            self.assertRaises(FileExistsError),
        ):
            move_file_exclusively(source, destination)
        self.assertEqual(source.read_bytes(), b"source recording")
        self.assertEqual(destination.read_bytes(), b"existing recording")

    def test_failed_cross_device_copy_removes_partial_destination(self) -> None:
        source = self.test_directory / "source.flac"
        source.write_bytes(b"complete recording")
        destination = self.test_directory / "destination.flac"
        with (
            patch(
                "sonora.core.files.os.link",
                side_effect=OSError(errno.EXDEV, "cross device"),
            ),
            patch(
                "sonora.core.files.shutil.copyfileobj", side_effect=OSError("disk full")
            ),
            self.assertRaisesRegex(OSError, "disk full"),
        ):
            move_file_exclusively(source, destination)
        self.assertEqual(source.read_bytes(), b"complete recording")
        self.assertFalse(destination.exists())

    def test_cross_device_move_refuses_existing_destination(self) -> None:
        source = self.test_directory / "source.flac"
        source.write_bytes(b"first recording")
        destination = self.test_directory / "destination.flac"
        destination.write_bytes(b"other recording")
        with (
            patch(
                "sonora.core.files.os.link",
                side_effect=OSError(errno.EXDEV, "cross device"),
            ),
            self.assertRaises(FileExistsError),
        ):
            move_file_exclusively(source, destination)
        self.assertEqual(source.read_bytes(), b"first recording")
        self.assertEqual(destination.read_bytes(), b"other recording")


class TestLibraryStateVault(IsolatedTestCase):
    def setUp(self) -> None:
        self.db_path = self.test_directory / "test_state.db"
        self.vault = LibraryStateVault(db_path=self.db_path)
        self.addCleanup(self.vault.close)

        self.test_file1 = self.test_directory / "song1.flac"
        self.test_file2 = self.test_directory / "song2.flac"
        self.test_file1.write_bytes(b"dummy audio 1")
        self.test_file2.write_bytes(b"dummy audio 2")

    def test_storage_stats_do_not_create_a_database_for_an_unused_library(self) -> None:
        state_path = get_default_state_db_path()
        self.assertFalse(state_path.exists())
        self.assertEqual(get_state_storage_stats(), (0, 0))
        self.assertFalse(state_path.exists())

    def test_storage_stats_include_active_state_and_reopen_existing_database(
        self,
    ) -> None:
        library_state = get_library_state()
        library_state.record_track_state(self.test_file1, status="TAGGED_OK")
        for reopen in (False, True):
            with self.subTest(reopen=reopen):
                if reopen:
                    reset_library_state()
                entry_count, storage_bytes = get_state_storage_stats()
                self.assertEqual(entry_count, 1)
                self.assertGreater(storage_bytes, 0)

    def test_storage_stats_report_io_failures(self) -> None:
        with (
            patch.object(Path, "stat", side_effect=PermissionError("locked file")),
            self.assertRaises(OSError),
        ):
            get_state_storage_stats()
        library_state = get_library_state()
        with (
            patch.object(Path, "stat", side_effect=PermissionError("locked file")),
            self.assertRaises(OSError),
        ):
            get_state_storage_stats()
        self.assertIs(get_library_state(), library_state)

    def test_state_lifecycle(self) -> None:
        self.assertFalse(self.vault.is_track_up_to_date(self.test_file1))
        outdated = self.vault.filter_outdated_tracks([self.test_file1, self.test_file2])
        self.assertEqual(len(outdated), 2)

        self.vault.record_track_state(self.test_file1, status="TAGGED_OK")
        self.assertTrue(self.vault.is_track_up_to_date(self.test_file1))
        self.assertFalse(self.vault.is_track_up_to_date(self.test_file2))

        outdated = self.vault.filter_outdated_tracks([self.test_file1, self.test_file2])
        self.assertEqual(outdated, [self.test_file2])

        self.vault.record_tracks_state_batch([self.test_file2], status="TAGGED_OK")
        self.assertTrue(self.vault.is_track_up_to_date(self.test_file2))

        outdated_empty = self.vault.filter_outdated_tracks(
            [self.test_file1, self.test_file2]
        )
        self.assertEqual(len(outdated_empty), 0)

        self.test_file1.write_bytes(b"modified audio data")
        self.assertFalse(self.vault.is_track_up_to_date(self.test_file1))
        outdated_modified = self.vault.filter_outdated_tracks(
            [self.test_file1, self.test_file2]
        )
        self.assertEqual(outdated_modified, [self.test_file1])

    def test_is_track_up_to_date_supports_hash_status(self) -> None:
        self.vault.record_tracks_state_batch(
            [self.test_file1], status="TAGGED_OK:abcdef123456"
        )
        self.assertTrue(self.vault.is_track_up_to_date(self.test_file1))
        self.assertTrue(
            self.vault.is_track_up_to_date(
                self.test_file1, expected_status="TAGGED_OK:abcdef123456"
            )
        )
        self.assertFalse(
            self.vault.is_track_up_to_date(
                self.test_file1, expected_status="TAGGED_OK:different_hash"
            )
        )

    def test_artist_alias_storage_lifecycle(self) -> None:
        self.assertEqual(self.vault.get_artist_alias_count(), 0)
        self.assertIsNone(self.vault.get_artist_alias("k-dot"))

        self.vault.record_artist_alias("k-dot", "Kendrick Lamar", source="musicbrainz")
        self.assertEqual(self.vault.get_artist_alias_count(), 2)
        self.assertEqual(self.vault.get_artist_alias("k-dot"), "Kendrick Lamar")
        self.assertEqual(self.vault.get_artist_alias("kdot"), "Kendrick Lamar")

        self.vault.record_artist_aliases(
            [
                ("colson baker", "Machine Gun Kelly"),
                ("mgk", "Machine Gun Kelly"),
            ],
            source="musicbrainz",
        )
        self.assertEqual(self.vault.get_artist_alias_count(), 5)
        self.assertEqual(self.vault.get_artist_alias("mgk"), "Machine Gun Kelly")
        self.assertEqual(
            self.vault.get_artist_alias("colson baker"), "Machine Gun Kelly"
        )

        all_aliases = self.vault.get_all_artist_aliases()
        self.assertIn("kdot", all_aliases)
        self.assertIn("mgk", all_aliases)

        cleared = self.vault.clear_artist_aliases()
        self.assertEqual(cleared, 5)
        self.assertEqual(self.vault.get_artist_alias_count(), 0)
        self.assertIsNone(self.vault.get_artist_alias("k-dot"))

    def test_library_state_survives_clear_purge_and_reuse(self) -> None:
        self.assertEqual(self.vault.get_state_count(), 0)
        self.assertGreater(self.vault.get_state_size(), 0)

        dummy_track = self.test_directory / "song.mp3"
        dummy_track.write_text("dummy audio")
        self.vault.record_track_state(dummy_track, status="TAGGED_OK")

        self.assertEqual(self.vault.get_state_count(), 1)
        self.assertTrue(self.vault.is_track_up_to_date(dummy_track))

        cleared_count = self.vault.clear_state(purge=False)
        self.assertEqual(cleared_count, 1)
        self.assertEqual(self.vault.get_state_count(), 0)
        self.assertTrue(self.db_path.exists())

        self.vault.record_track_state(dummy_track, status="TAGGED_OK")
        purged_count = self.vault.clear_state(purge=True)
        self.assertEqual(purged_count, 1)
        self.assertFalse(self.db_path.exists())

        self.vault.record_track_state(dummy_track, status="TAGGED_OK")
        self.assertEqual(self.vault.get_state_count(), 1)
        self.assertTrue(self.vault.is_track_up_to_date(dummy_track))

    def test_missing_and_changed_files_remain_outdated(self) -> None:
        missing = self.test_directory / "missing.flac"
        self.assertFalse(self.vault.is_track_up_to_date(missing))
        self.assertEqual(self.vault.filter_outdated_tracks([missing]), [missing])
        self.assertEqual(self.vault.filter_outdated_tracks([]), [])
        self.vault.record_tracks_state_batch([missing])
        self.vault.record_tracks_state_batch([])
        self.assertEqual(self.vault.get_state_count(), 0)
        self.vault.record_track_state(self.test_file1)
        self.test_file1.write_bytes(b"different audio")
        self.assertEqual(
            self.vault.filter_outdated_tracks([self.test_file1]), [self.test_file1]
        )

    def test_database_read_failure_does_not_skip_processing(self) -> None:
        with patch.object(
            self.vault,
            "_get_connection",
            side_effect=sqlite3.OperationalError("locked database"),
        ):
            self.assertFalse(self.vault.is_track_up_to_date(self.test_file1))
            self.assertEqual(
                self.vault.filter_outdated_tracks([self.test_file1]), [self.test_file1]
            )
            with self.assertRaisesRegex(OSError, "Cannot count library state entries"):
                self.vault.get_state_count()
            self.vault.record_track_state(self.test_file1)
            self.vault.record_tracks_state_batch([self.test_file1])
        self.assertEqual(self.vault.get_state_count(), 0)

    def test_failed_transaction_discards_connection_and_can_reconnect(self) -> None:
        connection = self.vault._get_connection()
        with (
            self.assertRaises(sqlite3.OperationalError),
            self.vault._connection() as transaction,
        ):
            transaction.execute("SELECT * FROM missing_table")
        self.assertIsNot(self.vault._get_connection(), connection)
        self.vault.record_track_state(self.test_file1)
        self.assertTrue(self.vault.is_track_up_to_date(self.test_file1))

    def test_simultaneous_first_calls_create_one_shared_vault(self) -> None:
        barrier = threading.Barrier(2)
        lock = threading.RLock()
        coordinated_lock = Mock()
        coordinated_lock.__enter__ = Mock(
            side_effect=partial(_synchronize_initialization, barrier, lock)
        )
        coordinated_lock.__exit__ = Mock(side_effect=lambda *arguments: lock.release())
        with (
            patch.object(state, "_STATE_LOCK", coordinated_lock),
            ThreadPoolExecutor(max_workers=2) as workers,
        ):
            futures = [workers.submit(state.get_library_state) for _ in range(2)]
            vaults = [future.result(timeout=5) for future in futures]
        self.assertIs(vaults[0], vaults[1])
        self.assertIsInstance(vaults[0], LibraryStateVault)


class TestCacheAndStateErrorBoundaries(IsolatedTestCase):
    def test_cache_size_counts_files_below_nested_directories(self) -> None:
        directory = self.test_directory / "nested"
        directory.mkdir()
        (directory / "record.val").write_bytes(b"recording")
        self.assertEqual(cache_module._get_api_cache_size(self.test_directory), 9)

    def test_cache_stats_report_unavailable_client(self) -> None:
        cache_module.get_cache()
        with (
            patch.object(cache_module, "get_cache", return_value=None),
            self.assertRaisesRegex(OSError, "Cannot read API cache statistics"),
        ):
            cache_module.get_cache_stats()

    def test_cache_purge_tolerates_directory_removal_after_close(self) -> None:
        cache_module.set_cached_api("recording", "metadata")
        api_directory = cache_module.get_api_cache_dir()
        with patch.object(
            cache_module,
            "close_cache",
            side_effect=partial(_close_and_remove_api_directory, api_directory),
        ):
            cleared = cache_module.clear_cache(purge=True, clear_memory=False)
        self.assertEqual(cleared.api_entries_cleared, 1)
        self.assertFalse(api_directory.exists())

    def test_cache_maintenance_skips_a_concurrently_removed_database(self) -> None:
        c = cache_module.get_cache()
        assert c is not None
        database_path = cache_module.get_api_cache_dir() / "cache.db"
        with patch.object(
            c, "check", side_effect=lambda **options: database_path.unlink()
        ):
            cleared = cache_module.clear_cache(clear_memory=False)
        self.assertEqual(cleared.api_entries_cleared, 0)
        self.assertFalse(database_path.exists())

    def test_unwritable_cache_initialization_is_a_cache_miss(self) -> None:
        cache_module.close_cache()
        with patch("pathlib.Path.mkdir", side_effect=PermissionError("read only")):
            self.assertIsNone(cache_module.get_cached_api("recording"))
            cache_module.set_cached_api("recording", {"title": "Title"})
        cache_module.set_cached_api("recording", {"title": "Title"})
        self.assertEqual(cache_module.get_cached_api("recording"), {"title": "Title"})

    def test_missing_cache_storage_cannot_report_successful_clear(self) -> None:
        with (
            patch("sonora.core.cache.get_cache", return_value=None),
            self.assertRaisesRegex(OSError, "Cannot open API cache"),
        ):
            cache_module.clear_cache(clear_memory=False)

    def test_cache_size_handles_missing_files_and_io_failures(self) -> None:
        self.assertEqual(
            cache_module._get_api_cache_size(self.test_directory / "missing"), 0
        )
        audio_path = self.test_directory / "recording.val"
        audio_path.write_bytes(b"abc")
        self.assertEqual(cache_module._get_api_cache_size(audio_path), 3)
        self.assertEqual(cache_module._get_api_cache_size(self.test_directory), 3)
        with patch(
            "pathlib.Path.rglob", side_effect=PermissionError("locked directory")
        ):
            self.assertEqual(cache_module._get_api_cache_size(self.test_directory), 0)
        with patch("pathlib.Path.stat", side_effect=PermissionError("locked file")):
            self.assertEqual(cache_module._get_api_cache_size(self.test_directory), 0)
        with patch(
            "pathlib.Path.exists", side_effect=PermissionError("locked directory")
        ):
            self.assertEqual(cache_module._get_api_cache_size(self.test_directory), 0)

    def test_cache_size_skips_unreadable_entries_and_counts_readable_files(
        self,
    ) -> None:
        readable_entry = self.test_directory / "readable.val"
        readable_entry.write_bytes(b"abc")
        for io_error in (
            PermissionError("locked file"),
            FileNotFoundError("removed file"),
        ):
            unreadable_entry = Mock(spec=Path)
            unreadable_entry.is_file.return_value = True
            unreadable_entry.stat.side_effect = io_error
            with (
                self.subTest(io_error=type(io_error).__name__),
                patch.object(
                    Path, "rglob", return_value=iter((unreadable_entry, readable_entry))
                ),
            ):
                self.assertEqual(
                    cache_module._get_api_cache_size(self.test_directory), 3
                )
            unreadable_entry.stat.assert_called_once_with()

    def test_cache_stats_report_database_length_failure(self) -> None:
        cache_module.get_cache()
        failing_cache = Mock()
        failing_cache.__len__ = Mock(
            side_effect=sqlite3.OperationalError("locked database")
        )
        with (
            patch("sonora.core.cache.get_cache", return_value=failing_cache),
            self.assertRaisesRegex(OSError, "Cannot count API cache entries"),
        ):
            cache_module.get_cache_stats()

    def test_cache_close_failure_does_not_reuse_failed_client(self) -> None:
        c = cache_module.get_cache()
        assert c is not None
        with patch.object(
            c, "close", side_effect=sqlite3.OperationalError("locked database")
        ):
            cache_module.close_cache()
        c.close()
        self.assertIsNot(cache_module.get_cache(), c)

    def test_cache_programming_errors_are_not_silenced(self) -> None:
        for exception in (RuntimeError("internal bug"), ValueError("internal bug")):
            with (
                self.subTest(exception=exception),
                patch("sonora.core.cache.get_cache") as c,
            ):
                c.return_value.get.side_effect = exception
                c.return_value.set.side_effect = exception
                with self.assertRaises(type(exception)):
                    get_cached_api("track")
                with self.assertRaises(type(exception)):
                    set_cached_api("track", {"title": "Title"})

    def test_unavailable_database_is_an_operational_error(self) -> None:
        with patch("sonora.core.cache.get_cache") as c:
            c.return_value.get.side_effect = sqlite3.OperationalError("locked")
            c.return_value.set.side_effect = sqlite3.OperationalError("locked")
            self.assertIsNone(get_cached_api("track"))
            set_cached_api("track", {"title": "Title"})

    def test_corrupted_serialized_cache_entry_is_an_operational_error(self) -> None:
        with patch("sonora.core.cache.get_cache") as c:
            c.return_value.get.side_effect = CacheDecodeError("corrupted entry")
            self.assertIsNone(get_cached_api("track"))
