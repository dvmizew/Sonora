import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from typing import Any
from unittest.mock import patch

import msgpack

from sonora.cli.main import main
from sonora.core import cache as cache_module
from sonora.core.cache import (
    clear_cache,
    close_cache,
    get_api_cache_dir,
    get_cache,
    get_cached_api,
    set_cached_api,
)
from sonora.core.cache_storage import CacheDecodeError, MetadataDisk
from sonora.core.state import LibraryStateVault, get_library_state, reset_library_state
from tests.support import IsolatedTestCase

"""
Unit tests for Sonora caching layer, XDG Base Directory compliance, and CLI cache management.
"""


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


class TestSqliteVacuumAndCheckpoint(IsolatedTestCase):
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


class TestCacheMaintenance(IsolatedTestCase):
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
        cache = get_cache()
        assert cache is not None
        external = get_api_cache_dir().parent / "outside.val"
        external.write_bytes(b"keep")
        for operation in ("clear", "delete", "cull"):
            for filename in ("../outside.val", str(external)):
                with self.subTest(operation=operation, filename=filename):
                    cache.set("artwork", b"x" * 100_000)
                    with (
                        closing(
                            sqlite3.connect(get_api_cache_dir() / "cache.db")
                        ) as database,
                        database,
                    ):
                        database.execute("UPDATE Cache SET filename = ?", (filename,))
                    with self.assertRaises(CacheDecodeError):
                        if operation == "clear":
                            cache.clear()
                        elif operation == "delete":
                            del cache["artwork"]
                        else:
                            cache.reset("size_limit", 0)
                            try:
                                cache.cull()
                            finally:
                                cache.reset("size_limit", 1 << 30)
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
        cache = get_cache()
        assert cache is not None
        with (
            patch.object(
                cache, "clear", side_effect=sqlite3.OperationalError("locked cache")
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
