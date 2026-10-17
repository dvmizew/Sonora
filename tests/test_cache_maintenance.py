"""
Unit tests for Sonora caching layer, XDG Base Directory compliance, and CLI cache management.
"""

import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

from sonora.core.cache import (
    clear_cache,
    close_cache,
    get_api_cache_dir,
    set_cached_api,
)
from sonora.core.state import LibraryStateVault, reset_library_state
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
