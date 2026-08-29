import contextlib
import sqlite3
import threading
import time
from collections.abc import Generator
from pathlib import Path

from sonora.core.logger import LOG

_DB_PATH = Path.home() / ".cache" / "sonora" / "library_state.db"
_STATE_LOCK = threading.RLock()
_STATE_INSTANCE: "LibraryStateManager | None" = None


class LibraryStateManager:
    """Persistent SQLite-backed state tracker for library files."""

    def __init__(self, db_path: Path = _DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextlib.contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA temp_store=MEMORY;")
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with _STATE_LOCK, self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS track_state (
                    file_path TEXT PRIMARY KEY,
                    mtime_ns INTEGER NOT NULL,
                    file_size INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    last_tagged_timestamp REAL NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_track_path ON track_state(file_path);"
            )

    def is_track_up_to_date(self, file_path: Path) -> bool:
        """Check if file on disk has not changed since it was last successfully tagged."""
        try:
            stat = file_path.stat()
            current_mtime = stat.st_mtime_ns
            current_size = stat.st_size
            path_str = str(file_path.resolve())

            with _STATE_LOCK, self._connection() as conn:
                cursor = conn.execute(
                    "SELECT mtime_ns, file_size, status FROM track_state WHERE file_path = ?;",
                    (path_str,),
                )
                row = cursor.fetchone()
                if row:
                    mtime_ns, file_size, status = row
                    return bool(
                        mtime_ns == current_mtime
                        and file_size == current_size
                        and status == "TAGGED_OK"
                    )
        except (OSError, sqlite3.Error) as error:
            LOG.debug(f"State check failed for {file_path}: {error}")
        return False

    def filter_outdated_tracks(self, file_paths: list[Path]) -> list[Path]:
        """
        Efficiently batch-check a list of file paths against the SQLite state index.
        Returns only the file paths that are missing or have been modified.
        """
        if not file_paths:
            return []

        outdated: list[Path] = []
        path_to_stat: dict[str, tuple[Path, int, int]] = {}

        for p in file_paths:
            try:
                stat = p.stat()
                path_to_stat[str(p.resolve())] = (
                    p,
                    stat.st_mtime_ns,
                    stat.st_size,
                )
            except OSError:
                outdated.append(p)

        if not path_to_stat:
            return outdated

        try:
            with _STATE_LOCK, self._connection() as conn:
                resolved_keys = [(k,) for k in path_to_stat]
                state_map: dict[str, tuple[int, int, str]] = {}

                conn.execute(
                    "CREATE TEMP TABLE IF NOT EXISTS _batch_paths (path TEXT PRIMARY KEY);"
                )
                conn.execute("DELETE FROM _batch_paths;")
                conn.executemany(
                    "INSERT INTO _batch_paths (path) VALUES (?);", resolved_keys
                )
                cursor = conn.execute(
                    "SELECT t.file_path, t.mtime_ns, t.file_size, t.status FROM track_state t INNER JOIN _batch_paths b ON t.file_path = b.path;"
                )
                for row in cursor.fetchall():
                    state_map[row[0]] = (row[1], row[2], row[3])

                for path_str, (
                    orig_path,
                    current_mtime,
                    current_size,
                ) in path_to_stat.items():
                    if path_str in state_map:
                        cached_mtime, cached_size, status = state_map[path_str]
                        if (
                            cached_mtime == current_mtime
                            and cached_size == current_size
                            and status == "TAGGED_OK"
                        ):
                            continue
                    outdated.append(orig_path)
        except (sqlite3.Error, OSError) as error:
            LOG.debug(f"Batch state filter error: {error}")
            return file_paths

        return outdated

    def record_track_state(self, file_path: Path, status: str = "TAGGED_OK") -> None:
        """Record successful tagging state for a single file."""
        try:
            stat = file_path.stat()
            path_str = str(file_path.resolve())
            now = time.time()
            with _STATE_LOCK, self._connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO track_state (file_path, mtime_ns, file_size, status, last_tagged_timestamp)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    (path_str, stat.st_mtime_ns, stat.st_size, status, now),
                )
        except (OSError, sqlite3.Error) as error:
            LOG.debug(f"Failed to record state for {file_path}: {error}")

    def record_tracks_state_batch(
        self, file_paths: list[Path], status: str = "TAGGED_OK"
    ) -> None:
        """Batch record successful tagging state for multiple files."""
        if not file_paths:
            return
        records = []
        now = time.time()
        for p in file_paths:
            try:
                stat = p.stat()
                records.append(
                    (
                        str(p.resolve()),
                        stat.st_mtime_ns,
                        stat.st_size,
                        status,
                        now,
                    )
                )
            except OSError:
                pass

        if not records:
            return

        try:
            with _STATE_LOCK, self._connection() as conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO track_state (file_path, mtime_ns, file_size, status, last_tagged_timestamp)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    records,
                )
        except sqlite3.Error as error:
            LOG.debug(f"Failed to batch record state: {error}")


def get_library_state() -> LibraryStateManager:
    global _STATE_INSTANCE
    if _STATE_INSTANCE is None:
        with _STATE_LOCK:
            if _STATE_INSTANCE is None:
                _STATE_INSTANCE = LibraryStateManager()
    return _STATE_INSTANCE
