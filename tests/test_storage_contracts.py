import errno
import shutil
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from unittest.mock import Mock, patch

import msgpack

from sonora.core import cache as cache_module
from sonora.core import state
from sonora.core.cache import close_cache
from sonora.core.cache_storage import CacheDecodeError, MetadataDisk
from sonora.core.files import atomic_write_bytes, move_file_exclusively
from sonora.core.state import LibraryStateVault
from tests.support import IsolatedTestCase


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


class TestConcurrentLibraryStateContracts(IsolatedTestCase):
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


class TestAtomicFileContracts(IsolatedTestCase):
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


class TestCacheFailureContracts(IsolatedTestCase):
    def test_cache_size_counts_files_below_nested_directories(self) -> None:
        directory = self.test_directory / "nested"
        directory.mkdir()
        (directory / "record.val").write_bytes(b"recording")
        self.assertEqual(cache_module._get_api_cache_size(self.test_directory), 9)

    def test_unavailable_cache_client_has_zero_entries_in_stats(self) -> None:
        cache_module.get_cache()
        with patch.object(cache_module, "get_cache", return_value=None):
            self.assertEqual(cache_module.get_cache_stats().api_entries, 0)

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
        cache = cache_module.get_cache()
        assert cache is not None
        database_path = cache_module.get_api_cache_dir() / "cache.db"
        with patch.object(
            cache, "check", side_effect=lambda **options: database_path.unlink()
        ):
            cleared = cache_module.clear_cache(clear_memory=False)
        self.assertEqual(cleared.api_entries_cleared, 0)
        self.assertFalse(database_path.exists())

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

    def test_cache_stats_tolerate_database_length_failure(self) -> None:
        cache_module.get_cache()
        failing_cache = Mock()
        failing_cache.__len__ = Mock(
            side_effect=sqlite3.OperationalError("locked database")
        )
        with patch("sonora.core.cache.get_cache", return_value=failing_cache):
            self.assertEqual(cache_module.get_cache_stats().api_entries, 0)

    def test_cache_close_failure_does_not_reuse_failed_client(self) -> None:
        cache = cache_module.get_cache()
        assert cache is not None
        with patch.object(
            cache, "close", side_effect=sqlite3.OperationalError("locked database")
        ):
            cache_module.close_cache()
        cache.close()
        self.assertIsNot(cache_module.get_cache(), cache)


class TestLibraryStateFailureContracts(IsolatedTestCase):
    def setUp(self) -> None:
        self.vault = LibraryStateVault(self.test_directory / "state.db")
        self.addCleanup(self.vault.close)
        self.audio_path = self.test_directory / "recording.flac"
        self.audio_path.write_bytes(b"audio")

    def test_missing_and_changed_files_remain_outdated(self) -> None:
        missing = self.test_directory / "missing.flac"
        self.assertFalse(self.vault.is_track_up_to_date(missing))
        self.assertEqual(self.vault.filter_outdated_tracks([missing]), [missing])
        self.assertEqual(self.vault.filter_outdated_tracks([]), [])
        self.vault.record_tracks_state_batch([missing])
        self.vault.record_tracks_state_batch([])
        self.assertEqual(self.vault.get_state_count(), 0)
        self.vault.record_track_state(self.audio_path)
        self.audio_path.write_bytes(b"different audio")
        self.assertEqual(
            self.vault.filter_outdated_tracks([self.audio_path]), [self.audio_path]
        )

    def test_database_read_failure_does_not_skip_processing(self) -> None:
        with patch.object(
            self.vault,
            "_get_connection",
            side_effect=sqlite3.OperationalError("locked database"),
        ):
            self.assertFalse(self.vault.is_track_up_to_date(self.audio_path))
            self.assertEqual(
                self.vault.filter_outdated_tracks([self.audio_path]), [self.audio_path]
            )
            self.assertEqual(self.vault.get_state_count(), 0)
            self.vault.record_track_state(self.audio_path)
            self.vault.record_tracks_state_batch([self.audio_path])
        self.assertEqual(self.vault.get_state_count(), 0)

    def test_failed_transaction_discards_connection_and_can_reconnect(self) -> None:
        connection = self.vault._get_connection()
        with (
            self.assertRaises(sqlite3.OperationalError),
            self.vault._connection() as transaction,
        ):
            transaction.execute("SELECT * FROM missing_table")
        self.assertIsNot(self.vault._get_connection(), connection)
        self.vault.record_track_state(self.audio_path)
        self.assertTrue(self.vault.is_track_up_to_date(self.audio_path))

    def test_state_recreates_schema_after_external_removal(self) -> None:
        self.vault.close()
        self.vault.db_path.unlink()
        self.assertEqual(self.vault.get_state_count(), 0)
        self.assertEqual(self.vault.get_state_size(), 0)
        self.assertEqual(self.vault.clear_state(), 0)
        self.vault.record_track_state(self.audio_path)
        self.assertTrue(self.vault.is_track_up_to_date(self.audio_path))

    def test_state_size_tolerates_disappearing_or_locked_files(self) -> None:
        with patch("pathlib.Path.stat", side_effect=PermissionError("locked state")):
            self.assertEqual(self.vault.get_state_size(), 0)
