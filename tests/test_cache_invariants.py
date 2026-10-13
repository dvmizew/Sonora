"""
Unit tests for Sonora caching layer, XDG Base Directory compliance, and CLI cache management.
"""

import os
import sqlite3
import tempfile
import threading
import warnings
from pathlib import Path
from unittest.mock import patch

import diskcache

from sonora.core.cache import (
    clear_cache,
    close_cache,
    get_api_cache_dir,
    get_cache,
    get_cache_dir,
    get_cached_api,
    set_cached_api,
)
from sonora.core.state import reset_library_state
from tests.support import IsolatedTestCase


class TestXDGCacheCompliance(IsolatedTestCase):
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
                self.assertEqual(api_dir, base_cache_dir / "api-msgpack-v1")

                cache_instance = get_cache()
                self.assertIsNotNone(cache_instance)
                assert cache_instance is not None
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


class TestCacheConcurrencyAndStorageInvariants(IsolatedTestCase):
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
                self.assertEqual(api_dir, base_dir / "api-msgpack-v1")

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
                assert cache_inst is not None
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
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
        """Confirmed misses survive round trips while absent entries remain distinguishable."""
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
                self.assertEqual(get_cached_api("neg_none_val"), "")
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
                assert cache_inst is not None

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
