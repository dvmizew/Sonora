import io
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import msgpack

from sonora.core.cache import (
    get_api_cache_dir,
    get_cache,
    get_cached_api,
    set_cached_api,
)
from sonora.core.cache_storage import CacheDecodeError, MetadataDisk
from tests.support import IsolatedTestCase


class TestMetadataCacheStorage(IsolatedTestCase):
    def test_catalog_indexes_and_artwork_types_survive_reopening(self) -> None:
        from sonora.core.cache import close_cache

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
        cache = get_cache()
        assert cache is not None
        cache.set(("album", 2, 1), {"title": "Second disc"})
        self.assertEqual(list(cache.iterkeys()), [("album", 2, 1)])
        self.assertEqual(cache.get(("album", 2, 1)), {"title": "Second disc"})

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
