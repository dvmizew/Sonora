import errno
from pathlib import Path
from typing import Any

import diskcache
import msgpack

_TUPLE_EXTENSION = 1


class CacheDecodeError(ValueError):
    """A transient cache entry violates the MessagePack metadata schema."""


def _encode_tuple(value: object) -> msgpack.ExtType:
    # Lists hold API records; tuples hold disc/position keys and artwork pairs.
    if type(value) is not tuple:
        raise TypeError(f"Unsupported metadata cache type: {type(value).__name__}")
    return msgpack.ExtType(
        _TUPLE_EXTENSION,
        msgpack.packb(
            list(value), strict_types=True, use_bin_type=True, default=_encode_tuple
        ),
    )


def _decode_tuple(extension: int, encoded_values: bytes) -> tuple[Any, ...]:
    if extension != _TUPLE_EXTENSION:
        raise CacheDecodeError(f"Unknown metadata cache extension: {extension}")
    decoded_values = _decode_metadata(encoded_values)
    if not isinstance(decoded_values, list):
        raise CacheDecodeError("A metadata tuple must contain an array")
    return tuple(decoded_values)


def _is_metadata_key(value: object) -> bool:
    if value is None or isinstance(value, (str, bytes, int, float)):
        return True
    return isinstance(value, tuple) and all(_is_metadata_key(part) for part in value)


def _decode_map(pairs: list[tuple[Any, Any]]) -> dict[Any, Any]:
    metadata: dict[Any, Any] = {}
    for key, value in pairs:
        if not _is_metadata_key(key):
            raise CacheDecodeError("Metadata map keys must be scalars or scalar tuples")
        if key in metadata:
            raise CacheDecodeError("Metadata maps cannot contain duplicate keys")
        metadata[key] = value
    return metadata


def _decode_metadata(encoded_metadata: bytes) -> Any:
    try:
        return msgpack.unpackb(
            encoded_metadata,
            raw=False,
            strict_map_key=False,
            ext_hook=_decode_tuple,
            object_pairs_hook=_decode_map,
        )
    except (ValueError, msgpack.UnpackException) as error:
        raise CacheDecodeError("Invalid MessagePack metadata cache entry") from error


class MetadataDisk(diskcache.Disk):
    """Persist metadata without pickle, retaining catalog indexes and binary assets."""

    def __init__(self, directory: str, **settings: Any) -> None:
        super().__init__(directory, **settings)
        self.cache_directory = Path(directory).resolve()

    def _cache_file(self, filename: str) -> Path:
        relative_path = Path(filename)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise CacheDecodeError("Metadata cache file escapes its directory")
        cache_file = self.cache_directory / relative_path
        resolved_path = cache_file.resolve()
        if resolved_path == self.cache_directory or not resolved_path.is_relative_to(
            self.cache_directory
        ):
            raise CacheDecodeError("Metadata cache file escapes its directory")
        return cache_file

    def remove(self, filename: str | None) -> None:
        if filename is None:
            return
        cache_file = self._cache_file(filename)
        cache_file.unlink(missing_ok=True)
        parent = cache_file.parent
        while parent != self.cache_directory:
            try:
                parent.rmdir()
            except OSError as error:
                if error.errno not in (errno.ENOTEMPTY, errno.EEXIST, errno.ENOENT):
                    raise
                break
            parent = parent.parent

    def put(self, key: Any) -> tuple[Any, bool]:
        encoded_key = msgpack.packb(
            key, strict_types=True, use_bin_type=True, default=_encode_tuple
        )
        database_key, raw = super().put(encoded_key)
        return database_key, raw

    def get(self, key: Any, raw: bool) -> Any:
        if not raw or not isinstance(key, (bytes, memoryview)):
            raise CacheDecodeError("Metadata cache keys must use raw binary storage")
        return _decode_metadata(bytes(key))

    def store(
        self, value: Any, read: bool, key: Any = diskcache.UNKNOWN
    ) -> tuple[int, int, str | None, Any]:
        if read:
            raise TypeError("The metadata cache accepts values, not file streams")
        encoded_value = msgpack.packb(
            value, strict_types=True, use_bin_type=True, default=_encode_tuple
        )
        size, mode, filename, database_value = super().store(
            encoded_value, False, key=key
        )
        return size, mode, filename, database_value

    def fetch(self, mode: int, filename: str | None, value: Any, read: bool) -> Any:
        if read:
            raise TypeError("The metadata cache returns values, not file streams")
        # Never dispatch on a persisted mode through Disk.fetch: that can unpickle.
        if filename is None:
            if not isinstance(value, (bytes, memoryview)):
                raise CacheDecodeError("Metadata cache values must use binary storage")
            encoded_value = bytes(value)
        else:
            encoded_value = self._cache_file(filename).read_bytes()
        return _decode_metadata(encoded_value)
