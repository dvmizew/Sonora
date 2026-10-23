import errno
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO


@contextmanager
def atomic_audio_edit(audio_path: Path) -> Iterator[Path]:
    """Keep the original audio intact until every tag operation has succeeded."""
    original_stat = audio_path.stat()
    with tempfile.NamedTemporaryFile(
        dir=audio_path.parent,
        prefix=f".{audio_path.stem}.",
        suffix=audio_path.suffix,
        delete=False,
    ) as staging_file:
        staging_path = Path(staging_file.name)
    try:
        shutil.copy2(audio_path, staging_path)
        yield staging_path
        current_stat = audio_path.stat()
        if (current_stat.st_mtime_ns, current_stat.st_size) != (
            original_stat.st_mtime_ns,
            original_stat.st_size,
        ):
            raise OSError(f"Audio file changed during tagging: {audio_path}")
        with staging_path.open("rb") as audio_output:
            os.fsync(audio_output.fileno())
        staging_path.replace(audio_path)
    finally:
        staging_path.unlink(missing_ok=True)


@contextmanager
def atomic_write_file(destination: Path) -> Iterator[IO[bytes]]:
    """Publish streamed output only after writing, flushing and syncing succeeds."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staging_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}."
    )
    staging_path = Path(staging_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            yield output
            output.flush()
            os.fsync(output.fileno())
        staging_path.replace(destination)
    finally:
        staging_path.unlink(missing_ok=True)


def atomic_write_bytes(destination: Path, payload: bytes) -> None:
    with atomic_write_file(destination) as output:
        output.write(payload)


def move_file_exclusively(source: Path, destination: Path) -> None:
    """Publish a file without overwriting an existing destination, even across devices."""
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        with destination.open("xb") as output:
            try:
                with source.open("rb") as audio_input:
                    shutil.copyfileobj(audio_input, output)
                output.flush()
                os.fsync(output.fileno())
            except OSError:
                destination.unlink(missing_ok=True)
                raise
        shutil.copystat(source, destination)
    source.unlink()
