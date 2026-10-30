import dataclasses
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from sonora.audio import metadata as metadata_module
from sonora.core.models import TrackInfo
from sonora.core.state import LibraryStateVault
from sonora.modules import tagger
from sonora.modules import track_processing as track_processing_module
from tests.regression_support import AudioRegressionTestCase


def _resume_state_worker(
    vault: LibraryStateVault,
    audio_path: Path,
    started: threading.Event,
    resumed: threading.Event,
) -> bool:
    vault.record_track_state(audio_path)
    started.set()
    resumed.wait(3)
    return vault.is_track_up_to_date(audio_path)


class TestRuntimeStateRegressions(AudioRegressionTestCase):
    def test_preview_and_normalize_do_not_mark_tagged(self) -> None:
        audio_path = self.audio("track.wav")
        original = TrackInfo(audio_path, artist="Alpha", title="Song", album="Album")
        for dry_run in (True, False):
            with (
                self.subTest(dry_run=dry_run),
                patch.object(
                    metadata_module, "read_track_metadata", return_value=original
                ),
                patch.object(tagger, "get_library_state") as vault,
            ):
                track_processing_module.normalize_single_track(
                    audio_path, dry_run=dry_run
                )
                vault.return_value.record_track_state.assert_not_called()

    def test_worker_reconnects_after_main_shutdown(self) -> None:
        vault = LibraryStateVault(self.root / "state.db")
        audio_path = self.audio("track.wav")
        started = threading.Event()
        resumed = threading.Event()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                _resume_state_worker, vault, audio_path, started, resumed
            )
            self.assertTrue(started.wait(3))
            vault.close()
            resumed.set()
            self.assertTrue(future.result(timeout=3))
        vault.close()

    def test_cancel_releases_paused_workers_before_join(self) -> None:
        cancellation_script = """
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from sonora.core import logger
stop_listener = Event()
logger._PAUSE_EVENT.clear()
worker_started = Event()
def paused_worker():
    worker_started.set()
    logger.wait_if_paused()
executor = ThreadPoolExecutor(max_workers=1)
worker = executor.submit(paused_worker)
assert worker_started.wait(timeout=1)
logger.resume_workers(stop_listener)
executor.shutdown(wait=True, cancel_futures=True)
assert stop_listener.is_set()
assert worker.done()
"""
        completed = subprocess.run(
            [sys.executable, "-c", cancellation_script],
            capture_output=True,
            timeout=5,
            check=True,
        )
        self.assertEqual(completed.returncode, 0)

    def test_state_requires_provider_evidence_and_requested_features(self) -> None:
        audio_path = self.audio("track.wav")
        cases: tuple[tuple[dict[str, str], bool, float | None, bool], ...] = (
            ({"title": "MusicBrainz"}, False, None, True),
            ({"title": "MusicBrainz"}, True, None, False),
            ({"title": "MusicBrainz"}, True, 120.0, True),
            ({}, False, None, False),
        )
        completed_statuses: list[str] = []
        for sources, fetch_bpm, bpm, completed in cases:
            with self.subTest(sources=sources, fetch_bpm=fetch_bpm, bpm=bpm):
                track = TrackInfo(
                    audio_path, artist="Alpha", title="Song", album="Album", bpm=bpm
                )
                if sources:
                    track.metadata_sources["title"] = "MusicBrainz"
                with (
                    patch.object(tagger, "get_library_state") as state,
                    patch.object(
                        tagger,
                        "_resolve_album_metadata_context",
                        return_value=tagger._AlbumMetadataContext(
                            sample_artist="Alpha", sample_album="Album"
                        ),
                    ),
                    patch.object(tagger, "process_single_track", return_value=track),
                    patch.object(
                        metadata_module, "read_track_metadata", return_value=track
                    ),
                    patch.object(tagger, "_harmonize_album_genres"),
                    patch.object(tagger, "_harmonize_album_artists"),
                    patch.object(tagger, "_finalize_album_assets_and_replaygain"),
                ):
                    tagger.tag_album_folder(
                        self.root,
                        fetch_bpm=fetch_bpm,
                        fetch_key=False,
                        fetch_replaygain=False,
                        fetch_lyrics=False,
                        fetch_itunes_art=False,
                        force=True,
                    )
                    recorded_paths = (
                        state.return_value.record_tracks_state_batch.call_args.args[0]
                    )
                    self.assertEqual(recorded_paths, [audio_path] if completed else [])
                    if completed:
                        completed_statuses.append(
                            state.return_value.record_tracks_state_batch.call_args.kwargs[
                                "status"
                            ]
                        )
        self.assertNotEqual(completed_statuses[0], completed_statuses[1])

    def test_state_requires_persisted_artwork_when_requested(self) -> None:
        audio_path = self.audio("art-state.flac")
        for actual_dimension, expected_completion in (
            (None, False),
            (300, False),
            (600, True),
        ):
            with self.subTest(actual_dimension=actual_dimension):
                proposed = TrackInfo(
                    audio_path,
                    artist="Alpha",
                    title="Song",
                    album="Album",
                    art_width=900,
                    art_height=900,
                )
                proposed.metadata_sources["title"] = "MusicBrainz"
                persisted = dataclasses.replace(
                    proposed, art_width=actual_dimension, art_height=actual_dimension
                )
                with (
                    patch.object(tagger, "get_library_state") as state,
                    patch.object(
                        tagger,
                        "_resolve_album_metadata_context",
                        return_value=tagger._AlbumMetadataContext(
                            sample_artist="Alpha", sample_album="Album"
                        ),
                    ),
                    patch.object(tagger, "process_single_track", return_value=proposed),
                    patch.object(
                        metadata_module, "read_track_metadata", return_value=persisted
                    ),
                    patch.object(tagger, "_harmonize_album_genres"),
                    patch.object(tagger, "_harmonize_album_artists"),
                    patch.object(tagger, "_finalize_album_assets_and_replaygain"),
                ):
                    tagger.tag_album_folder(
                        self.root,
                        fetch_bpm=False,
                        fetch_key=False,
                        fetch_replaygain=False,
                        fetch_lyrics=False,
                        fetch_itunes_art=True,
                        force=True,
                    )
                    self.assertEqual(
                        state.return_value.record_tracks_state_batch.call_args.args[0],
                        [audio_path] if expected_completion else [],
                    )
