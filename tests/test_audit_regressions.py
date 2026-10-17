import dataclasses
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from unittest.mock import patch

import httpx

from sonora.audio import art
from sonora.core.cancellation import InterruptedOperationError
from sonora.core.http import SESSION
from sonora.core.models import TrackInfo
from sonora.modules import metadata_consensus, tagger
from sonora.services import theaudiodb
from tests.support import IsolatedTestCase

MBID = "11111111-2222-3333-4444-555555555555"
OTHER_MBID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def _observe_shutdown(
    calls: list[tuple[ThreadPoolExecutor, bool, bool]],
    original: Callable[..., None],
    executor: ThreadPoolExecutor,
    wait: bool = True,
    *,
    cancel_futures: bool = False,
) -> None:
    calls.append((executor, wait, cancel_futures))
    original(executor, wait=wait, cancel_futures=cancel_futures)


class TestAuditRegressions(IsolatedTestCase):
    def test_duration_conflict_is_checked_without_artist_or_title_changes(self) -> None:
        for duration, accepted in (
            (180.0, True),
            (190.0, True),
            (60.0, False),
            (None, True),
        ):
            with self.subTest(duration=duration):
                recording = TrackInfo(
                    Path("recording.flac"),
                    artist="Alpha",
                    title="Original Song",
                    duration=180,
                )
                proposal = dataclasses.replace(
                    recording,
                    duration=duration,
                    genre="Rock",
                    metadata_sources={"genre": "iTunes"},
                )
                metadata_consensus.merge_provider_proposals(recording, [proposal])
                self.assertEqual(recording.genre, "Rock" if accepted else None)
                self.assertEqual(
                    "identity" in recording.metadata_conflicts, not accepted
                )
                self.assertEqual(recording.duration, 180)

    def test_album_finalization_does_not_download_assets_for_blocked_tracks(
        self,
    ) -> None:
        for conflict in ("identity", "isrc", None):
            with self.subTest(conflict=conflict):
                recording = TrackInfo(
                    Path("recording.flac"),
                    artist="Alpha",
                    title="Original Song",
                    metadata_conflicts={conflict: "Review needed"} if conflict else {},
                )
                with (
                    patch.object(art, "process_artist_artwork") as artist,
                    patch.object(art, "process_label_artwork") as label,
                ):
                    tagger._finalize_album_assets_and_replaygain(
                        self.test_directory,
                        [recording],
                        [],
                        None,
                        {"label_mbid": MBID},
                        False,
                        False,
                        False,
                        1,
                    )
                self.assertEqual(artist.call_count, 0 if conflict else 1)
                self.assertEqual(label.call_count, 0 if conflict else 1)

    def test_audiodb_fallback_cannot_contradict_known_recording_id(self) -> None:
        for candidate_id, accepted in ((MBID, True), (OTHER_MBID, False), (None, True)):
            candidate = {
                "strArtist": "Alpha",
                "strTrack": "Original Song",
                "strMood": "Calm",
            }
            if candidate_id:
                candidate["strMusicBrainzID"] = candidate_id
            responses = [
                httpx.Response(
                    200,
                    json={"track": None},
                    request=httpx.Request(
                        "GET", "https://www.theaudiodb.com/track-mb.php"
                    ),
                ),
                httpx.Response(
                    200,
                    json={"track": [candidate]},
                    request=httpx.Request(
                        "GET", "https://www.theaudiodb.com/searchtrack.php"
                    ),
                ),
            ]
            with (
                self.subTest(candidate_id=candidate_id),
                patch.object(theaudiodb, "get_cached_api", return_value=None),
                patch.object(theaudiodb, "set_cached_api"),
                patch.object(SESSION, "get", side_effect=responses),
            ):
                result = theaudiodb.fetch_theaudiodb_track_details(
                    "Alpha", "Original Song", recording_mbid=MBID
                )
            self.assertEqual(result is not None, accepted)

    def test_interrupt_cancels_all_worker_queues_before_joining_any_pool(self) -> None:
        path = self.test_directory / "recording.flac"
        path.touch()
        recording = TrackInfo(path, artist="Alpha", title="Original Song")
        calls: list[tuple[ThreadPoolExecutor, bool, bool]] = []
        original = ThreadPoolExecutor.shutdown
        with (
            patch.object(
                tagger,
                "_resolve_album_metadata_context",
                return_value=tagger._AlbumMetadataContext(),
            ),
            patch.object(tagger, "process_single_track", return_value=recording),
            patch.object(tagger, "as_completed", side_effect=KeyboardInterrupt),
            patch.object(
                ThreadPoolExecutor,
                "shutdown",
                autospec=True,
                side_effect=partial(_observe_shutdown, calls, original),
            ),
            self.assertRaises(InterruptedOperationError),
        ):
            tagger.tag_album_folder(
                self.test_directory,
                force=True,
                dry_run=True,
                max_threads=1,
                fetch_itunes_art=False,
                fetch_replaygain=False,
            )
        self.assertEqual(
            [(wait, cancel) for _, wait, cancel in calls[:3]], [(False, True)] * 3
        )
        self.assertEqual(len({id(executor) for executor, _, _ in calls[:3]}), 3)
        self.assertTrue(all(wait and cancel for _, wait, cancel in calls[3:6]))
