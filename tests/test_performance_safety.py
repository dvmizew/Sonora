import dataclasses
import multiprocessing
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from unittest.mock import patch

import httpx

from sonora.core import cache, coordination
from sonora.core.http import SESSION
from sonora.core.models import MetadataSource, TrackInfo
from sonora.core.state import LibraryStateVault
from sonora.modules import catalog_metadata, metadata_merge, tagger
from sonora.modules.metadata_merge import build_provider_proposal
from sonora.modules.track_processing import _record_persisted_album
from sonora.services import acoustid, deezer, itunes, musicbrainz
from tests.support import IsolatedTestCase


@coordination.coalesce_lookup
def _counted_lookup(marker: Path) -> str | None:
    key = f"test-provider:{marker}"
    cached = cache.get_cached_api(key)
    if cached is not None:
        return str(cached) or None
    with marker.open("a") as output:
        output.write("request\n")
    cache.set_cached_api(key, None)
    return None


def _run_process_lookup(marker: str) -> None:
    _counted_lookup(Path(marker))


def _itunes_after_barrier(barrier: threading.Barrier) -> dict[str, object] | None:
    barrier.wait(timeout=5)
    return itunes.lookup_itunes_track("123")


def _propose_genre(
    track: TrackInfo,
    genre: str,
    source: MetadataSource,
    barrier: threading.Barrier | None = None,
    **keywords: object,
) -> None:
    if barrier:
        barrier.wait(timeout=5)
    track.genre = genre
    track.metadata_sources["genre"] = source


def _replace_during_fingerprint(path: Path) -> tuple[float, str]:
    path.write_bytes(b"changed audio")
    return 100, "fingerprint"


def _mark_identity_conflict(track_info: TrackInfo, **keywords: object) -> None:
    track_info.artist = "Wrong Artist"
    track_info.genre = "Wrong Genre"
    track_info.metadata_conflicts["identity"] = "Conflicting identity"


class TestPerformanceSafety(IsolatedTestCase):
    def test_proposals_retain_only_changes_and_cannot_modify_audio_properties(
        self,
    ) -> None:
        recording = TrackInfo(
            Path("recording.flac"),
            artist="Alpha",
            title="Original Song",
            genre="Rock",
            sample_rate=44100,
            is_lossless=True,
        )
        candidate = dataclasses.replace(
            recording,
            genre="Jazz",
            sample_rate=8000,
            is_lossless=False,
            metadata_sources={"genre": "Deezer"},
        )
        proposal = build_provider_proposal(recording, candidate)
        self.assertEqual(proposal.changes, {"genre": ("Jazz", "Deezer")})
        metadata_merge.merge_provider_proposals(recording, [proposal])
        self.assertEqual(recording.genre, "Jazz")
        self.assertEqual((recording.sample_rate, recording.is_lossless), (44100, True))
        self.assertEqual(build_provider_proposal(recording, recording).changes, {})

    def test_provider_confirmation_is_retained_without_changing_the_tag(self) -> None:
        recording = TrackInfo(Path("recording.flac"), genre="Rock")
        candidate = dataclasses.replace(recording, metadata_sources={"genre": "iTunes"})
        proposal = build_provider_proposal(recording, candidate)
        self.assertEqual(proposal.changes, {"genre": ("Rock", "iTunes")})
        metadata_merge.merge_provider_proposals(recording, [proposal])
        self.assertEqual(recording.metadata_sources, {"genre": "iTunes"})

    def test_proposals_agree_or_preserve_conflicts_independent_of_order(self) -> None:
        for genres, expected in (
            (("Rock", "Rock"), "Rock"),
            (("Rock", "Jazz"), "Original"),
        ):
            original = TrackInfo(
                Path("recording.flac"),
                artist="Alpha",
                title="Original Song",
                genre="Original",
            )
            sources: tuple[MetadataSource, ...] = ("iTunes", "Deezer")
            proposals = [
                metadata_merge.collect_provider_proposal(
                    original, partial(_propose_genre, genre=genre, source=source)
                )
                for genre, source in zip(genres, sources)
            ]
            self.assertEqual(original.genre, "Original")
            self.assertEqual(original.metadata_sources, {})
            for ordered in (proposals, proposals[::-1]):
                merged = dataclasses.replace(
                    original, metadata_sources={}, metadata_conflicts={}
                )
                metadata_merge.merge_provider_proposals(merged, ordered)
                self.assertEqual(merged.genre, expected)
                self.assertEqual(
                    bool(merged.metadata_conflicts), genres[0] != genres[1]
                )
        metadata_merge.merge_provider_proposals(original, [])
        self.assertEqual(original.genre, "Original")

    def test_foreign_identity_and_conflicting_isrc_withhold_metadata_updates(
        self,
    ) -> None:
        original = TrackInfo(
            Path("recording.flac"),
            artist="Alpha",
            title="Original Song",
            isrc="USAAA2400001",
            metadata_sources={"isrc": "MusicBrainz"},
        )
        for title, isrc in (
            ("Different Song", original.isrc),
            (original.title, "USAAA2400002"),
        ):
            with self.subTest(title=title, isrc=isrc):
                merged = dataclasses.replace(
                    original,
                    metadata_sources=original.metadata_sources.copy(),
                    metadata_conflicts={},
                )
                proposal = dataclasses.replace(
                    original,
                    title=title,
                    isrc=isrc,
                    genre="Rock",
                    metadata_sources={"isrc": "Deezer", "genre": "Deezer"},
                )
                metadata_merge.merge_provider_proposals(
                    merged,
                    [build_provider_proposal(merged, proposal)],
                )
                self.assertIsNone(merged.genre)
                self.assertEqual(merged.isrc, original.isrc)
                self.assertTrue(merged.metadata_conflicts)
                self.assertTrue(merged.to_dict()["metadata_conflicts"])

    def test_parallel_provider_proposals_do_not_mutate_shared_identity(self) -> None:
        barrier = threading.Barrier(6)
        original = TrackInfo(
            Path("recording.flac"), artist="Alpha", title="Original Song"
        )
        with (
            patch.object(tagger, "_update_musicbrainz"),
            patch.object(tagger, "_update_acoustid"),
            patch.object(tagger, "_update_shazam"),
            patch.object(
                tagger,
                "_update_itunes",
                side_effect=partial(
                    _propose_genre, genre="Rock", source="iTunes", barrier=barrier
                ),
            ),
            patch.object(
                tagger,
                "_update_deezer",
                side_effect=partial(
                    _propose_genre, genre="Jazz", source="Deezer", barrier=barrier
                ),
            ),
            patch.object(
                tagger,
                "_update_lastfm",
                side_effect=partial(
                    _propose_genre, genre="Rock", source="Last.fm", barrier=barrier
                ),
            ),
            patch.object(
                tagger,
                "_update_discogs",
                side_effect=partial(
                    _propose_genre, genre="Rock", source="Discogs", barrier=barrier
                ),
            ),
            patch.object(
                tagger,
                "_update_genius",
                side_effect=partial(
                    _propose_genre, genre="Rock", source="Genius", barrier=barrier
                ),
            ),
            patch.object(
                tagger,
                "_update_theaudiodb",
                side_effect=partial(
                    _propose_genre, genre="Rock", source="TheAudioDB", barrier=barrier
                ),
            ),
            ThreadPoolExecutor(max_workers=6) as executor,
        ):
            tagger._update_metadata(
                original,
                original.file_path,
                None,
                None,
                None,
                None,
                True,
                provider_executor=executor,
                album_context=tagger._AlbumMetadataContext(),
            )
        self.assertIsNone(original.genre)
        self.assertEqual(original.artist, "Alpha")
        self.assertTrue(original.metadata_conflicts)

    def test_identical_threaded_lookups_make_one_provider_request(self) -> None:
        barrier = threading.Barrier(6)
        response = httpx.Response(
            200,
            json={
                "results": [
                    {
                        "trackId": 123,
                        "trackName": "Original Song",
                        "artistName": "Alpha",
                    }
                ]
            },
            request=httpx.Request("GET", "https://itunes.apple.com/lookup"),
        )
        with (
            patch.object(SESSION, "get", return_value=response) as request,
            ThreadPoolExecutor(max_workers=6) as executor,
        ):
            results = list(executor.map(_itunes_after_barrier, [barrier] * 6))
        self.assertTrue(all(result == results[0] for result in results))
        self.assertIsNotNone(results[0])
        request.assert_called_once()

    def test_duplicate_processes_reuse_confirmed_negative_answer(self) -> None:
        marker = self.test_directory / "requests.txt"
        context = multiprocessing.get_context("spawn")
        processes = [
            context.Process(target=_run_process_lookup, args=(str(marker),))
            for _ in range(3)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()
                self.fail("Lookup process did not finish")
            self.assertEqual(process.exitcode, 0)
        self.assertEqual(marker.read_text(), "request\n")

    def test_confirmed_misses_expire_earlier_than_successes(self) -> None:
        for key, value in (
            ("miss", None),
            ("empty", []),
            ("success", {"title": "Original Song"}),
        ):
            cache.set_cached_api(key, value)
        storage = cache.get_cache()
        assert storage is not None
        for key, expected in (("miss", 1800), ("empty", 1800), ("success", 2419200)):
            _, expiry = storage.get(key, expire_time=True)
            self.assertAlmostEqual(expiry - time.time(), expected, delta=2)
        self.assertIsNone(cache.get_cached_api("absent"))
        self.assertEqual(cache.get_cached_api("miss"), "")

    def test_provider_failures_are_not_negative_cached_but_not_found_is(self) -> None:
        for status, expected_calls in ((503, 2), (404, 1)):
            response = httpx.Response(
                status,
                request=httpx.Request("GET", "https://api.deezer.com/track/isrc:TEST"),
            )
            with (
                self.subTest(status=status),
                patch.object(SESSION, "get", return_value=response) as request,
            ):
                for _ in range(2):
                    self.assertIsNone(
                        deezer.fetch_deezer_track_by_isrc(f"TEST{status}")
                    )
            self.assertEqual(request.call_count, expected_calls)

    def test_persistent_fingerprint_survives_memory_clear_and_invalidates_changed_file(
        self,
    ) -> None:
        path = self.test_directory / "recording.flac"
        path.write_bytes(b"audio")
        with patch.object(
            acoustid,
            "_fingerprint_file_with_timeout",
            return_value=(100, "fingerprint"),
        ) as decode:
            first = acoustid.fingerprint_audio_file(path)
            acoustid._ACOUSTID_CACHE.clear()
            self.assertEqual(acoustid.fingerprint_audio_file(path), first)
            decode.assert_called_once()
            path.write_bytes(b"new audio")
            self.assertEqual(acoustid.fingerprint_audio_file(path), first)
            self.assertEqual(decode.call_count, 2)

    def test_file_changes_during_fingerprinting_are_rejected(self) -> None:
        path = self.test_directory / "recording.flac"
        path.write_bytes(b"audio")
        with (
            patch.object(
                acoustid,
                "_fingerprint_file_with_timeout",
                side_effect=_replace_during_fingerprint,
            ),
            self.assertRaisesRegex(OSError, "changed during fingerprinting"),
        ):
            acoustid.fingerprint_audio_file(path)

    def test_provider_quota_is_shared_and_other_hosts_are_independent(self) -> None:
        with (
            patch("sonora.core.coordination.time.time", return_value=100),
            patch("sonora.core.coordination.time.sleep") as sleep,
        ):
            self.assertEqual(coordination.wait_for_provider("provider-a", 1), 0)
            self.assertEqual(coordination.wait_for_provider("provider-b", 1), 0)
            self.assertEqual(coordination.wait_for_provider("provider-a", 1), 1)
        sleep.assert_called_once_with(1)

    def test_album_prefetch_is_bounded_and_skips_unchanged_albums(self) -> None:
        folders = {
            self.test_directory / str(index): [
                self.test_directory / str(index) / "recording.flac"
            ]
            for index in range(5)
        }
        changed = {path for paths in list(folders.values())[1:] for path in paths}
        with (
            patch.object(
                tagger,
                "_resolve_album_metadata_context",
                return_value=tagger._AlbumMetadataContext(),
            ) as resolve,
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            contexts = tagger._prefetch_album_contexts(
                folders,
                changed,
                executor,
                discogs_user_token=None,
                fetch_itunes_art=False,
                force=False,
                dry_run=True,
                window=2,
            )
            _, _, first = next(contexts)
            self.assertIsNone(first)
            _, _, second = next(contexts)
            assert second is not None
            second.result()
            self.assertLessEqual(resolve.call_count, 2)
            list(contexts)
        self.assertEqual(resolve.call_count, 4)

    def test_identity_conflict_prevents_tag_and_asset_writes(self) -> None:
        path = self.test_directory / "recording.flac"
        path.touch()
        recording = TrackInfo(path, artist="Alpha", title="Original Song")
        with (
            patch("sonora.audio.metadata.read_track_metadata", return_value=recording),
            patch(
                "sonora.services.artist_resolution.resolve_artist_name",
                return_value="Alpha",
            ),
            patch.object(
                tagger,
                "_update_metadata",
                side_effect=_mark_identity_conflict,
            ),
            patch.object(tagger, "_process_audio_and_save_tags") as persist,
        ):
            result = tagger.process_single_track(path, force=True)
        self.assertEqual(result.artist, "Alpha")
        self.assertIsNone(result.genre)
        self.assertIn("identity", result.metadata_conflicts)
        persist.assert_not_called()

    def test_conflicts_survive_reporting_and_are_not_marked_complete(self) -> None:
        path = self.test_directory / "recording.flac"
        path.touch()
        recording = TrackInfo(
            path,
            artist="Alpha",
            title="Original Song",
            genre="Unknown",
            metadata_sources={"genre": "iTunes"},
            metadata_conflicts={"genre": "Disputed genre"},
        )
        vault = LibraryStateVault(self.test_directory / "state.db")
        try:
            with patch(
                "sonora.audio.metadata.read_track_metadata",
                return_value=dataclasses.replace(recording, metadata_conflicts={}),
            ):
                persisted = _record_persisted_album(
                    [recording],
                    vault,
                    "TAGGED_OK",
                    fetch_bpm=False,
                    fetch_key=False,
                    fetch_replaygain=False,
                    fetch_lyrics=False,
                    fetch_itunes_art=False,
                )
            self.assertEqual(
                persisted[0].metadata_conflicts, recording.metadata_conflicts
            )
            self.assertEqual(
                vault.filter_outdated_tracks([path], expected_status="TAGGED_OK"),
                [path],
            )
            with patch("sonora.audio.metadata.write_track_metadata") as write:
                tagger._harmonize_album_genres(
                    [
                        recording,
                        dataclasses.replace(
                            recording, genre="Rock", metadata_conflicts={}
                        ),
                    ],
                    None,
                    None,
                )
            self.assertEqual(recording.genre, "Unknown")
            write.assert_not_called()
        finally:
            vault.close()

    def test_standalone_itunes_cannot_replace_known_release_date_country_or_copyright(
        self,
    ) -> None:
        recording = TrackInfo(
            Path("recording.flac"),
            artist="Alpha",
            title="Original Song",
            date="2020",
            release_country="Romania",
            copyright="Original copyright",
        )
        with patch.object(
            itunes,
            "fetch_itunes_track_metadata",
            return_value={
                "date": "2025",
                "release_country": "United States",
                "copyright": "Reissue copyright",
            },
        ):
            catalog_metadata._update_itunes(recording, force=True)
        self.assertEqual(
            (recording.date, recording.release_country, recording.copyright),
            ("2020", "Romania", "Original copyright"),
        )
        with patch.object(
            itunes,
            "search_itunes",
            return_value=[
                {
                    "trackId": 123,
                    "trackName": "Original Song",
                    "artistName": "Alpha",
                    "country": "US",
                }
            ],
        ):
            payload = itunes.fetch_itunes_track_metadata("Alpha", "Original Song")
        assert payload is not None
        self.assertNotIn("release_country", payload)

    def test_musicbrainz_ties_abstain_and_cached_evidence_is_reranked(self) -> None:
        recordings = [
            {
                "id": identifier,
                "title": "Original Song",
                "artist-credit": [{"artist": {"name": "Alpha"}}],
            }
            for identifier in (
                "11111111-2222-3333-4444-555555555555",
                "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            )
        ]
        with patch.object(
            musicbrainz,
            "_call_musicbrainz",
            return_value={"recording-list": recordings},
        ) as request:
            self.assertIsNone(musicbrainz.fetch_track_mbid("Alpha", "Original Song"))
            self.assertIsNone(musicbrainz.fetch_track_mbid("Alpha", "Original Song"))
        request.assert_called_once()
