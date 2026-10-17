import dataclasses
import os
import sqlite3
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import diskcache
import msgpack
import numpy as np
import soundfile as sf
import taglib

from sonora.audio.metadata import read_track_metadata
from sonora.cli.main import main
from sonora.core import cache as cache_module
from sonora.core.cache import (
    api_cache_session,
    get_api_cache_dir,
    get_cache,
    get_cached_api,
    set_cached_api,
)
from sonora.core.cache_storage import CacheDecodeError, MetadataDisk
from sonora.core.config import (
    SonoraConfig,
    _default_policy,
    clear_config_cache,
    get_config,
)
from sonora.core.models import TrackInfo
from sonora.core.state import LibraryStateVault, get_library_state, tagging_state_status
from sonora.modules.matching import _resolve_album_track_position, is_alien_album_track
from sonora.modules.organizer import (
    SingleDeduplicator,
    _resolve_single_target_file,
    organize_library_singles,
)
from tests.support import IsolatedTestCase


def _worker_cache_session(fail: bool = False) -> diskcache.Cache:
    with api_cache_session():
        owned_cache = get_cache()
        assert owned_cache is not None
        with api_cache_session():
            assert get_cache() is owned_cache
            set_cached_api("worker-session", {"title": "Recording"})
        if fail:
            raise OSError("worker failure")
    assert get_cache() is None
    return owned_cache


class TestWorkerCacheLifecycle(IsolatedTestCase):
    def test_worker_closes_owned_cache_and_preserves_main_cache(self) -> None:
        shared_cache = get_cache()
        assert shared_cache is not None
        original_close = diskcache.Cache.close
        with (
            patch.object(
                diskcache.Cache, "close", autospec=True, side_effect=original_close
            ) as close,
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            owned_cache = executor.submit(_worker_cache_session).result()
            self.assertIsNot(owned_cache, shared_cache)
            self.assertTrue(
                any(call.args[0] is owned_cache for call in close.call_args_list)
            )
            self.assertFalse(
                any(call.args[0] is shared_cache for call in close.call_args_list)
            )
            self.assertIs(get_cache(), shared_cache)
            self.assertEqual(get_cached_api("worker-session"), {"title": "Recording"})

    def test_worker_exception_closes_session_and_allows_next_task(self) -> None:
        original_close = diskcache.Cache.close
        with (
            patch.object(
                diskcache.Cache, "close", autospec=True, side_effect=original_close
            ) as close,
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            with self.assertRaisesRegex(OSError, "worker failure"):
                executor.submit(_worker_cache_session, True).result()
            self.assertTrue(close.called)
            executor.submit(_worker_cache_session).result()
        self.assertEqual(get_cached_api("worker-session"), {"title": "Recording"})

    def test_main_session_reuses_shared_cache(self) -> None:
        shared_cache = get_cache()
        with api_cache_session():
            self.assertIs(get_cache(), shared_cache)
        self.assertIs(get_cache(), shared_cache)


class TestAlbumIdentity(IsolatedTestCase):
    def test_album_positions_reject_recording_variants(self) -> None:
        pairs = (
            ("The Journey Part 2", "The Journey Part 3"),
            ("Long Summer Night (Alpha Remix)", "Long Summer Night (Omega Remix)"),
            ("Long Summer Night", "Long Summer Night (Live)"),
            ("Intro", "Outro"),
        )
        for query, candidate in pairs:
            for position in (1, 99):
                with self.subTest(query=query, candidate=candidate, position=position):
                    track = dataclasses.replace(
                        self._track(query),
                        track_number=position,
                        file_path=Path(f"/music/{position:02d} - Track.flac"),
                    )
                    self.assertIsNone(
                        _resolve_album_track_position(
                            track,
                            track.file_path,
                            album_deezer_details={
                                "tracks_by_position": {1: {"title": candidate}}
                            },
                        )
                    )

    def test_album_identifiers_still_heal_corrupted_titles(self) -> None:
        track = dataclasses.replace(self._track("Untitled"), isrc="USAAA2400001")
        position = _resolve_album_track_position(
            track,
            track.file_path,
            album_deezer_details={
                "tracks_by_position": {
                    3: {
                        "title": "Correct recording",
                        "isrc": "USAAA2400001",
                        "disc_number": 1,
                    }
                }
            },
        )
        self.assertEqual(position, 3)

    def test_variant_rejections_also_shield_alien_tracks(self) -> None:
        for query, candidate in (
            ("The Journey Part 2", "The Journey Part 3"),
            ("Long Summer Night (Alpha Remix)", "Long Summer Night (Omega Remix)"),
        ):
            with self.subTest(query=query):
                track = dataclasses.replace(
                    self._track(query), album="Album", track_number=1
                )
                recordings = {
                    1: {"artist": "Artist", "title": candidate},
                    2: {"artist": "Artist", "title": "Different Recording"},
                    3: {"artist": "Artist", "title": "Another Recording"},
                }
                release = {
                    "artist": "Artist",
                    "title": "Album",
                    "tracks_by_position": recordings,
                }
                self.assertTrue(
                    is_alien_album_track(
                        track,
                        track.file_path,
                        "Artist",
                        "Album",
                        album_deezer_details=release,
                    )
                )
                track.isrc = "USAAA2400001"
                recordings[1]["isrc"] = track.isrc
                self.assertTrue(
                    is_alien_album_track(
                        track,
                        track.file_path,
                        "Artist",
                        "Album",
                        album_deezer_details=release,
                    )
                )

    def test_title_matching_tolerates_corrupt_artist_without_losing_variants(
        self,
    ) -> None:
        track = dataclasses.replace(
            self._track("Correct recording"), artist="Damaged artist"
        )
        self.assertEqual(
            _resolve_album_track_position(
                track,
                track.file_path,
                album_deezer_details={
                    "tracks_by_position": {
                        1: {
                            "title": "Correct recording",
                            "artist": "Actual artist",
                        }
                    }
                },
            ),
            1,
        )

    @staticmethod
    def _track(title: str) -> TrackInfo:
        return TrackInfo(
            file_path=Path("/music/01 - Track.flac"), artist="Artist", title=title
        )


class TestSingleIdentity(IsolatedTestCase):
    def test_guests_in_titles_artists_and_cobilling_remain_distinct(self) -> None:
        from sonora.core.models import TrackInfo

        for artist_a, artist_b, title_a, title_b in (
            ("Artist", "Artist", "Song (feat. Alice)", "Song (feat. Bob)"),
            ("Artist feat. Alice", "Artist feat. Bob", "Song", "Song"),
            ("Artist & Alice", "Artist & Bob", "Song", "Song"),
        ):
            with self.subTest(artist=artist_a, title=title_a):
                original = TrackInfo(
                    file_path=Path("/a.flac"), artist=artist_a, title=title_a
                )
                original_snapshot = dataclasses.asdict(original)
                registry = SingleDeduplicator()
                registry.register(original)
                self.assertFalse(
                    registry.is_duplicate(
                        TrackInfo(
                            file_path=Path("/b.flac"),
                            artist=artist_b,
                            title=title_b,
                        )
                    )
                )
                self.assertEqual(dataclasses.asdict(original), original_snapshot)
                other_edition = TrackInfo(
                    file_path=Path("/b.flac"), artist=artist_b, title=title_b
                )
                self.assertNotEqual(
                    _resolve_single_target_file(
                        original.file_path, original, Path("/Singles")
                    ),
                    _resolve_single_target_file(
                        other_edition.file_path, other_edition, Path("/Singles")
                    ),
                )

    def test_equivalent_feature_encodings_are_duplicates(self) -> None:
        from sonora.core.models import TrackInfo

        registry = SingleDeduplicator()
        registry.register(
            TrackInfo(
                file_path=Path("/a.flac"),
                artist="Artist",
                title="Song (feat. Guest)",
            )
        )
        self.assertTrue(
            registry.is_duplicate(
                TrackInfo(
                    file_path=Path("/b.flac"),
                    artist="Artist",
                    title="Song",
                    featured_artists="Guest",
                )
            )
        )

    def test_real_flac_preview_preserves_distinct_guests(self) -> None:
        source = get_api_cache_dir().parent / "fixtures"
        for index, guest in enumerate(("Alice", "Bob")):
            folder = source / f"Edition {index}"
            folder.mkdir(parents=True)
            audio_path = folder / "01 - Song.flac"
            sf.write(audio_path, np.zeros(4410), 44100, subtype="PCM_16")
            with taglib.File(str(audio_path)) as audio_tags:
                audio_tags.tags = {
                    "ARTIST": ["Artist"],
                    "TITLE": [f"Song (feat. {guest})"],
                }
                audio_tags.save()
            self.assertIsNone(read_track_metadata(audio_path).featured_artists)
        originals = {path: path.read_bytes() for path in source.rglob("*.flac")}
        self.assertEqual(
            organize_library_singles(
                source, source / "Singles", dry_run=True, max_threads=1
            ),
            2,
        )
        self.assertEqual({path: path.read_bytes() for path in originals}, originals)

    def test_layout_is_independent_of_library_size(self) -> None:
        from sonora.core.models import TrackInfo

        source = get_api_cache_dir().parent / "library"
        source.mkdir(parents=True)
        track = TrackInfo(file_path=source / "Song.flac", artist="Artist", title="Song")
        destination = source / "Singles"
        expected = destination / "Artist - Song" / "01 - Song.flac"
        for count in range(8):
            (source / f"Artist {count}").mkdir()
            self.assertEqual(
                _resolve_single_target_file(track.file_path, track, destination),
                expected,
            )


class TestPolicyConfiguration(IsolatedTestCase):
    def test_defaults_are_loaded_once(self) -> None:
        _default_policy.cache_clear()
        original_read = Path.read_text
        with patch.object(
            Path, "read_text", autospec=True, wraps=Path.read_text
        ) as read_text:
            read_text.side_effect = original_read
            SonoraConfig()
            SonoraConfig()
        self.assertEqual(read_text.call_count, 1)

    def test_all_string_arrays_reject_invalid_types(self) -> None:
        for policy in (
            "disc_folder_patterns",
            "generic_containers",
            "codec_rip_keywords",
            "featuring_conjunctions",
            "known_group_artists",
            "noise_genres",
            "outlier_genres",
        ):
            for invalid in ("string", [1], [True], [None]):
                with self.subTest(policy=policy, value=invalid):
                    clear_config_cache()
                    with (
                        patch(
                            "sonora.core.config._discover_config_file",
                            return_value=Path("config.json"),
                        ),
                        patch(
                            "sonora.core.config._parse_config_data",
                            return_value={policy: invalid},
                        ),
                        self.assertRaisesRegex(TypeError, policy),
                    ):
                        get_config()

    def test_configuration_requires_real_boolean(self) -> None:
        value: object
        for value in ("false", 0, 1, [], None):
            with self.subTest(value=value):
                clear_config_cache()
                with (
                    patch.dict(os.environ, {}, clear=True),
                    patch("sonora.core.config.load_app_environment"),
                    patch(
                        "sonora.core.config._discover_config_file",
                        return_value=Path("config.json"),
                    ),
                    patch(
                        "sonora.core.config._parse_config_data",
                        return_value={"enable_shazam": value},
                    ),
                    self.assertRaisesRegex(TypeError, "enable_shazam"),
                ):
                    get_config()

    def test_environment_boolean_and_valid_arrays(self) -> None:
        for environment_value, expected in (("false", False), ("true", True)):
            clear_config_cache()
            with (
                patch.dict(os.environ, {"ENABLE_SHAZAM": environment_value}),
                patch(
                    "sonora.core.config._discover_config_file",
                    return_value=Path("config.json"),
                ),
                patch(
                    "sonora.core.config._parse_config_data",
                    return_value={
                        "disc_folder_patterns": ["^disk[0-9]+$"],
                        "generic_containers": ["Downloads"],
                    },
                ),
            ):
                configuration = get_config()
                self.assertIs(configuration.enable_shazam, expected)
                self.assertTrue(configuration.is_disc_folder("disk2"))
                self.assertTrue(configuration.is_generic_container("downloads"))

    def test_matching_policies_validate_bounds(self) -> None:
        invalid_configs: tuple[Callable[[], SonoraConfig], ...] = (
            lambda: SonoraConfig(version_mismatch_penalty=-1),
            lambda: SonoraConfig(remix_modifier_mismatch_penalty=101),
            lambda: SonoraConfig(title_score_weight=1.1),
            lambda: SonoraConfig(single_duration_tolerance_seconds=float("inf")),
        )
        for invalid_config in invalid_configs:
            with self.assertRaises(ValueError):
                invalid_config()

    def test_fingerprint_is_stable_in_separate_interpreters(self) -> None:
        script = (
            "from sonora.core.config import SonoraConfig; "
            "from sonora.core.state import tagging_state_status; "
            "print(tagging_state_status(SonoraConfig(),fetch_bpm=False,fetch_key=False,"
            "fetch_replaygain=False,fetch_lyrics=False,fetch_itunes_art=False))"
        )
        statuses = []
        for seed in ("1", "2", "3"):
            result = subprocess.run(
                [sys.executable, "-c", script],
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONHASHSEED": seed},
            )
            statuses.append(result.stdout.strip())
        self.assertEqual(len(set(statuses)), 1)

    def test_fingerprint_changes_for_meaningful_policy_changes(self) -> None:
        flags = {
            "fetch_bpm": False,
            "fetch_key": False,
            "fetch_replaygain": False,
            "fetch_lyrics": False,
            "fetch_itunes_art": False,
        }
        original = tagging_state_status(SonoraConfig(), **flags)
        self.assertNotEqual(
            original,
            tagging_state_status(SonoraConfig(), **{**flags, "fetch_bpm": True}),
        )
        self.assertNotEqual(
            original,
            tagging_state_status(
                dataclasses.replace(SonoraConfig(), title_match_threshold=90), **flags
            ),
        )


class TestCacheMaintenance(IsolatedTestCase):
    def test_unhashable_map_keys_are_cache_misses(self) -> None:
        invalid_keys = (
            msgpack.packb([1]),
            msgpack.packb(msgpack.ExtType(1, msgpack.packb([[]]))),
        )
        for encoded_key in invalid_keys:
            assert encoded_key is not None
            encoded_value = msgpack.packb("value")
            assert encoded_value is not None
            set_cached_api("damaged", {"title": "Title"})
            with (
                closing(sqlite3.connect(get_api_cache_dir() / "cache.db")) as database,
                database,
            ):
                database.execute(
                    "UPDATE Cache SET value = ?, filename = NULL",
                    (b"\x81" + encoded_key + encoded_value,),
                )
            self.assertIsNone(get_cached_api("damaged"))

    def test_cache_removal_rejects_external_paths(self) -> None:
        cache = get_cache()
        assert cache is not None
        external = get_api_cache_dir().parent / "outside.val"
        external.write_bytes(b"keep")
        for operation in ("clear", "delete", "cull"):
            for filename in ("../outside.val", str(external)):
                with self.subTest(operation=operation, filename=filename):
                    cache.set("artwork", b"x" * 100_000)
                    with (
                        closing(
                            sqlite3.connect(get_api_cache_dir() / "cache.db")
                        ) as database,
                        database,
                    ):
                        database.execute("UPDATE Cache SET filename = ?", (filename,))
                    with self.assertRaises(CacheDecodeError):
                        if operation == "clear":
                            cache.clear()
                        elif operation == "delete":
                            del cache["artwork"]
                        else:
                            cache.reset("size_limit", 0)
                            try:
                                cache.cull()
                            finally:
                                cache.reset("size_limit", 1 << 30)
                    self.assertEqual(external.read_bytes(), b"keep")
                    with (
                        closing(
                            sqlite3.connect(get_api_cache_dir() / "cache.db")
                        ) as database,
                        database,
                    ):
                        database.execute(
                            "UPDATE Cache SET filename = NULL, value = ?",
                            (msgpack.packb(None),),
                        )

    def test_cache_removal_rejects_parent_symlink_escape(self) -> None:
        cache_directory = get_api_cache_dir()
        cache_directory.mkdir(parents=True)
        external = cache_directory.parent / "outside.val"
        external.write_bytes(b"keep")
        (cache_directory / "linked").symlink_to(
            cache_directory.parent, target_is_directory=True
        )
        with self.assertRaises(CacheDecodeError):
            MetadataDisk(str(cache_directory)).remove("linked/outside.val")
        self.assertEqual(external.read_bytes(), b"keep")

    def test_normal_shard_removal_preserves_cache_root(self) -> None:
        directory = get_api_cache_dir()
        shard = directory / "aa" / "bb" / "record.val"
        shard.parent.mkdir(parents=True)
        shard.write_bytes(b"record")
        MetadataDisk(str(directory)).remove("aa/bb/record.val")
        self.assertFalse(shard.exists())
        self.assertTrue(directory.is_dir())

    def test_purge_failure_surfaces_without_success_summary(self) -> None:
        set_cached_api("recording", {"title": "Title"})
        with (
            patch(
                "sonora.core.cache.shutil.rmtree",
                side_effect=PermissionError("locked cache"),
            ),
            patch("sonora.cli.cache.LOG.success") as success,
        ):
            self.assertEqual(main(["cache", "clear", "--api", "--purge"]), 1)
            success.assert_not_called()
        self.assertEqual(get_cached_api("recording"), {"title": "Title"})

    def test_clear_database_failure_surfaces_without_success_summary(self) -> None:
        cache = get_cache()
        assert cache is not None
        with (
            patch.object(
                cache, "clear", side_effect=sqlite3.OperationalError("locked cache")
            ),
            patch("sonora.cli.cache.LOG.success") as success,
        ):
            self.assertEqual(main(["cache", "clear", "--api"]), 1)
            success.assert_not_called()

    def test_successful_clear_uses_actual_removed_count(self) -> None:
        set_cached_api("first", "First")
        set_cached_api("second", "Second")
        result = cache_module.clear_cache(clear_memory=False)
        self.assertEqual(result.api_entries_cleared, 2)
        self.assertIsNone(get_cached_api("first"))

    def test_state_purge_failure_propagates(self) -> None:
        vault = get_library_state()
        track = vault.db_path.parent / "track.flac"
        track.write_bytes(b"audio")
        vault.record_track_state(track)
        with (
            patch.object(Path, "unlink", side_effect=PermissionError("locked state")),
            self.assertRaises(PermissionError),
        ):
            vault.clear_state(purge=True)
        self.assertEqual(vault.get_state_count(), 1)

    def test_state_primary_key_has_no_redundant_index(self) -> None:
        vault = LibraryStateVault(get_api_cache_dir().parent / "state.db")
        self.addCleanup(vault.close)
        with closing(sqlite3.connect(vault.db_path)) as database:
            indexes = list(database.execute("PRAGMA index_list('track_state')"))
        self.assertEqual(len(indexes), 1)
