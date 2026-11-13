import dataclasses
import json
import os
import subprocess
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import taglib
from mutagen.flac import FLAC
from mutagen.mp4 import MP4
from PIL import Image

from sonora.audio import art, bpm, key, metadata, replaygain
from sonora.audio import decode as decode_module
from sonora.audio.metadata import (
    clear_metadata_cache,
    read_track_metadata,
    write_track_metadata,
)
from sonora.core.artists import get_primary_artist
from sonora.core.audio_files import find_audio_files
from sonora.core.config import get_config
from sonora.core.models import NormalizeReport, TrackInfo
from sonora.core.state import LibraryStateVault
from sonora.core.titles import score_artist_title_match, strip_corrupt_brackets
from sonora.modules import (
    matching,
    metadata_merge,
    normalization,
    tagger,
    track_processing,
)
from sonora.modules import musicbrainz_metadata as musicbrainz_metadata_module
from sonora.modules import track_processing as track_processing_module
from sonora.modules.backup import backup_library_tags, restore_library_tags
from sonora.modules.catalog_metadata import _update_deezer, _update_shazam
from sonora.modules.matching import _resolve_album_track_position, is_alien_album_track
from sonora.modules.metadata_merge import (
    apply_metadata_mapping,
    build_provider_proposal,
)
from sonora.modules.musicbrainz_metadata import (
    _resolve_musicbrainz_track_id,
    _update_musicbrainz,
)
from sonora.modules.tagger import (
    _AlbumMetadataContext,
    _resolve_album_folder_identity,
    process_single_track,
    tag_album_folder,
)
from sonora.modules.track_processing import (
    _compute_tag_diffs,
    normalize_single_track,
)
from sonora.services import artist_resolution, deezer, discogs, itunes, musicbrainz
from sonora.services.artist_resolution import resolve_artist_name
from sonora.services.shazam import ShazamTrackInfo
from tests.fixtures import image_bytes, write_silent_wav
from tests.module_support import ModuleTestCase, OfflinePipelineTestCase
from tests.regression_support import AudioRegressionTestCase
from tests.support import IsolatedTestCase

MBID = "11111111-2222-3333-4444-555555555555"


class TestAlbumPipelineContracts(IsolatedTestCase):
    def test_process_track_reuses_album_metadata_without_mutating_context(self) -> None:
        path = self.test_directory / "cached.flac"
        path.touch()
        original = TrackInfo(path, title="Song", artist="Original")
        context = tagger._AlbumMetadataContext(track_metadata={path: original})
        with (
            patch.object(
                metadata,
                "read_track_metadata",
                side_effect=AssertionError("track metadata was read twice"),
            ),
            patch(
                "sonora.services.artist_resolution.resolve_artist_name",
                return_value="Canonical",
            ),
            patch.object(tagger, "_update_metadata"),
            patch.object(
                tagger,
                "_process_audio_and_save_tags",
                side_effect=lambda **values: values["track_info"],
            ),
        ):
            result = tagger.process_single_track(
                path,
                fetch_bpm=False,
                fetch_key=False,
                fetch_lyrics=False,
                fetch_itunes_art=False,
                album_context=context,
                dry_run=True,
            )
        self.assertEqual(result.artist, "Canonical")
        self.assertEqual(context.track_metadata[path].artist, "Original")

    def test_genre_agreement_abstains_on_conflicts_and_ties(self) -> None:
        for genres, deezer_album, itunes_album in (
            (["Unknown", "Rock"], {"genre": "Pop"}, {"genre": "Jazz"}),
            (["Rock", "Jazz", "Unknown", "Unknown"], None, None),
            (["Rock", "Unknown"], {"genre": "Unknown"}, None),
        ):
            recordings = [
                TrackInfo(self.test_directory / f"{index}.flac", genre=genre)
                for index, genre in enumerate(genres)
            ]
            with (
                self.subTest(genres=genres),
                patch.object(metadata, "write_track_metadata") as write,
            ):
                tagger._harmonize_album_genres(
                    recordings, deezer_album, itunes_album, dry_run=True
                )
                if deezer_album and itunes_album:
                    self.assertEqual([track.genre for track in recordings], genres)
                    self.assertTrue(
                        all("genre" in track.metadata_conflicts for track in recordings)
                    )
                elif len(genres) == 4:
                    self.assertEqual([track.genre for track in recordings], genres)
                else:
                    self.assertEqual(
                        [track.genre for track in recordings], ["Rock", "Rock"]
                    )
            write.assert_not_called()

    def test_disputed_track_genre_does_not_vote_in_album_counts(self) -> None:
        recordings = [
            TrackInfo(
                self.test_directory / "first.flac",
                genre="Rock",
                metadata_conflicts={"genre": "disputed"},
            ),
            TrackInfo(self.test_directory / "second.flac", genre="Unknown"),
        ]
        tagger._harmonize_album_genres(recordings, None, None, dry_run=True)
        self.assertEqual(recordings[1].genre, "Unknown")

    def test_root_directory_identity_and_blank_artist_agreement_remain_unknown(
        self,
    ) -> None:
        self.assertEqual(
            tagger._resolve_album_folder_identity(Path("/"), []), (None, None)
        )
        recordings = [
            TrackInfo(Path("first.flac"), artist=""),
            TrackInfo(Path("second.flac"), artist=" "),
        ]
        with patch.object(metadata, "write_track_metadata") as write:
            tagger._harmonize_album_artists(recordings)
        self.assertEqual([recording.artist for recording in recordings], ["", " "])
        write.assert_not_called()

    def test_missing_label_identifier_does_not_trigger_a_label_logo_request(
        self,
    ) -> None:
        with (
            patch.object(art, "process_artist_artwork") as artist,
            patch.object(art, "process_label_artwork") as label,
        ):
            tagger._finalize_album_assets_and_replaygain(
                self.test_directory,
                [self.recording],
                [self.recording],
                None,
                None,
                False,
                False,
                True,
                1,
            )
        artist.assert_called_once()
        label.assert_not_called()

    def test_partial_release_indexes_do_not_establish_identity(self) -> None:
        releases: tuple[dict[str, object], ...] = (
            {"tracks_by_disc_and_position": {(1, 1): {"recording_mbid": None}}},
            {"tracks_by_disc_and_position": {(1, 1): {}}},
        )
        for release in releases:
            recording = dataclasses.replace(self.recording, musicbrainz_albumid=MBID)
            with (
                self.subTest(release=release),
                patch.object(
                    tagger,
                    "_resolve_album_folder_identity",
                    return_value=("Primary", "Record"),
                ),
                patch.object(metadata, "read_track_metadata", return_value=recording),
                patch.object(
                    musicbrainz,
                    "fetch_musicbrainz_release_details",
                    return_value=release,
                ),
                patch.object(deezer, "fetch_deezer_album_details", return_value=None),
                patch.object(itunes, "fetch_itunes_album_details", return_value=None),
            ):
                context = tagger._resolve_album_metadata_context(
                    self.test_directory,
                    [self.recording.file_path],
                    None,
                    False,
                    False,
                    False,
                )
            self.assertIsNone(context.track_mbids)

    def test_itunes_album_track_artists_are_preserved_during_identity_updates(
        self,
    ) -> None:
        recording = TrackInfo(
            self.test_directory / "03 - Recording.flac",
            title="Recording",
            artist="Guest",
            album="Record",
            track_number=3,
        )
        itunes_release = {
            "tracks_by_disc_and_position": {
                (1, 3): {"title": "Recording", "artist": "Guest"}
            }
        }
        tagger._apply_album_metadata(
            recording,
            recording.file_path,
            True,
            album_context=tagger._AlbumMetadataContext(
                itunes_details=itunes_release,
                sample_artist="Primary",
                sample_album="Record",
            ),
        )
        self.assertEqual(recording.artist, "Guest")
        self.assertEqual(recording.album_artist, "Primary")

    def test_deezer_album_genre_can_override_unknown_track_genres(self) -> None:
        recordings = [
            TrackInfo(self.test_directory / "first.flac", genre="Unknown"),
            TrackInfo(self.test_directory / "second.flac", genre="Rock"),
        ]
        tagger._harmonize_album_genres(recordings, {"genre": "Pop"}, None, dry_run=True)
        self.assertEqual(recordings[0].genre, "Pop")
        self.assertEqual(recordings[1].genre, "Rock")

    def test_artist_agreement_preserves_distinct_artist_identity_in_compilations(
        self,
    ) -> None:
        recordings = [
            TrackInfo(
                self.test_directory / "first.flac",
                artist="Primary",
                album_artist="Various Artists",
            ),
            TrackInfo(
                self.test_directory / "second.flac",
                artist="Secondary",
                album_artist="Various Artists",
            ),
            TrackInfo(
                self.test_directory / "guest.flac", artist="Guest", album_artist="guest"
            ),
        ]
        with patch.object(metadata, "write_track_metadata") as write:
            tagger._harmonize_album_artists(recordings)
        self.assertEqual(
            (recordings[2].artist, recordings[2].album_artist), ("guest", "guest")
        )
        self.assertEqual(
            [recording.artist for recording in recordings[:2]], ["Primary", "Secondary"]
        )
        write.assert_called_once_with(recordings[2])

    def test_itunes_album_genre_is_an_authoritative_agreement(self) -> None:
        recordings = [
            TrackInfo(self.test_directory / "first.flac", genre="Unknown"),
            TrackInfo(self.test_directory / "second.flac", genre="Rock"),
        ]
        tagger._harmonize_album_genres(recordings, None, {"genre": "Pop"}, dry_run=True)
        self.assertEqual(recordings[0].genre, "Pop")
        self.assertEqual(recordings[1].genre, "Rock")

    def setUp(self) -> None:
        self.recording = TrackInfo(
            self.test_directory / "recording.flac",
            artist="Primary",
            title="Recording",
            album="Record",
        )

    def test_cached_album_read_failure_does_not_duplicate_partial_results(self) -> None:
        paths = [
            self.test_directory / f"0{number} - Recording.flac" for number in (1, 2)
        ]
        for path in paths:
            path.touch()
        recordings = [
            dataclasses.replace(self.recording, file_path=path) for path in paths
        ]
        with (
            patch.object(LibraryStateVault, "filter_outdated_tracks", return_value=[]),
            patch.object(
                tagger,
                "group_files_by_album_root",
                return_value={self.test_directory: paths},
            ),
            patch.object(
                metadata,
                "read_track_metadata",
                side_effect=[recordings[0], OSError("unreadable cached metadata")],
            ),
            patch.object(
                tagger,
                "_resolve_album_metadata_context",
                return_value=tagger._AlbumMetadataContext(),
            ),
            patch.object(
                tagger,
                "process_single_track",
                side_effect=lambda file_path, **options: dataclasses.replace(
                    self.recording, file_path=file_path
                ),
            ) as tag,
        ):
            results = tagger.tag_album_folder(
                self.test_directory,
                fetch_bpm=False,
                fetch_key=False,
                fetch_replaygain=False,
                fetch_lyrics=False,
                fetch_itunes_art=False,
                dry_run=True,
            )
        self.assertEqual(len(results), 2)
        self.assertEqual({recording.file_path for recording in results}, set(paths))
        self.assertEqual(tag.call_count, 2)

    def test_up_to_date_album_skips_metadata_updates_and_preserves_results(
        self,
    ) -> None:
        self.recording.file_path.touch()
        with (
            patch.object(LibraryStateVault, "filter_outdated_tracks", return_value=[]),
            patch.object(metadata, "read_track_metadata", return_value=self.recording),
            patch.object(tagger, "_resolve_album_metadata_context") as context,
        ):
            self.assertEqual(
                tagger.tag_album_folder(self.test_directory, dry_run=True),
                [self.recording],
            )
        context.assert_not_called()

    def test_failed_track_is_collected_and_programming_errors_propagate(self) -> None:
        self.recording.file_path.touch()
        failures: list[dict[str, str]] = []
        with (
            patch.object(
                tagger,
                "_resolve_album_metadata_context",
                return_value=tagger._AlbumMetadataContext(),
            ),
            patch.object(
                tagger, "process_single_track", side_effect=OSError("broken audio")
            ),
        ):
            self.assertEqual(
                tagger.tag_album_folder(
                    self.test_directory,
                    force=True,
                    dry_run=True,
                    fetch_itunes_art=False,
                    failures=failures,
                ),
                [],
            )
            self.assertEqual(
                tagger.tag_album_folder(
                    self.test_directory,
                    force=True,
                    dry_run=True,
                    fetch_itunes_art=False,
                ),
                [],
            )
        self.assertEqual(failures[0]["error_type"], "OSError")
        with (
            patch.object(
                tagger,
                "_resolve_album_metadata_context",
                side_effect=RuntimeError("pipeline bug"),
            ),
            self.assertRaisesRegex(RuntimeError, "pipeline bug"),
        ):
            tagger.tag_album_folder(self.test_directory, force=True, dry_run=True)

    def test_missing_track_fails_before_provider_lookup(self) -> None:
        with self.assertRaises(FileNotFoundError):
            tagger.process_single_track(self.recording.file_path)

    def test_folder_identity_can_ignore_generic_structure_and_unreadable_tags(
        self,
    ) -> None:
        folder = self.test_directory / "Unknown - Unknown"
        folder.mkdir()
        with patch.object(
            metadata, "read_track_metadata", side_effect=OSError("corrupt tags")
        ):
            self.assertEqual(
                tagger._resolve_album_folder_identity(
                    folder, [folder / "recording.flac"]
                ),
                (None, None),
            )

    def test_single_track_genre_and_distinct_album_artist_need_no_agreement(
        self,
    ) -> None:
        with patch.object(metadata, "write_track_metadata") as write:
            tagger._harmonize_album_genres([self.recording], None, None)
            tagger._harmonize_album_artists(
                [
                    dataclasses.replace(
                        self.recording, artist="Guest", album_artist="Primary"
                    )
                ],
                dry_run=True,
            )
        write.assert_not_called()

    def test_prefetch_uses_majority_release_id_and_canonical_track_indexes(
        self,
    ) -> None:
        paths = [
            self.test_directory / f"recording-{number}.flac" for number in range(3)
        ]
        tagged_by_path = {
            path: dataclasses.replace(
                self.recording,
                file_path=path,
                title=f"Song {number}",
                track_number=number,
                musicbrainz_albumid=MBID,
            )
            for number, path in enumerate(paths, 1)
        }
        recordings = {
            (1, number): {
                "title": f"Song {number}",
                "artist": "Primary",
                "recording_mbid": f"00000000-0000-0000-0000-{number:012d}",
            }
            for number in range(1, 4)
        }
        release = {"tracks_by_disc_and_position": recordings}
        with (
            patch.object(
                tagger,
                "_resolve_album_folder_identity",
                return_value=("Primary", "Record"),
            ),
            patch.object(
                metadata, "read_track_metadata", side_effect=tagged_by_path.__getitem__
            ) as read_metadata,
            patch.object(musicbrainz, "search_musicbrainz_release") as search,
            patch.object(
                musicbrainz, "fetch_musicbrainz_release_details", return_value=release
            ),
            patch.object(
                deezer, "fetch_deezer_album_details", return_value={"genre": "Rock"}
            ),
            patch.object(
                itunes, "fetch_itunes_album_details", return_value={"genre": "Rock"}
            ),
            patch.object(
                discogs, "search_discogs_release", return_value={"label": "Publisher"}
            ),
            patch.object(art, "process_album_cover_art", return_value=None),
        ):
            context = tagger._resolve_album_metadata_context(
                self.test_directory, paths, "fixture", True, True, True
            )
        search.assert_not_called()
        self.assertEqual(read_metadata.call_count, len(paths))
        self.assertEqual(context.musicbrainz_id, MBID)
        self.assertEqual(
            context.track_mbids,
            {
                (1, number): f"00000000-0000-0000-0000-{number:012d}"
                for number in range(1, 4)
            },
        )
        self.assertEqual(context.discogs_release, {"label": "Publisher"})
        self.assertEqual(context.sample_artist, "Primary")

    def test_prefetch_uses_majority_itunes_collection_id(self) -> None:
        paths = [
            self.test_directory / f"recording-{number}.flac" for number in range(3)
        ]
        tracks = {
            path: dataclasses.replace(
                self.recording,
                file_path=path,
                itunes_collectionid="777" if index < 2 else "888",
            )
            for index, path in enumerate(paths)
        }
        with (
            patch.object(
                tagger,
                "_resolve_album_folder_identity",
                return_value=("Primary", "Record"),
            ),
            patch.object(
                metadata, "read_track_metadata", side_effect=tracks.__getitem__
            ),
            patch.object(musicbrainz, "search_musicbrainz_release", return_value=None),
            patch.object(deezer, "fetch_deezer_album_details", return_value=None),
            patch.object(
                itunes, "fetch_itunes_album_details", return_value=None
            ) as fetch,
        ):
            tagger._resolve_album_metadata_context(
                self.test_directory, paths, None, False, False, True
            )
        self.assertEqual(fetch.call_args.kwargs["collection_id"], "777")

    def test_prefetch_missing_recording_indexes_withholds_release_metadata(
        self,
    ) -> None:
        with (
            patch.object(
                tagger,
                "_resolve_album_folder_identity",
                return_value=("Primary", "Record"),
            ),
            patch.object(
                metadata, "read_track_metadata", side_effect=OSError("unreadable")
            ),
            patch.object(
                musicbrainz, "search_musicbrainz_release", return_value={"id": MBID}
            ),
            patch.object(
                musicbrainz, "fetch_musicbrainz_release_details", return_value=None
            ),
            patch.object(deezer, "fetch_deezer_album_details", return_value=None),
            patch.object(itunes, "fetch_itunes_album_details", return_value=None),
        ):
            context = tagger._resolve_album_metadata_context(
                self.test_directory,
                [self.recording.file_path],
                None,
                False,
                False,
                True,
            )
        self.assertIsNone(context.track_mbids)

    def test_prefetch_empty_identity_and_network_error_preserve_empty_context(
        self,
    ) -> None:
        with patch.object(
            tagger, "_resolve_album_folder_identity", return_value=(None, None)
        ):
            context = tagger._resolve_album_metadata_context(
                self.test_directory, [], None, False, False, True
            )
        self.assertIsNone(context.musicbrainz_id)
        with patch.object(
            tagger,
            "_resolve_album_folder_identity",
            side_effect=OSError("directory locked"),
        ):
            context = tagger._resolve_album_metadata_context(
                self.test_directory, [], None, False, False, True
            )
        self.assertIsNone(context.musicbrainz_id)

    def test_genre_harmonization_honors_album_metadata_and_majority(self) -> None:
        cases = (
            (["Rock", "Rock", "unknown"], None, None, ["Rock", "Rock", "Rock"]),
            (["Rock", "Jazz"], None, {"genre": "pop"}, ["Rock", "Jazz"]),
            (["unknown", None], None, {"genre": "pop"}, ["Pop", "Pop"]),
            (["Rock", "Jazz", "Pop"], None, None, ["Rock", "Jazz", "Pop"]),
        )
        for genres, deezer_album, itunes_album, expected in cases:
            recordings = [
                dataclasses.replace(self.recording, genre=genre) for genre in genres
            ]
            with (
                self.subTest(genres=genres),
                patch.object(
                    metadata, "write_track_metadata", side_effect=OSError("read-only")
                ),
            ):
                tagger._harmonize_album_genres(recordings, deezer_album, itunes_album)
            self.assertEqual([recording.genre for recording in recordings], expected)
        tagger._harmonize_album_genres([], None, None)

    def test_artist_casing_harmonization_keeps_distinct_performers(self) -> None:
        recordings = [
            dataclasses.replace(
                self.recording, artist="PRIMARY", album_artist="PRIMARY"
            ),
            dataclasses.replace(
                self.recording, artist="Primary", album_artist="Primary"
            ),
            dataclasses.replace(self.recording, artist="Guest", album_artist="Primary"),
        ]
        with (
            patch(
                "sonora.services.artist_resolution.resolve_artist_name",
                side_effect=lambda name: name,
            ),
            patch.object(
                metadata, "write_track_metadata", side_effect=OSError("read-only")
            ),
        ):
            tagger._harmonize_album_artists(recordings)
        self.assertEqual(
            [recording.artist for recording in recordings],
            ["Primary", "Primary", "Guest"],
        )
        self.assertEqual(
            [recording.album_artist for recording in recordings], ["Primary"] * 3
        )
        tagger._harmonize_album_artists([])

    def test_finalization_excludes_alien_files_from_album_replaygain(self) -> None:
        alien = dataclasses.replace(
            self.recording, file_path=self.test_directory / "alien.flac", is_alien=True
        )
        with (
            patch.object(
                replaygain, "calculate_album_replaygain", return_value=True
            ) as calculate,
            patch.object(art, "process_artist_artwork", side_effect=OSError("offline")),
            patch.object(
                art, "process_label_artwork", side_effect=OSError("offline")
            ) as label,
        ):
            tagger._finalize_album_assets_and_replaygain(
                self.test_directory,
                [self.recording, alien],
                [self.recording],
                "Primary",
                {"label_mbid": MBID, "label": "Publisher"},
                True,
                True,
                True,
                2,
            )
        self.assertEqual(calculate.call_args.args[0], [self.recording.file_path])
        label.assert_called_once()

    def test_empty_album_can_configure_explicit_credentials_without_api_requests(
        self,
    ) -> None:
        self.assertEqual(
            tagger.tag_album_folder(
                self.test_directory,
                fanart_api_key="fixture",
                fanart_client_key="client",
                enable_shazam=False,
            ),
            [],
        )
        self.assertEqual(get_config().fanart_api_key, "fixture")
        self.assertEqual(get_config().fanart_client_key, "client")
        self.assertFalse(get_config().enable_shazam)
        self.assertEqual(os.environ["FANART_API_KEY"], "fixture")


class TestBlockedAlbumAssets(IsolatedTestCase):
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


class TestAlbumProviderFailures(IsolatedTestCase):
    def setUp(self) -> None:
        self.path = self.test_directory / "01 - Recording.wav"
        self.path.touch()
        self.recording = TrackInfo(
            self.path,
            artist="Primary",
            title="Recording",
            album="Record",
            album_artist="Primary",
            track_number=1,
        )

    def test_album_prefetch_continues_after_independent_provider_failures(self) -> None:
        providers = (
            (musicbrainz, "search_musicbrainz_release", None, None),
            (deezer, "fetch_deezer_album_details", "deezer_details", {"genre": "Rock"}),
            (itunes, "fetch_itunes_album_details", "itunes_details", {"genre": "Rock"}),
            (discogs, "search_discogs_release", "discogs_release", {"genre": "Rock"}),
        )
        for failed_provider in range(len(providers)):
            with self.subTest(failed_provider=failed_provider):
                with ExitStack() as patches:
                    patches.enter_context(
                        patch.object(
                            tagger,
                            "_resolve_album_folder_identity",
                            return_value=("Primary", "Record"),
                        )
                    )
                    patches.enter_context(
                        patch.object(
                            metadata, "read_track_metadata", return_value=self.recording
                        )
                    )
                    lookups = []
                    for index, (module, function, _, response) in enumerate(providers):
                        lookups.append(
                            patches.enter_context(
                                patch.object(
                                    module,
                                    function,
                                    return_value=response,
                                    side_effect=httpx.ConnectError("unavailable")
                                    if index == failed_provider
                                    else None,
                                )
                            )
                        )
                    context = tagger._resolve_album_metadata_context(
                        self.test_directory, [self.path], "token", False, False, True
                    )
                for lookup in lookups:
                    lookup.assert_called_once()
                for index, (_, _, field, response) in enumerate(providers):
                    if field:
                        self.assertEqual(
                            getattr(context, field),
                            None if index == failed_provider else response,
                        )


class TestMetadataContracts(IsolatedTestCase):
    def test_provider_album_capitalization_preserves_consistent_local_spelling(
        self,
    ) -> None:
        track = TrackInfo(Path("recording.flac"), album="Un Film De Oscar")
        apply_metadata_mapping(
            track, {"album": "Un film de oscar"}, {"album": "album"}, force=True
        )
        self.assertEqual(track.album, "Un Film De Oscar")
        apply_metadata_mapping(
            track,
            {"album": "Un Film De Oscar (Deluxe)"},
            {"album": "album"},
            force=True,
        )
        self.assertEqual(track.album, "Un Film De Oscar (Deluxe)")

    def test_report_export_owns_provenance_and_excludes_internal_changes(self) -> None:
        track = TrackInfo(
            Path("recording.flac"),
            metadata_sources={"title": "MusicBrainz"},
            _diff_descriptions=["changed"],
        )
        exported = track.to_dict()
        sources = exported["metadata_sources"]
        assert isinstance(sources, dict)
        sources.clear()
        self.assertEqual(track.metadata_sources, {"title": "MusicBrainz"})
        self.assertNotIn("_diff_descriptions", exported)
        self.assertEqual(exported["file_path"], "recording.flac")

    def test_failed_atomic_commit_keeps_pending_advisory_normalization(self) -> None:
        path = self.test_directory / "recording.flac"
        path.write_bytes(b"original audio")
        recording = TrackInfo(path, _advisory_needs_normalization=True)
        song = MagicMock(spec=taglib.File)
        song.tags = {}
        song.save.return_value = []
        with (
            patch("taglib.File") as open_audio,
            patch.object(Path, "replace", side_effect=OSError("disk full")),
            self.assertRaisesRegex(OSError, "disk full"),
        ):
            open_audio.return_value.__enter__.return_value = song
            metadata.write_track_metadata(recording)
        self.assertTrue(recording._advisory_needs_normalization)
        self.assertEqual(path.read_bytes(), b"original audio")

    def test_track_copy_owns_provenance_and_change_collections(self) -> None:
        original = TrackInfo(
            Path("recording.flac"),
            metadata_sources={"title": "MusicBrainz"},
            metadata_conflicts={"isrc": "Catalog disagreement"},
            _diff_descriptions=["Title normalized"],
        )
        copied = original.copy()
        self.assertEqual(copied, original)
        copied.metadata_sources.clear()
        copied.metadata_conflicts.clear()
        copied._diff_descriptions.clear()
        self.assertEqual(original.metadata_sources, {"title": "MusicBrainz"})
        self.assertEqual(original.metadata_conflicts, {"isrc": "Catalog disagreement"})
        self.assertEqual(original._diff_descriptions, ["Title normalized"])
        empty = TrackInfo(Path("empty.flac"))
        empty.copy().metadata_conflicts["title"] = "Review required"
        self.assertEqual(empty.metadata_conflicts, {})

    def test_normalization_report_counts_include_failures(self) -> None:
        report = NormalizeReport()
        self.assertEqual((report.total_scanned, report.total_unchanged), (0, 0))
        report.total_scanned += 1
        self.assertEqual((report.total_scanned, report.total_unchanged), (1, 1))
        report.modified_files["recording.flac"] = ["Title normalized"]
        self.assertEqual((report.total_modified, report.total_unchanged), (1, 0))
        report.total_scanned += 1
        report.failed_files += 1
        self.assertEqual((report.total_scanned, report.total_unchanged), (2, 0))

    def test_embedded_dimensions_and_explicit_feature_tags_need_no_fallback(
        self,
    ) -> None:
        path = self.test_directory / "recording.wav"
        path.touch()
        song = MagicMock(spec=taglib.File)
        song.tags = {
            "TITLE": ["Recording"],
            "ARTIST": ["Primary"],
            "FEATURED_ARTISTS": ["Guest"],
        }
        song.pictures = [
            taglib.Picture(data=b"", mime_type="image/png", width=600, height=600)
        ]
        song.sampleRate = 44100
        song.bitrate = 128
        song.channels = 2
        song.length = 1
        with patch("taglib.File") as open_audio:
            open_audio.return_value.__enter__.return_value = song
            recording = metadata.read_track_metadata(path)
        self.assertEqual((recording.art_width, recording.art_height), (600, 600))
        self.assertEqual(recording.featured_artists, "Guest")

    def test_missing_audio_cannot_be_written(self) -> None:
        with self.assertRaises(FileNotFoundError):
            metadata.write_track_metadata(
                TrackInfo(self.test_directory / "missing.flac")
            )

    def test_numbering_and_date_writes_remove_absent_disc_numbers(self) -> None:
        tags = {"DISCNUMBER": ["2"]}
        metadata._write_numbering_and_dates(
            tags, TrackInfo(Path("recording.flac"), disc_number=None, date="2020")
        )
        self.assertNotIn("DISCNUMBER", tags)
        self.assertEqual(tags["DATE"], ["2020"])

    def test_read_failure_from_concurrent_removal_preserves_file_not_found(
        self,
    ) -> None:
        path = self.test_directory / "recording.wav"
        path.touch()
        with (
            patch("taglib.File", side_effect=FileNotFoundError("removed concurrently")),
            self.assertRaises(FileNotFoundError),
        ):
            metadata.read_track_metadata(path)

    def test_artist_lists_deduplicate_overlapping_features_and_empty_album_credits(
        self,
    ) -> None:
        tags: dict[str, list[str]] = {}
        metadata._write_artist_tags(
            tags,
            TrackInfo(
                Path("recording.flac"),
                artist="Primary & Guest",
                featured_artists="Guest",
                album_artist=" ",
            ),
        )
        self.assertEqual(tags["ARTISTS"], ["Primary", "Guest"])
        self.assertNotIn("ALBUMARTISTS", tags)

    def test_filename_and_disc_folder_recover_missing_numbering(self) -> None:
        for relative_path, expected_disc, expected_track in (
            ("CD 2/recording.wav", 2, None),
            ("2-03 - Recording.wav", 2, 3),
        ):
            path = self.test_directory / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            song = MagicMock(spec=taglib.File)
            song.tags = {
                "TITLE": ["Recording"],
                "ARTIST": ["Primary"],
                "ARTISTS": ["Primary", "Guest"],
                "ADVISORY": ["1"],
            }
            song.pictures = [
                taglib.Picture(
                    data=image_bytes(), mime_type="image/png", picture_type="Back Cover"
                )
            ]
            song.length = 1
            song.bitrate = 128
            song.sampleRate = 44100
            song.channels = 2
            with (
                self.subTest(relative_path=relative_path),
                patch("taglib.File") as open_audio,
            ):
                open_audio.return_value.__enter__.return_value = song
                recording = metadata.read_track_metadata(path)
            self.assertEqual(
                (recording.disc_number, recording.track_number),
                (expected_disc, expected_track),
            )
            self.assertEqual(recording.featured_artists, "Guest")
            self.assertEqual(recording.advisory, "Explicit")

    def test_read_permission_and_cache_capacity_are_explicit(self) -> None:
        path = self.test_directory / "recording.wav"
        path.touch()
        with (
            patch.object(os, "access", return_value=False),
            self.assertRaises(PermissionError),
        ):
            metadata.write_track_metadata(TrackInfo(path))
        with (
            patch.object(metadata, "_MAX_METADATA_CACHE_SIZE", 0),
            patch("taglib.File") as open_audio,
        ):
            song = open_audio.return_value.__enter__.return_value
            song.tags = {}
            song.pictures = []
            song.length = 1
            song.bitrate = 128
            song.sampleRate = 44100
            song.channels = 2
            metadata.read_track_metadata(path)
        self.assertEqual(metadata.get_metadata_cache_size(), 1)

    def test_numbering_without_totals_and_original_year_are_written(self) -> None:
        tags: dict[str, list[str]] = {}
        metadata._write_numbering_and_dates(
            tags,
            TrackInfo(
                Path("recording.flac"),
                disc_number=2,
                track_number=3,
                date="2020-05-12",
                original_date="1999-01-02",
            ),
        )
        self.assertEqual(tags["DISCNUMBER"], ["2"])
        self.assertEqual(tags["TRACKNUMBER"], ["3"])
        self.assertEqual(tags["DATE"], ["2020-05-12"])
        self.assertEqual(tags["YEAR"], ["2020"])
        self.assertEqual(tags["ORIGINALYEAR"], ["1999"])
        self.assertNotIn("TRACKTOTAL", tags)
        self.assertNotIn("DISCTOTAL", tags)

    def test_duration_is_available_without_metadata_tags(self) -> None:
        path = self.test_directory / "untagged.wav"
        write_silent_wav(path)
        self.assertEqual(metadata.get_audio_duration(path), 0.1)

    def test_mutagen_duration_and_corrupt_embedded_cover(self) -> None:
        with patch(
            "mutagen.File",
            return_value=SimpleNamespace(info=SimpleNamespace(length=12.5)),
        ):
            self.assertEqual(metadata.get_audio_duration(Path("recording.flac")), 12.5)
        path = self.test_directory / "recording.wav"
        path.write_bytes(b"audio fixture")
        cover = self.test_directory / "cover.png"
        cover.write_bytes(image_bytes())
        with patch("taglib.File") as open_audio:
            song = open_audio.return_value.__enter__.return_value
            song.tags = {}
            song.pictures = [
                taglib.Picture(
                    data=b"corrupt", mime_type="image/png", picture_type="Front Cover"
                )
            ]
            song.save.return_value = []
            metadata.write_track_metadata(TrackInfo(path), cover)
            self.assertEqual(len(song.pictures), 1)
            self.assertEqual(song.pictures[0].data, cover.read_bytes())

    def test_cover_replacement_cannot_reduce_either_image_dimension(self) -> None:
        path = self.test_directory / "recording.wav"
        path.write_bytes(b"audio fixture")
        replacement = self.test_directory / "cover.png"
        replacement.write_bytes(image_bytes((1000, 1000)))
        for dimensions in ((1400, 600), (600, 1400)):
            embedded = taglib.Picture(
                data=image_bytes(dimensions),
                mime_type="image/png",
                picture_type="Front Cover",
            )
            with (
                self.subTest(dimensions=dimensions),
                patch("taglib.File") as open_audio,
            ):
                song = open_audio.return_value.__enter__.return_value
                song.tags = {}
                song.pictures = [embedded]
                song.save.return_value = []
                metadata.write_track_metadata(TrackInfo(path), replacement)
                self.assertEqual(song.pictures[0].data, embedded.data)

    def test_unchanged_album_artist_preserves_original_multivalue_credits(self) -> None:
        original = {"ALBUMARTIST": ["Primary"], "ALBUMARTISTS": ["Primary", "Guest"]}
        updated = {"ALBUMARTIST": ["Primary"]}
        metadata._preserve_unchanged_multivalues(original, updated)
        self.assertEqual(updated["ALBUMARTISTS"], ["Primary", "Guest"])

    def test_mp4_codec_distinguishes_lossless_from_lossy(self) -> None:
        for codec, expected in (("alac", True), ("aac", False)):
            with (
                self.subTest(codec=codec),
                patch.object(
                    metadata,
                    "MP4",
                    return_value=SimpleNamespace(info=SimpleNamespace(codec=codec)),
                ),
            ):
                self.assertEqual(
                    metadata._read_audio_encoding(Path("recording.m4a")),
                    (expected, None),
                )
        with patch.object(metadata, "MP4", side_effect=OSError("broken mp4")):
            self.assertEqual(
                metadata._read_audio_encoding(Path("recording.m4a")), (False, None)
            )
        self.assertEqual(
            metadata._read_audio_encoding(Path("recording.unknown")), (True, None)
        )

    def test_audio_properties_roundtrip_optional_values_and_compilation_policy(
        self,
    ) -> None:
        recording = TrackInfo(
            Path("recording.flac"), bpm=123.45, rating=8.5, compilation=True
        )
        tags: dict[str, list[str]] = {}
        metadata._write_audio_properties_tags(tags, recording)
        self.assertEqual(tags["BPM"], ["123.5"])
        self.assertEqual(tags["RATING"], ["8.5"])
        self.assertEqual(tags["COMPILATION"], ["1"])
        metadata._write_audio_properties_tags(
            tags, dataclasses.replace(recording, compilation=False)
        )
        self.assertEqual(tags["COMPILATION"], ["0"])
        metadata._write_audio_properties_tags(
            tags, TrackInfo(recording.file_path, album_artist="Various Artists")
        )
        self.assertEqual(tags["COMPILATION"], ["1"])
        self.assertNotIn("BPM", tags)
        metadata._write_audio_properties_tags(tags, TrackInfo(recording.file_path))
        self.assertNotIn("COMPILATION", tags)

    def test_schema_writer_removes_invalid_uuid_and_clean_advisory_aliases(
        self,
    ) -> None:
        tags = {
            "MUSICBRAINZ_TRACKID": ["old"],
            "ITUNESADVISORY": ["1"],
            "ADVISORY": ["Explicit"],
        }
        recording = TrackInfo(
            Path("recording.flac"),
            musicbrainz_trackid="invalid",
            advisory="Clean",
            disc_subtitle="Second movement",
        )
        metadata._write_schema_tags(tags, recording)
        self.assertNotIn("MUSICBRAINZ_TRACKID", tags)
        self.assertNotIn("ADVISORY", tags)
        self.assertNotIn("ITUNESADVISORY", tags)
        self.assertEqual(tags["DISCSUBTITLE"], ["Second movement"])
        self.assertEqual(tags["TSST"], ["Second movement"])

    def test_picture_dimensions_are_decoded_when_taglib_does_not_supply_them(
        self,
    ) -> None:
        path = self.test_directory / "recording.wav"
        path.write_bytes(b"audio fixture")
        song = MagicMock(spec=taglib.File)
        song.tags = {
            "TITLE": ["Recording"],
            "ARTIST": ["Primary"],
            "ADVISORY": ["clean"],
        }
        song.pictures = [
            taglib.Picture(
                data=image_bytes((640, 640)),
                mime_type="image/png",
                picture_type="Front Cover",
            )
        ]
        song.length = 1
        song.bitrate = 128
        song.sampleRate = 44100
        song.channels = 2
        with patch("taglib.File") as open_audio:
            open_audio.return_value.__enter__.return_value = song
            recording = metadata.read_track_metadata(path)
        self.assertEqual((recording.art_width, recording.art_height), (640, 640))
        self.assertIsNone(recording.advisory)
        metadata.clear_metadata_cache()
        song.pictures = [
            taglib.Picture(
                data=b"broken", mime_type="image/png", picture_type="Front Cover"
            )
        ]
        with patch("taglib.File") as open_audio:
            open_audio.return_value.__enter__.return_value = song
            recording = metadata.read_track_metadata(path)
        self.assertIsNone(recording.art_width)
        self.assertIsNone(recording.art_height)

    def test_failed_save_and_concurrent_file_removal_leave_audio_untouched(
        self,
    ) -> None:
        path = self.test_directory / "recording.flac"
        path.write_bytes(b"original audio")
        recording = TrackInfo(path)
        song = MagicMock(spec=taglib.File)
        song.tags = {}
        song.pictures = []
        song.save.return_value = ["TITLE"]
        with (
            patch("taglib.File") as open_audio,
            self.assertRaisesRegex(OSError, "Unsupported tags"),
        ):
            open_audio.return_value.__enter__.return_value = song
            metadata.write_track_metadata(recording)
        self.assertEqual(path.read_bytes(), b"original audio")
        with (
            patch("taglib.File", side_effect=FileNotFoundError("removed concurrently")),
            self.assertRaises(FileNotFoundError),
        ):
            metadata.write_track_metadata(recording)
        self.assertEqual(path.read_bytes(), b"original audio")


class TestAdvisoryPersistence(AudioRegressionTestCase):
    def test_mp4_advisory_and_codec_share_one_mutagen_read(self) -> None:
        source = self.audio("source.wav")
        path = self.root / "recording.m4a"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(source), "-c:a", "aac", str(path)],
            check=True,
            capture_output=True,
        )
        metadata.clear_metadata_cache(path)
        with patch.object(metadata, "MP4", wraps=MP4) as open_mp4:
            metadata.read_track_metadata(path)
        open_mp4.assert_called_once_with(path)

    def test_lyrics_only_write_preserves_every_other_raw_tag(self) -> None:
        path = self.audio("lyrics.flac")
        with taglib.File(str(path)) as recording:
            recording.tags.update(
                {
                    "ARTIST": ["Artist feat. Guest"],
                    "BPM": ["123.456"],
                    "TOTALTRACKS": ["99"],
                    "CUSTOM": ["Keep"],
                }
            )
            recording.save()
        with taglib.File(str(path)) as recording:
            before = dict(recording.tags)
        track = metadata.read_track_metadata(path)
        track.lyrics = "[00:00.00] Correct words"
        metadata.write_track_metadata(track, lyrics_only=True)
        with taglib.File(str(path)) as recording:
            after = dict(recording.tags)
        self.assertEqual(
            {k: v for k, v in after.items() if k not in {"LYRICS", "UNSYNCEDLYRICS"}},
            before,
        )
        self.assertEqual(after["LYRICS"], [track.lyrics])

    def test_embedded_catalog_isrc_respects_explicit_edition(self) -> None:
        path = self.audio("edition.flac")
        metadata.write_track_metadata(
            TrackInfo(
                path,
                artist="Artist",
                title="Song",
                album="Album",
                advisory="Explicit",
                isrc="USABC2100001",
            )
        )
        for explicit, expected in ((False, "USABC2100001"), (True, "USABC2100002")):
            with self.subTest(explicit=explicit):
                with taglib.File(str(path)) as recording:
                    recording.tags["TIDAL_DATA"] = [
                        json.dumps(
                            {
                                "artist": {"name": "Artist"},
                                "title": "Song",
                                "album": {"title": "Album"},
                                "duration": 1,
                                "explicit": explicit,
                                "isrc": "USABC2100002",
                            }
                        )
                    ]
                    recording.save()
                metadata.clear_metadata_cache()
                result = metadata.read_track_metadata(path)
                self.assertEqual(result.isrc, expected)
                self.assertEqual(result.advisory, "Explicit")

    def test_tag_write_invalidates_only_the_changed_file(self) -> None:
        first = self.audio("first.flac")
        second = self.audio("second.flac")
        metadata.read_track_metadata(first)
        metadata.read_track_metadata(second)
        self.assertEqual(metadata.get_metadata_cache_size(), 2)
        metadata.write_track_metadata(TrackInfo(first, title="Changed"))
        self.assertEqual(metadata.get_metadata_cache_size(), 1)
        with patch("taglib.File", side_effect=AssertionError("unrelated cache miss")):
            self.assertEqual(metadata.read_track_metadata(second).file_path, second)

    def test_clean_native_tags_are_removed_without_force_and_dry_run_is_read_only(
        self,
    ) -> None:
        path = self.audio("recording.flac")
        metadata.write_track_metadata(TrackInfo(path, artist="Alpha", title="Song"))
        with taglib.File(str(path)) as recording:
            recording.tags["ITUNESADVISORY"] = ["2"]
            recording.tags["ADVISORY"] = ["Clean"]
            recording.save()
        metadata.clear_metadata_cache()
        before = path.read_bytes()
        preview = normalize_single_track(path, dry_run=True)
        assert preview is not None
        self.assertTrue(
            any("advisory" in change for change in preview._diff_descriptions)
        )
        self.assertEqual(path.read_bytes(), before)
        result = normalize_single_track(path)
        assert result is not None
        self.assertIsNone(result.advisory)
        with taglib.File(str(path)) as recording:
            self.assertNotIn("ITUNESADVISORY", recording.tags)
            self.assertNotIn("ADVISORY", recording.tags)
        unchanged = path.read_bytes()
        result = normalize_single_track(path)
        assert result is not None
        self.assertFalse(result._diff_descriptions)
        self.assertEqual(path.read_bytes(), unchanged)

    def test_explicit_advisory_round_trips_native_flac_mp3_and_mp4_tags(self) -> None:
        source = self.audio("source.wav")
        for suffix, codec in (("flac", "flac"), ("mp3", "libmp3lame"), ("m4a", "aac")):
            with self.subTest(format=suffix):
                path = self.root / f"recording.{suffix}"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-i",
                        str(source),
                        "-c:a",
                        codec,
                        str(path),
                    ],
                    check=True,
                    capture_output=True,
                )
                for value in (" explicit ", "1", "Clean", None):
                    expected = "Explicit" if value in (" explicit ", "1") else None
                    metadata.write_track_metadata(
                        TrackInfo(path, artist="Alpha", title="Song", advisory=value)
                    )
                    self.assertEqual(
                        metadata.read_track_metadata(path).advisory, expected
                    )
                    if suffix == "m4a":
                        native = MP4(path)
                        self.assertEqual(native.get("rtng"), [4] if expected else None)
                        self.assertNotIn("----:com.apple.iTunes:ITUNESADVISORY", native)
                        self.assertNotIn("----:com.apple.iTunes:ADVISORY", native)
                    else:
                        with taglib.File(str(path)) as native_tags:
                            self.assertEqual(
                                native_tags.tags.get("ITUNESADVISORY"),
                                ["1"] if expected else None,
                            )

    def test_native_mp4_rating_takes_precedence_over_conflicting_freeform_tags(
        self,
    ) -> None:
        source = self.audio("source.wav")
        path = self.root / "recording.m4a"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(source), "-c:a", "aac", str(path)],
            check=True,
            capture_output=True,
        )
        for rating, expected in ((4, "Explicit"), (2, None), (0, None)):
            with self.subTest(rating=rating):
                native = MP4(path)
                native["rtng"] = [rating]
                native["----:com.apple.iTunes:ADVISORY"] = [b"Explicit"]
                native.save()
                metadata.clear_metadata_cache()
                recording = metadata.read_track_metadata(path)
                self.assertEqual(recording.advisory, expected)
                self.assertTrue(recording._advisory_needs_normalization)
                metadata.write_track_metadata(recording)
                self.assertEqual(MP4(path).get("rtng"), [4] if expected else None)

    def test_backup_restores_native_mp4_rating_and_its_absence(self) -> None:
        source = self.audio("source.wav")
        path = self.root / "recording.m4a"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(source), "-c:a", "alac", str(path)],
            check=True,
            capture_output=True,
        )
        archive = self.root / "backup.json.gz"
        for rating in ([4], None):
            with self.subTest(rating=rating):
                native = MP4(path)
                if rating is None:
                    native.pop("rtng", None)
                else:
                    native["rtng"] = rating
                native.save()
                backup_library_tags(self.root, archive, max_threads=1)
                native = MP4(path)
                native["rtng"] = [2]
                native.save()
                self.assertEqual(restore_library_tags(archive, max_threads=1), 2)
                self.assertEqual(MP4(path).get("rtng"), rating)


class TestMetadataIdentityRegressions(AudioRegressionTestCase):
    def test_date_write_removes_stale_year_and_reconciles_original_year(self) -> None:
        path = self.audio("01 - Recording.flac")
        with taglib.File(str(path)) as song:
            song.tags.update(
                {
                    "ARTIST": ["Primary"],
                    "TITLE": ["Recording"],
                    "ALBUM": ["Record"],
                    "DATE": ["2009"],
                    "YEAR": ["2018"],
                    "ORIGINALDATE": ["2007"],
                    "ORIGINALYEAR": ["2008"],
                    "COMMENT": ["Preserve this comment"],
                }
            )
            song.save()
        track = read_track_metadata(path)
        self.assertEqual(track.date, "2009")
        write_track_metadata(track)
        with taglib.File(str(path)) as song:
            self.assertEqual(song.tags["DATE"], ["2009"])
            self.assertEqual(song.tags["YEAR"], ["2009"])
            self.assertEqual(song.tags["ORIGINALDATE"], ["2007"])
            self.assertEqual(song.tags["ORIGINALYEAR"], ["2007"])
            self.assertEqual(song.tags["COMMENT"], ["Preserve this comment"])

    def test_same_title_different_artist_is_preserved(self) -> None:
        audio_path = self.root / "01 - Intro.flac"
        unrelated = TrackInfo(
            audio_path,
            artist="ZZZ",
            title="Intro",
            album="Different Album",
            track_number=1,
        )
        release = {
            "album_artist": "Alpha",
            "title": "Album",
            "tracks_by_disc_and_position": {
                (1, 1): {"artist": "Alpha", "title": "Intro", "isrc": "USAAA0000001"}
            },
        }
        self.assertTrue(
            matching.is_alien_album_track(
                unrelated,
                audio_path,
                "Alpha",
                "Album",
                album_mb_release_details=release,
            )
        )
        unrelated.isrc = "USAAA0000001"
        self.assertTrue(
            matching.is_alien_album_track(
                unrelated,
                audio_path,
                "Alpha",
                "Album",
                album_mb_release_details=release,
            )
        )

    def test_offline_artist_resolution_ignores_api_cache(self) -> None:
        with patch(
            "sonora.services.artist_resolution.get_cached_api", return_value=True
        ) as api_cache:
            self.assertEqual(get_primary_artist("Example & Partners"), "Example")
            api_cache.assert_not_called()

    def test_identifier_updates_corrupt_title_and_position(self) -> None:
        for id_field in ("isrc", "itunes_trackid"):
            with self.subTest(id_field=id_field):
                track = TrackInfo(
                    self.root / "99 - Corrupt.flac", title="Untitled", track_number=99
                )
                position = matching._match_position_by_authoritative_id(
                    track,
                    99,
                    id_field,
                    "KNOWN123",
                    [
                        {
                            "tracks_by_disc_and_position": {
                                (2, 4): {id_field: "known123", "title": "Original"}
                            }
                        }
                    ],
                )
                self.assertEqual(position, 4)
                self.assertEqual(track.disc_number, 2)

    def test_repeated_identifier_requires_unique_position(self) -> None:
        track = TrackInfo(self.root / "track.flac", title="Original", track_number=99)
        release = {
            "tracks_by_disc_and_position": {
                (1, 2): {"isrc": "KNOWN", "title": "Original"},
                (2, 4): {"isrc": "KNOWN", "title": "Original"},
            }
        }
        self.assertIsNone(
            matching._match_position_by_authoritative_id(
                track, 99, "isrc", "KNOWN", [release]
            )
        )
        track.disc_number = 2
        self.assertEqual(
            matching._match_position_by_authoritative_id(
                track, 4, "isrc", "KNOWN", [release]
            ),
            4,
        )

    def test_wrong_identifier_does_not_update_from_title(self) -> None:
        track = TrackInfo(self.root / "track.flac", title="Original", track_number=2)
        release = {
            "tracks_by_disc_and_position": {
                (1, 2): {"itunes_trackid": "different", "title": "Original"}
            }
        }
        self.assertIsNone(
            matching._match_position_by_authoritative_id(
                track, 2, "itunes_trackid", "KNOWN", [release]
            )
        )

    def test_multiple_musicbrainz_isrcs_preserve_valid_existing_code(self) -> None:
        track = TrackInfo(self.root / "track.flac", isrc="SECOND")
        recording = {"title": "Original", "isrc": "FIRST", "isrcs": ["FIRST", "SECOND"]}
        metadata_merge.apply_metadata_mapping(
            track, recording, {"isrc": "isrc"}, force=True, source="MusicBrainz"
        )
        self.assertEqual(track.isrc, "SECOND")
        self.assertTrue(matching._recording_isrc_matches("second", recording))
        self.assertFalse(matching._recording_isrc_matches("UNKNOWN", recording))
        self.assertFalse(matching._recording_isrc_matches(None, recording))
        self.assertFalse(matching._recording_isrc_matches("SECOND", {"isrcs": []}))
        self.assertEqual(
            matching._match_position_by_authoritative_id(
                track,
                None,
                "isrc",
                "SECOND",
                [{"tracks_by_disc_and_position": {(1, 3): recording}}],
            ),
            3,
        )

    def test_plain_diff_report_handles_add_remove_and_lyrics(self) -> None:
        original = TrackInfo(self.root / "track.flac", genre="Pop", composer="Composer")
        updated = dataclasses.replace(
            original,
            genre="Rock",
            composer=None,
            label="Label",
            lyrics="private lyric content",
        )
        descriptions = track_processing_module._describe_tag_diffs(
            track_processing_module._compute_tag_diffs(original, updated)
        )
        self.assertIn("genre: 'Pop' -> 'Rock'", descriptions)
        self.assertIn("composer: removed 'Composer'", descriptions)
        self.assertIn("label: added 'Label'", descriptions)
        self.assertTrue(
            any(
                "lyrics:" in description and "characters" in description
                for description in descriptions
            )
        )
        self.assertFalse(
            any("private lyric content" in description for description in descriptions)
        )
        self.assertEqual(
            track_processing_module._describe_tag_diffs(
                track_processing_module._compute_tag_diffs(original, original)
            ),
            [],
        )

    def test_album_verified_isrc_updates_missing_identity(self) -> None:
        recording = {
            "title": "Original Song",
            "artist": "Alpha",
            "isrc": "FIRST",
            "isrcs": ["FIRST", "SECOND"],
        }
        for isrc, expected_title in (
            ("SECOND", "Original Song"),
            ("UNKNOWN", "Original Song"),
            (None, "Original Song"),
        ):
            with (
                self.subTest(isrc=isrc),
                patch.object(
                    artist_resolution, "resolve_artist_name", return_value="Alpha"
                ),
            ):
                track = TrackInfo(
                    self.root / "track.flac",
                    artist="Unknown Artist",
                    title="Untitled",
                    isrc=isrc,
                    track_number=98,
                )
                musicbrainz_metadata_module._apply_mb_recording_metadata(
                    track, recording, True, None, False
                )
                self.assertEqual(track.title, expected_title)
                self.assertEqual(track.track_number, 98)
                if isrc == "SECOND":
                    self.assertEqual(track.artist, "Alpha")
                    self.assertEqual(track.isrc, "SECOND")

    def test_dsp_write_preserves_multi_value_text_and_aliases(self) -> None:
        audio_path = self.audio("multi.flac")
        original = {
            "ARTIST": ["Alpha", "Beta"],
            "ALBUMARTIST": ["Alpha", "Beta"],
            "ALBUMARTISTS": ["Alpha", "Beta"],
            "TITLE": ["Song"],
            "ALBUM": ["Album"],
            "GENRE": ["Pop", "Rock"],
            "COMPOSER": ["Composer One", "Composer Two"],
            "PUBLISHER": ["Indie", "Major"],
        }
        with taglib.File(str(audio_path)) as audio_tags:
            audio_tags.tags.update(original)
            self.assertFalse(audio_tags.save())
        track = read_track_metadata(audio_path)
        self.assertEqual(track.featured_artists, "Beta")
        track.bpm = 120.0
        write_track_metadata(track)
        with taglib.File(str(audio_path)) as audio_tags:
            for tag_key in (
                "ARTIST",
                "ALBUMARTIST",
                "ALBUMARTISTS",
                "GENRE",
                "COMPOSER",
            ):
                self.assertEqual(audio_tags.tags[tag_key], original[tag_key])
            self.assertEqual(audio_tags.tags["LABEL"], original["PUBLISHER"])
            self.assertNotIn("PUBLISHER", audio_tags.tags)
            self.assertEqual(audio_tags.tags["BPM"], ["120.0"])

    def test_multi_value_preservation_allows_explicit_identity_change(self) -> None:
        from sonora.audio.metadata import _preserve_unchanged_multivalues

        original = {
            "ARTIST": ["Alpha", "Beta"],
            "GENRE": ["Pop", "Rock"],
            "CUSTOM": ["One", "Two"],
        }
        updated = {"ARTIST": ["Gamma"], "GENRE": ["Metal"], "CUSTOM": ["One", "Two"]}
        _preserve_unchanged_multivalues(original, updated)
        self.assertEqual(updated["ARTIST"], ["Gamma"])
        self.assertEqual(updated["GENRE"], ["Metal"])
        self.assertEqual(updated["CUSTOM"], ["One", "Two"])
        _preserve_unchanged_multivalues({}, {})

    def test_partial_credit_responses_do_not_downgrade_existing_credits(self) -> None:
        track = TrackInfo(
            self.root / "track.flac",
            label="Indie / Major Records",
            producers="Producer One, Producer Two",
            copyright="2024 Major Records",
        )
        metadata_merge.apply_metadata_mapping(
            track,
            {
                "label": "Major Records",
                "producers": "Producer One",
                "copyright": "Major Records",
            },
            {"label": "label", "producers": "producers", "copyright": "copyright"},
            force=True,
            source="MusicBrainz",
        )
        self.assertEqual(track.label, "Indie / Major Records")
        self.assertEqual(track.producers, "Producer One, Producer Two")
        self.assertEqual(track.copyright, "2024 Major Records")
        self.assertNotIn("label", track.metadata_sources)
        blank = TrackInfo(track.file_path)
        metadata_merge.apply_metadata_mapping(
            blank, {"label": "Major Records"}, {"label": "label"}, force=True
        )
        self.assertEqual(blank.label, "Major Records")
        metadata_merge.apply_metadata_mapping(
            blank, {"label": "Different Records"}, {"label": "label"}, force=True
        )
        self.assertEqual(blank.label, "Different Records")

    def test_genre_aliases_and_noise_preserve_secondary_genres_on_dsp_write(
        self,
    ) -> None:
        for genres, expected in (
            (["RNB", "Rock"], ["R&B/Soul", "Rock"]),
            (["Other", "RNB", "Rock"], ["R&B/Soul", "Rock"]),
            (["Other", "Rock"], ["Rock"]),
        ):
            with self.subTest(genres=genres):
                audio_path = self.audio("genre-alias.flac")
                with taglib.File(str(audio_path)) as audio_tags:
                    audio_tags.tags["GENRE"] = genres
                    self.assertFalse(audio_tags.save())
                clear_metadata_cache()
                track = read_track_metadata(audio_path)
                self.assertEqual(track.genre, expected[0])
                track.bpm = 120.0
                write_track_metadata(track)
                with taglib.File(str(audio_path)) as audio_tags:
                    self.assertEqual(audio_tags.tags["GENRE"], expected)


class TestProviderDurationEvidence(IsolatedTestCase):
    def test_duration_conflict_is_checked_without_artist_or_title_changes(self) -> None:
        for duration, accepted in (
            (180.0, True),
            (182.0, True),
            (190.0, False),
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
                metadata_merge.merge_provider_proposals(
                    recording, [build_provider_proposal(recording, proposal)]
                )
                self.assertEqual(recording.genre, "Rock" if accepted else None)
                self.assertEqual(
                    "identity" in recording.metadata_conflicts, not accepted
                )
                self.assertEqual(recording.duration, 180)


class TestMetadataWorkflows(ModuleTestCase):
    @patch("sonora.services.artist_resolution.get_cached_api", return_value=None)
    @patch("sonora.services.musicbrainz.search_musicbrainz_artists")
    def test_resolve_artist_name(
        self, mock_search: MagicMock, mock_get_cache: MagicMock
    ) -> None:
        mock_search.return_value = [
            {"name": "M.G.L.", "alias-list": [{"alias": "mgl"}]},
            {"name": "Killa Fonic", "alias-list": []},
            {"name": "Nane", "alias-list": []},
        ]
        self.assertEqual(resolve_artist_name("mgl"), "M.G.L.")
        self.assertEqual(resolve_artist_name("killa fonic"), "Killa Fonic")
        self.assertEqual(resolve_artist_name("nane"), "Nane")

    def test_strip_corrupt_brackets(self) -> None:
        self.assertEqual(
            strip_corrupt_brackets("Song Title [Official Video]"), "Song Title"
        )
        self.assertEqual(strip_corrupt_brackets("Song Title [FLAC]"), "Song Title")
        self.assertEqual(
            strip_corrupt_brackets("Song Title (Remix)"), "Song Title (Remix)"
        )
        self.assertEqual(
            strip_corrupt_brackets("Candy (feat. Trippie Redd)"),
            "Candy (feat. Trippie Redd)",
        )
        self.assertEqual(strip_corrupt_brackets("Song (2011 Remaster)"), "Song")
        self.assertEqual(strip_corrupt_brackets(""), "")

    def test_is_alien_album_track(self) -> None:
        track_path = Path("/music/21 Savage - Issa Album/03 - Blue Jeans.wav")
        alien_track = TrackInfo(
            file_path=track_path,
            artist="Lana Del Rey",
            title="Blue Jeans",
            album="Born to Die",
        )
        mb_release: dict[str, object] = {
            "title": "Issa Album",
            "album_artist": "21 Savage",
            "tracks_by_disc_and_position": {
                (1, 1): {"title": "Famous", "artist": "21 Savage"},
                (1, 2): {"title": "Bank Account", "artist": "21 Savage"},
                (1, 3): {"title": "Close My Eyes", "artist": "21 Savage"},
            },
        }
        self.assertTrue(
            is_alien_album_track(
                track_info=alien_track,
                file_path=track_path,
                target_album_artist="21 Savage",
                target_album_title="Issa Album",
                album_mb_release_details=mb_release,
            )
        )
        valid_track = TrackInfo(
            file_path=Path("/music/21 Savage - Issa Album/02 - Bank Account.wav"),
            artist="21 Savage",
            title="Bank Account",
            album="Issa Album",
        )
        self.assertFalse(
            is_alien_album_track(
                track_info=valid_track,
                file_path=valid_track.file_path,
                target_album_artist="21 Savage",
                target_album_title="Issa Album",
                album_mb_release_details=mb_release,
            )
        )
        generic_track = TrackInfo(
            file_path=Path("/music/21 Savage - Issa Album/01.wav"),
            artist="Unknown Artist",
            title="Track 01",
            album="Unknown Album",
        )
        self.assertFalse(
            is_alien_album_track(
                track_info=generic_track,
                file_path=generic_track.file_path,
                target_album_artist="21 Savage",
                target_album_title="Issa Album",
                album_mb_release_details=mb_release,
            )
        )
        collab_track = TrackInfo(
            file_path=Path("/music/Drake & 21 Savage - Her Loss/01 - Rich Flex.wav"),
            artist="21 Savage",
            title="Rich Flex",
        )
        self.assertFalse(
            is_alien_album_track(
                track_info=collab_track,
                file_path=collab_track.file_path,
                target_album_artist="Drake & 21 Savage",
                target_album_title="Her Loss",
            )
        )
        updated_track = TrackInfo(
            file_path=Path(
                "/music/James Arthur - Back From the Edge/04 - Lust for Life.wav"
            ),
            artist="James Arthur",
            title="Lust for Life (feat. Lana Del Rey & The Weeknd)",
            album="Back From the Edge",
            isrc="DEE861600588",
            track_number=4,
        )
        mb_release_ja: dict[str, object] = {
            "title": "Back From the Edge",
            "album_artist": "James Arthur",
            "tracks_by_disc_and_position": {
                (1, 4): {
                    "title": "Can I Be Him",
                    "artist": "James Arthur",
                    "isrc": "DEE861600588",
                    "recording_mbid": "c8b03190-306c-4125-9b32-3f9d86d60a12",
                }
            },
        }
        self.assertFalse(
            is_alien_album_track(
                track_info=updated_track,
                file_path=updated_track.file_path,
                target_album_artist="James Arthur",
                target_album_title="Back From the Edge",
                album_mb_release_details=mb_release_ja,
            )
        )
        bonus_track = TrackInfo(
            file_path=Path("/music/21 Savage - Issa Album/15 - Bonus Track.wav"),
            artist="21 Savage",
            title="7 Min Freestyle (Bonus)",
            album="Issa Album",
        )
        self.assertFalse(
            is_alien_album_track(
                track_info=bonus_track,
                file_path=bonus_track.file_path,
                target_album_artist="21 Savage",
                target_album_title="Issa Album",
                album_mb_release_details=mb_release,
            )
        )
        conflicting_track = TrackInfo(
            file_path=Path("/music/XXXTENTACION - Revenge/04 - Revenge.flac"),
            artist="XXXTENTACION",
            title="Revenge",
            album="Revenge",
            track_number=4,
            isrc="USUYG1156895",
        )
        mb_release_rev: dict[str, object] = {
            "title": "Revenge",
            "album_artist": "XXXTENTACION",
            "tracks_by_disc_and_position": {
                (1, 1): {
                    "title": "Look at Me!",
                    "artist": "XXXTENTACION",
                    "isrc": "USUYG1131080",
                },
                (1, 2): {
                    "title": "I don't wanna do this anymore",
                    "artist": "XXXTENTACION",
                    "isrc": "USUYG1142046",
                },
                (1, 3): {
                    "title": "Looking for a Star",
                    "artist": "XXXTENTACION",
                    "isrc": "USUYG1142047",
                },
                (1, 4): {
                    "title": "KING",
                    "artist": "XXXTENTACION",
                    "isrc": "USUYG1142049",
                },
                (1, 5): {
                    "title": "Slipknot",
                    "artist": "XXXTENTACION feat. Kin$oul & Killstation",
                    "isrc": "USUYG1142050",
                },
                (1, 6): {
                    "title": "YuNg BrAtZ",
                    "artist": "XXXTENTACION",
                    "isrc": "USUYG1142051",
                },
                (1, 7): {
                    "title": 'R.I.P ROACH "EAST SIDE SOULJA"',
                    "artist": "XXXTENTACION",
                    "isrc": "USUYG1142052",
                },
            },
        }
        self.assertTrue(
            is_alien_album_track(
                track_info=conflicting_track,
                file_path=conflicting_track.file_path,
                target_album_artist="XXXTENTACION",
                target_album_title="Revenge",
                album_mb_release_details=mb_release_rev,
            )
        )

    def test_symfonium_extended_tags(self) -> None:
        wav_path = self.tmp_path / "extended.wav"
        write_silent_wav(wav_path)
        track_metadata = read_track_metadata(wav_path)
        track_metadata.artist_sort = "Artist, Test"
        track_metadata.album_artist_sort = "Artist, Test Album"
        track_metadata.total_tracks = 12
        track_metadata.total_discs = 2
        track_metadata.release_type = "Album"
        track_metadata.release_status = "Official"
        track_metadata.release_country = "US"
        track_metadata.musicbrainz_trackid = "c8b03190-306c-4125-9b32-3f9d86d60a12"
        track_metadata.musicbrainz_albumid = "a1b2c3d4-e5f6-4a5b-8c9d-0e1f2a3b4c5d"
        track_metadata.label = "Test Label"
        track_metadata.barcode = "123456789012"
        expected = track_metadata.to_dict()
        write_track_metadata(track_metadata)
        reloaded = read_track_metadata(wav_path)
        self.assertEqual(
            reloaded.musicbrainz_trackid, "c8b03190-306c-4125-9b32-3f9d86d60a12"
        )
        self.assertEqual(
            reloaded.musicbrainz_albumid, "a1b2c3d4-e5f6-4a5b-8c9d-0e1f2a3b4c5d"
        )
        for field in (
            "artist_sort",
            "album_artist_sort",
            "total_tracks",
            "total_discs",
            "release_type",
            "release_status",
            "release_country",
            "label",
            "barcode",
        ):
            with self.subTest(field=field):
                self.assertEqual(getattr(reloaded, field), expected[field])

    def test_find_audio_files_ignores_hidden_directories(self) -> None:
        normal_wav = self.tmp_path / "normal.wav"
        write_silent_wav(normal_wav)
        hidden_dir = self.tmp_path / ".venv" / "lib"
        hidden_dir.mkdir(parents=True, exist_ok=True)
        hidden_wav = hidden_dir / "test.wav"
        write_silent_wav(hidden_wav)
        dot_wav = self.tmp_path / "._hidden.flac"
        dot_wav.write_bytes(b"dummy")
        files = find_audio_files(self.tmp_path)
        self.assertEqual(files, [normal_wav])
        all_files = find_audio_files(self.tmp_path, include_hidden=True)
        self.assertIn(hidden_wav, all_files)
        self.assertIn(dot_wav, all_files)

    def test_match_score_short_artist_name_isolation(self) -> None:
        self.assertEqual(score_artist_title_match("Ian", "Song", "Brian", "Song"), 0.0)
        self.assertEqual(
            score_artist_title_match("Ian", "Song", "Arkanian", "Song"), 0.0
        )
        self.assertEqual(
            score_artist_title_match("Ian", "Song", "Christian", "Song"), 0.0
        )
        self.assertEqual(
            score_artist_title_match("Tyga", "Song", "Tyga Ichinose", "Song"), 0.0
        )
        self.assertEqual(
            score_artist_title_match("Aaryan Shah", "Song", "Kavinsky", "Song"), 0.0
        )
        self.assertGreaterEqual(
            score_artist_title_match(
                "21 Savage & Metro Boomin", "Song", "21 Savage", "Song"
            ),
            80.0,
        )
        self.assertGreaterEqual(
            score_artist_title_match(
                "Kendrick Lamar feat. Travis Scott", "Song", "Kendrick Lamar", "Song"
            ),
            80.0,
        )


class TestNumberingRepairSafety(AudioRegressionTestCase):
    def test_verified_same_total_reconciles_conflicting_inline_encoding(self) -> None:
        path = self.recording(
            "01 - Song.flac",
            {"TRACKNUMBER": ["1/14"], "TRACKTOTAL": ["17"], "TOTALTRACKS": ["17"]},
        )
        original = read_track_metadata(path)
        verified = original.copy()
        self.assertEqual(verified.total_tracks, 17)
        self.assertFalse(_compute_tag_diffs(original, verified))
        verified.metadata_sources["total_tracks"] = "iTunes"
        self.assertTrue(
            any(
                (
                    field == "numbering"
                    for field, _, _ in _compute_tag_diffs(original, verified)
                )
            )
        )
        write_track_metadata(verified)
        self.assertEqual(FLAC(path)["TRACKNUMBER"], ["1/17"])

    def recording(self, name: str, tags: dict[str, list[str]]) -> Path:
        path = self.audio(name)
        audio = FLAC(path)
        audio.update({"ARTIST": ["Performer"], "ALBUM": ["Album"], "TITLE": ["Song"]})
        audio.update(tags)
        audio.save()
        return path

    def test_corroborating_totals_repair_stale_alias_without_force(self) -> None:
        path = self.recording(
            "11 - Song.flac",
            {
                "TRACKNUMBER": ["11/17"],
                "TRACKTOTAL": ["9"],
                "TOTALTRACKS": ["17"],
                "USER_DATA": ["one", "two"],
            },
        )
        original_digest = FLAC(path).info.md5_signature
        recording = read_track_metadata(path)
        self.assertEqual(recording.total_tracks, 17)
        self.assertTrue(recording._numbering_needs_normalization)
        self.assertNotIn("_numbering_needs_normalization", recording.to_dict())
        result = normalize_single_track(path)
        assert result is not None
        self.assertTrue(any("numbering" in diff for diff in result._diff_descriptions))
        audio = FLAC(path)
        self.assertEqual(audio["TRACKNUMBER"], ["11/17"])
        self.assertEqual(audio["TRACKTOTAL"], ["17"])
        self.assertEqual(audio["TOTALTRACKS"], ["17"])
        self.assertEqual(audio["USER_DATA"], ["one", "two"])
        self.assertEqual(audio.info.md5_signature, original_digest)
        repeated = normalize_single_track(path)
        assert repeated is not None
        self.assertEqual(repeated._diff_descriptions, [])

    def test_impossible_majority_does_not_override_only_valid_stored_total(
        self,
    ) -> None:
        path = self.recording(
            "11 - Song.flac",
            {"TRACKNUMBER": ["11/9"], "TRACKTOTAL": ["9"], "TOTALTRACKS": ["17"]},
        )
        recording = read_track_metadata(path)
        self.assertEqual(recording.total_tracks, 17)
        self.assertTrue(recording._numbering_needs_normalization)

    def test_ambiguous_totals_survive_unrelated_forced_normalization(self) -> None:
        tags = {"TRACKNUMBER": ["1/8"], "TRACKTOTAL": ["10"], "TOTALTRACKS": ["12"]}
        path = self.recording("01 - Song.flac", tags)
        recording = read_track_metadata(path)
        self.assertIn("album_sequence", recording.metadata_conflicts)
        self.assertFalse(recording._numbering_needs_normalization)
        normalize_single_track(path, force=True)
        audio = FLAC(path)
        for tag_name, value in tags.items():
            self.assertEqual(audio[tag_name], value)
        repeated = normalize_single_track(path)
        assert repeated is not None
        self.assertEqual(repeated._diff_descriptions, [])
        self.assertIn("album_sequence", repeated.metadata_conflicts)

    def test_redundant_aliases_cannot_outvote_another_plausible_edition(self) -> None:
        path = self.recording(
            "01 - Song.flac",
            {"TRACKNUMBER": ["1/17"], "TRACKTOTAL": ["9"], "TOTALTRACKS": ["17"]},
        )
        recording = read_track_metadata(path)
        self.assertEqual(recording.total_tracks, 9)
        self.assertFalse(recording._numbering_needs_normalization)
        self.assertIn("album_sequence", recording.metadata_conflicts)

    def test_consistently_impossible_count_requires_release_evidence(self) -> None:
        tags = {"TRACKNUMBER": ["13/12"], "TRACKTOTAL": ["12"], "TOTALTRACKS": ["12"]}
        path = self.recording("13 - Song.flac", tags)
        normalize_single_track(path, force=True)
        recording = read_track_metadata(path)
        self.assertEqual((recording.track_number, recording.total_tracks), (13, 12))
        self.assertIn("album_sequence", recording.metadata_conflicts)
        repeated = normalize_single_track(path)
        assert repeated is not None
        self.assertEqual(repeated._diff_descriptions, [])

    def test_explicit_disc_one_is_not_replaced_by_filename_disc_two(self) -> None:
        path = self.recording(
            "2-01 - Song.flac", {"TRACKNUMBER": ["1/10"], "DISCNUMBER": ["1/1"]}
        )
        recording = read_track_metadata(path)
        self.assertEqual((recording.disc_number, recording.total_discs), (1, 1))
        normalize_single_track(path, force=True)
        self.assertEqual(FLAC(path)["DISCNUMBER"], ["1/1"])

    def test_nonpositive_and_nonfinite_bpm_are_unknown_and_not_written(self) -> None:
        for value in ("0.0", "-1", "nan", "inf"):
            with self.subTest(value=value):
                path = self.recording(f"{value}.flac", {"BPM": [value]})
                recording = read_track_metadata(path)
                self.assertIsNone(recording.bpm)
                recording.bpm = float(value)
                write_track_metadata(recording)
                self.assertNotIn("BPM", FLAC(path))


class TestTaggingWorkflows(ModuleTestCase):
    def test_tag_empty_directory_warns(self) -> None:
        album_dir = self.tmp_path / "empty"
        album_dir.mkdir()
        with patch("sonora.modules.tagger.LOG.warning") as warning:
            self.assertEqual(tag_album_folder(album_dir), [])
        warning.assert_called_once_with("No audio files found to tag.")

    @patch("sonora.audio.metadata.write_track_metadata")
    @patch("sonora.services.lyrics.fetch_synced_lyrics")
    @patch("sonora.services.musicbrainz.fetch_track_mbid")
    @patch("sonora.audio.metadata.read_track_metadata")
    def test_process_single_track(
        self,
        mock_read: MagicMock,
        mock_mbid: MagicMock,
        mock_lyrics: MagicMock,
        mock_write: MagicMock,
    ) -> None:
        wav_file = self.tmp_path / "song.wav"
        write_silent_wav(wav_file)
        mock_read.return_value = TrackInfo(
            file_path=wav_file, artist="nane", title="Piesa"
        )
        mock_mbid.return_value = "c8b03190-306c-4125-9b32-3f9d86d60a12"
        mock_lyrics.return_value = "[00:01.00] Vers"
        with (
            patch("sonora.audio.art.process_album_cover_art", return_value=None),
            patch(
                "sonora.services.theaudiodb.fetch_theaudiodb_track_details",
                return_value=None,
            ),
            patch(
                "sonora.services.genius.fetch_genius_song_details", return_value=None
            ),
            patch(
                "sonora.services.deezer.fetch_deezer_track_details", return_value=None
            ),
            patch(
                "sonora.services.deezer.fetch_deezer_album_details", return_value=None
            ),
            patch(
                "sonora.services.itunes.fetch_itunes_album_details", return_value=None
            ),
            patch(
                "sonora.services.musicbrainz.fetch_musicbrainz_recording_details",
                return_value={
                    "artist": "Nane",
                    "title": "Piesa",
                    "track_number": 3,
                    "date": "2021-04-30",
                    "isrc": "ROGRA2101575",
                    "advisory": "Explicit",
                },
            ),
            patch(
                "sonora.services.musicbrainz.search_musicbrainz_release",
                return_value=None,
            ),
            patch(
                "sonora.services.artist_resolution.resolve_artist_name",
                return_value="Nane",
            ),
            patch(
                "sonora.services.itunes.fetch_itunes_track_metadata", return_value=None
            ),
        ):
            track_metadata = process_single_track(wav_file, fetch_bpm=False)
        self.assertEqual(track_metadata.artist, "Nane")
        self.assertEqual(
            track_metadata.musicbrainz_trackid, "c8b03190-306c-4125-9b32-3f9d86d60a12"
        )
        mock_write.assert_called_once()

    @patch("sonora.services.itunes.fetch_itunes_album_details", return_value=None)
    @patch("sonora.modules.tagger.process_single_track")
    def test_tag_album_folder(
        self, mock_process: MagicMock, mock_itunes: MagicMock
    ) -> None:
        album_dir = self.tmp_path / "album"
        audio_file_1 = album_dir / "01.wav"
        audio_file_2 = album_dir / "02.wav"
        write_silent_wav(audio_file_1)
        write_silent_wav(audio_file_2)
        mock_process.side_effect = [
            TrackInfo(file_path=audio_file_1, artist="Artist", title="T1"),
            TrackInfo(file_path=audio_file_2, artist="Artist", title="T2"),
        ]
        with (
            patch(
                "sonora.services.musicbrainz.search_musicbrainz_release",
                return_value=None,
            ),
            patch(
                "sonora.services.deezer.fetch_deezer_album_details", return_value=None
            ),
        ):
            results = tag_album_folder(album_dir, max_threads=2, fetch_itunes_art=False)
        self.assertEqual(len(results), 2)
        states = [
            call.kwargs["lyrics_lookup_state"] for call in mock_process.call_args_list
        ]
        self.assertIsNotNone(states[0])
        self.assertIs(states[0], states[1])

    def test_cached_tracks_are_read_in_parallel(self) -> None:
        album_dir = self.tmp_path / "cached album"
        files = [album_dir / f"{index}.wav" for index in range(4)]
        for path in files:
            write_silent_wav(path)
        state = MagicMock()
        state.filter_outdated_tracks.return_value = []
        active = 0
        maximum_active = 0
        lock = threading.Lock()

        def read_track(path: Path) -> TrackInfo:
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return TrackInfo(file_path=path, artist="Artist", title=path.stem)

        with (
            patch("sonora.modules.tagger.get_library_state", return_value=state),
            patch(
                "sonora.modules.tagger.metadata_module.read_track_metadata",
                side_effect=read_track,
            ),
            patch("sonora.modules.tagger._resolve_album_metadata_context") as resolve,
        ):
            results = tag_album_folder(album_dir, max_threads=4)
        self.assertEqual(len(results), len(files))
        self.assertGreater(maximum_active, 1)
        resolve.assert_not_called()

    def test_resolve_album_folder_identity(self) -> None:
        album_dir = self.tmp_path / "Artist - Great Album"
        f1 = album_dir / "01 - Song.wav"
        f2 = album_dir / "02 - Other.wav"
        write_silent_wav(f1)
        write_silent_wav(f2)
        with patch("sonora.audio.metadata.read_track_metadata") as mock_read:
            mock_read.side_effect = [
                TrackInfo(file_path=f1, artist="Corrupt Artist", album="Corrupt Album"),
                TrackInfo(file_path=f2, artist="Artist", album="Great Album"),
            ]
            artist, album = _resolve_album_folder_identity(album_dir, [f1, f2])
            self.assertEqual(artist, "Artist")
            self.assertEqual(album, "Great Album")
        disc_dir = self.tmp_path / "Pink Floyd" / "The Wall" / "CD1"
        f_disc = disc_dir / "01.wav"
        write_silent_wav(f_disc)
        with patch("sonora.audio.metadata.read_track_metadata") as mock_read:
            mock_read.return_value = TrackInfo(
                file_path=f_disc, artist="Pink Floyd", album="The Wall"
            )
            artist, album = _resolve_album_folder_identity(disc_dir, [f_disc])
            self.assertEqual(artist, "Pink Floyd")
            self.assertEqual(album, "The Wall")
        singles_dir = self.tmp_path / "3 Doors Down" / "Singles"
        f_single = singles_dir / "01.wav"
        write_silent_wav(f_single)
        with patch("sonora.audio.metadata.read_track_metadata") as mock_read:
            mock_read.return_value = TrackInfo(
                file_path=f_single, artist="3 Doors Down", album="Unknown Album"
            )
            artist, album = _resolve_album_folder_identity(singles_dir, [f_single])
            self.assertEqual(artist, "3 Doors Down")
            self.assertIsNone(album)

    def test_progress_retains_estimate_for_tracks_longer_than_thirty_seconds(
        self,
    ) -> None:
        from sonora.core.logger import create_progress

        progress = create_progress()
        self.assertEqual(progress.speed_estimate_period, 300.0)
        clock = [0.0]
        with patch.object(progress, "get_time", side_effect=lambda: clock[0]):
            task_id = progress.add_task("Tagging", total=10)
            for moment in (70.0, 140.0, 210.0):
                clock[0] = moment
                progress.advance(task_id)
            self.assertIsNotNone(progress.tasks[0].speed)
            self.assertIsNotNone(progress.tasks[0].time_remaining)

    def test_artist_download_folder_has_no_shared_release_context(self) -> None:
        artist_dir = self.tmp_path / "Mornye" / "Daniel Di Angelo"
        paths = [artist_dir / f"{index}.flac" for index in range(3)]
        with patch("sonora.audio.metadata.read_track_metadata") as read:
            read.side_effect = [
                TrackInfo(paths[0], artist="Daniel Di Angelo", album="Addicted"),
                TrackInfo(paths[1], artist="Daniel Di Angelo", album="Addicted"),
                TrackInfo(paths[2], artist="Daniel Di Angelo", album="Comfort Mode"),
            ]
            artist, album = _resolve_album_folder_identity(artist_dir, paths)
        self.assertEqual(artist, "Daniel Di Angelo")
        self.assertIsNone(album)

    def test_resolve_album_track_position(self) -> None:
        mb_release: dict[str, object] = {
            "title": "That's My Name",
            "album_artist": "Akcent",
            "tracks_by_disc_and_position": {
                (1, 1): {"title": "Kylie", "artist": "Akcent", "isrc": "RO001"},
                (1, 2): {
                    "title": "That's My Name",
                    "artist": "Akcent",
                    "isrc": "RO002",
                },
                (1, 3): {"title": "My Passion", "artist": "Akcent", "isrc": "RO003"},
                (1, 4): {"title": "Kamelia", "artist": "Akcent", "isrc": "RO004"},
                (1, 5): {"title": "Te Quiero", "artist": "Akcent", "isrc": "RO005"},
                (1, 6): {"title": "Amor Gitana", "artist": "Akcent", "isrc": "RO006"},
            },
        }
        track_misnumbered = TrackInfo(
            file_path=Path("/music/Akcent - That's My Name/06 - That's My Name.flac"),
            artist="Akcent",
            title="That's My Name",
            track_number=6,
            album="That's My Name",
        )
        pos = _resolve_album_track_position(
            track_info=track_misnumbered,
            file_path=track_misnumbered.file_path,
            album_mb_release_details=mb_release,
        )
        self.assertEqual(pos, 2)
        track_corrupted = TrackInfo(
            file_path=Path("/music/Akcent - That's My Name/02 - Leaked Audio.flac"),
            artist="Akcent",
            title="Untitled",
            isrc="RO002",
            track_number=2,
            album="That's My Name",
        )
        pos = _resolve_album_track_position(
            track_info=track_corrupted,
            file_path=track_corrupted.file_path,
            album_mb_release_details=mb_release,
        )
        self.assertEqual(pos, 2)
        track_alien = TrackInfo(
            file_path=Path(
                "/music/Akcent - That's My Name/07 - Drake - Hotline Bling.flac"
            ),
            artist="Drake",
            title="Hotline Bling",
            track_number=7,
            album="Views",
        )
        pos = _resolve_album_track_position(
            track_info=track_alien,
            file_path=track_alien.file_path,
            album_mb_release_details=mb_release,
        )
        self.assertIsNone(pos)

    def test_update_musicbrainz_candidate_rec_mismatch_prevents_incorrect_metadata(
        self,
    ) -> None:
        track = TrackInfo(
            file_path=Path("/music/XXXTENTACION - Revenge/04 - Revenge.flac"),
            artist="XXXTENTACION",
            title="Revenge",
            album="Revenge",
            isrc="USUYG1156895",
            track_number=4,
        )
        mb_release: dict[str, object] = {
            "title": "Revenge",
            "album_artist": "XXXTENTACION",
            "tracks_by_disc_and_position": {
                (1, 4): {
                    "title": "KING",
                    "artist": "XXXTENTACION",
                    "isrc": "USUYG1142049",
                    "recording_mbid": "548be1ad-0000-0000-0000-000000000000",
                    "producers": "King Yosef",
                }
            },
        }
        _update_musicbrainz(
            track_info=track,
            album_mbid="9d90908d-0a74-4df0-8e76-13c6473b33e5",
            album_track_mbids={(1, 4): "548be1ad-0000-0000-0000-000000000000"},
            album_mb_release_details=mb_release,
            force=True,
        )
        self.assertEqual(track.title, "Revenge")
        self.assertEqual(track.isrc, "USUYG1156895")
        self.assertIsNone(track.musicbrainz_trackid)
        self.assertIsNone(track.producers)

    def test_standalone_musicbrainz_lookup_uses_track_isrc(self) -> None:
        track = TrackInfo(
            Path("song.flac"), artist="Artist", title="Song", isrc="USABC2500001"
        )
        recording_id = "12345678-1234-1234-1234-123456789abc"
        with patch(
            "sonora.modules.musicbrainz_metadata.musicbrainz_service.fetch_track_mbid",
            return_value=recording_id,
        ) as search:
            _resolve_musicbrainz_track_id(track, None, None)
        search.assert_called_once_with("Artist", "Song", "USABC2500001")
        self.assertEqual(track.musicbrainz_trackid, recording_id)

    def test_update_musicbrainz_updates_corrupted_title_via_isrc(self) -> None:
        track = TrackInfo(
            file_path=Path("dummy.wav"),
            artist="James Arthur",
            title="Untitled",
            album="Back From the Edge",
            isrc="DEE861600588",
            track_number=4,
            musicbrainz_trackid="56a71671-a4cd-4fe7-96aa-1aadab6cd94e",
        )
        mb_release: dict[str, object] = {
            "title": "Back From the Edge",
            "album_artist": "James Arthur",
            "tracks_by_disc_and_position": {
                (1, 4): {
                    "title": "Can I Be Him",
                    "artist": "James Arthur",
                    "isrc": "DEE861600588",
                    "recording_mbid": "c8b03190-306c-4125-9b32-3f9d86d60a12",
                }
            },
        }
        with patch(
            "sonora.services.musicbrainz.fetch_musicbrainz_recording_details",
            return_value={
                "artist": "James Arthur",
                "title": "A Different Recording",
                "isrc": "USAAA2400001",
            },
        ) as stored_lookup:
            _update_musicbrainz(
                track_info=track,
                album_mbid="c6b7d374-1425-4414-ba65-2ddcb4d98196",
                album_track_mbids={(1, 4): "c8b03190-306c-4125-9b32-3f9d86d60a12"},
                album_mb_release_details=mb_release,
                force=False,
            )
        stored_lookup.assert_called_once_with("56a71671-a4cd-4fe7-96aa-1aadab6cd94e")
        self.assertEqual(track.title, "Can I Be Him")
        self.assertEqual(
            track.musicbrainz_trackid, "c8b03190-306c-4125-9b32-3f9d86d60a12"
        )

    @patch("sonora.audio.metadata.write_track_metadata")
    @patch("sonora.audio.metadata.read_track_metadata")
    def test_process_single_track_preserves_alien_track(
        self, mock_read: MagicMock, mock_write: MagicMock
    ) -> None:
        track_file = self.tmp_path / "03 - Blue Jeans.wav"
        write_silent_wav(track_file)
        mock_read.return_value = TrackInfo(
            file_path=track_file,
            artist="Lana Del Rey",
            title="Blue Jeans",
            album="Born to Die",
            track_number=3,
        )
        mbids = {(1, 1): "uuid-1", (1, 2): "uuid-2", (1, 3): "uuid-3"}
        mb_release: dict[str, object] = {
            "title": "Issa Album",
            "album_artist": "21 Savage",
            "tracks_by_disc_and_position": {
                (1, 1): {"title": "Famous", "artist": "21 Savage"},
                (1, 2): {"title": "Bank Account", "artist": "21 Savage"},
                (1, 3): {"title": "Close My Eyes", "artist": "21 Savage"},
            },
        }
        with (
            patch("sonora.modules.tagger._update_musicbrainz"),
            patch("sonora.modules.tagger._update_acoustid"),
            patch("sonora.modules.tagger._update_deezer"),
            patch("sonora.modules.tagger._update_itunes"),
            patch("sonora.modules.tagger._update_theaudiodb"),
            patch("sonora.modules.tagger._update_genius"),
        ):
            track_metadata = process_single_track(
                track_file,
                album_context=_AlbumMetadataContext(
                    track_mbids=mbids,
                    mb_release_details=mb_release,
                    sample_artist="21 Savage",
                    sample_album="Issa Album",
                ),
                force=True,
                fetch_bpm=False,
                fetch_lyrics=False,
                fetch_itunes_art=False,
            )
            self.assertTrue(track_metadata.is_alien)
            self.assertEqual(track_metadata.artist, "Lana Del Rey")
            self.assertEqual(track_metadata.title, "Blue Jeans")

    def test_unverified_itunes_tracklist_cannot_mark_an_alternate_track_alien(
        self,
    ) -> None:
        track_file = self.tmp_path / "02 - Alternate Version.wav"
        write_silent_wav(track_file)
        track = TrackInfo(
            file_path=track_file,
            artist="Artist",
            title="Alternate Version",
            album="Album",
            track_number=2,
        )
        itunes_details = {
            "title": "Album",
            "album_artist": "Artist",
            "tracks_by_disc_and_position": {
                (1, 1): {"artist": "Artist", "title": "Track One"},
                (1, 2): {"artist": "Artist", "title": "Track Two"},
                (1, 3): {"artist": "Artist", "title": "Track Three"},
            },
        }
        with patch("sonora.modules.tagger._update_metadata"):
            result = process_single_track(
                track_file,
                album_context=_AlbumMetadataContext(
                    track_metadata={track_file: track},
                    itunes_details=itunes_details,
                    sample_artist="Artist",
                    sample_album="Album",
                    itunes_layout_verified=False,
                ),
                fetch_bpm=False,
                fetch_key=False,
                fetch_lyrics=False,
                fetch_itunes_art=False,
                dry_run=True,
            )
        self.assertFalse(result.is_alien)

    def test_tag_album_folder_nonexistent_directory(self) -> None:
        with self.assertRaises(FileNotFoundError):
            tag_album_folder(self.tmp_path / "nonexistent_dir_999")

    @patch("sonora.audio.metadata.write_track_metadata")
    @patch("sonora.services.discogs.search_discogs_release")
    @patch("sonora.services.acoustid.lookup_acoustid")
    @patch("sonora.services.musicbrainz.fetch_track_mbid")
    @patch("sonora.audio.metadata.read_track_metadata")
    def test_process_single_track_acoustid_discogs_fallback(
        self,
        mock_read: MagicMock,
        mock_mbid: MagicMock,
        mock_acoustid: MagicMock,
        mock_discogs: MagicMock,
        mock_write: MagicMock,
    ) -> None:
        wav_file = self.tmp_path / "song.wav"
        write_silent_wav(wav_file)
        mock_read.return_value = TrackInfo(
            file_path=wav_file, artist="Artist", title="Title", genre=None
        )
        mock_mbid.return_value = None
        mock_acoustid.return_value = "c8b03190-306c-4125-9b32-3f9d86d60a12"
        mock_discogs.return_value = {"id": 123, "year": 2024}
        with (
            patch("sonora.audio.art.process_album_cover_art", return_value=None),
            patch(
                "sonora.services.theaudiodb.fetch_theaudiodb_track_details",
                return_value=None,
            ),
            patch(
                "sonora.services.genius.fetch_genius_song_details", return_value=None
            ),
            patch(
                "sonora.services.deezer.fetch_deezer_track_details", return_value=None
            ),
            patch(
                "sonora.services.deezer.fetch_deezer_album_details", return_value=None
            ),
            patch(
                "sonora.services.musicbrainz.fetch_musicbrainz_recording_details",
                return_value=None,
            ),
            patch(
                "sonora.services.musicbrainz.search_musicbrainz_release",
                return_value=None,
            ),
            patch(
                "sonora.services.artist_resolution.resolve_artist_name",
                side_effect=lambda x: str(x).strip(),
            ),
            patch(
                "sonora.services.itunes.fetch_itunes_track_metadata", return_value=None
            ),
        ):
            track_metadata = process_single_track(
                wav_file,
                fetch_bpm=False,
                fetch_lyrics=False,
                fetch_itunes_art=False,
                acoustid_api_key="acoustid_key",
                discogs_user_token="discogs_token",
            )
        self.assertEqual(
            track_metadata.musicbrainz_trackid, "c8b03190-306c-4125-9b32-3f9d86d60a12"
        )
        self.assertIsNone(track_metadata.date)
        self.assertNotIn("discogs_release_id", track_metadata.metadata_conflicts)
        mock_acoustid.assert_called_once()
        mock_discogs.assert_called_once()
        mock_write.assert_called_once()

    @patch.dict(os.environ, {"ENABLE_SHAZAM": "true"})
    @patch("sonora.services.shazam.recognize_audio_track")
    def test_update_shazam_generic_and_force(self, mock_recognize: MagicMock) -> None:
        audio_file = self.tmp_path / "01.flac"
        audio_file.write_bytes(b"dummy")
        mock_recognize.return_value = ShazamTrackInfo(
            title="Recognized Title",
            artist="Recognized Artist",
            album="Recognized Album",
            genre="Electronic",
            apple_music_id="12345",
            isrc="USABC1234567",
            release_date="2024-01-01",
            label="Recognized Label",
            lyrics="Lyrics Line 1\nLyrics Line 2",
        )
        trk = TrackInfo(file_path=audio_file, title="Track 01", artist="Unknown Artist")
        _update_shazam(trk, audio_file, force=False)
        self.assertEqual(trk.title, "Recognized Title")
        self.assertEqual(trk.artist, "Recognized Artist")
        self.assertEqual(trk.album, "Recognized Album")
        self.assertEqual(trk.genre, "Electronic")
        self.assertEqual(trk.itunes_trackid, "12345")
        self.assertEqual(trk.isrc, "USABC1234567")
        self.assertEqual(trk.date, "2024-01-01")
        self.assertEqual(trk.label, "Recognized Label")
        self.assertEqual(trk.lyrics, "Lyrics Line 1\nLyrics Line 2")
        trk2 = TrackInfo(
            file_path=audio_file, title="Original Title", artist="Original Artist"
        )
        mock_recognize.reset_mock()
        _update_shazam(trk2, audio_file, force=False)
        self.assertEqual(trk2.title, "Original Title")
        self.assertEqual(trk2.artist, "Original Artist")
        mock_recognize.assert_not_called()
        _update_shazam(trk2, audio_file, force=True)
        self.assertEqual(trk2.title, "Original Title")
        self.assertEqual(trk2.artist, "Original Artist")

    def test_apply_mapping_genre_noise_rejection(self) -> None:
        trk = TrackInfo(
            file_path=Path("/dummy/song.flac"), artist="Artist", title="Title"
        )
        apply_metadata_mapping(trk, {"genre": "Karaoke"}, {"genre": "genre"})
        self.assertEqual(trk.genre, "Karaoke")
        apply_metadata_mapping(trk, {"genre": "Fitness & Workout"}, {"genre": "genre"})
        self.assertEqual(trk.genre, "Karaoke")
        apply_metadata_mapping(trk, {"genre": "Children's Music"}, {"genre": "genre"})
        self.assertEqual(trk.genre, "Karaoke")
        apply_metadata_mapping(
            trk, {"genre": "Hip Hop"}, {"genre": "genre"}, force=True
        )
        self.assertEqual(trk.genre, "Hip-Hop/Rap")

    @patch("sonora.modules.tagger._update_musicbrainz")
    @patch("sonora.modules.tagger._update_acoustid")
    @patch("sonora.modules.tagger._update_itunes")
    @patch("sonora.modules.tagger._update_deezer")
    @patch("sonora.modules.tagger._update_lastfm")
    @patch("sonora.modules.tagger._update_discogs")
    @patch("sonora.modules.tagger._update_genius")
    @patch("sonora.modules.tagger._update_theaudiodb")
    def test_album_context_identity_updates(
        self,
        mock_tadb: MagicMock,
        mock_genius: MagicMock,
        mock_discogs: MagicMock,
        mock_lastfm: MagicMock,
        mock_deezer: MagicMock,
        mock_itunes: MagicMock,
        mock_acoustid: MagicMock,
        mock_mb: MagicMock,
    ) -> None:
        album_dir = self.tmp_path / "Aaryan Shah - The Arrival"
        album_dir.mkdir()
        audio_file = album_dir / "05 - Renegade.wav"
        write_silent_wav(audio_file)
        trk = TrackInfo(
            file_path=audio_file,
            artist="Unknown Artist",
            title="Renegade",
            track_number=5,
            isrc="TCADM1875890",
        )
        write_track_metadata(trk)
        mock_deezer_details = {
            "title": "The Arrival",
            "tracks_by_disc_and_position": {
                (1, 5): {
                    "title": "Renegade",
                    "artist": "Aaryan Shah",
                    "isrc": "TCADM1875890",
                }
            },
        }
        result = process_single_track(
            file_path=audio_file,
            fetch_bpm=False,
            fetch_key=False,
            fetch_lyrics=False,
            fetch_itunes_art=False,
            album_context=_AlbumMetadataContext(
                deezer_details=mock_deezer_details,
                sample_artist="Aaryan Shah",
                sample_album="The Arrival",
            ),
        )
        self.assertEqual(result.artist, "Aaryan Shah")
        self.assertEqual(result.album, "The Arrival")

    def test_update_deezer_fallback_preserves_album_track_number(self) -> None:
        """Fallback track search from Deezer (e.g. single track_position=1) must not overwrite album track number."""
        trk = TrackInfo(
            file_path=Path("/music/album/17 - Dip.flac"),
            artist="Tyga",
            title="Dip",
            album="Legendary",
            track_number=17,
            disc_number=1,
        )
        incomplete_album = {
            "title": "Legendary",
            "tracks_by_disc_and_position": {
                (1, 1): {"title": "Too Many", "track_position": 1}
            },
            "tracks_by_title": {"too many": {"title": "Too Many", "track_position": 1}},
        }
        with patch("sonora.services.deezer.fetch_deezer_track_details") as mock_fetch:
            mock_fetch.return_value = {
                "title": "Dip",
                "artist": "Tyga",
                "track_position": 1,
                "disk_number": 1,
                "producers": "D.A. Got That Dope",
            }
            _update_deezer(
                trk,
                album_deezer_details=incomplete_album,
                has_album_context=True,
                force=True,
            )
        self.assertEqual(trk.track_number, 17)
        self.assertEqual(trk.disc_number, 1)
        self.assertEqual(trk.producers, "D.A. Got That Dope")

    def test_updater_preserves_credited_collaborative_artist(self) -> None:
        """Legitimate track artist on collaborative/multi-artist album must not be overwritten by album artist."""
        audio_file = self.tmp_path / "2-02 - What's in Your Heart.wav"
        write_silent_wav(audio_file)
        trk = TrackInfo(
            file_path=audio_file,
            artist="Swae Lee",
            title="What's in Your Heart?",
            album="SR3MM",
            track_number=2,
            disc_number=2,
        )
        write_track_metadata(trk)
        mock_mb_release: dict[str, object] = {
            "title": "SR3MM",
            "album_artist": "Rae Sremmurd",
            "tracks_by_disc_and_position": {
                (1, 2): {"title": "What's in Your Heart?", "artist": "Swae Lee"}
            },
        }
        with (
            patch("sonora.modules.tagger._update_musicbrainz"),
            patch("sonora.modules.tagger._update_acoustid"),
            patch("sonora.modules.tagger._update_deezer"),
            patch("sonora.modules.tagger._update_itunes"),
            patch("sonora.modules.tagger._update_theaudiodb"),
            patch("sonora.modules.tagger._update_genius"),
        ):
            track_metadata = process_single_track(
                file_path=audio_file,
                fetch_bpm=False,
                fetch_key=False,
                fetch_lyrics=False,
                fetch_itunes_art=False,
                album_context=_AlbumMetadataContext(
                    mb_release_details=mock_mb_release,
                    sample_artist="Rae Sremmurd",
                    sample_album="SR3MM",
                ),
                force=True,
            )
        self.assertEqual(track_metadata.artist, "Swae Lee")
        self.assertEqual(track_metadata.album_artist, "Rae Sremmurd")

    def test_album_genre_agreement_harmonization(self) -> None:
        """Album genre majority must harmonize isolated instrumental/unknown outliers to dominant genre."""
        album_dir = self.tmp_path / "Lewis Capaldi - Divinely Uninspired"
        album_dir.mkdir(parents=True, exist_ok=True)
        f1 = album_dir / "01 - Grace.wav"
        f2 = album_dir / "02 - Bruises.wav"
        f3 = album_dir / "03 - Hold Me While You Wait.wav"
        f4 = album_dir / "04 - Maybe.wav"
        for f in (f1, f2, f3, f4):
            write_silent_wav(f)
        write_track_metadata(
            TrackInfo(
                file_path=f1,
                artist="Lewis Capaldi",
                album="Divinely Uninspired",
                title="Grace",
                genre="Alternative",
                track_number=1,
            )
        )
        write_track_metadata(
            TrackInfo(
                file_path=f2,
                artist="Lewis Capaldi",
                album="Divinely Uninspired",
                title="Bruises",
                genre="Alternative",
                track_number=2,
            )
        )
        write_track_metadata(
            TrackInfo(
                file_path=f3,
                artist="Lewis Capaldi",
                album="Divinely Uninspired",
                title="Hold Me While You Wait",
                genre="Alternative",
                track_number=3,
            )
        )
        write_track_metadata(
            TrackInfo(
                file_path=f4,
                artist="Lewis Capaldi",
                album="Divinely Uninspired",
                title="Maybe",
                genre="Instrumental",
                track_number=4,
            )
        )
        with (
            patch(
                "sonora.services.musicbrainz.search_musicbrainz_release",
                return_value=None,
            ),
            patch(
                "sonora.services.deezer.fetch_deezer_album_details", return_value=None
            ),
            patch(
                "sonora.services.itunes.fetch_itunes_album_details", return_value=None
            ),
            patch("sonora.audio.art.process_artist_artwork"),
            patch("sonora.audio.replaygain.calculate_album_replaygain"),
            patch("sonora.modules.tagger._update_musicbrainz"),
            patch("sonora.modules.tagger._update_acoustid"),
            patch("sonora.modules.tagger._update_deezer"),
            patch("sonora.modules.tagger._update_itunes"),
            patch("sonora.modules.tagger._update_theaudiodb"),
            patch("sonora.modules.tagger._update_genius"),
        ):
            results = tag_album_folder(
                album_dir,
                fetch_bpm=False,
                fetch_key=False,
                fetch_replaygain=False,
                fetch_lyrics=False,
                fetch_itunes_art=False,
            )
        genres = {t.title: t.genre for t in results}
        self.assertEqual(
            genres,
            dict.fromkeys(
                ("Grace", "Bruises", "Hold Me While You Wait", "Maybe"), "Alternative"
            ),
        )

    def test_album_artist_casing_agreement_harmonization(self) -> None:
        """Album artist and track artist casing must harmonize to canonical majority."""
        album_dir = self.tmp_path / "Alex - Test Release"
        album_dir.mkdir(parents=True, exist_ok=True)
        track_file1 = album_dir / "01 - Intro.wav"
        track_file2 = album_dir / "02 - Main.wav"
        track_file3 = album_dir / "03 - Outro.wav"
        for track_file in (track_file1, track_file2, track_file3):
            write_silent_wav(track_file)
        write_track_metadata(
            TrackInfo(
                file_path=track_file1,
                artist="Alex",
                album_artist="Alex",
                album="Test Release",
                title="Intro",
                track_number=1,
            )
        )
        write_track_metadata(
            TrackInfo(
                file_path=track_file2,
                artist="ALEX",
                album_artist="Alex",
                album="Test Release",
                title="Main",
                track_number=2,
            )
        )
        write_track_metadata(
            TrackInfo(
                file_path=track_file3,
                artist="Alex",
                album_artist="ALEX",
                album="Test Release",
                title="Outro",
                track_number=3,
            )
        )
        with (
            patch(
                "sonora.services.musicbrainz.search_musicbrainz_release",
                return_value=None,
            ),
            patch(
                "sonora.services.deezer.fetch_deezer_album_details", return_value=None
            ),
            patch(
                "sonora.services.itunes.fetch_itunes_album_details", return_value=None
            ),
            patch("sonora.audio.art.process_artist_artwork"),
            patch("sonora.audio.replaygain.calculate_album_replaygain"),
            patch("sonora.modules.tagger._update_musicbrainz"),
            patch("sonora.modules.tagger._update_acoustid"),
            patch("sonora.modules.tagger._update_deezer"),
            patch("sonora.modules.tagger._update_itunes"),
            patch("sonora.modules.tagger._update_theaudiodb"),
            patch("sonora.modules.tagger._update_genius"),
            patch(
                "sonora.services.artist_resolution.resolve_artist_name",
                side_effect=lambda name, **kwargs: name,
            ),
        ):
            harmonized_tracks = tag_album_folder(
                album_dir,
                fetch_bpm=False,
                fetch_key=False,
                fetch_replaygain=False,
                fetch_lyrics=False,
                fetch_itunes_art=False,
            )
        self.assertEqual(len(harmonized_tracks), 3)
        for track_info_result in harmonized_tracks:
            self.assertEqual(track_info_result.artist, "Alex")
            self.assertEqual(track_info_result.album_artist, "Alex")


class TestTrackPreservation(OfflinePipelineTestCase):
    def test_process_single_track_rejects_mismatched_release_title_and_artist_sort(
        self,
    ) -> None:
        wav = self.tmp_path / "decebal.wav"
        write_silent_wav(wav)
        track_metadata = read_track_metadata(wav)
        track_metadata.artist = "RAVA"
        track_metadata.album = "LUCIFER"
        track_metadata.musicbrainz_albumid = "a48c628f-1b6e-4b73-b155-1ccb4cfdb7cd"
        write_track_metadata(track_metadata)
        fake_release = {
            "title": "Katcharpari",
            "artist": "Rava",
            "artist_sort": "Rava, Enrico",
            "date": "1973-01-10",
            "release_country": "Italy",
            "musicbrainz_releasegroupid": "f7b54840-2826-4a25-a821-945a9ee09669",
        }
        with patch(
            "sonora.services.musicbrainz.fetch_musicbrainz_release_details",
            return_value=fake_release,
        ):
            processed = process_single_track(
                wav,
                fetch_bpm=False,
                fetch_key=False,
                fetch_lyrics=False,
                fetch_itunes_art=False,
                force=True,
                dry_run=True,
                album_context=_AlbumMetadataContext(
                    sample_album="LUCIFER", sample_artist="RAVA"
                ),
            )
            self.assertIsNone(processed.musicbrainz_albumid)
            self.assertIsNone(processed.musicbrainz_releasegroupid)
            self.assertIsNone(processed.artist_sort)
            self.assertNotEqual(processed.date, "1973-01-10")

    def test_process_single_track_rejects_foreign_artist_sort_containment(self) -> None:
        wav = self.tmp_path / "fratemius.wav"
        write_silent_wav(wav)
        track_metadata = read_track_metadata(wav)
        track_metadata.artist = "Samurai"
        track_metadata.album = "Fratemius High"
        track_metadata.musicbrainz_albumid = "fc48d2ff-667d-468a-a5e2-760f8f5a1bcf"
        write_track_metadata(track_metadata)
        fake_release = {
            "title": "Fly High",
            "artist": "A Samurai In Tokyo",
            "artist_sort": "A Samurai In Tokyo",
            "date": "2023-02-22",
            "musicbrainz_releasegroupid": "d0078d28-0a2a-4025-a3a9-81ee90685e6a",
        }
        with patch(
            "sonora.services.musicbrainz.fetch_musicbrainz_release_details",
            return_value=fake_release,
        ):
            processed = process_single_track(
                wav,
                fetch_bpm=False,
                fetch_key=False,
                fetch_lyrics=False,
                fetch_itunes_art=False,
                force=True,
                dry_run=True,
                album_context=_AlbumMetadataContext(
                    sample_album="Fratemius High", sample_artist="Samurai"
                ),
            )
            self.assertIsNone(processed.musicbrainz_albumid)
            self.assertIsNone(processed.musicbrainz_releasegroupid)
            self.assertIsNone(processed.artist_sort)

    def test_process_single_track_preserves_valid_inverted_artist_sort(self) -> None:
        wav = self.tmp_path / "posty.wav"
        write_silent_wav(wav)
        track_metadata = read_track_metadata(wav)
        track_metadata.artist = "Post Malone"
        track_metadata.album = "Hollywood's Bleeding"
        track_metadata.musicbrainz_albumid = "11111111-2222-3333-4444-555555555555"
        write_track_metadata(track_metadata)
        fake_release = {
            "title": "Hollywood's Bleeding",
            "artist": "Post Malone",
            "artist_sort": "Malone, Post",
            "date": "2019-09-06",
            "musicbrainz_releasegroupid": "66666666-7777-8888-9999-000000000000",
        }
        with patch(
            "sonora.services.musicbrainz.fetch_musicbrainz_release_details",
            return_value=fake_release,
        ):
            processed = process_single_track(
                wav,
                fetch_bpm=False,
                fetch_key=False,
                fetch_lyrics=False,
                fetch_itunes_art=False,
                force=True,
                dry_run=True,
                album_context=_AlbumMetadataContext(
                    sample_album="Hollywood's Bleeding", sample_artist="Post Malone"
                ),
            )
            self.assertEqual(
                processed.musicbrainz_albumid, "11111111-2222-3333-4444-555555555555"
            )
            self.assertEqual(processed.artist_sort, "Malone, Post")

    @patch("sonora.audio.metadata.write_track_metadata")
    def test_process_single_track_triggers_write_on_art_upgrade(
        self, mock_write: MagicMock
    ) -> None:
        wav = self.tmp_path / "lowres_art.wav"
        write_silent_wav(wav)
        track_metadata = read_track_metadata(wav)
        track_metadata.artist = "Artist"
        track_metadata.title = "Title"
        track_metadata.album = "Album"
        track_metadata.art_width = 300
        track_metadata.art_height = 300
        write_track_metadata(track_metadata)
        cover_img = self.tmp_path / "cover.jpg"
        Image.new("RGB", (800, 800), "red").save(cover_img)
        with (
            patch(
                "sonora.services.musicbrainz.fetch_track_mbid",
                return_value=(None, None),
            ),
            patch("sonora.audio.metadata.read_track_metadata") as mock_read,
        ):
            mock_read.return_value = TrackInfo(
                file_path=wav,
                artist="Artist",
                title="Title",
                album="Album",
                album_artist="Artist",
                track_number=1,
                disc_number=1,
                art_width=300,
                art_height=300,
            )
            process_single_track(
                wav,
                fetch_bpm=False,
                fetch_key=False,
                fetch_lyrics=False,
                fetch_itunes_art=True,
                force=False,
                dry_run=False,
                album_context=_AlbumMetadataContext(
                    cover_path=cover_img, sample_album="Album", sample_artist="Artist"
                ),
            )
            self.assertTrue(mock_write.called)
            called_cover_path = mock_write.call_args[1].get("cover_art_path")
            self.assertEqual(called_cover_path, cover_img)


def _change_proposal_annotations(recording: TrackInfo) -> None:
    recording.metadata_sources["title"] = "Deezer"
    recording.metadata_conflicts.clear()
    recording._diff_descriptions.append("Provider change")


class TestTrackProcessingContracts(IsolatedTestCase):
    def test_provider_proposals_do_not_share_mutable_track_state(self) -> None:
        self.recording.metadata_sources["title"] = "MusicBrainz"
        self.recording.metadata_conflicts["album"] = "Review edition"
        self.recording._diff_descriptions.append("Existing change")
        proposal = metadata_merge.collect_provider_proposal(
            self.recording, _change_proposal_annotations
        )
        self.assertEqual(self.recording.metadata_sources, {"title": "MusicBrainz"})
        self.assertEqual(self.recording.metadata_conflicts, {"album": "Review edition"})
        self.assertEqual(self.recording._diff_descriptions, ["Existing change"])
        self.assertEqual(proposal.changes, {})
        self.assertEqual(proposal.conflicts, {})

    def test_blank_artist_and_album_fields_normalize_without_inventing_identity(
        self,
    ) -> None:
        recording = TrackInfo(
            self.path, artist=" ", title="", album="", album_artist=""
        )
        normalized = normalization.normalize_track_metadata(recording)
        self.assertEqual((normalized.artist, normalized.album), ("", ""))

    def test_normalization_preview_reports_changes_without_writing_tags(self) -> None:
        with (
            patch.object(
                metadata,
                "read_track_metadata",
                return_value=dataclasses.replace(self.recording, genre="rock"),
            ),
            patch.object(metadata, "write_track_metadata") as write,
        ):
            normalized = track_processing.normalize_single_track(
                self.path, dry_run=True
            )
        assert normalized is not None
        self.assertEqual(normalized.genre, "Rock")
        self.assertTrue(normalized._diff_descriptions)
        write.assert_not_called()

    def setUp(self) -> None:
        self.path = self.test_directory / "recording.flac"
        self.path.touch()
        self.recording = TrackInfo(
            self.path, artist="Primary", title="Recording", album="Record"
        )

    def test_cuesheets_use_explicit_content_then_companion_without_replacing_tags(
        self,
    ) -> None:
        companion = self.test_directory / "record.cue"
        companion.write_text('TITLE "Record"')
        track_processing._update_cuesheet(self.recording, self.path, None)
        self.assertEqual(self.recording.cuesheet, 'TITLE "Record"')
        companion.write_text('TITLE "Changed"')
        track_processing._update_cuesheet(self.recording, self.path, None)
        self.assertEqual(self.recording.cuesheet, 'TITLE "Record"')
        track_processing._update_cuesheet(self.recording, self.path, 'TITLE "Explicit"')
        self.assertEqual(self.recording.cuesheet, 'TITLE "Explicit"')

    def test_artwork_metadata_updates_can_be_disabled_and_recovers_from_io_errors(
        self,
    ) -> None:
        with patch.object(
            art, "process_album_cover_art", side_effect=OSError("offline")
        ) as fetch:
            self.assertIsNone(
                track_processing._update_artwork(self.recording, self.path, False)
            )
            fetch.assert_not_called()
            self.assertIsNone(
                track_processing._update_artwork(self.recording, self.path, True)
            )
        fetch.assert_called_once()

    def test_unchanged_normalization_is_successful_and_replaygain_is_forwarded(
        self,
    ) -> None:
        with (
            patch.object(metadata, "read_track_metadata", return_value=self.recording),
            patch.object(metadata, "write_track_metadata") as write,
            patch.object(replaygain, "calculate_album_replaygain") as gain,
        ):
            result = track_processing.normalize_single_track(self.path)
            assert result is not None
            self.assertEqual(result._diff_descriptions, [])
            report = track_processing.normalize_library(
                self.test_directory, fetch_replaygain=True, dry_run=True
            )
        self.assertEqual(report.total_scanned, 1)
        write.assert_not_called()
        gain.assert_called_once_with(
            [self.path], force=False, dry_run=True, max_threads=4
        )

    def test_normalization_handles_empty_identity_and_existing_sort_names(self) -> None:
        for artist, album_artist, title in (
            ("", "", ""),
            ("!!!", "!!!", "Recording (prod.   )"),
            ("Primary", "Primary", "Recording feat. Guest"),
        ):
            recording = TrackInfo(
                self.path,
                artist=artist,
                album_artist=album_artist,
                title=title,
                artist_sort=artist,
                album_artist_sort=album_artist,
            )
            with (
                self.subTest(artist=artist, title=title),
                patch(
                    "sonora.services.artist_resolution.resolve_artist_name",
                    side_effect=lambda name: name,
                ) as resolve,
            ):
                normalized = normalization.normalize_track_metadata(recording)
            self.assertIs(normalized, recording)
            self.assertEqual(normalized.artist, artist)
            resolve.assert_not_called()
            self.assertIsNone(normalized.producers)
        self.assertEqual(normalized.featured_artists, "Guest")

    def test_dsp_metadata_updates_accepts_results_and_handles_decode_failures(
        self,
    ) -> None:
        with patch.object(bpm, "calculate_bpm", return_value=123.4):
            track_processing._update_bpm(self.recording, self.path, True)
        self.assertEqual(self.recording.bpm, 123.4)
        with patch.object(bpm, "calculate_bpm") as calculate:
            track_processing._update_bpm(self.recording, self.path, True)
        calculate.assert_not_called()
        with patch.object(key, "detect_musical_key", return_value="Am"):
            track_processing._update_key(self.recording, self.path, True)
        self.assertEqual(self.recording.initial_key, "Am")
        for result in (None, OSError("decode failed")):
            with (
                self.subTest(result=result),
                patch.object(
                    key, "detect_musical_key", return_value=None, side_effect=result
                ),
                patch.object(
                    bpm, "calculate_bpm", return_value=None, side_effect=result
                ),
            ):
                track_processing._update_key(
                    self.recording, self.path, True, force=True
                )
                track_processing._update_bpm(
                    self.recording, self.path, True, force=True
                )
            self.assertEqual(self.recording.initial_key, "Am")
            self.assertEqual(self.recording.bpm, 123.4)

    def test_audio_analysis_shares_one_decode_when_bpm_and_key_are_missing(
        self,
    ) -> None:
        recording = TrackInfo(self.path, artist="Artist", title="Song")
        samples = object()
        with (
            patch.object(
                decode_module, "load_audio", return_value=(samples, 44100)
            ) as load_audio,
            patch.object(bpm, "calculate_bpm", return_value=123.4) as calculate_bpm,
            patch.object(key, "detect_musical_key", return_value="Am") as detect_key,
        ):
            track_processing._update_audio_analysis(
                recording, self.path, fetch_bpm=True, fetch_key=True
            )
        load_audio.assert_called_once_with(self.path, mono=True, max_seconds=70.0)
        self.assertIs(calculate_bpm.call_args.kwargs["audio_data"][0], samples)
        self.assertIs(detect_key.call_args.kwargs["audio_data"][0], samples)
        self.assertEqual((recording.bpm, recording.initial_key), (123.4, "Am"))

    def test_audio_analysis_does_not_prefetch_for_a_present_empty_key(self) -> None:
        recording = TrackInfo(self.path, artist="Artist", title="Song", initial_key="")
        with (
            patch.object(decode_module, "load_audio") as load_audio,
            patch.object(bpm, "calculate_bpm", return_value=123.4) as calculate_bpm,
            patch.object(key, "detect_musical_key") as detect_key,
        ):
            track_processing._update_audio_analysis(
                recording, self.path, fetch_bpm=True, fetch_key=True
            )
        load_audio.assert_not_called()
        calculate_bpm.assert_called_once_with(self.path, audio_data=None)
        detect_key.assert_not_called()

    def test_lyrics_updates_accept_provider_text_and_preserve_it_on_failure(
        self,
    ) -> None:
        for existing, force in ((False, False), (True, False), (True, True)):
            if existing:
                self.path.with_suffix(".lrc").write_text("[00:01] Existing")
            with (
                self.subTest(existing=existing, force=force),
                patch.object(metadata, "get_audio_duration", return_value=100),
                patch.object(
                    track_processing,
                    "process_track_lyrics",
                    return_value=("[00:01] Lyrics", "synced"),
                ),
            ):
                track_processing._update_lyrics(
                    self.recording, self.path, True, force=force
                )
            self.assertEqual(self.recording.lyrics, "[00:01] Lyrics")
        with patch.object(
            track_processing, "process_track_lyrics", side_effect=OSError("offline")
        ):
            track_processing._update_lyrics(self.recording, self.path, True)
        self.assertEqual(self.recording.lyrics, "[00:01] Lyrics")

    def test_normalization_read_failure_and_failed_write_produce_no_success(
        self,
    ) -> None:
        self.assertIsNone(
            track_processing.normalize_single_track(self.path.with_name("missing.flac"))
        )
        with patch.object(
            metadata, "read_track_metadata", side_effect=OSError("bad metadata")
        ):
            self.assertIsNone(track_processing.normalize_single_track(self.path))
        with (
            patch.object(
                metadata,
                "read_track_metadata",
                return_value=dataclasses.replace(self.recording, genre="rock"),
            ),
            patch.object(
                metadata, "write_track_metadata", side_effect=OSError("disk full")
            ),
        ):
            self.assertIsNone(track_processing.normalize_single_track(self.path))

    def test_empty_library_and_worker_failures_preserve_domain_errors(self) -> None:
        with self.assertRaises(ValueError):
            track_processing.normalize_library(self.path)
        empty_directory = self.test_directory / "empty"
        empty_directory.mkdir()
        self.assertEqual(
            track_processing.normalize_library(empty_directory).total_scanned, 0
        )
        with patch.object(
            track_processing, "normalize_single_track", return_value=None
        ):
            report = track_processing.normalize_library(self.test_directory)
            self.assertEqual(
                (report.total_scanned, report.failed_files, report.total_unchanged),
                (1, 1, 0),
            )
        with (
            patch.object(
                track_processing,
                "normalize_single_track",
                side_effect=RuntimeError("worker bug"),
            ),
            self.assertRaisesRegex(RuntimeError, "worker bug"),
        ):
            track_processing.normalize_library(self.test_directory)

    def test_canonical_normalization_preserves_known_credits_and_drops_wrong_sorts(
        self,
    ) -> None:
        recording = TrackInfo(
            self.path,
            artist="Primary feat. Guest",
            title="Recording (prod. Producer) [Explicit]",
            album="Record",
            album_artist="Primary",
            artist_sort="Unrelated",
            album_artist_sort="Unrelated",
            producers="Verified",
            release_type="single",
            total_tracks=1,
            track_number=4,
        )
        normalized = normalization.normalize_track_metadata(recording)
        self.assertEqual(normalized.title, "Recording")
        self.assertEqual(
            (normalized.artist, normalized.featured_artists, normalized.advisory),
            ("Primary", "Guest", "Explicit"),
        )
        self.assertEqual(normalized.producers, "Verified")
        self.assertEqual((normalized.track_number, normalized.total_tracks), (1, 1))
        self.assertIsNone(normalized.artist_sort)
        self.assertIsNone(normalized.album_artist_sort)
        before = dataclasses.replace(normalized)
        self.assertEqual(normalization.normalize_track_metadata(normalized), before)
