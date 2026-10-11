import dataclasses
import os
from pathlib import Path
from unittest.mock import patch

from sonora.audio import art, metadata, replaygain
from sonora.core.config import get_config
from sonora.core.models import TrackInfo
from sonora.core.state import LibraryStateVault
from sonora.modules import tagger
from sonora.services import deezer, discogs, itunes, musicbrainz
from tests.support import IsolatedTestCase

MBID = "11111111-2222-3333-4444-555555555555"


class TestAlbumPipelineContracts(IsolatedTestCase):
    def test_root_directory_identity_and_blank_artist_consensus_remain_unknown(
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

    def test_partial_release_indexes_fall_back_to_the_recording_id_lookup(self) -> None:
        for release in (
            {"tracks_by_position": {1: {"recording_mbid": None}}},
            {"tracks_by_disc_and_position": {(1, 1): {"recording_mbid": None}}},
        ):
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
                patch.object(
                    musicbrainz, "fetch_album_track_mbids", return_value={1: MBID}
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
            self.assertEqual(context.track_mbids, {1: MBID})

    def test_itunes_album_track_artists_are_preserved_during_identity_healing(
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
            "tracks_by_number": {3: {"title": "Recording", "artist": "Guest"}}
        }
        tagger._heal_and_align_album_track(
            recording,
            recording.file_path,
            None,
            None,
            itunes_release,
            None,
            "Primary",
            "Record",
            True,
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

    def test_artist_consensus_preserves_distinct_artist_identity_in_compilations(
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

    def test_itunes_album_genre_is_an_authoritative_consensus(self) -> None:
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

    def test_up_to_date_album_skips_enrichment_and_preserves_results(self) -> None:
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

    def test_single_track_genre_and_distinct_album_artist_need_no_consensus(
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
        tagged = dataclasses.replace(self.recording, musicbrainz_albumid=MBID)
        release = {
            "tracks_by_position": {1: {"recording_mbid": MBID}, 2: {}},
            "tracks_by_disc_and_position": {(1, 1): {"recording_mbid": MBID}},
        }
        with (
            patch.object(
                tagger,
                "_resolve_album_folder_identity",
                return_value=("Primary", "Record"),
            ),
            patch.object(metadata, "read_track_metadata", return_value=tagged),
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
        self.assertEqual(context.musicbrainz_id, MBID)
        self.assertEqual(context.track_mbids, {(1, 1): MBID, 1: MBID})
        self.assertEqual(context.discogs_release, {"label": "Publisher"})
        self.assertEqual(context.sample_artist, "Primary")

    def test_prefetch_missing_recording_indexes_fetches_mbids_separately(self) -> None:
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
            patch.object(
                musicbrainz, "fetch_album_track_mbids", return_value={(1, 1): MBID}
            ) as fetch,
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
        fetch.assert_called_once_with(MBID)
        self.assertEqual(context.track_mbids, {(1, 1): MBID})

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
                "sonora.core.artists.resolve_artist_name",
                side_effect=lambda name, *, allow_network=True: name,
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
