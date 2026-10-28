import dataclasses
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import httpx

from sonora.core.config import SonoraConfig
from sonora.core.models import TrackInfo
from sonora.modules import catalog_metadata as catalog
from sonora.modules import matching, metadata_merge
from sonora.services import deezer as deezer_service
from sonora.services import discogs as discogs_service
from sonora.services import genius as genius_service
from sonora.services import itunes as itunes_service
from sonora.services import lastfm as lastfm_service
from sonora.services import shazam as shazam_service
from sonora.services import theaudiodb as theaudiodb_service
from sonora.services.shazam import ShazamTrackInfo
from tests.support import IsolatedTestCase

MBID = "11111111-2222-3333-4444-555555555555"


class TestMetadataMappingContracts(IsolatedTestCase):
    def test_provider_placeholders_cannot_become_metadata_or_destroy_a_title(
        self,
    ) -> None:
        from sonora.modules.normalization import normalize_track_metadata

        track = TrackInfo(
            Path("Song.flac"), artist="Primary", title="?", mood="...", style="…"
        )
        normalize_track_metadata(track)
        self.assertEqual(track.title, "?")
        self.assertIsNone(track.mood)
        self.assertIsNone(track.style)
        metadata_merge.apply_metadata_mapping(
            track,
            {"mood": "...", "style": "…", "composer": "Unknown"},
            {"mood": "mood", "style": "style", "composer": "composer"},
            source="TheAudioDB",
        )
        self.assertIsNone(track.mood)
        self.assertIsNone(track.style)
        self.assertIsNone(track.composer)

    def test_mapping_normalizes_valid_metadata_and_records_provenance(self) -> None:
        recording = TrackInfo(Path("recording.flac"), artist="Primary")
        provider_fields = {
            "genre": "rock",
            "date": "2021-02-03",
            "original_date": "2020",
            "release_country": "US",
            "language": "eng",
            "script": "Latn",
            "title": "Song (feat. Guest)",
            "album": "The ’Album’",
            "album_artist": "Primary",
            "advisory": "explicit",
            "initial_key": "Am",
            "featured_artists": "Another Guest",
            "track_number": "2",
            "disc_number": "1",
            "total_tracks": "8",
            "total_discs": "1",
            "label": "Publisher",
            "isrc": "USAAA2100001",
        }
        metadata_merge.apply_metadata_mapping(
            recording,
            provider_fields,
            {name: name for name in provider_fields},
            source="Deezer",
        )
        expected = {
            "genre": "Rock",
            "date": "2021-02-03",
            "original_date": "2020",
            "release_country": "United States",
            "language": "English",
            "script": "Latin",
            "title": "Song",
            "album": "The 'Album'",
            "album_artist": "Primary",
            "advisory": "Explicit",
            "initial_key": "Am",
            "featured_artists": "Guest, Another Guest",
            "track_number": 2,
            "disc_number": 1,
            "total_tracks": 8,
            "total_discs": 1,
            "label": "Publisher",
            "isrc": "USAAA2100001",
        }
        for name, value in expected.items():
            with self.subTest(field=name):
                self.assertEqual(getattr(recording, name), value)
        self.assertEqual(recording.metadata_sources["label"], "Deezer")
        for name in ("track_number", "disc_number", "total_tracks", "total_discs"):
            self.assertEqual(recording.metadata_sources[name], "Deezer")

    def test_invalid_and_absent_provider_values_preserve_existing_tags(self) -> None:
        cases: tuple[tuple[str, object], ...] = (
            ("label", None),
            ("label", " "),
            ("label", "null"),
            ("label", "none"),
            ("genre", "unknown"),
            ("date", "not a date"),
            ("advisory", "clean"),
            ("initial_key", "not a key"),
            ("track_number", "unknown"),
            ("featured_artists", "Primary"),
        )
        for field, value in cases:
            recording = TrackInfo(Path("recording.flac"), artist="Primary")
            before = dataclasses.replace(recording)
            with self.subTest(field=field, value=value):
                metadata_merge.apply_metadata_mapping(
                    recording, {"value": value}, {"value": field}, force=True
                )
                self.assertEqual(recording, before)

    def test_force_preserves_richer_credits_and_equivalent_isrc(self) -> None:
        recording = TrackInfo(
            Path("recording.flac"),
            artist="Primary",
            label="Publisher Records",
            isrc="USAAA2100001",
            track_number=3,
            disc_number=1,
        )
        metadata_merge.apply_metadata_mapping(
            recording,
            {
                "label": "Publisher",
                "isrc": "usaaa2100001",
                "track_number": 2,
                "disc_number": 2,
            },
            {name: name for name in ("label", "isrc", "track_number", "disc_number")},
            force=True,
            source="Deezer",
        )
        self.assertEqual(recording.label, "Publisher Records")
        self.assertEqual(recording.isrc, "USAAA2100001")
        self.assertEqual((recording.track_number, recording.disc_number), (2, 2))

    def test_existing_tags_respect_force_and_source_precedence(self) -> None:
        recording = TrackInfo(
            Path("recording.flac"), label="Existing", track_number=3, disc_number=1
        )
        metadata_merge.apply_metadata_mapping(
            recording,
            {"label": "Replacement", "track_number": 2, "disc_number": 1},
            {name: name for name in ("label", "track_number", "disc_number")},
            source="Deezer",
        )
        self.assertEqual((recording.label, recording.track_number), ("Existing", 3))
        metadata_merge.apply_metadata_mapping(
            recording, {"label": "Existing"}, {"label": "label"}, source="Deezer"
        )
        self.assertEqual(recording.metadata_sources["label"], "Deezer")
        metadata_merge.apply_metadata_mapping(
            recording, {"label": "Existing"}, {"label": "label"}, source="Discogs"
        )
        self.assertEqual(recording.metadata_sources["label"], "Deezer")
        metadata_merge.apply_metadata_mapping(
            recording, {"label": "Different"}, {"label": "label"}, force=True
        )
        self.assertEqual(recording.label, "Different")


class TestCatalogMetadataContracts(IsolatedTestCase):
    def test_shazam_cannot_invent_an_album_when_recognition_has_none(self) -> None:
        recording = TrackInfo(Path("recording.flac"))
        with (
            patch.object(
                catalog, "get_config", return_value=SonoraConfig(enable_shazam=True)
            ),
            patch.object(
                shazam_service,
                "recognize_audio_track",
                return_value=ShazamTrackInfo("Recording", "Primary"),
            ),
        ):
            catalog._update_shazam(
                recording, recording.file_path, has_album_context=True
            )
        self.assertEqual(
            (recording.title, recording.artist, recording.album),
            ("Recording", "Primary", "Unknown Album"),
        )

    def test_unmatched_discogs_track_credits_preserve_the_recording(self) -> None:
        recording = dataclasses.replace(self.recording)
        catalog._update_discogs(
            recording,
            "fixture",
            {"id": 1, "track_credits": {"other": {"producers": "Wrong"}}},
        )
        self.assertIsNone(recording.discogs_release_id)
        self.assertIsNone(recording.producers)

    def test_deezer_base_album_cannot_supply_deluxe_release_selectors(self) -> None:
        track = TrackInfo(
            Path("Song.flac"), artist="Primary", title="Song", album="Album (Deluxe)"
        )
        with patch.object(
            deezer_service, "fetch_deezer_track_details", return_value=None
        ):
            catalog._update_deezer(
                track,
                {
                    "title": "Album",
                    "nb_tracks": 18,
                    "barcode": "wrong-base-upc",
                    "release_date": "2016-11-25",
                    "label": "Base Edition Label",
                },
                has_album_context=True,
            )
        self.assertIsNone(track.barcode)
        self.assertIsNone(track.date)
        self.assertIsNone(track.label)
        self.assertEqual(track.album, "Album (Deluxe)")

    def test_deezer_preserves_authoritative_dates_and_consistent_album_totals(
        self,
    ) -> None:
        album = {
            "title": "Record",
            "nb_tracks": 5,
            "release_date": "2021",
            "tracks_by_disc_and_position": {(1, 3): {"title": "Song"}},
        }
        recording = dataclasses.replace(
            self.recording,
            track_number=3,
            total_tracks=10,
            date="2020",
            musicbrainz_albumid=MBID,
        )
        catalog._update_deezer(recording, album, has_album_context=True)
        self.assertEqual(
            (recording.album, recording.total_tracks, recording.date),
            ("Record", 10, "2020"),
        )
        recording.musicbrainz_albumid = None
        catalog._update_deezer(recording, album, has_album_context=True)
        catalog._update_deezer(recording, album, has_album_context=True, force=True)
        self.assertEqual(recording.total_tracks, 10)

    def test_deezer_preserves_conflicting_isrc_even_for_exact_duration_match(
        self,
    ) -> None:
        recording = TrackInfo(
            Path("i lied - sped up.flac"),
            artist="Cheyanne & Ex Habit",
            title="i lied - sped up",
            album="i lied",
            isrc="UKZGC2506039",
            duration=147.698,
            producers="Verified producer",
            musicbrainz_trackid=MBID,
        )
        conflicting = {
            "artist": "Cheyanne & Ex Habit",
            "title": "i lied",
            "duration": 165,
            "isrc": "UKZGC2506039",
        }
        exact_edit = {
            "artist": "Cheyanne & Ex Habit",
            "title": "i lied - sped up",
            "duration": 148,
            "isrc": "UKZGC2506040",
        }
        with (
            patch.object(
                deezer_service,
                "fetch_deezer_track_by_isrc",
                return_value=conflicting,
            ),
            patch.object(
                deezer_service,
                "fetch_deezer_track_details",
                return_value=exact_edit,
            ) as lookup,
        ):
            catalog._update_deezer(recording, None, has_album_context=True)
        lookup.assert_called_once_with(
            "Cheyanne & Ex Habit", "i lied - sped up", duration=147.698
        )
        self.assertEqual(recording.isrc, "UKZGC2506039")
        self.assertNotIn("isrc", recording.metadata_sources)
        self.assertIn("isrc", recording.metadata_conflicts)

    def test_deezer_keeps_conflicting_isrc_when_search_result_is_not_exact(
        self,
    ) -> None:
        recording = TrackInfo(
            Path("i lied - sped up.flac"),
            artist="Cheyanne & Ex Habit",
            title="i lied - sped up",
            isrc="UKZGC2506039",
            duration=147.698,
            producers="Verified producer",
        )
        with (
            patch.object(
                deezer_service,
                "fetch_deezer_track_by_isrc",
                return_value={
                    "artist": "Cheyanne & Ex Habit",
                    "title": "i lied",
                    "duration": 165,
                    "isrc": "UKZGC2506039",
                },
            ),
            patch.object(
                deezer_service,
                "fetch_deezer_track_details",
                return_value={
                    "artist": "Different Artist",
                    "title": "Other Song",
                    "duration": 148,
                    "isrc": "UKZGC2506040",
                },
            ),
        ):
            catalog._update_deezer(recording, None, has_album_context=True)
        self.assertEqual(recording.isrc, "UKZGC2506039")
        self.assertIn("isrc", recording.metadata_conflicts)

    def test_clean_title_index_matches_without_a_position(self) -> None:
        recording = {"title": "Recording"}
        self.assertEqual(
            matching._find_track_in_album_mapping(
                {"recording": recording}, None, "Recording (feat. Guest)", None
            ),
            (recording, True),
        )

    def test_album_lookup_supports_disc_positions_and_raw_title_indexes(self) -> None:
        recording = {"title": "Recording"}
        for title_index, position_index, disc_index, title, expected in (
            ({}, {}, {(2, 3): recording}, "Recording", recording),
            ({}, {(2, 3): recording}, None, "Recording", recording),
            (
                {"recording feat guest": recording},
                {},
                None,
                "Recording (feat. Guest)",
                recording,
            ),
            ({}, {}, None, "Untitled", None),
        ):
            with self.subTest(
                title_index=title_index,
                position_index=position_index,
                disc_index=disc_index,
            ):
                matched, from_album = matching._find_track_in_album_mapping(
                    title_index, disc_index or position_index, title, 3, 2
                )
                self.assertEqual(matched, expected)
                self.assertEqual(from_album, expected is not None)

    def test_album_shazam_without_an_album_keeps_known_identity_and_credits(
        self,
    ) -> None:
        recognized = ShazamTrackInfo("Recognized", "Primary")
        for has_context, title, artist, expected_title in (
            (True, "Song", "Primary", "Song"),
            (True, "", "", "Recognized"),
            (False, "Song", "Primary", "Song"),
        ):
            recording = TrackInfo(
                Path("recording.flac"),
                title=title,
                artist=artist,
                album="Record",
                genre="Rock",
                label="Publisher",
                lyrics="Verified",
                isrc="USAAA2100001",
                itunes_trackid="123",
                date="2020",
            )
            with (
                self.subTest(has_context=has_context, title=title),
                patch.object(
                    catalog, "get_config", return_value=SonoraConfig(enable_shazam=True)
                ),
                patch.object(
                    shazam_service, "recognize_audio_track", return_value=recognized
                ),
            ):
                catalog._update_shazam(
                    recording,
                    recording.file_path,
                    has_album_context=has_context,
                    force=True,
                )
            self.assertEqual(
                (recording.title, recording.artist, recording.album),
                (expected_title, "Primary", "Record"),
            )
            self.assertEqual(
                (recording.genre, recording.label, recording.lyrics),
                ("Rock", "Publisher", "Verified"),
            )

    def test_redundant_lastfm_styles_do_not_create_empty_style_tags(self) -> None:
        recording = dataclasses.replace(self.recording, mood="Verified")
        with patch.object(
            lastfm_service, "fetch_lastfm_tags", return_value=["rock", "Rock"]
        ):
            catalog._update_lastfm(recording, "fixture")
        self.assertEqual(recording.genre, "Rock")
        self.assertIsNone(recording.style)
        self.assertEqual(recording.mood, "Verified")

    @patch.object(itunes_service, "fetch_itunes_track_metadata", return_value=None)
    @patch.object(deezer_service, "fetch_deezer_track_details", return_value=None)
    def test_album_structure_preserves_existing_totals_and_authoritative_dates(
        self,
        _deezer_lookup: MagicMock,
        _itunes_lookup: MagicMock,
    ) -> None:
        recording = dataclasses.replace(
            self.recording,
            track_number=3,
            disc_number=2,
            total_discs=2,
            total_tracks=10,
            date="2020",
        )
        catalog._update_itunes(
            recording,
            {
                "tracks_by_disc_and_position": {
                    (1, 3): {
                        "title": "Song",
                        "date": "2021",
                        "disc_number": 1,
                        "total_discs": 1,
                    }
                }
            },
        )
        self.assertEqual(
            (recording.disc_number, recording.total_discs, recording.date),
            (2, 2, "2020"),
        )
        catalog._update_deezer(
            recording,
            {
                "title": "Record",
                "nb_tracks": "bad",
                "tracks_by_disc_and_position": {(1, 3): {"title": "Song"}},
            },
            has_album_context=True,
            force=True,
        )
        self.assertEqual(recording.total_tracks, 10)

    def test_known_genius_dates_and_lyricists_are_not_replaced(self) -> None:
        recording = dataclasses.replace(
            self.recording, date="2020", lyricist="Verified"
        )
        with patch.object(
            genius_service,
            "fetch_genius_song_details",
            return_value={"writers": "Writer", "release_date": "2021"},
        ):
            catalog._update_genius(recording, "fixture")
        self.assertEqual((recording.date, recording.lyricist), ("2020", "Verified"))

    def test_theaudiodb_existing_rating_and_genre_survive_metadata_updates(self) -> None:
        recording = dataclasses.replace(self.recording, genre="Rock", rating=8.5)
        with patch.object(
            theaudiodb_service,
            "fetch_theaudiodb_track_details",
            return_value={"rating": 1, "genre": "Pop", "mood": "Chill"},
        ):
            catalog._update_theaudiodb(recording)
        self.assertEqual(
            (recording.genre, recording.rating, recording.mood), ("Rock", 8.5, "Chill")
        )

    def setUp(self) -> None:
        artist_resolution = patch(
            "sonora.services.artist_resolution.resolve_artist_name",
            side_effect=lambda artist: artist,
        )
        artist_resolution.start()
        self.addCleanup(artist_resolution.stop)
        self.recording = TrackInfo(
            Path("recording.flac"),
            artist="Primary",
            title="Song",
            album="Record",
            disc_number=None,
        )

    def test_shazam_recognition_fills_missing_metadata(self) -> None:
        recognized = ShazamTrackInfo(
            "Song",
            "Primary",
            "Album",
            "Rock",
            "123",
            isrc="USAAA2100001",
            release_date="2021",
            label="Publisher",
            lyrics="[00:01] Song",
        )
        recording = TrackInfo(Path("recording.flac"))
        with (
            patch.object(
                catalog, "get_config", return_value=SonoraConfig(enable_shazam=True)
            ),
            patch.object(
                shazam_service, "recognize_audio_track", return_value=recognized
            ),
        ):
            catalog._update_shazam(recording, recording.file_path)
        self.assertEqual(
            (recording.title, recording.artist, recording.album),
            ("Song", "Primary", "Album"),
        )
        self.assertEqual(
            (recording.genre, recording.itunes_trackid, recording.isrc),
            ("Rock", "123", "USAAA2100001"),
        )
        self.assertEqual(
            (recording.date, recording.label, recording.lyrics),
            ("2021", "Publisher", "[00:01] Song"),
        )

    def test_shazam_album_context_and_sample_hijacking_protection(self) -> None:
        cases = (
            (
                TrackInfo(Path("missing.flac")),
                None,
                None,
                "Recognized",
                "Primary",
                "Found Album",
            ),
            (
                TrackInfo(Path("missing.flac")),
                "Album Artist",
                "Album Title",
                "Recognized",
                "Album Artist",
                "Album Title",
            ),
            (
                TrackInfo(
                    Path("existing.flac"),
                    artist="Primary",
                    title="Song",
                    album="Record",
                    disc_number=None,
                ),
                "Primary",
                "Record",
                "Song",
                "Primary",
                "Record",
            ),
        )
        with (
            patch.object(
                catalog, "get_config", return_value=SonoraConfig(enable_shazam=True)
            ),
            patch.object(
                shazam_service,
                "recognize_audio_track",
                return_value=ShazamTrackInfo("Recognized", "Primary", "Found Album"),
            ),
        ):
            for (
                recording,
                artist,
                album,
                title,
                expected_artist,
                expected_album,
            ) in cases:
                with self.subTest(path=recording.file_path, album=album):
                    catalog._update_shazam(
                        recording,
                        recording.file_path,
                        artist,
                        album,
                        has_album_context=True,
                        force=True,
                    )
                    self.assertEqual(
                        (recording.title, recording.artist, recording.album),
                        (title, expected_artist, expected_album),
                    )
            with patch.object(
                shazam_service,
                "recognize_audio_track",
                return_value=ShazamTrackInfo("Wrong", "Unrelated Artist"),
            ):
                catalog._update_shazam(
                    self.recording, self.recording.file_path, "Primary", force=True
                )
                self.assertEqual(
                    (self.recording.artist, self.recording.title), ("Primary", "Song")
                )

    def test_shazam_skips_complete_metadata_and_empty_recognition(self) -> None:
        with (
            patch.object(
                catalog, "get_config", return_value=SonoraConfig(enable_shazam=True)
            ),
            patch.object(
                shazam_service, "recognize_audio_track", return_value=None
            ) as recognize,
        ):
            catalog._update_shazam(self.recording, self.recording.file_path)
            recognize.assert_not_called()
            catalog._update_shazam(self.recording, self.recording.file_path, force=True)
            recognize.assert_called_once()

    def test_itunes_album_structure_and_musicbrainz_date_precedence(self) -> None:
        track = {
            "genre": "Rock",
            "advisory": "Explicit",
            "date": "2020",
            "disc_number": 2,
            "total_discs": 2,
            "title": "Found",
            "artist": "Primary",
            "itunes_trackid": "123",
        }
        album = {"tracks_by_disc_and_position": {(1, 1): track}}
        self.recording.title = "Untitled"
        self.recording.itunes_trackid = "123"
        self.recording.track_number = 1
        with patch.object(itunes_service, "fetch_itunes_track_metadata") as lookup:
            catalog._update_itunes(self.recording, album, force=True)
        lookup.assert_not_called()
        self.assertEqual(
            (
                self.recording.title,
                self.recording.disc_number,
                self.recording.total_discs,
                self.recording.date,
            ),
            ("Found", 2, 2, "2020"),
        )
        self.recording.musicbrainz_albumid = MBID
        self.recording.date = "2019"
        with (
            patch.object(itunes_service, "lookup_itunes_track", return_value=None),
            patch.object(
                itunes_service, "fetch_itunes_track_metadata", return_value=None
            ),
        ):
            catalog._update_itunes(self.recording, album, force=True)
        self.assertEqual(self.recording.date, "2019")

    def test_itunes_single_fallback_does_not_copy_album_positions(self) -> None:
        with patch.object(
            itunes_service,
            "fetch_itunes_track_metadata",
            return_value={
                "genre": "Rock",
                "disc_number": 4,
                "total_discs": 4,
                "date": "2020",
                "artist": "Primary",
                "title": "Song",
            },
        ):
            catalog._update_itunes(self.recording)
        self.assertEqual(self.recording.genre, "Rock")
        self.assertIsNone(self.recording.disc_number)
        self.assertIsNone(self.recording.total_discs)

    def test_lastfm_genre_styles_and_mood(self) -> None:
        for tags, expected in (
            ([], (None, None, None)),
            (["rock"], ("Rock", None, None)),
            (["unknown", "rock", "chill"], (None, "rock, chill", "Chill")),
            (["rock", "Rock", "chill", "dark"], ("Rock", "chill, dark", "Chill")),
        ):
            recording = dataclasses.replace(self.recording)
            with (
                self.subTest(tags=tags),
                patch.object(lastfm_service, "fetch_lastfm_tags", return_value=tags),
            ):
                catalog._update_lastfm(recording, "fixture")
            self.assertEqual(
                (recording.genre, recording.style, recording.mood), expected
            )

    def test_discogs_release_and_track_credits(self) -> None:
        release: dict[str, Any] = {
            "id": 123,
            "styles": ["Alternative", "Indie"],
            "genres": ["rock"],
            "year": 2020,
            "track_credits": {"2": {"title": "Found Song", "producers": "Producer"}},
            "country": "US",
        }
        self.recording.title = "Found Song"
        self.recording.track_number = 2
        self.recording.discogs_release_id = "123"
        catalog._update_discogs(
            self.recording, "fixture", release, has_album_context=True
        )
        self.assertEqual(
            (
                self.recording.discogs_release_id,
                self.recording.style,
                self.recording.genre,
                self.recording.date,
            ),
            ("123", "Alternative, Indie", "Rock", "2020"),
        )
        self.assertEqual(
            (self.recording.title, self.recording.producers), ("Found Song", "Producer")
        )
        self.recording.musicbrainz_albumid = MBID
        release["year"] = 2022
        release["track_credits"] = {
            "found song": {"title": "Found Song", "composer": "Composer"}
        }
        catalog._update_discogs(self.recording, "fixture", release, force=True)
        self.assertEqual(self.recording.date, "2020")
        self.assertEqual(self.recording.composer, "Composer")

    def test_discogs_album_match_cannot_invent_pressing_metadata(self) -> None:
        release = {
            "id": 123,
            "barcode": "001234567890",
            "catalog_number": "OTHER-1",
            "country": "JP",
            "media": "CD",
            "year": 2016,
            "label": "Other Label",
            "styles": ["House"],
        }
        for force in (False, True):
            for identifiers in (
                {},
                {"barcode": "999999999999"},
                {"discogs_release_id": "123", "barcode": "999999999999"},
            ):
                with self.subTest(force=force, identifiers=identifiers):
                    track = TrackInfo(
                        Path("Song.flac"),
                        artist="Primary",
                        album="Album",
                        **identifiers,
                    )
                    catalog._update_discogs(track, "fixture", release, force=force)
                    self.assertEqual(track.style, "House")
                    self.assertIsNone(track.catalog_number)
                    self.assertIsNone(track.media)
                    self.assertIsNone(track.date)
                    self.assertIsNone(track.label)
                    self.assertEqual(track.barcode, identifiers.get("barcode"))
                    self.assertEqual(
                        track.discogs_release_id, identifiers.get("discogs_release_id")
                    )
                    self.assertEqual(
                        "discogs_release_id" in track.metadata_conflicts,
                        bool(identifiers),
                    )

    def test_discogs_matching_identifier_does_not_override_source_medium(self) -> None:
        track = TrackInfo(
            Path("Song.flac"),
            artist="Primary",
            album="Album",
            barcode="1234567890",
            media="Digital Media",
        )
        catalog._update_discogs(
            track,
            "fixture",
            {
                "id": 123,
                "barcode": "1234567890",
                "catalog_number": "CD-1",
                "media": "CD, Album",
            },
            force=True,
        )
        self.assertIsNone(track.discogs_release_id)
        self.assertIsNone(track.catalog_number)
        self.assertEqual(track.media, "Digital Media")
        self.assertIn("discogs_release_id", track.metadata_conflicts)

    def test_discogs_matching_barcode_allows_edition_metadata(self) -> None:
        track = TrackInfo(
            Path("Song.flac"), artist="Primary", album="Album", barcode="001234567890"
        )
        catalog._update_discogs(
            track,
            "fixture",
            {
                "id": 123,
                "barcode": "1234567890",
                "catalog_number": "KNOWN-1",
                "media": "CD",
            },
        )
        self.assertEqual(track.discogs_release_id, "123")
        self.assertEqual(track.catalog_number, "KNOWN-1")
        self.assertEqual(track.media, "CD")
        self.assertNotIn("discogs_release_id", track.metadata_conflicts)

    def test_discogs_does_not_search_other_albums_in_album_context(self) -> None:
        with patch.object(
            discogs_service, "search_discogs_release", return_value=None
        ) as search:
            catalog._update_discogs(
                self.recording, "fixture", None, has_album_context=True
            )
            search.assert_not_called()
            catalog._update_discogs(self.recording, "fixture", None)
            search.assert_called_once()

    def test_deezer_album_metadata_and_structure(self) -> None:
        recording = TrackInfo(
            Path("recording.flac"),
            artist="Primary",
            title="Untitled",
            isrc="USAAA2100001",
            track_number=1,
            disc_number=2,
        )
        track = {
            "title": "Found Song",
            "artist": "Primary",
            "track_position": 1,
            "disk_number": 2,
            "explicit_lyrics": True,
            "bpm": 100.15,
            "isrc": "USAAA2100001",
            "producers": "Producer",
        }
        album = {
            "title": "Found Album",
            "nb_tracks": 9,
            "genre": "Rock",
            "release_date": "2021",
            "tracks_by_disc_and_position": {
                (1, 1): {},
                **{(2, position): track for position in range(1, 9)},
            },
        }
        catalog._update_deezer(recording, album, has_album_context=True, force=True)
        self.assertEqual(
            (
                recording.album,
                recording.title,
                recording.total_tracks,
                recording.disc_number,
            ),
            ("Found Album", "Found Song", 8, 2),
        )
        self.assertEqual(
            (recording.advisory, recording.bpm, recording.isrc),
            ("Explicit", 100.2, "USAAA2100001"),
        )
        self.assertEqual(
            (recording.genre, recording.date, recording.producers),
            ("Rock", "2021", "Producer"),
        )

    def test_deezer_identity_lookup_and_context_isolation(self) -> None:
        self.recording.isrc = "USAAA2100001"
        for has_context in (False, True):
            recording = dataclasses.replace(self.recording)
            with (
                self.subTest(has_context=has_context),
                patch.object(
                    deezer_service,
                    "fetch_deezer_album_details",
                    return_value=None,
                ),
                patch.object(
                    deezer_service,
                    "fetch_deezer_track_by_isrc",
                    return_value={
                        "artist": recording.artist,
                        "title": recording.title,
                        "isrc": recording.isrc,
                        "track_position": 7,
                        "disk_number": 3,
                    },
                ) as lookup,
            ):
                catalog._update_deezer(recording, None, has_album_context=has_context)
            lookup.assert_called_once_with(recording.isrc)
            self.assertIsNone(recording.track_number)
            self.assertIsNone(recording.disc_number)

    def test_deezer_single_fallback_preserves_authoritative_credits(self) -> None:
        self.recording.musicbrainz_trackid = MBID
        self.recording.producers = "Verified Producer"
        with (
            patch.object(
                deezer_service, "fetch_deezer_album_details", return_value=None
            ),
            patch.object(
                deezer_service,
                "fetch_deezer_track_details",
                return_value={
                    "artist": self.recording.artist,
                    "title": self.recording.title,
                    "producers": "Wrong Producer",
                    "release_date": "2021",
                    "bpm": "invalid",
                },
            ),
        ):
            catalog._update_deezer(self.recording, None, force=True)
        self.assertEqual(self.recording.producers, "Verified Producer")
        self.assertEqual(self.recording.date, "2021")
        self.assertIsNone(self.recording.bpm)

    def test_genius_credits_and_missing_rating(self) -> None:
        with patch.object(
            genius_service,
            "fetch_genius_song_details",
            return_value={
                "writers": "Writer",
                "producers": "Producer",
                "release_date": "2020",
            },
        ):
            catalog._update_genius(self.recording, "fixture")
        self.assertEqual(
            (
                self.recording.composer,
                self.recording.lyricist,
                self.recording.producers,
            ),
            ("Writer", "Writer", "Producer"),
        )
        with patch.object(
            theaudiodb_service,
            "fetch_theaudiodb_track_details",
            return_value={
                "rating": "8.5",
                "initial_key": "Am",
                "music_video_url": "https://example.com",
                "mood": "Chill",
                "genre": "rock",
                "description": "Description",
            },
        ):
            catalog._update_theaudiodb(self.recording)
        self.assertEqual(
            (self.recording.rating, self.recording.initial_key, self.recording.genre),
            (8.5, "Am", "Rock"),
        )

    def test_optional_services_skip_unnecessary_requests_and_empty_matches(
        self,
    ) -> None:
        self.recording.composer = "Composer"
        self.recording.producers = "Producer"
        with patch.object(
            genius_service, "fetch_genius_song_details", return_value=None
        ) as genius:
            catalog._update_genius(self.recording, "fixture")
            genius.assert_called_once()
        self.recording.music_video_url = "https://example.com"
        self.recording.mood = "Chill"
        self.recording.initial_key = "Am"
        self.recording.rating = 8.5
        self.recording.comment = "Description"
        with patch.object(
            theaudiodb_service,
            "fetch_theaudiodb_track_details",
            return_value=None,
        ) as lookup:
            catalog._update_theaudiodb(self.recording)
            lookup.assert_not_called()
            catalog._update_theaudiodb(self.recording, force=True)
            lookup.assert_called_once()
        self.recording.rating = None
        with patch.object(
            theaudiodb_service,
            "fetch_theaudiodb_track_details",
            return_value={"rating": "invalid"},
        ):
            catalog._update_theaudiodb(self.recording)
        self.assertIsNone(self.recording.rating)

    def test_provider_network_errors_leave_tags_unchanged(self) -> None:
        providers: tuple[tuple[ModuleType, str, Callable[[TrackInfo], None]], ...] = (
            (
                itunes_service,
                "fetch_itunes_track_metadata",
                catalog._update_itunes,
            ),
            (
                lastfm_service,
                "fetch_lastfm_tags",
                partial(catalog._update_lastfm, lastfm_api_key="fixture"),
            ),
            (
                discogs_service,
                "search_discogs_release",
                partial(
                    catalog._update_discogs,
                    discogs_user_token="fixture",
                    album_discogs_release=None,
                ),
            ),
            (
                deezer_service,
                "fetch_deezer_album_details",
                partial(catalog._update_deezer, album_deezer_details=None),
            ),
            (
                genius_service,
                "fetch_genius_song_details",
                partial(catalog._update_genius, genius_api_token="fixture"),
            ),
            (
                theaudiodb_service,
                "fetch_theaudiodb_track_details",
                catalog._update_theaudiodb,
            ),
        )
        for service, name, enrich in providers:
            recording = dataclasses.replace(self.recording)
            with (
                self.subTest(provider=name),
                patch.object(
                    service, name, side_effect=httpx.ConnectTimeout("offline")
                ),
            ):
                enrich(recording)
            self.assertEqual(recording, self.recording)
        with (
            patch.object(
                catalog, "get_config", return_value=SonoraConfig(enable_shazam=True)
            ),
            patch.object(
                shazam_service,
                "recognize_audio_track",
                side_effect=OSError("decode failed"),
            ),
        ):
            catalog._update_shazam(self.recording, self.recording.file_path, force=True)
        self.assertEqual(self.recording.title, "Song")
