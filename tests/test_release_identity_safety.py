from pathlib import Path
from unittest.mock import patch

from sonora.core.models import TrackInfo
from sonora.modules import musicbrainz_enrichment
from sonora.services import artist_resolution, musicbrainz
from tests.support import IsolatedTestCase

RECORDING_ID = "11111111-1111-4111-8111-111111111111"
OTHER_RECORDING_ID = "22222222-2222-4222-8222-222222222222"
RELEASE_ID = "33333333-3333-4333-8333-333333333333"
RELEASE_GROUP_ID = "44444444-4444-4444-8444-444444444444"


class TestReleaseIdentitySafety(IsolatedTestCase):
    def test_clean_album_recording_requires_local_edition_evidence(self) -> None:
        for force in (False, True):
            for advisory in (None, "Explicit"):
                with self.subTest(force=force, advisory=advisory):
                    track = TrackInfo(
                        Path("01 - Song.flac"),
                        artist="Performer",
                        title="Song",
                        album="Album",
                        track_number=1,
                        duration=200,
                        advisory=advisory,
                        musicbrainz_trackid=RECORDING_ID,
                    )
                    recording = {
                        "artist": "Performer",
                        "title": "Song",
                        "duration": 200,
                        "recording_mbid": RECORDING_ID,
                        "disambiguation": "clean",
                    }
                    release = {
                        "title": "Album",
                        "album_artist": "Performer",
                        "tracks_by_disc_and_position": {(1, 1): recording},
                        "tracks_by_mbid": {RECORDING_ID: recording},
                    }
                    with patch.object(
                        musicbrainz, "fetch_musicbrainz_recording_details"
                    ) as fetch:
                        musicbrainz_enrichment._enrich_musicbrainz(
                            track,
                            RELEASE_ID,
                            {(1, 1): RECORDING_ID},
                            release,
                            force=force,
                        )
                    fetch.assert_not_called()
                    self.assertEqual(track.musicbrainz_trackid, RECORDING_ID)
                    self.assertIsNone(track.musicbrainz_albumid)
                    self.assertIsNone(track.disambiguation)
                    self.assertEqual(track.advisory, advisory)
                    self.assertIn("identity", track.metadata_conflicts)
                    self.assertNotIn("musicbrainz_trackid", track.metadata_sources)

    def test_acoustid_clean_result_requires_edition_evidence(self) -> None:
        track = TrackInfo(
            Path("Song.flac"), artist="Performer", title="Song", duration=200
        )
        with (
            patch.object(
                musicbrainz_enrichment.acoustid_service,
                "lookup_acoustid",
                return_value=RECORDING_ID,
            ),
            patch.object(
                musicbrainz,
                "fetch_musicbrainz_recording_details",
                return_value={
                    "artist": "Performer",
                    "title": "Song",
                    "duration": 200,
                    "disambiguation": "clean",
                },
            ),
        ):
            musicbrainz_enrichment._enrich_acoustid(track, track.file_path, "test-key")
        self.assertIsNone(track.musicbrainz_trackid)
        self.assertIn("identity", track.metadata_conflicts)

    def test_standalone_clean_recording_is_not_verified_by_stored_id(self) -> None:
        track = TrackInfo(
            Path("Song.flac"),
            artist="Performer",
            title="Song",
            duration=200,
            musicbrainz_trackid=RECORDING_ID,
        )
        with patch.object(
            musicbrainz,
            "fetch_musicbrainz_recording_details",
            return_value={
                "artist": "Performer",
                "title": "Song",
                "duration": 200,
                "disambiguation": "clean",
            },
        ):
            self.assertIsNone(musicbrainz_enrichment._fetch_standalone_recording(track))
        self.assertEqual(track.musicbrainz_trackid, RECORDING_ID)
        self.assertIn("identity", track.metadata_conflicts)
        self.assertNotIn("musicbrainz_trackid", track.metadata_sources)

    def test_edition_guard_preserves_supported_clean_and_uncensored_recordings(
        self,
    ) -> None:
        for local, advisory, candidate, conflict in (
            ("clean", None, "cleaned version", False),
            ("clean", None, "explicit", True),
            (None, "Explicit", "uncensored", False),
            (None, None, "studio", False),
            ("clean", "Explicit", "clean", True),
        ):
            with self.subTest(local=local, advisory=advisory, candidate=candidate):
                track = TrackInfo(
                    Path("Song.flac"), disambiguation=local, advisory=advisory
                )
                self.assertEqual(
                    musicbrainz_enrichment._release_edition_conflicts(
                        track, {"disambiguation": candidate}
                    ),
                    conflict,
                )

    def test_base_release_search_cannot_assign_deluxe_identifiers_or_dates(
        self,
    ) -> None:
        track = TrackInfo(Path("Song.flac"), artist="Performer", album="Album (Deluxe)")
        with patch.object(
            musicbrainz,
            "search_musicbrainz_release",
            return_value={
                "id": RELEASE_ID,
                "title": "Album",
                "date": "2016-11-25",
            },
        ):
            musicbrainz_enrichment._resolve_musicbrainz_album_id(track, None, None)
        self.assertIsNone(track.musicbrainz_albumid)
        self.assertIsNone(track.date)
        for title, expected in (
            ("Album", False),
            ("Album (Deluxe)", True),
            ("Album (Deluxe Edition)", True),
            ("Album (Live)", False),
        ):
            with self.subTest(title=title):
                self.assertEqual(
                    musicbrainz_enrichment._validate_mb_release_title(
                        title, track.album
                    ),
                    expected,
                )

    def test_clean_qualifier_is_evidence_but_song_title_word_is_not(self) -> None:
        for title, album, expected in (
            ("Song (Clean)", "Album", False),
            ("Song", "Album [Clean Version]", False),
            ("Clean", "Album", True),
        ):
            with self.subTest(title=title, album=album):
                track = TrackInfo(Path("Song.flac"), title=title, album=album)
                self.assertEqual(
                    musicbrainz_enrichment._release_edition_conflicts(
                        track, {"disambiguation": "clean"}
                    ),
                    expected,
                )

    def test_release_disambiguation_can_establish_deluxe_edition(self) -> None:
        track = TrackInfo(Path("Song.flac"), artist="Primary", album="Album (Deluxe)")
        with patch.object(
            musicbrainz,
            "search_musicbrainz_release",
            return_value={
                "id": RELEASE_ID,
                "title": "Album",
                "disambiguation": "deluxe, explicit",
            },
        ):
            musicbrainz_enrichment._resolve_musicbrainz_album_id(track, None, None)
        self.assertEqual(track.musicbrainz_albumid, RELEASE_ID)
        self.assertFalse(
            musicbrainz_enrichment._validate_mb_release_title(
                "Album",
                "Album (Deluxe)",
                "Japanese bonus track",
            )
        )

    def test_album_slot_cannot_replace_verified_or_unavailable_stored_recording(
        self,
    ) -> None:
        for stored_details in (
            {"artist": "Performer", "title": "Introduction", "duration": 103},
            None,
        ):
            with self.subTest(stored_details=stored_details):
                track = TrackInfo(
                    Path("01 - Introduction.flac"),
                    artist="Performer",
                    title="Introduction",
                    album="Album",
                    album_artist="Performer",
                    track_number=1,
                    duration=103,
                    musicbrainz_trackid=RECORDING_ID,
                    musicbrainz_albumid=RELEASE_ID,
                )
                candidate = {
                    "artist": "Performer",
                    "title": "Introduction",
                    "duration": 103,
                    "recording_mbid": OTHER_RECORDING_ID,
                }
                release = {
                    "title": "Album",
                    "album_artist": "Performer",
                    "tracks_by_disc_and_position": {(1, 1): candidate},
                    "tracks_by_mbid": {OTHER_RECORDING_ID: candidate},
                }
                with (
                    patch.object(
                        musicbrainz,
                        "fetch_musicbrainz_recording_details",
                        return_value=stored_details,
                    ),
                    patch.object(musicbrainz, "fetch_track_mbid") as search,
                    patch.object(
                        musicbrainz, "search_musicbrainz_release"
                    ) as release_search,
                    patch.object(
                        artist_resolution,
                        "resolve_artist_name",
                        return_value="Performer",
                    ),
                ):
                    musicbrainz_enrichment._enrich_musicbrainz(
                        track,
                        RELEASE_ID,
                        {(1, 1): OTHER_RECORDING_ID},
                        release,
                        force=True,
                    )
                self.assertEqual(track.musicbrainz_trackid, RECORDING_ID)
                self.assertIsNone(track.musicbrainz_albumid)
                self.assertIn("musicbrainz_albumid", track.metadata_conflicts)
                search.assert_not_called()
                release_search.assert_not_called()

    def test_force_preserves_full_edition_beyond_fuzzy_base_title(self) -> None:
        base_title = "A Long Summer Evening With The Orchestra"
        for edition in ("Deluxe", "Live", "Remixes", "Radio Edit"):
            with self.subTest(edition=edition):
                track = TrackInfo(
                    Path("recording.flac"),
                    artist="Performer",
                    album_artist="Performer",
                    album=f"{base_title} ({edition})",
                    musicbrainz_albumid=RELEASE_ID,
                )
                with patch.object(
                    artist_resolution, "resolve_artist_name", return_value="Performer"
                ):
                    musicbrainz_enrichment._apply_mb_release_metadata(
                        track,
                        RELEASE_ID,
                        {"title": base_title, "album_artist": "Performer"},
                        force=True,
                    )
                self.assertEqual(track.album, f"{base_title} ({edition})")

    def test_force_validates_stored_recording_before_searching_same_title(self) -> None:
        track = TrackInfo(
            Path("01 - Introduction.flac"),
            artist="Performer",
            title="Introduction",
            album="Second Album",
            duration=103,
            musicbrainz_trackid=RECORDING_ID,
        )
        recording = {
            "artist": "Performer",
            "title": "Introduction",
            "duration": 103,
        }
        with (
            patch.object(
                musicbrainz, "fetch_track_mbid", return_value=OTHER_RECORDING_ID
            ) as search,
            patch.object(
                musicbrainz,
                "fetch_musicbrainz_recording_details",
                return_value=recording,
            ) as fetch,
        ):
            musicbrainz_enrichment._resolve_musicbrainz_track_id(track, None, None)
            self.assertEqual(
                musicbrainz_enrichment._fetch_standalone_recording(track, force=True),
                recording,
            )
        search.assert_not_called()
        fetch.assert_called_once_with(RECORDING_ID)
        self.assertEqual(track.musicbrainz_trackid, RECORDING_ID)

    def test_release_with_same_title_other_recording_does_not_match_verified_id(
        self,
    ) -> None:
        track = TrackInfo(
            Path("01 - Introduction.flac"),
            artist="Performer",
            title="Introduction",
            album="Album",
            album_artist="Performer",
            duration=103,
            musicbrainz_trackid=RECORDING_ID,
            musicbrainz_albumid=RELEASE_ID,
            musicbrainz_releasegroupid=RELEASE_GROUP_ID,
            metadata_sources={
                "musicbrainz_trackid": "MusicBrainz",
                "musicbrainz_albumid": "MusicBrainz",
                "musicbrainz_releasegroupid": "MusicBrainz",
            },
        )
        release = {
            "title": "Album",
            "album_artist": "Performer",
            "musicbrainz_releasegroupid": RELEASE_GROUP_ID,
            "tracks_by_mbid": {
                OTHER_RECORDING_ID: {
                    "artist": "Performer",
                    "title": "Introduction",
                    "duration": 103,
                }
            },
        }
        with patch.object(
            artist_resolution, "resolve_artist_name", return_value="Performer"
        ):
            musicbrainz_enrichment._apply_mb_release_metadata(
                track, RELEASE_ID, release, force=True
            )
        self.assertEqual(track.musicbrainz_trackid, RECORDING_ID)
        self.assertEqual(track.metadata_sources["musicbrainz_trackid"], "MusicBrainz")
        self.assertIsNone(track.musicbrainz_albumid)
        self.assertIsNone(track.musicbrainz_releasegroupid)
        self.assertNotIn("musicbrainz_albumid", track.metadata_sources)
        self.assertNotIn("musicbrainz_releasegroupid", track.metadata_sources)
        self.assertIn("musicbrainz_albumid", track.metadata_conflicts)

    def test_membership_failure_preserves_verified_recording_even_without_force(
        self,
    ) -> None:
        track = TrackInfo(
            Path("recording.flac"),
            artist="Performer",
            title="Original Song",
            album="Album",
            album_artist="Performer",
            musicbrainz_trackid=RECORDING_ID,
            musicbrainz_albumid=RELEASE_ID,
            metadata_sources={"musicbrainz_trackid": "AcoustID"},
        )
        musicbrainz_enrichment._apply_mb_release_metadata(
            track,
            RELEASE_ID,
            {
                "title": "Album",
                "album_artist": "Performer",
                "tracks_by_mbid": {
                    OTHER_RECORDING_ID: {"artist": "Performer", "title": "Other Song"}
                },
            },
        )
        self.assertEqual(track.musicbrainz_trackid, RECORDING_ID)
        self.assertIsNone(track.musicbrainz_albumid)
