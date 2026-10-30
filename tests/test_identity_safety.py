import dataclasses
from pathlib import Path
from unittest.mock import patch

import httpx

from sonora.audio import metadata
from sonora.core import artists
from sonora.core.config import SonoraConfig
from sonora.core.http import SESSION
from sonora.core.models import CheckReport, TrackInfo
from sonora.modules import catalog_metadata, checker, matching, tagger
from sonora.modules.metadata_merge import (
    build_provider_proposal,
    merge_provider_proposals,
)
from sonora.modules.normalization import normalize_track_metadata
from sonora.services import (
    acoustid,
    artist_resolution,
    itunes,
    lyrics,
    musicbrainz,
    theaudiodb,
)
from sonora.services.itunes import _parse_itunes_track
from tests.support import IsolatedTestCase

MBID = "11111111-2222-3333-4444-555555555555"
OTHER_MBID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


class TestRecordingSafety(IsolatedTestCase):
    def test_numeric_song_names_are_preserved_and_do_not_match_other_numbers(
        self,
    ) -> None:
        for title in ("23", "777", "007", "2020", "123 [FLAC]"):
            with self.subTest(title=title):
                self.assertFalse(matching._is_generic_title(title))
                self.assertTrue(
                    matching._is_strong_track_match("Alpha", title, "Alpha", title)
                )
                self.assertFalse(
                    matching._is_strong_track_match("Alpha", title, "Alpha", "9999")
                )

    def test_matching_isrc_cannot_override_a_meaningful_catalog_title(self) -> None:
        correct = {"title": "Original Song", "artist": "Alpha", "isrc": "USAAA2400001"}
        wrong = {**correct, "title": "Different Song"}
        self.assertEqual(
            matching._find_track_in_album_mapping(
                {"original song": correct},
                {(1, 1): wrong},
                "Original Song",
                1,
                disc_number=1,
                artist="Alpha",
                isrc="USAAA2400001",
            ),
            (correct, True),
        )
        self.assertEqual(
            matching._find_track_in_album_mapping(
                {},
                {(1, 1): wrong},
                "Original Song",
                1,
                disc_number=1,
                artist="Alpha",
                isrc="USAAA2400001",
            ),
            (None, False),
        )

    def test_itunes_direct_lookup_validates_identity_before_returning_metadata(
        self,
    ) -> None:
        for name, length, expected in (
            ("Original Song", 180000, "123"),
            ("Different Song", 180000, None),
            ("Original Song", 90000, None),
        ):
            payload = {
                "results": [
                    {
                        "trackId": 123,
                        "trackName": name,
                        "artistName": "Alpha",
                        "trackTimeMillis": length,
                    }
                ]
            }
            with (
                self.subTest(name=name, length=length),
                patch.object(itunes, "get_cached_api", return_value=None),
                patch.object(itunes, "set_cached_api"),
                patch.object(
                    SESSION,
                    "get",
                    return_value=httpx.Response(
                        200,
                        json=payload,
                        request=httpx.Request("GET", "https://itunes.apple.com/lookup"),
                    ),
                ) as request,
                patch.object(itunes, "search_itunes", return_value=[]),
            ):
                metadata = itunes.fetch_itunes_track_metadata(
                    "Alpha", "Original Song", track_id="123", duration=180
                )
            self.assertEqual(
                metadata.get("itunes_trackid") if metadata else None, expected
            )
            self.assertEqual(request.call_args.kwargs["params"]["id"], "123")

    def test_conflicting_stored_itunes_id_is_cleared_only_with_provider_evidence(
        self,
    ) -> None:
        for response, expected_id in (
            ({"artistName": "Alpha", "trackName": "Different Song"}, None),
            ({"artistName": "Alpha", "trackName": "Original Song"}, "123"),
            (None, "123"),
        ):
            track = TrackInfo(
                file_path=Path("recording.flac"),
                artist="Alpha",
                title="Original Song",
                itunes_trackid="123",
                itunes_collectionid="456",
            )
            with (
                self.subTest(response=response),
                patch.object(
                    itunes,
                    "lookup_itunes_track",
                    return_value=_parse_itunes_track(response)
                    if response is not None
                    else None,
                ),
                patch.object(itunes, "fetch_itunes_track_metadata", return_value=None),
            ):
                catalog_metadata._update_itunes(track, force=True)
            self.assertEqual(track.itunes_trackid, expected_id)
            self.assertEqual(track.itunes_collectionid, "456" if expected_id else None)

    def test_itunes_search_abstains_from_equally_supported_recordings(self) -> None:
        candidates = [
            {"trackId": identifier, "trackName": "Original Song", "artistName": "Alpha"}
            for identifier in (123, 456)
        ]
        with patch.object(itunes, "search_itunes", return_value=candidates):
            self.assertIsNone(
                itunes.fetch_itunes_track_metadata("Alpha", "Original Song")
            )
        with patch.object(itunes, "search_itunes", return_value=candidates[:1]):
            metadata = itunes.fetch_itunes_track_metadata("Alpha", "Original Song")
        assert metadata is not None
        self.assertEqual(metadata["itunes_trackid"], "123")

    def test_acoustic_near_ties_and_wrong_durations_do_not_select_an_id(self) -> None:
        for scores, length, expected in (
            ((0.99, 0.98), 180, None),
            ((0.99, 0.90), 180, MBID),
            ((0.99, 0.90), 90, None),
        ):
            payload = {
                "status": "ok",
                "results": [
                    {
                        "score": score,
                        "recordings": [
                            {
                                "id": identifier,
                                "title": "Original Song",
                                "artists": [{"name": "Alpha"}],
                                "duration": length,
                            }
                        ],
                    }
                    for score, identifier in zip(scores, (MBID, OTHER_MBID))
                ],
            }
            with (
                self.subTest(scores=scores, length=length),
                patch.object(
                    acoustid,
                    "fingerprint_audio_file",
                    return_value=(180, "fingerprint"),
                ),
                patch.object(acoustid, "get_cached_api", return_value=None),
                patch.object(acoustid, "set_cached_api"),
                patch.object(acoustid._ACOUSTID_LIMITER, "wait"),
                patch("acoustid.lookup", return_value=payload) as lookup,
            ):
                self.assertEqual(
                    acoustid.lookup_acoustid(
                        Path("recording.flac"), "fixture", "Alpha", "Original Song"
                    ),
                    expected,
                )
            self.assertEqual(lookup.call_args.args[2], 180)
            self.assertEqual(lookup.call_args.kwargs["meta"], ["recordings", "isrcs"])

    def test_theaudiodb_validates_recording_id_text_and_duration(self) -> None:
        for title, length, identifier, expected in (
            ("Original Song", 180000, MBID, "Rock"),
            ("Wrong Song", 180000, MBID, None),
            ("Original Song", 90000, MBID, None),
            ("Original Song", 180000, OTHER_MBID, None),
        ):
            raw_track = {
                "strArtist": "Alpha",
                "strTrack": title,
                "strMusicBrainzID": identifier,
                "intDuration": length,
                "strGenre": "Rock",
            }
            with (
                self.subTest(title=title, length=length, identifier=identifier),
                patch.object(theaudiodb, "get_cached_api", return_value=None),
                patch.object(theaudiodb, "set_cached_api"),
                patch.object(
                    SESSION,
                    "get",
                    side_effect=[
                        httpx.Response(200, json={"track": [raw_track]}),
                        httpx.Response(200, json={"track": None}),
                    ],
                ) as request,
            ):
                metadata = theaudiodb.fetch_theaudiodb_track_details(
                    "Alpha", "Original Song", recording_mbid=MBID, duration=180
                )
            self.assertEqual(metadata.get("genre") if metadata else None, expected)
            self.assertIn("track-mb.php", request.call_args_list[0].args[0])

    def test_check_flags_shared_ids_on_different_songs_and_accepts_repeat_recordings(
        self,
    ) -> None:
        for second_title, expected_issues in (
            ("Original Song", 0),
            ("Different Song", 2),
        ):
            first = TrackInfo(
                Path("first.flac"),
                artist="Alpha",
                title="Original Song",
                itunes_trackid="123",
                duration=180,
            )
            second = TrackInfo(
                Path("second.flac"),
                artist="Alpha",
                title=second_title,
                itunes_trackid="123",
                duration=180,
            )
            report = CheckReport()
            checker._check_identifier_conflicts(
                {Path("album"): [(first.file_path, first), (second.file_path, second)]},
                report,
            )
            self.assertEqual(len(report.issues), expected_issues)
        report = CheckReport()
        checker._check_identifier_conflicts({}, report)
        self.assertEqual(report.issues, {})

    def test_unverified_lyrics_fallback_requires_opt_in(self) -> None:
        for enabled in (False, True):
            with (
                self.subTest(enabled=enabled),
                patch.object(
                    lyrics,
                    "get_config",
                    return_value=SonoraConfig(allow_unverified_lyrics=enabled),
                ),
                patch.object(lyrics, "_query_lrclib", return_value=None),
                patch.object(lyrics._LYRICS_LIMITER, "wait"),
                patch(
                    "syncedlyrics.search", return_value="[00:01.00] Lyrics"
                ) as search,
            ):
                self.assertEqual(
                    lyrics._query_syncedlyrics(
                        "Alpha - Original Song", False, False, False, None, None
                    ),
                    "[00:01.00] Lyrics" if enabled else None,
                )
            self.assertEqual(search.call_count, int(enabled))

    def test_release_verification_requires_every_track_and_unique_assignment(
        self,
    ) -> None:
        paths = [Path("first.flac"), Path("second.flac")]
        originals = {
            paths[0]: TrackInfo(
                paths[0], artist="Alpha", title="First Song", track_number=1
            ),
            paths[1]: TrackInfo(
                paths[1], artist="Alpha", title="Second Song", track_number=2
            ),
        }
        for second_title, expected in (
            ("Second Song", True),
            ("Different Song", False),
        ):
            release = {
                "tracks_by_disc_and_position": {
                    (1, 1): {"artist": "Alpha", "title": "First Song"},
                    (1, 2): {"artist": "Alpha", "title": second_title},
                }
            }
            with patch.object(
                metadata,
                "read_track_metadata",
                side_effect=originals.__getitem__,
            ):
                self.assertEqual(
                    tagger._release_matches_album_tracks(release, paths), expected
                )
                self.assertFalse(
                    tagger._release_matches_album_tracks(release, paths[:1])
                )
        self.assertFalse(tagger._release_matches_album_tracks({}, paths))

    def test_cached_acoustic_evidence_is_rechecked_against_current_policy(self) -> None:
        payload = {
            "status": "ok",
            "results": [
                {
                    "score": 0.95,
                    "recordings": [
                        {
                            "id": MBID,
                            "title": "Original Song",
                            "artists": [{"name": "Alpha"}],
                            "duration": 180,
                        }
                    ],
                }
            ],
        }
        for threshold, expected in ((0.8, MBID), (0.99, None)):
            with (
                self.subTest(threshold=threshold),
                patch.object(
                    acoustid,
                    "get_config",
                    return_value=SonoraConfig(acoustid_confidence_threshold=threshold),
                ),
                patch.object(
                    acoustid,
                    "fingerprint_audio_file",
                    return_value=(180, "fingerprint"),
                ),
                patch.object(acoustid, "get_cached_api", return_value=payload),
                patch.object(acoustid, "set_cached_api"),
                patch("acoustid.lookup") as request,
            ):
                self.assertEqual(
                    acoustid.lookup_acoustid(
                        Path("recording.flac"), "fixture", "Alpha", "Original Song"
                    ),
                    expected,
                )
            request.assert_not_called()


class TestProviderConflictHandling(IsolatedTestCase):
    def test_provider_conflict_preserves_disputed_field_and_allows_other_evidence(
        self,
    ) -> None:
        track = TrackInfo(
            Path("recording.flac"), artist="Artist", title="Song", total_tracks=14
        )
        disputed = dataclasses.replace(
            track,
            total_tracks=12,
            metadata_conflicts={"total_tracks": "Release count excludes this track"},
        )
        agreed = dataclasses.replace(
            track, genre="Pop", metadata_sources={"genre": "iTunes"}
        )
        merge_provider_proposals(
            track,
            [
                build_provider_proposal(track, candidate)
                for candidate in [disputed, agreed]
            ],
        )
        self.assertEqual(track.total_tracks, 14)
        self.assertEqual(
            track.metadata_conflicts,
            {"total_tracks": "Release count excludes this track"},
        )
        self.assertEqual(track.genre, "Pop")
        self.assertEqual(track.metadata_sources.get("genre"), "iTunes")

    def test_rejected_recording_cannot_inject_field_conflicts(self) -> None:
        track = TrackInfo(Path("recording.flac"), artist="Artist", title="Song")
        unrelated = dataclasses.replace(
            track,
            title="Entirely Different Recording",
            metadata_conflicts={"genre": "Unrelated conflict"},
        )
        merge_provider_proposals(
            track,
            [build_provider_proposal(track, unrelated)],
        )
        self.assertIn("identity", track.metadata_conflicts)
        self.assertNotIn("genre", track.metadata_conflicts)

    def test_existing_conflict_cannot_be_overwritten_by_a_later_provider(self) -> None:
        track = TrackInfo(
            Path("recording.flac"),
            artist="Artist",
            title="Song",
            genre="Jazz",
            metadata_conflicts={"genre": "Album providers disagree"},
        )
        proposal = dataclasses.replace(
            track, genre="Pop", metadata_sources={"genre": "iTunes"}
        )
        merge_provider_proposals(
            track,
            [build_provider_proposal(track, proposal)],
        )
        self.assertEqual(track.genre, "Jazz")
        self.assertNotIn("genre", track.metadata_sources)


class TestOfflineArtistNormalization(IsolatedTestCase):
    def test_normalization_preserves_configured_groups_in_artist_and_title(
        self,
    ) -> None:
        policy = SonoraConfig(known_group_artists=frozenset({"Example & Partners"}))
        for artist, title in (
            ("Primary feat. Example & Partners", "Recording"),
            ("Primary", "Recording (feat. Example & Partners)"),
            ("Primary", "Recording feat. Example & Partners"),
        ):
            with (
                self.subTest(artist=artist, title=title),
                patch.object(artists, "get_config", return_value=policy),
            ):
                recording = TrackInfo(Path("song.flac"), artist=artist, title=title)
                normalize_track_metadata(recording)
                self.assertEqual(recording.artist, "Primary")
                self.assertEqual(recording.title, "Recording")
                self.assertEqual(recording.featured_artists, "Example & Partners")

    def test_online_artist_resolution_uses_local_alias_before_providers(self) -> None:
        with (
            patch.object(
                artists, "_load_user_overrides", return_value={"legal name": "Stage"}
            ),
            patch.object(
                artist_resolution,
                "_load_user_overrides",
                return_value={"legal name": "Stage"},
            ),
            patch.object(artist_resolution, "get_cached_api") as cache_lookup,
            patch.object(SESSION, "get") as http_request,
            patch.object(musicbrainz, "search_musicbrainz_artists") as artist_search,
        ):
            self.assertEqual(
                artist_resolution.resolve_artist_name("Legal Name"), "Stage"
            )
            cache_lookup.assert_not_called()
            http_request.assert_not_called()
            artist_search.assert_not_called()

    def test_offline_artist_functions_ignore_external_identity_cache(self) -> None:
        for cached_identity in ("Unrelated Artist", True, False):
            with (
                self.subTest(cached_identity=cached_identity),
                patch.object(
                    artist_resolution, "get_cached_api", return_value=cached_identity
                ) as cache_lookup,
            ):
                self.assertEqual(artists.normalize_artist_name("Example"), "Example")
                self.assertEqual(
                    artists.get_primary_artist("Example & Partners"), "Example"
                )
                self.assertFalse(artists.is_single_group_artist("Example & Partners"))
                cache_lookup.assert_not_called()

    def test_configured_groups_preserve_primary_artist(self) -> None:
        policy = SonoraConfig(known_group_artists=frozenset({"Example & Partners"}))
        with patch.object(artists, "get_config", return_value=policy):
            self.assertEqual(
                artists.get_primary_artist("Example & Partners"), "Example & Partners"
            )
            self.assertEqual(artists.get_primary_artist(None), "Unknown")

    def test_local_aliases_work_without_online_artist_resolution(self) -> None:
        with (
            patch.object(
                artists, "_load_user_overrides", return_value={"legal name": "Stage"}
            ),
            patch.object(
                artist_resolution, "get_cached_api", return_value="Wrong Artist"
            ) as cache_lookup,
        ):
            self.assertEqual(artists.normalize_artist_name("Legal Name"), "Stage")
            self.assertEqual(artists.get_primary_artist("Legal Name"), "Stage")
            self.assertEqual(artists.normalize_artist_name("Other Name"), "Other Name")
            cache_lookup.assert_not_called()

    def test_normalization_does_not_resolve_identity_online(self) -> None:
        recording = TrackInfo(
            Path("song.flac"), artist="Legal Name", album_artist="Legal Name"
        )
        with patch.object(artist_resolution, "resolve_artist_name") as resolver:
            normalize_track_metadata(recording)
        self.assertEqual(recording.artist, "Legal Name")
        self.assertEqual(recording.album_artist, "Legal Name")
        resolver.assert_not_called()
