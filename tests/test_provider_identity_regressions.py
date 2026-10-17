import dataclasses
import subprocess
from collections.abc import Iterator
from functools import partial
from unittest.mock import patch

import httpx

from sonora.audio.metadata import read_track_metadata, write_track_metadata
from sonora.core.http import RetryTransport, _retry_delay
from sonora.core.models import TrackInfo
from sonora.core.titles import recording_identity_matches, score_artist_title_match
from sonora.modules import catalog_enrichment, tagger
from sonora.modules.metadata_consensus import (
    collect_provider_proposal,
    merge_provider_proposals,
)
from sonora.services import acoustid, deezer, genius, itunes
from tests.regression_support import AudioRegressionTestCase


def _validate_genius_identifier(track_info: TrackInfo, **keywords: object) -> None:
    proposal = collect_provider_proposal(
        track_info,
        partial(catalog_enrichment._enrich_genius, genius_api_token="fixture"),
    )
    merge_provider_proposals(track_info, [proposal])


class Stream(httpx.SyncByteStream):
    def __init__(self) -> None:
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        yield b"response"

    def close(self) -> None:
        self.closed = True


class TestProviderIdentityRegressions(AudioRegressionTestCase):
    def test_generic_album_version_label_does_not_create_a_false_isrc_conflict(
        self,
    ) -> None:
        for title, accepted in (
            ("Original Song (Album Version)", True),
            ("Original Song [Album Version]", True),
            ("Original Song (Radio Edit)", False),
            ("Original Song (Clean Version)", False),
            ("Original Song (Album Version - Edited)", False),
        ):
            with self.subTest(title=title):
                self.assertEqual(
                    score_artist_title_match("Alpha", "Original Song", "Alpha", title)
                    > 0,
                    accepted,
                )

    def test_verified_apple_advisory_clears_stale_explicit_and_keeps_unknown(
        self,
    ) -> None:
        for explicitness, expected in (
            ("notexplicit", None),
            ("explicit", "Explicit"),
            ("", "Explicit"),
        ):
            with self.subTest(explicitness=explicitness):
                track = TrackInfo(
                    self.audio("recording.flac"),
                    artist="Alpha",
                    title="Original Song",
                    duration=1.0,
                    advisory="Explicit",
                )
                with patch.object(
                    itunes,
                    "fetch_itunes_track_metadata",
                    return_value={
                        "artist": "Alpha",
                        "title": "Original Song",
                        "duration": 1.0,
                        "itunes_trackid": "123",
                        "explicitness": explicitness,
                    },
                ):
                    catalog_enrichment._enrich_itunes(track, force=True)
                self.assertEqual(track.advisory, expected)
                if explicitness:
                    self.assertEqual(track.metadata_sources["advisory"], "iTunes")

    def test_deezer_advisory_requires_boolean_evidence(self) -> None:
        for explicit_lyrics, expected in (
            (False, None),
            (True, "Explicit"),
            ("false", "Explicit"),
            (None, "Explicit"),
        ):
            with self.subTest(explicit_lyrics=explicit_lyrics):
                track = TrackInfo(
                    self.audio("recording.flac"),
                    artist="Alpha",
                    title="Original Song",
                    duration=1.0,
                    advisory="Explicit",
                )
                with patch.object(
                    deezer,
                    "fetch_deezer_track_details",
                    return_value={
                        "artist": "Alpha",
                        "title": "Original Song",
                        "duration": 1.0,
                        "explicit_lyrics": explicit_lyrics,
                    },
                ):
                    catalog_enrichment._enrich_deezer(
                        track, None, has_album_context=True, force=True
                    )
                self.assertEqual(track.advisory, expected)

    def test_advisory_provider_disagreement_preserves_existing_rating(self) -> None:
        track = TrackInfo(
            self.audio("recording.flac"),
            artist="Alpha",
            title="Original Song",
            advisory="Explicit",
        )
        explicit = dataclasses.replace(
            track, metadata_sources={"advisory": "iTunes"}, metadata_conflicts={}
        )
        nonexplicit = dataclasses.replace(
            track,
            advisory=None,
            metadata_sources={"advisory": "Deezer"},
            metadata_conflicts={},
        )
        merge_provider_proposals(track, [explicit, nonexplicit])
        self.assertEqual(track.advisory, "Explicit")
        self.assertIn("advisory", track.metadata_conflicts)
        agreed = dataclasses.replace(track, metadata_conflicts={})
        merge_provider_proposals(
            agreed,
            [
                nonexplicit,
                dataclasses.replace(
                    nonexplicit, metadata_sources={"advisory": "iTunes"}
                ),
            ],
        )
        self.assertIsNone(agreed.advisory)

    def test_retries_close_discarded_streams(self) -> None:
        streams = [Stream(), Stream(), Stream()]
        responses = [
            httpx.Response(status, stream=stream)
            for status, stream in zip((503, 429, 200), streams)
        ]
        transport = RetryTransport(backoff_factor=0)
        with patch("httpx.HTTPTransport.handle_request", side_effect=responses):
            response = transport.handle_request(
                httpx.Request("GET", "https://example.invalid/")
            )
        response.close()
        transport.close()
        self.assertTrue(all(stream.closed for stream in streams))

    def test_retry_after_delay_not_truncated(self) -> None:
        for header, expected in (("60", 60.0), ("invalid", 1.5), ("-1", 0.0)):
            with self.subTest(header=header):
                self.assertEqual(_retry_delay(header, 1.5), expected)

    def test_fingerprint_timeout_does_not_restart_decoder(self) -> None:
        with (
            patch(
                "sonora.services.acoustid.subprocess.run",
                side_effect=subprocess.TimeoutExpired("fpcalc", 20),
            ),
            patch("sonora.services.acoustid.acoustid.fingerprint_file") as unbounded,
            self.assertRaises(OSError),
        ):
            acoustid._fingerprint_file_with_timeout(self.root / "track.flac")
        unbounded.assert_not_called()

    def test_deezer_identity_rejection_precedes_release_bonuses(self) -> None:
        candidate = {
            "title": "Same Album",
            "artist": {"name": "Unrelated"},
            "id": 123,
            "nb_tracks": 10,
        }
        response = httpx.Response(
            200,
            json={"data": [candidate]},
            request=httpx.Request("GET", "https://example.invalid/"),
        )
        with (
            patch.object(deezer, "get_cached_api", return_value=None),
            patch(
                "sonora.services.deezer.SESSION.get", return_value=response
            ) as request,
        ):
            self.assertIsNone(
                deezer.fetch_deezer_album_details("Original", "Same Album", 10)
            )
            self.assertEqual(request.call_count, 1)

    def test_deezer_reads_complete_paginated_album(self) -> None:
        search = {"data": [{"id": 1, "title": "Album", "artist": {"name": "Alpha"}}]}
        album = {
            "title": "Album",
            "artist": {"name": "Alpha"},
            "tracks": {
                "data": [{"id": 1, "title": "First", "track_position": 1}],
                "next": "http://api.deezer.com/album/1/tracks?index=1",
            },
        }
        page = {"data": [{"id": 2, "title": "Second", "track_position": 2}]}
        replies = [
            httpx.Response(
                200,
                json=payload,
                request=httpx.Request("GET", "https://api.deezer.com/album/1"),
            )
            for payload in (search, album, page)
        ]
        with patch(
            "sonora.services.deezer.SESSION.get", side_effect=replies
        ) as provider:
            details = deezer.fetch_deezer_album_details("Alpha", "Album")
        assert details is not None
        self.assertEqual(
            details["tracks_by_disc_and_position"][(1, 2)]["title"], "Second"
        )
        self.assertTrue(provider.call_args.args[0].startswith("https://"))

    def test_deezer_rejects_invalid_pagination(self) -> None:
        search = {"data": [{"id": 1, "title": "Album", "artist": {"name": "Alpha"}}]}
        for next_page in (
            "https://other.example/tracks",
            "https://api.deezer.com/album/1/tracks",
        ):
            with self.subTest(next_page=next_page):
                tracks: dict[str, object] = {"data": [], "next": next_page}
                album = {"tracks": tracks}
                replies = [
                    httpx.Response(
                        200,
                        json=payload,
                        request=httpx.Request("GET", "https://api.deezer.com/album/1"),
                    )
                    for payload in (search, album, tracks)
                ]
                with (
                    patch("sonora.services.deezer.SESSION.get", side_effect=replies),
                    self.assertRaises(ValueError),
                ):
                    deezer.fetch_deezer_album_details("Alpha", "Album")

    def test_deezer_pagination_timeout_does_not_cache_partial_album(self) -> None:
        search = {"data": [{"id": 1, "title": "Album", "artist": {"name": "Alpha"}}]}
        album = {
            "tracks": {
                "data": [{"title": "First"}],
                "next": "https://api.deezer.com/album/1/tracks",
            }
        }
        replies = [
            httpx.Response(
                200,
                json=payload,
                request=httpx.Request("GET", "https://api.deezer.com/album/1"),
            )
            for payload in (search, album)
        ]
        with (
            patch(
                "sonora.services.deezer.SESSION.get",
                side_effect=[*replies, httpx.ReadTimeout("provider timeout")],
            ),
            patch.object(deezer, "set_cached_api") as cache_writer,
        ):
            self.assertIsNone(deezer.fetch_deezer_album_details("Alpha", "Album"))
        cache_writer.assert_not_called()

    def test_genius_does_not_accept_placeholder_descriptions(self) -> None:
        search = {
            "response": {
                "hits": [
                    {
                        "result": {
                            "title": "Song",
                            "primary_artist": {"name": "Alpha"},
                            "api_path": "/songs/1",
                        }
                    }
                ]
            }
        }
        for description in ("?", " ? ", "", "Meaningful story"):
            with self.subTest(description=description):
                song = {
                    "response": {
                        "song": {
                            "id": 1,
                            "title": "Song",
                            "primary_artist": {"name": "Alpha"},
                            "description": {"plain": description},
                        }
                    }
                }
                replies = [
                    httpx.Response(
                        200,
                        json=payload,
                        request=httpx.Request("GET", "https://api.genius.com/songs/1"),
                    )
                    for payload in (search, song)
                ]
                with (
                    patch("sonora.services.genius.SESSION.get", side_effect=replies),
                    patch.object(genius, "get_cached_api", return_value=None),
                ):
                    details = genius.fetch_genius_song_details(
                        "Alpha", "Song", api_token="test-token"
                    )
                assert details is not None
                self.assertEqual(
                    details["description"],
                    "Meaningful story" if description == "Meaningful story" else None,
                )

    def test_genius_cached_description_obeys_same_quality_rules(self) -> None:
        for description in ("?", "", "Meaningful story"):
            with (
                self.subTest(description=description),
                patch.object(
                    genius,
                    "get_cached_api",
                    return_value={
                        "artist": "Alpha",
                        "title": "Song",
                        "description": description,
                    },
                ),
            ):
                details = genius.fetch_genius_song_details(
                    "Alpha", "Song", api_token="test-token"
                )
                assert details is not None
                self.assertEqual(
                    details["description"],
                    "Meaningful story" if description == "Meaningful story" else None,
                )
        with self.assertRaises(TypeError):
            genius._normalize_genius_description({"html": "unexpected format"})

    def test_reordered_album_enrichment_saves_only_matching_recording_credits(
        self,
    ) -> None:
        recording = TrackInfo(
            self.audio("recording.flac"),
            artist="Alpha",
            title="Original Song",
            track_number=2,
            disc_number=1,
            isrc="USAAA2400001",
        )
        wrong_recording = {
            "title": "Different Song",
            "isrc": "USAAA2400002",
            "composer": "Wrong Writer",
            "explicit_lyrics": True,
        }
        correct_recording = {
            "title": "Original Song",
            "isrc": "USAAA2400001",
            "composer": "Correct Writer",
            "explicit_lyrics": False,
        }
        album = {
            "tracks_by_disc_and_position": {(1, 2): wrong_recording},
            "tracks_by_title": {"original song": correct_recording},
        }
        with patch.object(deezer, "fetch_deezer_track_by_isrc") as lookup:
            catalog_enrichment._enrich_deezer(recording, album, has_album_context=True)
        lookup.assert_not_called()
        write_track_metadata(recording)
        saved_recording = read_track_metadata(recording.file_path)
        self.assertEqual(saved_recording.composer, "Correct Writer")
        self.assertEqual(saved_recording.isrc, "USAAA2400001")
        self.assertNotEqual(saved_recording.advisory, "Explicit")
        self.assertEqual(saved_recording.title, "Original Song")

    def test_unverified_album_positions_preserve_existing_recording_tags(self) -> None:
        for candidate in (
            {"title": "Original Song", "isrc": "USAAA2400002"},
            {"title": "Original Song", "artist": "Unrelated Artist"},
            {},
        ):
            with self.subTest(candidate=candidate):
                recording = TrackInfo(
                    self.audio("recording.flac"),
                    artist="Alpha",
                    title="Original Song",
                    track_number=1,
                    isrc="USAAA2400001",
                    composer="Verified Writer",
                )
                album = {
                    "tracks_by_position": {
                        1: {
                            **candidate,
                            "composer": "Wrong Writer",
                            "explicit_lyrics": True,
                        }
                    }
                }
                with (
                    patch.object(
                        deezer, "fetch_deezer_track_by_isrc", return_value=None
                    ) as lookup,
                    patch.object(
                        deezer, "fetch_deezer_track_details", return_value=None
                    ),
                ):
                    catalog_enrichment._enrich_deezer(
                        recording, album, has_album_context=True, force=True
                    )
                lookup.assert_called_once_with("USAAA2400001")
                self.assertEqual(recording.composer, "Verified Writer")
                self.assertIsNone(recording.advisory)
                self.assertEqual(recording.isrc, "USAAA2400001")

    def test_itunes_conflicting_track_id_cannot_override_title_match(self) -> None:
        recording = TrackInfo(
            self.audio("recording.flac"),
            artist="Alpha",
            title="Original Song",
            track_number=1,
            disc_number=1,
            itunes_trackid="123",
        )
        album = {
            "tracks_by_disc_and_position": {
                (1, 1): {
                    "trackName": "Original Song",
                    "trackId": 456,
                    "advisory": "Explicit",
                }
            },
            "tracks_by_title": {
                "original song": {
                    "trackName": "Original Song",
                    "trackId": 123,
                    "genre": "Jazz",
                }
            },
        }
        with patch.object(itunes, "fetch_itunes_track_metadata") as lookup:
            catalog_enrichment._enrich_itunes(recording, album, force=True)
        lookup.assert_not_called()
        self.assertEqual(recording.genre, "Jazz")
        self.assertIsNone(recording.advisory)
        self.assertEqual(recording.itunes_trackid, "123")

    def test_itunes_validated_track_repairs_associated_ids_without_force(self) -> None:
        for collection, artist_id in (("stale", "stale"), (None, None), ("22", "33")):
            with self.subTest(collection=collection, artist_id=artist_id):
                recording = TrackInfo(
                    self.audio("recording.flac"),
                    artist="Alpha",
                    title="Original Song",
                    duration=180,
                    itunes_trackid="11",
                    itunes_collectionid=collection,
                    itunes_artistid=artist_id,
                    genre="Rock",
                    advisory="Explicit",
                )
                with (
                    patch.object(
                        itunes,
                        "lookup_itunes_track",
                        return_value={
                            "trackId": 11,
                            "artistName": "Alpha",
                            "trackName": "Original Song",
                            "trackTimeMillis": 180000,
                            "collectionId": 22,
                            "artistId": 33,
                        },
                    ),
                    patch.object(itunes, "fetch_itunes_track_metadata") as search,
                ):
                    catalog_enrichment._enrich_itunes(recording)
                search.assert_not_called()
                self.assertEqual(
                    (
                        recording.itunes_trackid,
                        recording.itunes_collectionid,
                        recording.itunes_artistid,
                    ),
                    ("11", "22", "33"),
                )

    def test_itunes_rejected_recording_clears_its_artist_pointer(self) -> None:
        recording = TrackInfo(
            self.audio("recording.flac"),
            artist="Alpha",
            title="Original Song",
            itunes_trackid="11",
            itunes_collectionid="22",
            itunes_artistid="33",
            metadata_sources={"itunes_artistid": "iTunes"},
        )
        with (
            patch.object(
                itunes,
                "lookup_itunes_track",
                return_value={
                    "artistName": "Unrelated",
                    "trackName": "Different Song",
                    "trackId": 11,
                },
            ),
            patch.object(itunes, "fetch_itunes_track_metadata", return_value=None),
        ):
            catalog_enrichment._enrich_itunes(recording)
        self.assertIsNone(recording.itunes_artistid)
        self.assertNotIn("itunes_artistid", recording.metadata_sources)

    def test_genius_failed_search_validates_stored_id_without_importing_wrong_credits(
        self,
    ) -> None:
        for stored, expected in (
            (
                {
                    "artist": "Unrelated",
                    "title": "Wrong Song",
                    "writers": "Wrong Writer",
                },
                None,
            ),
            (
                {
                    "artist": "Alpha",
                    "title": "Original Song",
                    "genius_song_id": 11,
                    "writers": "Correct Writer",
                },
                "11",
            ),
            (None, "11"),
            ({"title": "Original Song"}, "11"),
        ):
            with self.subTest(stored=stored):
                recording = TrackInfo(
                    self.audio("recording.flac"),
                    artist="Alpha",
                    title="Original Song",
                    genius_song_id="11",
                    metadata_sources={"genius_song_id": "Genius"},
                )
                original = dataclasses.replace(
                    recording,
                    metadata_sources=recording.metadata_sources.copy(),
                    metadata_conflicts=recording.metadata_conflicts.copy(),
                )
                with (
                    patch.object(
                        genius, "fetch_genius_song_details", return_value=None
                    ),
                    patch.object(
                        genius, "lookup_genius_song_details", return_value=stored
                    ),
                ):
                    catalog_enrichment._enrich_genius(recording, "fixture")
                self.assertEqual(recording.genius_song_id, expected)
                if expected is None:
                    self.assertIsNone(recording.composer)
                    self.assertIn(
                        "genius_song_id_validation", recording.metadata_conflicts
                    )
                    self.assertNotIn("genius_song_id", recording.metadata_sources)
                elif stored and stored.get("writers"):
                    self.assertEqual(recording.composer, "Correct Writer")
                merge_provider_proposals(original, [recording])
                self.assertEqual(original.genius_song_id, expected)

    def test_discogs_position_cannot_supply_another_song_or_unidentified_credits(
        self,
    ) -> None:
        for positional in (
            {"title": "Different Song", "composer": "Wrong Writer"},
            {"title": "", "composer": "Wrong Writer"},
            {"composer": "Wrong Writer"},
        ):
            with self.subTest(positional=positional):
                recording = TrackInfo(
                    self.audio("recording.flac"),
                    artist="Alpha",
                    title="Original Song",
                    track_number=1,
                )
                release = {
                    "id": 1,
                    "track_credits": {
                        "1": positional,
                        "original song": {
                            "title": "Original Song",
                            "composer": "Correct Writer",
                        },
                    },
                }
                catalog_enrichment._enrich_discogs(
                    recording, "fixture", release, has_album_context=True
                )
                self.assertEqual(recording.composer, "Correct Writer")
        recording = TrackInfo(
            self.audio("recording.flac"),
            artist="Alpha",
            title="Untitled",
            track_number=1,
        )
        catalog_enrichment._enrich_discogs(
            recording,
            "fixture",
            {
                "id": 1,
                "track_credits": {
                    "1": {"title": "Different Song", "composer": "Wrong Writer"}
                },
            },
            has_album_context=True,
        )
        self.assertEqual(recording.title, "Untitled")
        self.assertIsNone(recording.composer)

    def test_feature_credits_do_not_change_numbered_title_identity(self) -> None:
        for original, featured in (
            ("X", "X (feat. Future)"),
            ("Part II", "Part II (feat. Guest)"),
            ("Volume 2", "Volume 2 feat. Guest 3"),
        ):
            with self.subTest(original=original, featured=featured):
                self.assertTrue(
                    recording_identity_matches("Alpha", original, "Alpha", featured)
                )
                self.assertTrue(
                    recording_identity_matches("Alpha", featured, "Alpha", original)
                )
        self.assertFalse(
            recording_identity_matches(
                "Alpha", "Part II", "Alpha", "Part III (feat. Guest)"
            )
        )
        self.assertFalse(
            recording_identity_matches(
                "Alpha", "Original Song", "Alpha", "Original Song (Live)"
            )
        )

    def test_deezer_isrc_contradictions_are_visible_without_clearing_unverified_values(
        self,
    ) -> None:
        for candidate in (
            {"artist": "Unrelated", "title": "Original Song", "duration": 180},
            {"artist": "Alpha", "title": "Different Song", "duration": 180},
            {"artist": "Alpha", "title": "Original Song", "duration": 90},
        ):
            with self.subTest(candidate=candidate):
                recording = TrackInfo(
                    self.audio("recording.flac"),
                    artist="Alpha",
                    title="Original Song",
                    duration=180,
                    isrc="USAAA2400001",
                )
                with (
                    patch.object(
                        deezer, "fetch_deezer_track_by_isrc", return_value=candidate
                    ),
                    patch.object(
                        deezer, "fetch_deezer_track_details", return_value=None
                    ),
                ):
                    catalog_enrichment._enrich_deezer(
                        recording, {}, has_album_context=True
                    )
                self.assertIn("isrc", recording.metadata_conflicts)
                self.assertEqual(recording.isrc, "USAAA2400001")
                self.assertIsNone(recording.composer)

    def test_censored_apple_titles_require_review_without_validating_or_deleting_id(
        self,
    ) -> None:
        for candidate_artist, duration, preserved in (
            ("21 Savage & Metro Boomin", 185467, True),
            ("Unrelated", 185467, False),
            ("21 Savage & Metro Boomin", 90000, False),
            ("21 Savage & Metro Boomin", None, False),
        ):
            with self.subTest(candidate_artist=candidate_artist, duration=duration):
                recording = TrackInfo(
                    self.audio("recording.flac"),
                    artist="21 Savage & Metro Boomin",
                    title="Real Nigga",
                    duration=185.4667,
                    itunes_trackid="1133782487",
                )
                with (
                    patch.object(
                        itunes,
                        "lookup_itunes_track",
                        return_value={
                            "trackId": 1133782487,
                            "artistName": candidate_artist,
                            "trackName": "Real Ni**a",
                            "trackTimeMillis": duration,
                        },
                    ),
                    patch.object(
                        itunes, "fetch_itunes_track_metadata", return_value=None
                    ),
                ):
                    catalog_enrichment._enrich_itunes(recording, force=True)
                self.assertEqual(
                    recording.itunes_trackid, "1133782487" if preserved else None
                )
                self.assertNotIn("itunes_trackid", recording.metadata_sources)
                if preserved:
                    self.assertIn("itunes_trackid", recording.metadata_conflicts)

    def test_confirmed_genius_rejection_survives_the_file_write_pipeline(self) -> None:
        for provider, expected in (
            ({"artist": "Unrelated", "title": "Wrong Song"}, None),
            (None, "11"),
            ({"artist": "Alpha", "title": "Original Song", "genius_song_id": 11}, "11"),
        ):
            with self.subTest(provider=provider):
                path = self.audio("recording.flac")
                write_track_metadata(
                    TrackInfo(
                        path, artist="Alpha", title="Original Song", genius_song_id="11"
                    )
                )
                with (
                    patch.object(
                        tagger,
                        "_run_metadata_enrichment_pipeline",
                        side_effect=_validate_genius_identifier,
                    ),
                    patch(
                        "sonora.services.artist_resolution.resolve_artist_name",
                        return_value="Alpha",
                    ),
                    patch.object(
                        genius, "fetch_genius_song_details", return_value=None
                    ),
                    patch.object(
                        genius, "lookup_genius_song_details", return_value=provider
                    ),
                ):
                    result = tagger.process_single_track(
                        path,
                        force=True,
                        fetch_bpm=False,
                        fetch_key=False,
                        fetch_lyrics=False,
                        fetch_itunes_art=False,
                    )
                self.assertEqual(result.genius_song_id, expected)
                self.assertEqual(read_track_metadata(path).genius_song_id, expected)
                self.assertEqual(
                    (result.artist, result.title), ("Alpha", "Original Song")
                )

    def test_clean_apple_editions_do_not_enrich_explicit_recordings(self) -> None:
        for album_context in (False, True):
            for explicitness in ("cleaned", "explicit", "notExplicit"):
                with self.subTest(
                    album_context=album_context, explicitness=explicitness
                ):
                    recording = TrackInfo(
                        self.audio("recording.flac"),
                        artist="Alpha",
                        title="Original Song",
                        duration=180,
                        advisory="Explicit",
                        itunes_trackid="11",
                    )
                    raw = {
                        "trackId": 11,
                        "artistName": "Alpha",
                        "trackName": "Original Song",
                        "trackTimeMillis": 180000,
                        "trackExplicitness": explicitness,
                        "collectionId": 22,
                        "artistId": 33,
                    }
                    normalized = {
                        "title": "Original Song",
                        "artist": "Alpha",
                        "duration": 180,
                        "itunes_trackid": "11",
                        "itunes_collectionid": "22",
                        "itunes_artistid": "33",
                        "genre": "Rock",
                        "explicitness": explicitness,
                    }
                    album = (
                        {"tracks_by_title": {"original song": normalized}}
                        if album_context
                        else None
                    )
                    with (
                        patch.object(itunes, "lookup_itunes_track", return_value=raw),
                        patch.object(
                            itunes,
                            "fetch_itunes_track_metadata",
                            return_value=normalized,
                        ),
                    ):
                        catalog_enrichment._enrich_itunes(recording, album, force=True)
                    self.assertEqual(recording.itunes_trackid, "11")
                    self.assertEqual(recording.advisory, "Explicit")
                    if explicitness == "cleaned":
                        self.assertIsNone(recording.genre)
                        self.assertIn("itunes_trackid", recording.metadata_conflicts)
                        self.assertNotIn("itunes_trackid", recording.metadata_sources)
                    else:
                        self.assertEqual(recording.genre, "Rock")

    def test_rejected_recording_clears_only_its_verified_dependent_work_id(
        self,
    ) -> None:
        from sonora.modules import musicbrainz_enrichment
        from sonora.services import musicbrainz

        recording_id = "11111111-2222-3333-4444-555555555555"
        work_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        for linked_work, expected in (
            (work_id, None),
            (None, work_id),
            (recording_id, work_id),
        ):
            with self.subTest(linked_work=linked_work):
                recording = TrackInfo(
                    self.audio("recording.flac"),
                    artist="Alpha",
                    title="Original Song",
                    musicbrainz_trackid=recording_id,
                    musicbrainz_workid=work_id,
                    metadata_sources={"musicbrainz_workid": "MusicBrainz"},
                )
                with patch.object(
                    musicbrainz,
                    "fetch_musicbrainz_recording_details",
                    return_value={
                        "artist": "Unrelated",
                        "title": "Different Song",
                        "musicbrainz_workid": linked_work,
                    },
                ):
                    self.assertIsNone(
                        musicbrainz_enrichment._fetch_standalone_recording(recording)
                    )
                self.assertIsNone(recording.musicbrainz_trackid)
                self.assertEqual(recording.musicbrainz_workid, expected)
                if expected is None:
                    self.assertNotIn("musicbrainz_workid", recording.metadata_sources)

    def test_clean_musicbrainz_release_blocks_force_writes_to_explicit_audio(
        self,
    ) -> None:
        import hashlib

        for description in ("clean", "16bit clean", "cleaned version"):
            with self.subTest(description=description):
                path = self.audio("recording.flac")
                write_track_metadata(
                    TrackInfo(
                        path,
                        artist="Alpha",
                        title="Original Song",
                        album="Album",
                        advisory="Explicit",
                    )
                )
                before = hashlib.sha256(path.read_bytes()).hexdigest()
                with (
                    patch(
                        "sonora.services.artist_resolution.resolve_artist_name",
                        return_value="Alpha",
                    ),
                    patch.object(itunes, "fetch_itunes_track_metadata") as catalog,
                    patch("sonora.audio.metadata.write_track_metadata") as persist,
                ):
                    result = tagger.process_single_track(
                        path,
                        force=True,
                        fetch_bpm=False,
                        fetch_key=False,
                        fetch_lyrics=False,
                        fetch_itunes_art=False,
                        album_mb_release_details={
                            "artist": "Alpha",
                            "title": "Album",
                            "disambiguation": description,
                        },
                    )
                persist.assert_not_called()
                catalog.assert_not_called()
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
                self.assertEqual(result.title, "Original Song")
                self.assertIn("identity", result.metadata_conflicts)

    def test_clean_release_found_by_track_search_also_blocks_force_writes(self) -> None:
        import hashlib

        from sonora.services import musicbrainz

        path = self.audio("recording.flac")
        write_track_metadata(
            TrackInfo(
                path,
                artist="Alpha",
                title="Original Song",
                album="Album",
                advisory="Explicit",
            )
        )
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        with (
            patch(
                "sonora.services.artist_resolution.resolve_artist_name",
                return_value="Alpha",
            ),
            patch.object(musicbrainz, "fetch_track_mbid", return_value=None),
            patch.object(
                musicbrainz,
                "search_musicbrainz_release",
                return_value={
                    "id": "11111111-2222-3333-4444-555555555555",
                    "title": "Album",
                },
            ),
            patch.object(
                musicbrainz,
                "fetch_musicbrainz_release_details",
                return_value={
                    "artist": "Alpha",
                    "title": "Album",
                    "disambiguation": "clean",
                    "barcode": "rejected barcode",
                },
            ),
            patch.object(itunes, "fetch_itunes_track_metadata") as catalog,
        ):
            result = tagger.process_single_track(
                path,
                force=True,
                fetch_bpm=False,
                fetch_key=False,
                fetch_lyrics=False,
                fetch_itunes_art=False,
            )
        catalog.assert_not_called()
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
        self.assertIsNone(result.musicbrainz_albumid)
        self.assertIn("identity", result.metadata_conflicts)

    def test_release_membership_is_required_for_recording_specific_metadata(
        self,
    ) -> None:
        from sonora.modules import musicbrainz_enrichment

        for candidate, accepted in (
            ("Original Song", True),
            ("Different Song", False),
            (None, False),
        ):
            with self.subTest(candidate=candidate):
                identifier = "11111111-2222-3333-4444-555555555555"
                track = TrackInfo(
                    self.audio("recording.flac"),
                    artist="Alpha",
                    title="Original Song",
                    album="Album",
                    duration=100,
                    musicbrainz_albumid=identifier,
                )
                recordings = (
                    {
                        identifier: {
                            "recording_mbid": identifier,
                            "artist": "Alpha",
                            "title": candidate,
                            "duration": 100,
                        }
                    }
                    if candidate
                    else {}
                )
                musicbrainz_enrichment._apply_mb_release_metadata(
                    track,
                    identifier,
                    {
                        "artist": "Alpha",
                        "title": "Album",
                        "barcode": "verified-barcode",
                        "tracks_by_mbid": recordings,
                    },
                    force=True,
                )
                self.assertEqual(
                    track.musicbrainz_albumid, identifier if accepted else None
                )
                self.assertEqual(
                    track.barcode, "verified-barcode" if accepted else None
                )
                self.assertEqual(
                    "musicbrainz_albumid" in track.metadata_conflicts, not accepted
                )

    def test_album_release_date_cannot_postdate_an_existing_original_date(self) -> None:
        from sonora.modules import musicbrainz_enrichment

        for candidate, expected in (
            ("2021-09-24", "2020-10-16"),
            ("2019-01-01", "2019-01-01"),
            (None, "2020-10-16"),
        ):
            with self.subTest(candidate=candidate):
                identifier = "11111111-2222-3333-4444-555555555555"
                track = TrackInfo(
                    self.audio("recording.flac"),
                    artist="Alpha",
                    title="Original Song",
                    album="Album",
                    original_date="2020-10-16",
                    musicbrainz_albumid=identifier,
                )
                musicbrainz_enrichment._apply_mb_release_metadata(
                    track,
                    identifier,
                    {"artist": "Alpha", "title": "Album", "original_date": candidate},
                    force=True,
                )
                self.assertEqual(track.original_date, expected)

    def test_normalization_preserves_extended_version_designations(self) -> None:
        from sonora.modules.normalization import normalize_track_metadata

        for title in ("Song [Extended]", "Song (Extended)", "Song [Extended Mix]"):
            with self.subTest(title=title):
                track = TrackInfo(
                    self.audio("recording.flac"), artist="Alpha", title=title
                )
                normalize_track_metadata(track)
                self.assertEqual(track.title, title)

    def test_genius_does_not_merge_titles_that_differ_by_a_semantic_word(self) -> None:
        for requested, candidate in (
            ("Când câinii de pază dorm", "Când câinii de pază NU dorm"),
            ("Long Enough Story", "Long Enough Story Again"),
        ):
            with self.subTest(requested=requested):
                cached = {
                    "artist": "Alpha",
                    "title": candidate,
                    "genius_song_id": 11,
                    "writers": "Wrong Writer",
                }
                with patch.object(genius, "get_cached_api", return_value=cached):
                    self.assertIsNone(
                        genius.fetch_genius_song_details(
                            "Alpha", requested, api_token="fixture"
                        )
                    )
                recording = TrackInfo(
                    self.audio("recording.flac"),
                    artist="Alpha",
                    title=requested,
                    genius_song_id="11",
                )
                with (
                    patch.object(
                        genius, "fetch_genius_song_details", return_value=None
                    ),
                    patch.object(
                        genius, "lookup_genius_song_details", return_value=cached
                    ),
                ):
                    catalog_enrichment._enrich_genius(recording, "fixture")
                self.assertIsNone(recording.composer)
                self.assertEqual(
                    recording.genius_song_id,
                    "11" if requested.startswith("Când") else None,
                )
                self.assertIn("genius_song_id_validation", recording.metadata_conflicts)
