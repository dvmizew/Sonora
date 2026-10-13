import subprocess
from collections.abc import Iterator
from unittest.mock import patch

import httpx

from sonora.audio.metadata import read_track_metadata, write_track_metadata
from sonora.core.http import RetryTransport, _retry_delay
from sonora.core.models import TrackInfo
from sonora.modules import catalog_enrichment
from sonora.services import acoustid, deezer, genius, itunes
from tests.regression_support import AudioRegressionTestCase


class Stream(httpx.SyncByteStream):
    def __init__(self) -> None:
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        yield b"response"

    def close(self) -> None:
        self.closed = True


class TestProviderIdentityRegressions(AudioRegressionTestCase):
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
                    genius, "get_cached_api", return_value={"description": description}
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
