from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from sonora.core.config import SonoraConfig
from sonora.core.http import SESSION
from sonora.core.models import TrackInfo
from sonora.modules.catalog_enrichment import _enrich_genius
from sonora.services import genius
from sonora.services.genius import fetch_genius_song_details
from tests.service_support import ServiceResponseTestCase
from tests.support import IsolatedTestCase


class TestGeniusResponses(ServiceResponseTestCase):
    @patch("sonora.core.http.SESSION.get")
    def test_fetch_genius_description(self, mock_get: MagicMock) -> None:
        mock_search_response = MagicMock()
        mock_search_response.json.return_value = {
            "response": {
                "hits": [
                    {
                        "result": {
                            "api_path": "/songs/123",
                            "title": "Title",
                            "primary_artist": {"name": "Artist"},
                        }
                    }
                ]
            }
        }
        mock_song_response = MagicMock()
        mock_song_response.json.return_value = {
            "response": {
                "song": {
                    "id": 123,
                    "title": "Title",
                    "primary_artist": {"name": "Artist"},
                    "description": {"plain": "Song story description"},
                    "featured_artists": [{"name": "Feat Artist"}],
                    "producer_artists": [{"name": "Producer 1"}],
                    "writer_artists": [{"name": "Writer 1"}],
                    "release_date": "2023-01-01",
                }
            }
        }
        mock_get.side_effect = [mock_search_response, mock_song_response]

        song_details = fetch_genius_song_details(
            "Artist", "Title", api_token="dummy_genius_token"
        )
        self.assertIsNotNone(song_details)
        self.assertEqual(mock_get.call_args.kwargs["params"], {"text_format": "plain"})
        if song_details:
            self.assertEqual(song_details["description"], "Song story description")
            self.assertEqual(song_details["producers"], "Producer 1")
            self.assertEqual(song_details["writers"], "Writer 1")

    @patch("sonora.services.genius.get_cached_api")
    def test_fetch_genius_song_details_cached(self, mock_cache: MagicMock) -> None:
        mock_cache.return_value = {
            "artist": "Artist",
            "title": "Title",
            "genius_song_id": 999,
            "description": "Cached story",
            "featured_artists": None,
            "producers": None,
            "writers": None,
            "release_date": None,
        }
        details = fetch_genius_song_details("Artist", "Title", api_token="dummy_token")
        self.assertIsNotNone(details)
        assert details is not None
        self.assertEqual(details["description"], "Cached story")


class TestGeniusSongIdentity(IsolatedTestCase):
    def test_primary_artist_array_preserves_joint_billing(self) -> None:
        song: dict[str, object] = {
            "id": 222,
            "title": "Original Song",
            "primary_artist": {"name": "Beta"},
            "primary_artists": [{"name": "Beta"}, {"name": "Alpha"}],
            "writer_artists": [{"name": "Verified Writer"}],
        }
        with patch.object(genius, "get_cached_api", return_value=song):
            details = genius.lookup_genius_song_details("222", "token")
        assert details is not None
        self.assertEqual(details["artist"], "Beta & Alpha")
        self.assertEqual(
            genius.score_genius_song_match("Alpha & Beta", "Original Song", details),
            100,
        )
        self.assertEqual(
            genius.score_genius_song_match("Alpha & Gamma", "Original Song", details), 0
        )
        track = TrackInfo(
            Path("song.flac"),
            artist="Alpha & Beta",
            title="Original Song",
            genius_song_id="222",
        )
        with (
            patch.object(genius, "fetch_genius_song_details", return_value=None),
            patch.object(genius, "lookup_genius_song_details", return_value=details),
        ):
            _enrich_genius(track, "token")
        self.assertEqual(track.composer, "Verified Writer")
        self.assertFalse(track.metadata_conflicts)

    def test_collaboration_billing_order_preserves_only_complete_credits(self) -> None:
        for candidate_artist, featured, accepted in (
            ("Beta", "Alpha", True),
            ("Alpha", "Beta", True),
            ("Alpha", None, False),
            ("Beta", None, False),
            ("Beta", "Gamma", False),
        ):
            with self.subTest(artist=candidate_artist, featured=featured):
                song: dict[str, object] = {
                    "artist": candidate_artist,
                    "title": "Original Song",
                    "featured_artists": featured,
                    "genius_song_id": 222,
                    "writers": "Verified Writer",
                }
                with patch.object(genius, "get_cached_api", return_value=song):
                    result = genius.fetch_genius_song_details(
                        "Alpha & Beta", "Original Song", "token"
                    )
                self.assertEqual(result is not None, accepted)
                track = TrackInfo(
                    Path("song.flac"),
                    artist="Alpha & Beta",
                    title="Original Song",
                    genius_song_id="222",
                )
                with (
                    patch.object(
                        genius, "fetch_genius_song_details", return_value=None
                    ),
                    patch.object(
                        genius, "lookup_genius_song_details", return_value=song
                    ),
                ):
                    _enrich_genius(track, "token")
                self.assertEqual(track.genius_song_id, "222")
                self.assertEqual(
                    track.composer, "Verified Writer" if accepted else None
                )

    def test_validated_song_id_repairs_complete_credits_without_replacing_them(
        self,
    ) -> None:
        for previous_id in ("111", "222", None):
            with self.subTest(previous_id=previous_id):
                track = TrackInfo(
                    Path("song.flac"),
                    artist="Alpha",
                    title="Original Song",
                    genius_song_id=previous_id,
                    composer="Existing Writer",
                    producers="Existing Producer",
                )
                with patch.object(
                    genius,
                    "fetch_genius_song_details",
                    return_value={
                        "genius_song_id": 222,
                        "writers": "Other Writer",
                        "producers": "Other Producer",
                    },
                ) as lookup:
                    _enrich_genius(track, "token")
                lookup.assert_called_once()
                self.assertEqual(track.genius_song_id, "222")
                self.assertEqual(track.metadata_sources.get("genius_song_id"), "Genius")
                self.assertEqual(track.composer, "Existing Writer")
                self.assertEqual(track.producers, "Existing Producer")

    def test_search_abstains_on_equal_distinct_api_paths_but_deduplicates_repeats(
        self,
    ) -> None:
        for paths, expected in (
            (["/songs/1", "/songs/2"], False),
            (["/songs/1", "/songs/1"], True),
            ([], False),
        ):
            with self.subTest(paths=paths):
                search = MagicMock()
                search.json.return_value = {
                    "response": {
                        "hits": [
                            {
                                "result": {
                                    "api_path": path,
                                    "title": "Original Song",
                                    "primary_artist": {"name": "Alpha"},
                                }
                            }
                            for path in paths
                        ]
                    }
                }
                details = MagicMock()
                details.json.return_value = {
                    "response": {
                        "song": {
                            "id": 1,
                            "title": "Original Song",
                            "primary_artist": {"name": "Alpha"},
                            "description": {"plain": "Story"},
                        }
                    }
                }
                with (
                    patch.object(genius, "get_cached_api", return_value=None),
                    patch.object(genius, "set_cached_api"),
                    patch.object(
                        SESSION, "get", side_effect=[search, details]
                    ) as request,
                ):
                    result = genius.fetch_genius_song_details(
                        "Alpha", "Original Song", "token"
                    )
                self.assertEqual(result is not None, expected)
                self.assertEqual(request.call_count, 2 if expected else 1)

    def test_cached_evidence_is_revalidated_without_mutating_cached_record(
        self,
    ) -> None:
        for artist, title, expected in (
            ("Alpha", "Original Song", True),
            ("Beta", "Original Song", False),
            ("Alpha", "Different Song", False),
        ):
            with self.subTest(artist=artist, title=title):
                cached = {
                    "artist": artist,
                    "title": title,
                    "genius_song_id": 1,
                    "description": "?",
                }
                with (
                    patch.object(
                        genius, "get_cached_api", return_value=cached
                    ) as cache_lookup,
                    patch.object(SESSION, "get") as request,
                ):
                    result = genius.fetch_genius_song_details(
                        "Alpha", "Original Song", "token"
                    )
                self.assertEqual(result is not None, expected)
                self.assertIn("genius_song:v7:", cache_lookup.call_args.args[0])
                self.assertEqual(cached["description"], "?")
                if result is not None:
                    self.assertIsNone(result["description"])
                request.assert_not_called()


class TestGeniusDirectLookup(IsolatedTestCase):
    def test_specific_song_lookup_returns_identity_and_caches_original_evidence(
        self,
    ) -> None:
        song = {
            "id": 8119626,
            "title": "BLEED",
            "primary_artist": {"name": "2Scratch"},
            "description": {"plain": "?"},
            "writer_artists": [{"name": "Writer"}],
        }
        response = httpx.Response(
            200,
            json={"response": {"song": song}},
            request=httpx.Request("GET", "https://api.genius.com/songs/8119626"),
        )
        with (
            patch.object(genius, "get_cached_api", return_value=None),
            patch.object(genius, "set_cached_api") as cache_write,
            patch.object(SESSION, "get", return_value=response) as request,
        ):
            details = genius.lookup_genius_song_details("8119626", "token")
        assert details is not None
        self.assertEqual((details["artist"], details["title"]), ("2Scratch", "BLEED"))
        self.assertIsNone(details["description"])
        self.assertEqual(details["writers"], "Writer")
        self.assertEqual(
            request.call_args.args, ("https://api.genius.com/songs/8119626",)
        )
        self.assertEqual(request.call_args.kwargs["params"], {"text_format": "plain"})
        cache_write.assert_called_once_with("genius_recording:v1:8119626", song)

    def test_cached_specific_song_preserves_raw_provider_identity(self) -> None:
        song = {
            "id": 1,
            "title": "Another Recording",
            "primary_artist": {"name": "Another Artist"},
        }
        with (
            patch.object(genius, "get_cached_api", return_value=song),
            patch.object(SESSION, "get") as request,
        ):
            details = genius.lookup_genius_song_details("1", "token")
        assert details is not None
        self.assertEqual(details["title"], "Another Recording")
        request.assert_not_called()

    def test_invalid_inputs_and_response_id_mismatch_do_not_supply_false_ids(
        self,
    ) -> None:
        with patch.object(SESSION, "get") as request:
            for identifier in ("", "0", "-1", "1.0", "١", "garbage"):
                with self.subTest(identifier=identifier):
                    self.assertIsNone(
                        genius.lookup_genius_song_details(identifier, "token")
                    )
            self.assertIsNone(genius.lookup_genius_song_details("1", None))
        request.assert_not_called()
        for returned_id in (2, 3):
            song = {
                "id": returned_id,
                "title": "Recording",
                "primary_artist": {"name": "Primary"},
            }
            with (
                self.subTest(returned_id=returned_id),
                patch.object(genius, "get_cached_api", return_value=song),
                patch.object(genius, "set_cached_api") as store,
            ):
                self.assertIsNone(genius.lookup_genius_song_details("1", "token"))
            store.assert_not_called()

    def test_transport_failure_is_not_cached_as_a_missing_song(self) -> None:
        with (
            patch.object(genius, "get_cached_api", return_value=None),
            patch.object(SESSION, "get", side_effect=httpx.ConnectTimeout("offline")),
            patch.object(genius, "set_cached_api") as store,
        ):
            self.assertIsNone(genius.lookup_genius_song_details("1", "token"))
        store.assert_not_called()

    def test_malformed_provider_identifier_fails_explicitly(self) -> None:
        for identifier in (None, "1", True, 0):
            song = {
                "id": identifier,
                "title": "Recording",
                "primary_artist": {"name": "Primary"},
            }
            with (
                self.subTest(identifier=identifier),
                patch.object(genius, "get_cached_api", return_value=song),
                self.assertRaisesRegex(TypeError, "positive integer"),
            ):
                genius.lookup_genius_song_details("1", "token")

    def test_cached_search_obeys_current_confidence_policy(self) -> None:
        cached = {
            "artist": "Primari",
            "title": "Original Recording",
            "genius_song_id": 1,
        }
        for threshold, accepted in ((90.0, True), (99.0, False)):
            with (
                self.subTest(threshold=threshold),
                patch.object(genius, "get_cached_api", return_value=cached),
                patch.object(
                    genius,
                    "get_config",
                    return_value=SonoraConfig(genius_match_threshold=threshold),
                ),
                patch.object(SESSION, "get") as request,
            ):
                result = genius.fetch_genius_song_details(
                    "Primary", "Original Recording", "token"
                )
            self.assertEqual(result is not None, accepted)
            request.assert_not_called()

    def test_rejected_or_ambiguous_search_is_negative_cached(self) -> None:
        for artist, paths in (
            ("Unrelated", ["/songs/1"]),
            ("Primary", ["/songs/1", "/songs/2"]),
        ):
            response = httpx.Response(
                200,
                json={
                    "response": {
                        "hits": [
                            {
                                "result": {
                                    "api_path": path,
                                    "title": "Original Recording",
                                    "primary_artist": {"name": artist},
                                }
                            }
                            for path in paths
                        ]
                    }
                },
                request=httpx.Request("GET", "https://api.genius.com/search"),
            )
            with (
                self.subTest(artist=artist, paths=paths),
                patch.object(genius, "get_cached_api", return_value=None),
                patch.object(genius, "set_cached_api") as store,
                patch.object(SESSION, "get", return_value=response) as request,
            ):
                self.assertIsNone(
                    genius.fetch_genius_song_details(
                        "Primary", "Original Recording", "token"
                    )
                )
            self.assertEqual(request.call_count, 1)
            store.assert_called_once_with(
                "genius_song:v7:primary:original recording", None
            )
