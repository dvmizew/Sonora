from pathlib import Path
from unittest.mock import MagicMock, patch

from sonora.core.http import SESSION
from sonora.core.models import TrackInfo
from sonora.modules.catalog_enrichment import _enrich_genius
from sonora.services import genius
from tests.support import IsolatedTestCase


class TestGeniusAlbumIdentity(IsolatedTestCase):
    def test_featured_artist_separates_same_title_versions_and_stored_ids(self) -> None:
        for guest in ("Lil Tjay", "NAV"):
            with self.subTest(guest=guest):
                track = TrackInfo(
                    Path("Fall Back.flac"),
                    artist="Lithe",
                    title="Fall Back",
                    album="Fall Back (feat. NAV)",
                    featured_artists="NAV",
                    genius_song_id="10834991",
                )
                song: dict[str, object] = {
                    "artist": "Lithe",
                    "title": "Fall Back",
                    "featured_artists": guest,
                    "genius_song_id": "10834991",
                    "writers": f"Lithe, {guest}",
                }
                with (
                    patch.object(genius, "get_cached_api", return_value=song) as cache,
                    patch.object(
                        genius, "lookup_genius_song_details", return_value=song
                    ),
                ):
                    _enrich_genius(track, "token")
                self.assertIn("nav", cache.call_args.args[0])
                self.assertEqual(track.featured_artists, "NAV")
                if guest == "NAV":
                    self.assertEqual(track.composer, "Lithe, NAV")
                    self.assertNotIn(
                        "genius_song_id_validation", track.metadata_conflicts
                    )
                else:
                    self.assertIsNone(track.composer)
                    self.assertIsNone(track.lyricist)
                    self.assertIn("genius_song_id_validation", track.metadata_conflicts)

    def test_stored_same_title_id_for_another_album_is_withheld(self) -> None:
        track = TrackInfo(
            Path("song.flac"),
            artist="Alpha",
            title="Introduction",
            album="Later Album",
            genius_song_id="1",
            comment="Existing comment",
        )
        with (
            patch.object(genius, "fetch_genius_song_details", return_value=None),
            patch.object(
                genius,
                "lookup_genius_song_details",
                return_value={
                    "artist": "Alpha",
                    "title": "Introduction",
                    "album": "Earlier Album",
                    "genius_song_id": 1,
                    "writers": "Unrelated Writer",
                    "description": "Unrelated description",
                },
            ),
        ):
            _enrich_genius(track, "token")
        self.assertEqual(track.genius_song_id, "1")
        self.assertEqual(track.comment, "Existing comment")
        self.assertIsNone(track.composer)
        self.assertIn("genius_song_id_validation", track.metadata_conflicts)

    def test_same_title_on_another_album_is_not_identity_proof(self) -> None:
        song: dict[str, object] = {
            "artist": "Artist",
            "title": "Introduction",
            "album": "Earlier Album",
        }
        self.assertEqual(
            genius.score_genius_song_match(
                "Artist", "Introduction", song, album="Later Album"
            ),
            0,
        )
        self.assertEqual(
            genius.score_genius_song_match(
                "Artist", "Introduction", song, album="Earlier Album"
            ),
            100,
        )

    def test_structural_title_needs_album_evidence_when_local_album_is_known(
        self,
    ) -> None:
        for title in ("Intro", "Introduction", "Outro", "Interlude"):
            with self.subTest(title=title):
                song: dict[str, object] = {"artist": "Artist", "title": title}
                self.assertEqual(
                    genius.score_genius_song_match(
                        "Artist", title, song, album="Local Album"
                    ),
                    0,
                )

    def test_album_context_separates_cached_selections(self) -> None:
        song = {
            "artist": "Artist",
            "title": "Introduction",
            "album": "Earlier Album",
            "genius_song_id": 1,
        }
        with patch.object(genius, "get_cached_api", return_value=song) as cache:
            self.assertIsNotNone(
                genius.fetch_genius_song_details(
                    "Artist", "Introduction", "token", album="Earlier Album"
                )
            )
            self.assertIsNone(
                genius.fetch_genius_song_details(
                    "Artist", "Introduction", "token", album="Later Album"
                )
            )
        self.assertNotEqual(cache.call_args_list[0], cache.call_args_list[1])

    def test_album_resolves_equal_title_hits_without_accepting_another_album(
        self,
    ) -> None:
        response = MagicMock()
        response.json.return_value = {
            "response": {
                "hits": [
                    {
                        "result": {
                            "api_path": f"/songs/{identifier}",
                            "title": "Introduction",
                            "primary_artist": {"name": "Artist"},
                        }
                    }
                    for identifier in (1, 2)
                ]
            }
        }
        songs: list[dict[str, object]] = [
            {
                "artist": "Artist",
                "title": "Introduction",
                "album": album,
                "genius_song_id": identifier,
            }
            for identifier, album in ((1, "Earlier Album"), (2, "Later Album"))
        ]
        for local_album, expected in (("Later Album", 2), ("Unknown Album", None)):
            with (
                self.subTest(album=local_album),
                patch.object(genius, "get_cached_api", return_value=None),
                patch.object(genius, "set_cached_api"),
                patch.object(SESSION, "get", return_value=response),
                patch.object(genius, "lookup_genius_song_details", side_effect=songs),
            ):
                result = genius.fetch_genius_song_details(
                    "Artist", "Introduction", "token", album=local_album
                )
                self.assertEqual(
                    result.get("genius_song_id") if result else None, expected
                )

    def test_multiple_hits_without_album_evidence_remain_unresolved(self) -> None:
        response = MagicMock()
        response.json.return_value = {
            "response": {
                "hits": [
                    {
                        "result": {
                            "api_path": f"/songs/{identifier}",
                            "title": "Original Song",
                            "primary_artist": {"name": "Artist"},
                        }
                    }
                    for identifier in (1, 2)
                ]
            }
        }
        with (
            patch.object(genius, "get_cached_api", return_value=None),
            patch.object(genius, "set_cached_api"),
            patch.object(SESSION, "get", return_value=response),
            patch.object(
                genius,
                "lookup_genius_song_details",
                return_value={"artist": "Artist", "title": "Original Song"},
            ),
        ):
            self.assertIsNone(
                genius.fetch_genius_song_details(
                    "Artist", "Original Song", "token", album="Local Album"
                )
            )

    def test_raw_album_schema_is_validated_and_description_stays_available(
        self,
    ) -> None:
        payload: dict[str, object] = {
            "id": 1,
            "title": "Introduction",
            "primary_artist": {"name": "Artist"},
            "album": {"name": "Earlier Album"},
            "description": {"plain": "Community description"},
        }
        with patch.object(genius, "get_cached_api", return_value=payload):
            result = genius.lookup_genius_song_details("1", "token")
        assert result is not None
        self.assertEqual(result["album"], "Earlier Album")
        self.assertEqual(result["description"], "Community description")
        payload["album"] = {"name": 123}
        with (
            patch.object(genius, "get_cached_api", return_value=payload),
            self.assertRaisesRegex(TypeError, "album name"),
        ):
            genius.lookup_genius_song_details("1", "token")

    def test_unavailable_candidate_does_not_make_a_tie_look_unique(self) -> None:
        response = MagicMock()
        response.json.return_value = {
            "response": {
                "hits": [
                    {
                        "result": {
                            "api_path": f"/songs/{identifier}",
                            "title": "Introduction",
                            "primary_artist": {"name": "Artist"},
                        }
                    }
                    for identifier in (1, 2)
                ]
            }
        }
        with (
            patch.object(genius, "get_cached_api", return_value=None),
            patch.object(genius, "set_cached_api") as store,
            patch.object(SESSION, "get", return_value=response),
            patch.object(
                genius,
                "lookup_genius_song_details",
                side_effect=[
                    {"artist": "Artist", "title": "Introduction", "album": "Album"},
                    None,
                ],
            ),
        ):
            self.assertIsNone(
                genius.fetch_genius_song_details(
                    "Artist", "Introduction", "token", album="Album"
                )
            )
        store.assert_not_called()

    def test_audited_puya_pouya_intro_collision_requires_artist_and_album(self) -> None:
        # Both catalog artists have Intro recordings; spelling similarity is not evidence.
        for album, candidate_artist, candidate_album, accepted in (
            (None, "Puya", "Românisme, partea II", False),
            (None, "Pouya", "Stunna", False),
            ("Românisme, partea II", "Pouya", "Românisme, partea II", False),
            ("Românisme, partea II", "Puya", "Românisme, partea II", True),
        ):
            with self.subTest(album=album, artist=candidate_artist):
                self.assertEqual(
                    genius.score_genius_song_match(
                        "Puya",
                        "Intro",
                        {
                            "artist": candidate_artist,
                            "title": "Intro",
                            "album": candidate_album,
                        },
                        album=album,
                    )
                    > 0,
                    accepted,
                )

    def test_complete_primary_collaboration_does_not_lose_its_delimiters(self) -> None:
        self.assertEqual(
            genius.score_genius_song_match(
                "Quality Control",
                "Intro",
                {
                    "artist": "Quality Control & Migos",
                    "title": "Intro",
                    "album": "Vol. 2",
                },
                album="Vol. 2",
            ),
            100,
        )
