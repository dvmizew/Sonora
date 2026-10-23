from unittest.mock import MagicMock, patch

from sonora.audio import art
from sonora.core.http import SESSION
from sonora.services import deezer, discogs, genius, itunes, lyrics, theaudiodb
from tests.fixtures import image_bytes
from tests.service_support import ServiceResponseTestCase


class TestCatalogReview(ServiceResponseTestCase):
    def test_repeated_track_titles_do_not_select_an_arbitrary_edition(self) -> None:
        response = MagicMock(status_code=200)
        for titles in (
            ("Recording", "Recording"),
            ("Recording", "Second"),
            ("Recording", "Recording", "Recording"),
        ):
            with self.subTest(titles=titles):
                response.json.return_value = {
                    "results": [
                        {
                            "wrapperType": "track",
                            "trackNumber": position,
                            "trackName": title,
                            "trackId": position,
                        }
                        for position, title in enumerate(titles, 1)
                    ]
                }
                with (
                    patch.object(
                        itunes,
                        "search_itunes",
                        return_value=[
                            {
                                "collectionId": 1,
                                "collectionName": "Album",
                                "artistName": "Primary",
                            }
                        ],
                    ),
                    patch.object(SESSION, "get", return_value=response),
                ):
                    album = itunes.fetch_itunes_album_details("Primary", "Album")
                assert album is not None
                position_index = album["tracks_by_disc_and_position"]
                title_index = album["tracks_by_title"]
                assert isinstance(position_index, dict)
                assert isinstance(title_index, dict)
                self.assertEqual(len(position_index), len(titles))
                self.assertEqual(
                    "recording" in title_index,
                    titles.count("Recording") == 1,
                )
                search_response = MagicMock(status_code=200)
                search_response.json.return_value = {
                    "data": [{"id": 1, "title": "Album", "artist": {"name": "Primary"}}]
                }
                response.json.return_value = {
                    "tracks": {
                        "data": [
                            {"id": position, "track_position": position, "title": title}
                            for position, title in enumerate(titles, 1)
                        ]
                    }
                }
                with patch.object(
                    SESSION, "get", side_effect=[search_response, response]
                ):
                    album = deezer.fetch_deezer_album_details("Primary", "Album")
                assert album is not None
                title_index = album["tracks_by_title"]
                assert isinstance(title_index, dict)
                self.assertEqual(
                    "recording" in title_index,
                    titles.count("Recording") == 1,
                )
                response.json.return_value = {
                    "id": 1,
                    "tracklist": [
                        {"position": str(position), "title": title}
                        for position, title in enumerate(titles, 1)
                    ],
                }
                with patch.object(SESSION, "get", return_value=response):
                    release = discogs.fetch_discogs_release_details(1)
                assert release is not None
                self.assertEqual(
                    "recording" in release["track_credits"],
                    titles.count("Recording") == 1,
                )
                self.assertIn("1", release["track_credits"])

    def test_artist_artwork_requires_unambiguous_artist_identity(self) -> None:
        artist_mbid = "11111111-2222-3333-4444-555555555555"
        with patch.object(SESSION, "get") as request:
            for artist_name in ("", " "):
                self.assertEqual(
                    theaudiodb.fetch_artist_images(artist_name), (None, None)
                )
        request.assert_not_called()
        for artist_name, candidate_mbid, expected in (
            ("Primary", artist_mbid, (b"portrait", None)),
            ("Unrelated", artist_mbid, (None, None)),
            ("Primary", "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", (None, None)),
        ):
            response = MagicMock(status_code=200)
            response.json.return_value = {
                "artists": [
                    {
                        "strArtist": artist_name,
                        "strMusicBrainzID": candidate_mbid,
                        "strArtistThumb": "https://example.com/avatar.jpg",
                    }
                ]
            }
            with (
                self.subTest(artist_name=artist_name, candidate_mbid=candidate_mbid),
                patch.object(SESSION, "get", return_value=response) as request,
                patch.object(
                    theaudiodb, "_download_artwork_bytes", return_value=b"portrait"
                ),
            ):
                self.assertEqual(
                    theaudiodb.fetch_artist_images("Primary", artist_mbid=artist_mbid),
                    expected,
                )
            self.assertIn("artist-mb.php", request.call_args.args[0])
        response.json.return_value = {
            "artists": [{"strArtist": "Primary"}, {"strArtist": "Primary"}]
        }
        with (
            patch.object(SESSION, "get", return_value=response),
            patch.object(theaudiodb, "_download_artwork_bytes") as download,
        ):
            self.assertEqual(theaudiodb.fetch_artist_images("Primary"), (None, None))
        download.assert_not_called()

    def test_discogs_audio_count_checks_nested_tracks_and_rejects_wrong_editions(
        self,
    ) -> None:
        tracklist: list[dict[str, object]] = [
            {"type_": "heading", "title": "Side One"},
            {
                "type_": "index",
                "title": "Suite",
                "sub_tracks": [
                    {"type_": "track", "position": "1", "title": "First"},
                    {"type_": "track", "position": "2", "title": "Second"},
                ],
            },
            {"type_": "track", "position": "Video 1", "title": "Music Video"},
        ]
        self.assertEqual(discogs._count_discogs_audio_tracks(tracklist), 2)
        self.assertEqual(discogs._count_discogs_audio_tracks([]), 0)
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "results": [{"id": 1, "title": "Primary - Recording", "format": ["File"]}]
        }
        for expected_count, details, accepted in (
            (2, {"audio_track_count": 2}, True),
            (3, {"audio_track_count": 2}, False),
            (2, None, False),
        ):
            with (
                self.subTest(expected_count=expected_count, details=details),
                patch.object(SESSION, "get", return_value=response),
                patch.object(
                    discogs, "fetch_discogs_release_details", return_value=details
                ),
            ):
                self.assertEqual(
                    discogs.search_discogs_release(
                        "Primary", "Recording", "token", expected_count
                    )
                    is not None,
                    accepted,
                )

    def test_cover_upgrade_never_reduces_either_dimension(self) -> None:
        cover = self.test_directory / "cover.jpg"
        original = image_bytes((1000, 1000))
        for replacement_dimensions, accepted in (
            ((1200, 1200), True),
            ((2000, 600), False),
            ((600, 2000), False),
        ):
            cover.write_bytes(original)
            replacement = image_bytes(replacement_dimensions)
            with (
                self.subTest(dimensions=replacement_dimensions),
                patch.object(
                    art,
                    "_fetch_candidate_artwork",
                    return_value=(replacement, replacement_dimensions),
                ),
            ):
                art.process_album_cover_art(
                    self.test_directory, "Primary", "Recording", force=True
                )
            self.assertEqual(cover.read_bytes(), replacement if accepted else original)

    def test_equal_album_candidates_abstain_across_catalogs(self) -> None:
        for candidate_ids in ((1, 2), (2, 1)):
            with self.subTest(candidate_ids=candidate_ids):
                albums = [
                    {
                        "collectionId": identifier,
                        "collectionName": "Recording",
                        "artistName": "Primary",
                    }
                    for identifier in candidate_ids
                ]
                with (
                    patch.object(itunes, "search_itunes", return_value=albums),
                    patch.object(SESSION, "get") as request,
                ):
                    self.assertIsNone(
                        itunes.fetch_itunes_album_details("Primary", "Recording")
                    )
                request.assert_not_called()
                response = MagicMock(status_code=200)
                response.json.return_value = {
                    "data": [
                        {
                            "id": identifier,
                            "title": "Recording",
                            "artist": {"name": "Primary"},
                        }
                        for identifier in candidate_ids
                    ]
                }
                with patch.object(SESSION, "get", return_value=response) as request:
                    self.assertIsNone(
                        deezer.fetch_deezer_album_details("Primary", "Recording")
                    )
                self.assertEqual(request.call_count, 1)
                response.json.return_value = {
                    "results": [
                        {
                            "id": identifier,
                            "title": "Primary - Recording",
                            "format": ["File"],
                        }
                        for identifier in candidate_ids
                    ]
                }
                with (
                    patch.object(SESSION, "get", return_value=response),
                    patch.object(discogs, "fetch_discogs_release_details") as details,
                ):
                    self.assertIsNone(
                        discogs.search_discogs_release("Primary", "Recording", "token")
                    )
                details.assert_not_called()

    def test_genius_preserves_version_constraints_and_revalidates_details(self) -> None:
        search = MagicMock()
        search.json.return_value = {
            "response": {
                "hits": [
                    {
                        "result": {
                            "api_path": "/songs/1",
                            "title": "Recording",
                            "primary_artist": {"name": "Primary"},
                        }
                    }
                ]
            }
        }
        for title in ("Recording (Clean Version)", "Recording (Explicit Version)"):
            with (
                self.subTest(title=title),
                patch.object(SESSION, "get", return_value=search) as request,
            ):
                self.assertIsNone(
                    genius.fetch_genius_song_details("Primary", title, "token")
                )
            self.assertEqual(request.call_count, 1)
        for candidate_title, candidate_artist, accepted in (
            ("Recording", "Primary", True),
            ("Different", "Primary", False),
            ("Recording", "Unrelated", False),
        ):
            details = MagicMock()
            details.json.return_value = {
                "response": {
                    "song": {
                        "title": candidate_title,
                        "primary_artist": {"name": candidate_artist},
                        "id": 1,
                    }
                }
            }
            with (
                self.subTest(
                    candidate_title=candidate_title, candidate_artist=candidate_artist
                ),
                patch.object(SESSION, "get", side_effect=[search, details]),
            ):
                self.assertEqual(
                    genius.fetch_genius_song_details("Primary", "Recording", "token")
                    is not None,
                    accepted,
                )

    def test_lyrics_retry_preserves_editions_and_allows_generic_labels(self) -> None:
        for title in (
            "Recording (Clean Version)",
            "Recording (Album Version)",
            "Recording (feat. Guest)",
        ):
            with (
                self.subTest(title=title),
                patch.object(lyrics, "_query_syncedlyrics", return_value=None) as query,
            ):
                self.assertIsNone(lyrics.fetch_synced_lyrics("Primary", title))
            self.assertEqual(
                query.call_count,
                2 if "feat." in title or "Album Version" in title else 1,
            )
