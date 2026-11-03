from unittest.mock import MagicMock, call, patch

import musicbrainzngs

from sonora.services.musicbrainz import (
    fetch_cover_art_archive_url,
    fetch_musicbrainz_release_details,
    fetch_track_mbid,
    search_musicbrainz_release,
)
from tests.service_support import ServiceResponseTestCase


class TestMusicbrainzResponses(ServiceResponseTestCase):
    @patch("sonora.services.musicbrainz.musicbrainzngs")
    def test_fetch_track_mbid_uses_isrc_before_text_search(
        self, mock_mb: MagicMock
    ) -> None:
        mock_mb.search_recordings.return_value = {
            "recording-list": [
                {
                    "id": "12345678-1234-1234-1234-123456789abc",
                    "title": "Title",
                    "artist-credit": [{"artist": {"name": "Artist"}}],
                }
            ]
        }

        cache: dict[str, object] = {}

        with (
            patch(
                "sonora.services.musicbrainz.get_cached_api",
                side_effect=cache.get,
            ),
            patch(
                "sonora.services.musicbrainz.set_cached_api",
                side_effect=lambda key, value, **_kwargs: cache.__setitem__(key, value),
            ),
        ):
            mbid = fetch_track_mbid("Artist", "Title", isrc="USABC2500001")
            cached_mbid = fetch_track_mbid("Artist", "Title", isrc="USABC2500001")

        self.assertEqual(mbid, "12345678-1234-1234-1234-123456789abc")
        self.assertEqual(cached_mbid, mbid)
        mock_mb.search_recordings.assert_called_once_with(isrc="USABC2500001", limit=5)

    @patch("sonora.services.musicbrainz.musicbrainzngs")
    def test_fetch_track_mbid_normalizes_display_isrc(self, mock_mb: MagicMock) -> None:
        mock_mb.search_recordings.return_value = {"recording-list": []}

        fetch_track_mbid("Artist", "Title", isrc="ISRC US-ABC-25-00001")

        self.assertEqual(
            mock_mb.search_recordings.call_args_list,
            [
                call(isrc="USABC2500001", limit=5),
                call(artist="Artist", recording="Title", limit=5),
            ],
        )

    @patch("sonora.services.musicbrainz.musicbrainzngs")
    def test_invalid_isrc_uses_artist_and_title_search(
        self, mock_mb: MagicMock
    ) -> None:
        mock_mb.search_recordings.return_value = {"recording-list": []}

        fetch_track_mbid("Artist", "Title", isrc="not-an-isrc")

        mock_mb.search_recordings.assert_called_once_with(
            artist="Artist", recording="Title", limit=5
        )

    @patch("sonora.services.musicbrainz.musicbrainzngs")
    def test_missing_isrc_recording_falls_back_to_artist_and_title(
        self, mock_mb: MagicMock
    ) -> None:
        mock_mb.search_recordings.side_effect = [
            {"recording-list": []},
            {
                "recording-list": [
                    {
                        "id": "12345678-1234-1234-1234-123456789abc",
                        "title": "Title",
                        "artist-credit": [{"artist": {"name": "Artist"}}],
                    }
                ]
            },
        ]

        mbid = fetch_track_mbid("Artist", "Title", isrc="USABC2500001")

        self.assertEqual(mbid, "12345678-1234-1234-1234-123456789abc")
        self.assertEqual(
            mock_mb.search_recordings.call_args_list,
            [
                call(isrc="USABC2500001", limit=5),
                call(artist="Artist", recording="Title", limit=5),
            ],
        )

    @patch("sonora.services.musicbrainz.musicbrainzngs")
    def test_isrc_hit_with_different_identity_is_not_overridden_by_text_search(
        self, mock_mb: MagicMock
    ) -> None:
        mock_mb.search_recordings.return_value = {
            "recording-list": [
                {
                    "id": "12345678-1234-1234-1234-123456789abc",
                    "title": "Wrong Title",
                    "artist-credit": [{"artist": {"name": "Other Artist"}}],
                }
            ]
        }

        self.assertIsNone(fetch_track_mbid("Artist", "Title", isrc="USABC2500001"))
        mock_mb.search_recordings.assert_called_once_with(isrc="USABC2500001", limit=5)

    @patch("sonora.services.musicbrainz.musicbrainzngs")
    def test_fetch_track_mbid(self, mock_mb: MagicMock) -> None:
        mock_mb.search_recordings.return_value = {
            "recording-list": [
                {
                    "id": "12345678-1234-1234-1234-123456789abc",
                    "title": "Title",
                    "artist-credit": [{"artist": {"name": "Artist"}}],
                }
            ]
        }

        mbid = fetch_track_mbid("Artist", "Title")
        self.assertEqual(mbid, "12345678-1234-1234-1234-123456789abc")

    @patch("sonora.services.musicbrainz.musicbrainzngs")
    def test_musicbrainz_error_handling(self, mock_mb: MagicMock) -> None:
        mock_mb.search_recordings.side_effect = musicbrainzngs.MusicBrainzError(
            "MusicBrainz server 500"
        )
        mbid_result = fetch_track_mbid("Artist", "Title")
        self.assertIsNone(mbid_result)

    @patch("sonora.services.musicbrainz.musicbrainzngs")
    def test_musicbrainz_multi_disc_release_details(self, mock_mb: MagicMock) -> None:
        mock_release_payload = {
            "release": {
                "id": "11111111-2222-3333-4444-555555555555",
                "title": "Double Album",
                "artist-credit": [{"artist": {"name": "Artist"}}],
                "date": "2024",
                "release-group": {"first-release-date": "2024"},
                "medium-list": [
                    {
                        "position": 1,
                        "track-count": 2,
                        "title": "Disc One",
                        "track-list": [
                            {
                                "position": 1,
                                "recording": {
                                    "id": "aaaaaaa1-0000-0000-0000-000000000001",
                                    "title": "Disc 1 Track 1",
                                },
                            },
                            {
                                "position": 2,
                                "recording": {
                                    "id": "aaaaaaa2-0000-0000-0000-000000000002",
                                    "title": "Disc 1 Track 2",
                                },
                            },
                        ],
                    },
                    {
                        "position": 2,
                        "track-count": 2,
                        "title": "Disc Two",
                        "track-list": [
                            {
                                "position": 1,
                                "recording": {
                                    "id": "bbbbbbb1-0000-0000-0000-000000000001",
                                    "title": "Disc 2 Track 1",
                                },
                            },
                            {
                                "position": 2,
                                "recording": {
                                    "id": "bbbbbbb2-0000-0000-0000-000000000002",
                                    "title": "Disc 2 Track 2",
                                },
                            },
                        ],
                    },
                ],
            }
        }
        mock_mb.get_release_by_id.return_value = mock_release_payload

        details = fetch_musicbrainz_release_details(
            "11111111-2222-3333-4444-555555555555"
        )
        assert details is not None
        self.assertEqual(details["total_discs"], 2)
        self.assertEqual(details["total_tracks"], 4)

        tracks_by_disc = details["tracks_by_disc_and_position"]
        assert isinstance(tracks_by_disc, dict)
        track_1_1 = tracks_by_disc[(1, 1)]
        assert isinstance(track_1_1, dict)
        self.assertEqual(track_1_1["title"], "Disc 1 Track 1")

        track_2_1 = tracks_by_disc[(2, 1)]
        assert isinstance(track_2_1, dict)
        self.assertEqual(track_2_1["title"], "Disc 2 Track 1")
        self.assertEqual(track_2_1["disc_subtitle"], "Disc Two")
        self.assertEqual(track_2_1["disc_number"], 2)
        self.assertEqual(track_2_1["total_discs"], 2)

    @patch("sonora.services.musicbrainz.SESSION.head")
    def test_fetch_cover_art_archive_url(self, mock_head: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.url = "https://ia8000.us.archive.org/cover.jpg"
        mock_head.return_value = mock_response

        url = fetch_cover_art_archive_url("c8b03190-306c-4125-9b32-3f9d86d60a12")
        self.assertEqual(url, "https://ia8000.us.archive.org/cover.jpg")

    def test_fetch_cover_art_archive_invalid_uuid_returns_none(self) -> None:
        self.assertIsNone(fetch_cover_art_archive_url("invalid-uuid"))

    @patch("sonora.services.musicbrainz.musicbrainzngs")
    def test_search_musicbrainz_release(self, mock_mb: MagicMock) -> None:
        mock_mb.search_releases.return_value = {
            "release-list": [
                {
                    "id": "c8b03190-306c-4125-9b32-3f9d86d60a12",
                    "title": "Album Title",
                    "date": "2024-01-01",
                    "country": "US",
                    "barcode": "123456789012",
                    "artist-credit": [
                        {
                            "artist": {
                                "name": "Artist",
                                "id": "a1b2c3d4-1234-1234-1234-123456789abc",
                            }
                        }
                    ],
                }
            ]
        }
        release_result = search_musicbrainz_release("Artist", "Album Title")
        assert release_result is not None
        self.assertEqual(release_result["id"], "c8b03190-306c-4125-9b32-3f9d86d60a12")
        self.assertEqual(release_result["country"], "US")

    def test_call_musicbrainz_retries_on_transient_network_error(self) -> None:
        import musicbrainzngs

        from sonora.services.musicbrainz import _call_musicbrainz

        mock_api = MagicMock(
            side_effect=[
                musicbrainzngs.NetworkError("Temporary dropout"),
                {"recording-list": [{"id": "uuid-123", "title": "Song"}]},
            ]
        )
        with patch("time.sleep"):
            result = _call_musicbrainz(mock_api, "param")
        self.assertEqual(
            result, {"recording-list": [{"id": "uuid-123", "title": "Song"}]}
        )
        self.assertEqual(mock_api.call_count, 2)

    def test_call_musicbrainz_does_not_retry_on_404(self) -> None:
        import email.message
        import io
        import urllib.error

        import musicbrainzngs

        from sonora.services.musicbrainz import _call_musicbrainz

        http_404 = urllib.error.HTTPError(
            "https://musicbrainz.org",
            404,
            "Not Found",
            email.message.Message(),
            io.BytesIO(b""),
        )
        self.addCleanup(http_404.close)
        err_404 = musicbrainzngs.ResponseError(cause=http_404)
        mock_api = MagicMock(side_effect=err_404)

        with patch("time.sleep"), self.assertRaises(musicbrainzngs.ResponseError):
            _call_musicbrainz(mock_api)
        self.assertEqual(mock_api.call_count, 1)
