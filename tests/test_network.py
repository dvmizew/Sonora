import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import musicbrainzngs

from sonora.core.constants import USER_AGENT
from sonora.core.http import SESSION
from sonora.core.models import TrackInfo
from sonora.core.utils import RateLimiter
from sonora.modules.tagger import process_single_track
from sonora.services.deezer import (
    fetch_deezer_album_details,
    fetch_deezer_cover_art_url,
    fetch_deezer_track_details,
)
from sonora.services.discogs import (
    fetch_discogs_release_details,
    search_discogs_release,
)
from sonora.services.genius import fetch_genius_song_details
from sonora.services.itunes import (
    fetch_itunes_cover_art_url,
    fetch_itunes_track_metadata,
    search_itunes,
)
from sonora.services.lastfm import fetch_lastfm_tags
from sonora.services.lyrics import fetch_synced_lyrics, process_track_lyrics
from sonora.services.musicbrainz import (
    fetch_album_track_mbids,
    fetch_cover_art_archive_url,
    fetch_musicbrainz_recording_details,
    fetch_musicbrainz_release_details,
    fetch_track_mbid,
)
from sonora.services.theaudiodb import (
    _download_artwork_bytes,
    fetch_artist_images,
    fetch_theaudiodb_track_details,
)


class TestNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        RateLimiter.set_disabled(True)

    @classmethod
    def tearDownClass(cls) -> None:
        RateLimiter.set_disabled(False)

    def setUp(self) -> None:
        services = [
            "sonora.services.itunes",
            "sonora.services.lyrics",
            "sonora.services.musicbrainz",
            "sonora.services.discogs",
            "sonora.services.acoustid",
            "sonora.services.lastfm",
            "sonora.services.genius",
            "sonora.services.deezer",
            "sonora.services.theaudiodb",
        ]
        for service in services:
            patcher = patch(f"{service}.get_cached_api", return_value=None)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_http_session_transport_configuration(self) -> None:
        self.assertEqual(SESSION.headers.get("User-Agent"), USER_AGENT)
        transport = SESSION._transport
        self.assertIsInstance(transport, httpx.HTTPTransport)
        if isinstance(transport, httpx.HTTPTransport):
            pool = getattr(transport, "_pool", None)
            if pool and hasattr(pool, "_http2"):
                self.assertTrue(pool._http2)

    # --- TheAudioDB Tests ---

    @patch("sonora.services.theaudiodb.SESSION.get")
    def test_theaudiodb_server_disconnect(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = httpx.RemoteProtocolError("Server disconnected")
        thumb, banner = fetch_artist_images("Emil Lassaria")
        self.assertIsNone(thumb)
        self.assertIsNone(banner)

    @patch("sonora.services.theaudiodb.SESSION.get")
    def test_theaudiodb_image_download_disconnect(self, mock_get: MagicMock) -> None:
        mock_json_response = MagicMock()
        mock_json_response.status_code = 200
        mock_json_response.json.return_value = {
            "artists": [
                {
                    "strArtistThumb": "https://www.theaudiodb.com/images/thumb.jpg",
                    "strArtistBanner": "https://www.theaudiodb.com/images/banner.jpg",
                }
            ]
        }
        mock_get.side_effect = [
            mock_json_response,
            httpx.ConnectTimeout("Connection timed out"),
            httpx.RemoteProtocolError("Server disconnected"),
        ]

        thumb, banner = fetch_artist_images("Artist Name")
        self.assertIsNone(thumb)
        self.assertIsNone(banner)

    @patch("sonora.services.theaudiodb.SESSION.get")
    def test_theaudiodb_track_details_503_and_broken_json(
        self, mock_get: MagicMock
    ) -> None:
        mock_response_503 = MagicMock()
        mock_response_503.status_code = 503
        mock_get.return_value = mock_response_503
        self.assertIsNone(fetch_theaudiodb_track_details("Artist", "Title"))

        mock_broken_json = MagicMock()
        mock_broken_json.status_code = 200
        mock_broken_json.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_broken_json
        self.assertIsNone(fetch_theaudiodb_track_details("Artist", "Title"))

    def test_download_artwork_bytes_empty_url(self) -> None:
        self.assertIsNone(_download_artwork_bytes(None))
        self.assertIsNone(_download_artwork_bytes(""))

    # --- MusicBrainz & Cover Art Archive Tests ---

    @patch("sonora.services.musicbrainz.musicbrainzngs")
    def test_musicbrainz_recording_lookup_network_errors(
        self, mock_mb: MagicMock
    ) -> None:
        mock_mb.search_recordings.side_effect = musicbrainzngs.MusicBrainzError(
            "Service Unavailable 503"
        )
        self.assertIsNone(fetch_track_mbid("Artist", "Title"))

        mock_mb.get_recording_by_id.side_effect = OSError("Connection reset by peer")
        self.assertIsNone(
            fetch_musicbrainz_recording_details("c8b03190-306c-4125-9b32-3f9d86d60a12")
        )

        mock_mb.get_release_by_id.side_effect = musicbrainzngs.MusicBrainzError(
            "Gateway Timeout 504"
        )
        self.assertEqual(
            fetch_album_track_mbids("c8b03190-306c-4125-9b32-3f9d86d60a12"), {}
        )
        self.assertIsNone(
            fetch_musicbrainz_release_details("c8b03190-306c-4125-9b32-3f9d86d60a12")
        )

    @patch("sonora.services.musicbrainz.SESSION.head")
    def test_cover_art_archive_network_timeout(self, mock_head: MagicMock) -> None:
        mock_head.side_effect = httpx.ConnectTimeout("Connect timeout")
        self.assertIsNone(
            fetch_cover_art_archive_url("c8b03190-306c-4125-9b32-3f9d86d60a12")
        )

    # --- iTunes Tests ---

    @patch("sonora.services.itunes.SESSION.get")
    def test_itunes_rate_limit_and_timeout(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = httpx.HTTPStatusError(
            "429 Too Many Requests",
            request=MagicMock(),
            response=MagicMock(status_code=429),
        )
        self.assertEqual(search_itunes("Artist", "Title"), [])
        self.assertIsNone(fetch_itunes_cover_art_url("Artist", "Album"))
        self.assertIsNone(fetch_itunes_track_metadata("Artist", "Title"))

        mock_get.side_effect = httpx.ReadTimeout("Read timed out")
        self.assertEqual(search_itunes("Artist", "Title"), [])

    # --- Deezer & Discogs Tests ---

    @patch("sonora.services.deezer.SESSION.get")
    def test_deezer_server_disconnect(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = httpx.RemoteProtocolError("Server disconnected")
        self.assertIsNone(fetch_deezer_cover_art_url("Artist", "Album"))
        self.assertIsNone(fetch_deezer_album_details("Artist", "Album"))
        self.assertIsNone(fetch_deezer_track_details("Artist", "Title"))

    @patch("sonora.services.discogs.SESSION.get")
    def test_discogs_network_errors(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = httpx.HTTPError("401 Unauthorized")
        self.assertIsNone(
            search_discogs_release("Artist", "Album", user_token="bad_token")
        )
        self.assertIsNone(
            fetch_discogs_release_details("12345", user_token="bad_token")
        )

    # --- Last.fm & Genius Tests ---

    @patch("sonora.services.lastfm.SESSION.get")
    def test_lastfm_disconnect(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = httpx.RemoteProtocolError("Server disconnected")
        self.assertEqual(
            fetch_lastfm_tags("Artist", "Title", api_key="dummy_lastfm_key"), []
        )

    @patch("sonora.services.genius.SESSION.get")
    def test_genius_network_error(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = httpx.ConnectError("Connection refused")
        self.assertIsNone(
            fetch_genius_song_details("Artist", "Title", api_token="dummy_token")
        )

    # --- SyncedLyrics Tests ---

    @patch("sonora.services.lyrics.syncedlyrics.search")
    def test_lyrics_upstream_exception(self, mock_search: MagicMock) -> None:
        mock_search.side_effect = httpx.RemoteProtocolError(
            "Musixmatch connection dropped"
        )
        with self.assertRaises(RuntimeError):
            fetch_synced_lyrics("Artist", "Title")

        lyrics_text, lyrics_type = process_track_lyrics(
            Path("/tmp/nonexistent.flac"), "Artist", "Title"
        )
        self.assertIsNone(lyrics_text)
        self.assertIsNone(lyrics_type)

    # --- Concurrency & Rate Limiting Tests ---

    def test_rate_limiter_spacing(self) -> None:
        RateLimiter.set_disabled(False)
        try:
            limiter = RateLimiter(interval_seconds=0.05)
            timestamps: list[float] = []
            lock = threading.Lock()

            def worker() -> None:
                limiter.wait()
                with lock:
                    timestamps.append(time.monotonic())

            threads = [threading.Thread(target=worker) for _ in range(6)]
            start_time = time.monotonic()
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            elapsed = time.monotonic() - start_time
            self.assertGreaterEqual(elapsed, 0.20)
            self.assertEqual(len(timestamps), 6)

            sorted_timestamps = sorted(timestamps)
            for i in range(1, len(sorted_timestamps)):
                diff = sorted_timestamps[i] - sorted_timestamps[i - 1]
                self.assertGreaterEqual(diff, 0.03)
        finally:
            RateLimiter.set_disabled(True)

    # --- Smart Field Gating Tests ---

    @patch("sonora.modules.tagger.fetch_theaudiodb_track_details")
    @patch("sonora.modules.tagger.read_track_metadata")
    def test_smart_gating_skips_complete_tracks(
        self,
        mock_read: MagicMock,
        mock_tadb: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            song_path = Path(tmp_dir) / "song.flac"
            song_path.write_bytes(b"dummy audio data")

            complete_track = TrackInfo(
                file_path=song_path,
                artist="Artist",
                title="Title",
                album="Album",
                mood="Energetic",
                style="Pop",
                initial_key="8A",
                rating=5.0,
                music_video_url="https://youtube.com/watch?v=123",
                comment="Existing comment",
            )
            mock_read.return_value = complete_track

            with (
                patch(
                    "sonora.modules.tagger.fetch_genius_song_details",
                    return_value=None,
                ),
                patch(
                    "sonora.modules.tagger.fetch_deezer_track_details",
                    return_value=None,
                ),
                patch(
                    "sonora.modules.tagger.fetch_deezer_album_details",
                    return_value=None,
                ),
                patch(
                    "sonora.modules.tagger.fetch_musicbrainz_recording_details",
                    return_value=None,
                ),
                patch("sonora.modules.tagger.fetch_track_mbid", return_value=None),
                patch(
                    "sonora.modules.tagger.search_musicbrainz_release",
                    return_value=None,
                ),
                patch(
                    "sonora.modules.tagger.fetch_itunes_track_metadata",
                    return_value=None,
                ),
                patch(
                    "sonora.modules.tagger.resolve_artist_name",
                    side_effect=lambda x: x,
                ),
                patch("sonora.modules.tagger.write_track_metadata"),
            ):
                result = process_single_track(
                    file_path=song_path,
                    fetch_bpm=False,
                    fetch_lyrics=False,
                    fetch_itunes_art=False,
                    force=False,
                )

            mock_tadb.assert_not_called()
            self.assertEqual(result.mood, "Energetic")
            self.assertEqual(result.initial_key, "8A")


if __name__ == "__main__":
    unittest.main()
