import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp

from sonora.core.config import SonoraConfig
from sonora.core.rate_limit import RateLimiter
from sonora.services.shazam import _recognize_async, recognize_audio_track
from tests.service_support import ServiceResponseTestCase
from tests.support import IsolatedTestCase


class TestShazamResponses(ServiceResponseTestCase):
    @patch.dict(os.environ, {"ENABLE_SHAZAM": "true"})
    @patch("sonora.services.shazam._recognize_async")
    def test_recognize_audio_track(self, mock_shazam_async: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_file = Path(tmp_dir) / "test.flac"
            audio_file.write_bytes(b"dummy flac content")

            # 1. Happy path: successful recognition
            mock_shazam_async.return_value = {
                "track": {
                    "title": "Starboy",
                    "subtitle": "The Weeknd",
                    "genres": {"primary": "R&B/Soul"},
                    "images": {"coverarthq": "https://shazam.com/cover.jpg"},
                    "key": "7654321",
                    "hub": {"actions": [{"type": "applemusicplay", "id": "123456"}]},
                    "isrc": "USUM71607007",
                    "sections": [
                        {
                            "type": "SONG",
                            "metadata": [
                                {"title": "Album", "text": "Starboy"},
                                {"title": "Released", "text": "2016"},
                                {"title": "Label", "text": "Republic Records"},
                            ],
                        },
                        {
                            "type": "LYRICS",
                            "text": [
                                "I'm tryna put you in the worst mood, ah",
                                "P1 cleaner than your church shoes, ah",
                            ],
                        },
                    ],
                }
            }

            recognized_track = recognize_audio_track(audio_file)
            self.assertIsNotNone(recognized_track)
            if recognized_track:
                self.assertEqual(recognized_track.title, "Starboy")
                self.assertEqual(recognized_track.artist, "The Weeknd")
                self.assertEqual(recognized_track.album, "Starboy")
                self.assertEqual(recognized_track.genre, "R&B/Soul")
                self.assertEqual(recognized_track.isrc, "USUM71607007")
                self.assertEqual(recognized_track.apple_music_id, "123456")
                self.assertEqual(
                    recognized_track.cover_art_url, "https://shazam.com/cover.jpg"
                )
                self.assertEqual(recognized_track.release_date, "2016")
                self.assertEqual(recognized_track.label, "Republic Records")
                self.assertIsNotNone(recognized_track.lyrics)
                if recognized_track.lyrics:
                    self.assertIn("worst mood", recognized_track.lyrics)

            # 2. Edge case: empty or invalid track payload
            mock_shazam_async.return_value = {"matches": []}
            self.assertIsNone(recognize_audio_track(audio_file))

            # 3. Edge case: non-existent file
            non_existent = Path(tmp_dir) / "non_existent.flac"
            self.assertIsNone(recognize_audio_track(non_existent))

            # 4. Edge case: disabled by config
            with patch("sonora.services.shazam.get_config") as mock_cfg:
                mock_cfg.return_value = MagicMock(enable_shazam=False)
                self.assertIsNone(recognize_audio_track(audio_file))


class TestShazamContracts(IsolatedTestCase):
    def setUp(self) -> None:
        RateLimiter.set_disabled(True)
        self.addCleanup(RateLimiter.set_disabled, False)
        config_patch = patch(
            "sonora.services.shazam.get_config",
            return_value=SonoraConfig(enable_shazam=True),
        )
        config_patch.start()
        self.addCleanup(config_patch.stop)

    def test_client_closed_after_success(self) -> None:
        client = MagicMock()
        client.__aenter__.return_value = client
        client.recognize = AsyncMock(return_value={"matches": []})
        with patch("sonora.services.shazam.Shazam", return_value=client):
            self.assertEqual(
                asyncio.run(_recognize_async(Path("track.flac"))), {"matches": []}
            )
        client.__aexit__.assert_awaited_once()

    def test_programming_errors_propagate_and_client_closes(self) -> None:
        for error in (RuntimeError("internal bug"), ValueError("bad internal state")):
            with self.subTest(error=error):
                client = MagicMock()
                client.__aenter__.return_value = client
                client.recognize = AsyncMock(side_effect=error)
                with (
                    patch("sonora.services.shazam.Shazam", return_value=client),
                    self.assertRaises(type(error)),
                ):
                    asyncio.run(_recognize_async(Path("track.flac")))
                client.__aexit__.assert_awaited_once()

    def test_operational_errors_return_none_and_client_closes(self) -> None:
        for error in (
            aiohttp.ClientConnectionError("offline"),
            TimeoutError(),
            OSError("unreadable audio"),
        ):
            with self.subTest(error=error):
                client = MagicMock()
                client.__aenter__.return_value = client
                client.recognize = AsyncMock(side_effect=error)
                with patch("sonora.services.shazam.Shazam", return_value=client):
                    self.assertIsNone(asyncio.run(_recognize_async(Path("track.flac"))))
                client.__aexit__.assert_awaited_once()

    def test_invalid_recognition_envelope_raises(self) -> None:
        client = MagicMock()
        client.__aenter__.return_value = client
        client.recognize = AsyncMock(return_value=[])
        with (
            patch("sonora.services.shazam.Shazam", return_value=client),
            self.assertRaises(TypeError),
        ):
            asyncio.run(_recognize_async(Path("track.flac")))

    def test_shazam_key_is_never_an_apple_music_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "track.flac"
            audio_path.write_bytes(b"audio")
            with patch(
                "sonora.services.shazam._recognize_async",
                new=AsyncMock(
                    return_value={
                        "track": {
                            "title": "Title",
                            "subtitle": "Artist",
                            "key": "7654321",
                        }
                    }
                ),
            ):
                recognized = recognize_audio_track(audio_path)
            self.assertIsNotNone(recognized)
            assert recognized is not None
            self.assertIsNone(recognized.apple_music_id)

    def test_invalid_track_schema_is_not_cached(self) -> None:
        track: object
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "track.flac"
            audio_path.write_bytes(b"audio")
            for track in (
                [],
                {"title": None, "subtitle": "Artist"},
                {"title": "Title", "subtitle": ""},
            ):
                with (
                    self.subTest(track=track),
                    patch(
                        "sonora.services.shazam._recognize_async",
                        new=AsyncMock(return_value={"track": track}),
                    ),
                    patch("sonora.services.shazam.set_cached_api") as write_cache,
                ):
                    with self.assertRaises(TypeError):
                        recognize_audio_track(audio_path)
                    write_cache.assert_not_called()

    def test_cache_distinguishes_equal_basenames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio_paths = [
                Path(directory) / artist / "track.flac"
                for artist in ("First", "Second")
            ]
            for audio_path in audio_paths:
                audio_path.parent.mkdir()
                audio_path.write_bytes(b"audio")
            with patch(
                "sonora.services.shazam._recognize_async",
                new=AsyncMock(
                    side_effect=[
                        {"track": {"title": "First", "subtitle": "Artist"}},
                        {"track": {"title": "Second", "subtitle": "Artist"}},
                    ]
                ),
            ) as recognize:
                first_match = recognize_audio_track(audio_paths[0])
                second_match = recognize_audio_track(audio_paths[1])
            assert first_match is not None and second_match is not None
            self.assertEqual(
                (first_match.title, second_match.title), ("First", "Second")
            )
            self.assertEqual(recognize.await_count, 2)

    def test_missing_and_empty_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "missing.flac"
            self.assertIsNone(recognize_audio_track(audio_path))
            audio_path.touch()
            self.assertIsNone(recognize_audio_track(audio_path))
