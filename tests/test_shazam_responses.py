import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from sonora.services.shazam import recognize_audio_track
from tests.service_support import ServiceResponseTestCase


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
