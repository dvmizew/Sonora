import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from sonora.services.acoustid import fingerprint_audio_file, lookup_acoustid
from tests.service_support import ServiceResponseTestCase


class TestAcoustidResponses(ServiceResponseTestCase):
    @patch("sonora.services.acoustid.acoustid")
    def test_lookup_acoustid(self, mock_acoustid: MagicMock) -> None:
        mock_acoustid.fingerprint_file.return_value = (120.0, "fingerprint_data_str")
        mock_acoustid.lookup.return_value = {}
        mock_acoustid.parse_lookup_result.return_value = [
            (0.95, "c8b03190-306c-4125-9b32-3f9d86d60a12", "Title", "Artist")
        ]

        with patch(
            "sonora.services.acoustid._fingerprint_file_with_timeout",
            return_value=(120.0, "fingerprint"),
        ):
            mbid = lookup_acoustid(Path(__file__), api_key="dummy_key")
        self.assertEqual(mbid, "c8b03190-306c-4125-9b32-3f9d86d60a12")

    def test_acoustid_no_api_key_returns_none(self) -> None:
        self.assertIsNone(lookup_acoustid(Path(__file__), api_key=""))

    @patch("sonora.services.acoustid.acoustid")
    def test_acoustid_low_score_returns_none(self, mock_acoustid: MagicMock) -> None:
        mock_acoustid.fingerprint_file.return_value = (100.0, "fp_data")
        mock_acoustid.lookup.return_value = {}
        mock_acoustid.parse_lookup_result.return_value = [
            (0.5, "c8b03190-306c-4125-9b32-3f9d86d60a12", "Title", "Artist")
        ]

        self.assertIsNone(lookup_acoustid(Path(__file__), api_key="dummy_key"))

    @patch("sonora.services.acoustid.acoustid")
    def test_acoustid_rejects_fingerprint_collision(
        self, mock_acoustid: MagicMock
    ) -> None:
        mock_acoustid.fingerprint_file.return_value = (216.0, "fp_data")
        mock_acoustid.lookup.return_value = {}
        # AcoustID returns high audio score (0.95) for unrelated song (e.g. Diego Modena - Brazil)
        mock_acoustid.parse_lookup_result.return_value = [
            (0.95, "cbe3bed2-4dc5-461d-b6e3-a5e11c9845d5", "Brazil", "Diego Modena")
        ]

        # When searching for Laurentiu Duta - Shining Heart, the collision must be rejected
        mbid = lookup_acoustid(
            Path(__file__),
            api_key="dummy_key",
            expected_title="Shining Heart",
            expected_artist="Laurentiu Duta",
        )
        self.assertIsNone(mbid)

    @patch("sonora.services.acoustid._fingerprint_file_with_timeout")
    def test_fingerprint_in_memory_cache(self, mock_fp: MagicMock) -> None:
        mock_fp.return_value = (180.0, "AQADtEmSJEqiJE")
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_file = Path(tmp_dir) / "track.flac"
            audio_file.write_bytes(b"dummy audio data for fingerprint")

            dur1, fp1 = fingerprint_audio_file(audio_file)
            self.assertEqual(dur1, 180.0)
            self.assertEqual(fp1, "AQADtEmSJEqiJE")
            self.assertEqual(mock_fp.call_count, 1)

            # Second call hits in-memory cache without calling acoustid.fingerprint_file again
            dur2, fp2 = fingerprint_audio_file(audio_file)
            self.assertEqual(dur2, 180.0)
            self.assertEqual(fp2, "AQADtEmSJEqiJE")
            self.assertEqual(mock_fp.call_count, 1)

    @patch("sonora.services.acoustid.acoustid")
    def test_lookup_acoustid_retries_on_webservice_error(
        self, mock_acoustid: MagicMock
    ) -> None:
        import acoustid

        from sonora.services.acoustid import lookup_acoustid

        mock_acoustid.fingerprint_file.return_value = (180.0, "fp_test")
        mock_acoustid.lookup.side_effect = [
            acoustid.WebServiceError("HTTP 503 Service Unavailable"),
            {
                "status": "ok",
                "results": [{"score": 0.95, "recordings": [{"id": "uuid-test"}]}],
            },
        ]
        mock_acoustid.parse_lookup_result.return_value = [
            (0.95, "c8b03190-306c-4125-9b32-3f9d86d60a12", "Track", "Artist")
        ]

        with (
            patch(
                "sonora.services.acoustid._fingerprint_file_with_timeout",
                return_value=(120.0, "fingerprint"),
            ),
            patch("sonora.services.acoustid.get_cached_api", return_value=None),
            patch("sonora.services.acoustid.set_cached_api"),
            patch("time.sleep"),
        ):
            mbid = lookup_acoustid(Path(__file__), api_key="dummy_key")

        self.assertEqual(mbid, "c8b03190-306c-4125-9b32-3f9d86d60a12")
        self.assertEqual(mock_acoustid.lookup.call_count, 2)
