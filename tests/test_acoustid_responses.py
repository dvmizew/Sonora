import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from sonora.services import acoustid as fingerprint_service
from sonora.services.acoustid import fingerprint_audio_file, lookup_acoustid
from tests.service_support import ServiceResponseTestCase


class TestAcoustidResponses(ServiceResponseTestCase):
    def test_duration_evidence_cannot_be_borrowed_from_another_acoustic_score(
        self,
    ) -> None:
        recording_id = "c8b03190-306c-4125-9b32-3f9d86d60a12"
        for high_duration, low_duration, expected in (
            (180, 100, None),
            (100, 180, recording_id),
            (0, 100, None),
        ):
            for reverse in (False, True):
                results = [
                    {
                        "score": score,
                        "recordings": [
                            {
                                "id": recording_id,
                                "title": "Original Song",
                                "artists": [{"name": "Alpha"}],
                                "duration": duration,
                            }
                        ],
                    }
                    for score, duration in ((0.99, high_duration), (0.1, low_duration))
                ]
                if reverse:
                    results.reverse()
                with (
                    self.subTest(high_duration=high_duration, reverse=reverse),
                    patch.object(
                        fingerprint_service,
                        "fingerprint_audio_file",
                        return_value=(100, "fingerprint"),
                    ),
                    patch.object(
                        fingerprint_service,
                        "get_cached_api",
                        return_value={"status": "ok", "results": results},
                    ),
                    patch.object(fingerprint_service, "set_cached_api"),
                ):
                    self.assertEqual(
                        lookup_acoustid(
                            Path(__file__), "token", "Alpha", "Original Song"
                        ),
                        expected,
                    )

    def test_conflicting_durations_for_same_score_require_abstention(self) -> None:
        recording_id = "c8b03190-306c-4125-9b32-3f9d86d60a12"
        for durations, expected in (
            ((100, 180), None),
            ((180, 100), None),
            ((100, 100), recording_id),
        ):
            payload = {
                "status": "ok",
                "results": [
                    {
                        "score": 0.99,
                        "recordings": [
                            {
                                "id": recording_id,
                                "title": "Original Song",
                                "artists": [{"name": "Alpha"}],
                                "duration": duration,
                            }
                        ],
                    }
                    for duration in durations
                ],
            }
            with (
                self.subTest(durations=durations),
                patch.object(
                    fingerprint_service,
                    "fingerprint_audio_file",
                    return_value=(100, "fingerprint"),
                ),
                patch.object(
                    fingerprint_service, "get_cached_api", return_value=payload
                ),
                patch.object(fingerprint_service, "set_cached_api"),
            ):
                self.assertEqual(
                    lookup_acoustid(Path(__file__), "token", "Alpha", "Original Song"),
                    expected,
                )

    def test_known_artist_rejects_reported_polluted_fingerprint_links(self) -> None:
        for (
            title,
            candidate_artist,
            recording_id,
            audio_duration,
            candidate_duration,
        ) in (
            (
                "Ice Cream",
                "Teenage Joans",
                "c9f7d8fd-dd46-44ac-8277-ebd4e4162085",
                205.714,
                210.0,
            ),
            (
                "Blame",
                "Whallex",
                "5a9fc94b-ec0c-4619-acc8-388a022630d0",
                168.379,
                166.4,
            ),
            (
                "Rich",
                "Kid Ink",
                "36779797-ec67-4e3f-88cf-fe49d4f0b129",
                163.689,
                162.462,
            ),
        ):
            payload = {
                "status": "ok",
                "results": [
                    {
                        "score": 1.0,
                        "recordings": [
                            {
                                "id": recording_id,
                                "title": title,
                                "artists": [{"name": candidate_artist}],
                                "duration": candidate_duration,
                            }
                        ],
                    }
                ],
            }
            with (
                self.subTest(title=title),
                patch.object(
                    fingerprint_service,
                    "fingerprint_audio_file",
                    return_value=(audio_duration, "fingerprint"),
                ),
                patch.object(
                    fingerprint_service, "get_cached_api", return_value=payload
                ),
                patch.object(fingerprint_service, "set_cached_api"),
            ):
                self.assertIsNone(
                    lookup_acoustid(Path(__file__), "token", "2Scratch", title)
                )

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
