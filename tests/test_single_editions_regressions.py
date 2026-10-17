import dataclasses
from unittest.mock import patch

from sonora.core.models import TrackInfo
from sonora.modules import organizer
from tests.regression_support import AudioRegressionTestCase


class TestSingleEditionsRegressions(AudioRegressionTestCase):
    def test_destination_collision_does_not_quarantine_other_isrc(self) -> None:
        source = self.root / "incoming.flac"
        target = self.root / "existing.flac"
        source.touch()
        target.touch()
        original = TrackInfo(target, artist="Alpha", title="Song", isrc="USAAA0000001")
        incoming = dataclasses.replace(original, file_path=source, isrc="USAAA0000002")
        registry = organizer.SingleDeduplicator()
        registry.register(original)
        with patch.object(
            organizer, "_read_file_info", return_value=(target, original)
        ):
            skipped, count = organizer._handle_single_deduplication(
                source,
                incoming,
                target,
                "Alpha Song",
                registry,
                self.root / ".duplicates",
                False,
            )
        self.assertFalse(skipped)
        self.assertEqual(count, 0)
        self.assertTrue(source.exists())

    def test_duplicate_quality_prefers_lossless_and_higher_lossy_bitrate(self) -> None:
        quality_cases = (
            (True, 16, 44100, 1000, False, 24, 96000, 320, False),
            (False, 24, 96000, 320, True, 16, 44100, 1000, True),
            (False, 16, 44100, 128, False, 16, 44100, 320, True),
        )
        for (
            original_lossless,
            original_bits,
            original_rate,
            original_bitrate,
            incoming_lossless,
            incoming_bits,
            incoming_rate,
            incoming_bitrate,
            replace_existing,
        ) in quality_cases:
            with self.subTest(
                replace_existing=replace_existing, incoming_lossless=incoming_lossless
            ):
                source = self.root / "incoming.flac"
                existing = self.root / "original.flac"
                source.touch()
                existing.touch()
                prior = TrackInfo(
                    existing,
                    artist="Alpha",
                    title="Song",
                    isrc="KNOWN",
                    is_lossless=original_lossless,
                    bits_per_sample=original_bits,
                    sample_rate=original_rate,
                    bitrate=original_bitrate,
                )
                incoming = dataclasses.replace(
                    prior,
                    file_path=source,
                    is_lossless=incoming_lossless,
                    bits_per_sample=incoming_bits,
                    sample_rate=incoming_rate,
                    bitrate=incoming_bitrate,
                )
                registry = organizer.SingleDeduplicator()
                registry.register(prior)
                with patch.object(
                    organizer, "_quarantine_duplicate_single"
                ) as quarantine:
                    skipped, count = organizer._handle_single_deduplication(
                        source,
                        incoming,
                        self.root / "destination.flac",
                        "Alpha Song",
                        registry,
                        self.root / ".duplicates",
                        False,
                    )
                self.assertEqual(skipped, not replace_existing)
                self.assertEqual(count, 1)
                self.assertEqual(
                    quarantine.call_args.args[0],
                    existing if replace_existing else source,
                )

    def test_missing_duplicate_does_not_discard_incoming_recording(self) -> None:
        source = self.root / "incoming.flac"
        source.touch()
        incoming = TrackInfo(source, artist="Alpha", title="Song")
        registry = organizer.SingleDeduplicator()
        registry.register(
            dataclasses.replace(incoming, file_path=self.root / "missing.flac")
        )
        with patch.object(organizer, "_quarantine_duplicate_single") as quarantine:
            skipped, count = organizer._handle_single_deduplication(
                source,
                incoming,
                self.root / "destination.flac",
                "Alpha Song",
                registry,
                self.root / ".duplicates",
                False,
            )
        self.assertFalse(skipped)
        self.assertEqual(count, 0)
        quarantine.assert_not_called()

    def test_organizer_reports_corrupt_files(self) -> None:
        (self.root / "broken.flac").write_bytes(b"not audio")
        failures: list[str] = []
        self.assertEqual(
            organizer.organize_library_singles(
                self.root, self.root / "Singles", dry_run=True, failures=failures
            ),
            0,
        )
        self.assertEqual(failures, [str(self.root / "broken.flac")])

    def test_dry_run_replacement_counts_same_destination_once(self) -> None:
        target = self.root / "existing.flac"
        target.touch()
        incoming = TrackInfo(
            self.root / "incoming.flac",
            artist="Alpha",
            title="Song",
            bits_per_sample=24,
        )
        registry = organizer.SingleDeduplicator()
        registry.register(
            dataclasses.replace(incoming, file_path=target, bits_per_sample=16)
        )
        with patch.object(organizer, "_quarantine_duplicate_single") as quarantine:
            skipped, removed_count = organizer._handle_single_deduplication(
                incoming.file_path,
                incoming,
                target,
                "Alpha Song",
                registry,
                self.root / ".duplicates",
                True,
            )
        self.assertFalse(skipped)
        self.assertEqual(removed_count, 1)
        quarantine.assert_called_once()
