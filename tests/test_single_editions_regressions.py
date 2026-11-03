import dataclasses
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import taglib

from sonora.audio.metadata import read_track_metadata
from sonora.core.cache import (
    get_api_cache_dir,
)
from sonora.core.logger import LOG
from sonora.core.models import TrackInfo
from sonora.modules import organizer
from sonora.modules.organizer import (
    SingleDeduplicator,
    _resolve_single_target_file,
    organize_library_singles,
)
from tests.regression_support import AudioRegressionTestCase
from tests.support import IsolatedTestCase


class TestSingleEditionsRegressions(AudioRegressionTestCase):
    def test_dry_run_labels_duplicate_quarantine_as_a_preview(self) -> None:
        source = self.root / "library"
        paths = [
            source / artist / "Singles" / "Alpha - Song" / "01 - Song.flac"
            for artist in ("Artist A", "Artist B")
        ]
        for path in paths:
            path.parent.mkdir(parents=True)
            path.touch()

        def read_file(path: Path) -> tuple[Path, TrackInfo]:
            return path, TrackInfo(path, artist="Alpha", title="Song")

        with (
            patch.object(organizer, "_read_file_info", side_effect=read_file),
            patch.object(LOG, "detail") as detail,
        ):
            organize_library_singles(
                source, source / "Singles", dry_run=True, max_threads=1
            )

        messages = [str(call.args[0]) for call in detail.call_args_list]
        self.assertIn("Would quarantine 1 duplicate single(s).", messages)
        self.assertNotIn("Removed 1 duplicate single(s).", messages)
        self.assertTrue(all(path.exists() for path in paths))
        self.assertFalse((source / "Singles" / ".duplicates").exists())

    def test_quarantine_keeps_lyrics_paired_when_names_collide(self) -> None:
        for collision_suffix in (".flac", ".lrc"):
            with self.subTest(collision_suffix=collision_suffix):
                source = self.root / collision_suffix[1:] / "Song.flac"
                source.parent.mkdir()
                source.write_bytes(b"incoming audio")
                source.with_suffix(".lrc").write_text("incoming lyrics")
                quarantine = source.parent / ".duplicates"
                quarantine.mkdir()
                collision = quarantine / f"Song{collision_suffix}"
                collision.write_bytes(b"existing recording")

                target = organizer._quarantine_file(source, quarantine)

                self.assertEqual(target.name, "Song (2).flac")
                self.assertEqual(target.read_bytes(), b"incoming audio")
                self.assertEqual(
                    target.with_suffix(".lrc").read_text(), "incoming lyrics"
                )
                self.assertEqual(collision.read_bytes(), b"existing recording")
                self.assertFalse(source.exists())
                self.assertFalse(source.with_suffix(".lrc").exists())

    def test_dry_run_reserves_destinations_for_distinct_recordings(self) -> None:
        source = self.root / "incoming"
        target = self.root / "Singles"
        originals = []
        for index in range(3):
            path = source / f"recording-{index}.flac"
            path.parent.mkdir(exist_ok=True)
            sf.write(path, np.zeros(4410 + index * 176400), 44100)
            with taglib.File(str(path)) as audio:
                audio.tags.update(
                    ARTIST=["Alpha"],
                    TITLE=["Song"],
                    ISRC=[f"USAAA000000{min(index, 1)}"],
                )
                audio.save()
            originals.append(path)
        expected = {
            target / "Alpha - Song" / "01 - Song.flac",
            target / "Alpha - Song [USAAA0000001]" / "01 - Song.flac",
            target / "Alpha - Song [USAAA0000001]" / "01 - Song (2).flac",
        }
        with patch.object(LOG, "detail") as detail:
            self.assertEqual(
                organize_library_singles(source, target, dry_run=True, max_threads=1),
                3,
            )
        previews = {
            str(call.args[0]).split(" -> ", 1)[1]
            for call in detail.call_args_list
            if str(call.args[0]).startswith("[DRY-RUN] Would move ")
        }
        self.assertEqual(previews, {str(path) for path in expected})
        self.assertTrue(all(path.exists() for path in originals))
        self.assertFalse(target.exists())
        self.assertEqual(organize_library_singles(source, target, max_threads=1), 3)
        self.assertEqual(set(target.rglob("*.flac")), expected)

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


class TestSingleIdentity(IsolatedTestCase):
    def test_guests_in_titles_artists_and_cobilling_remain_distinct(self) -> None:
        for artist_a, artist_b, title_a, title_b in (
            ("Artist", "Artist", "Song (feat. Alice)", "Song (feat. Bob)"),
            ("Artist feat. Alice", "Artist feat. Bob", "Song", "Song"),
            ("Artist & Alice", "Artist & Bob", "Song", "Song"),
        ):
            with self.subTest(artist=artist_a, title=title_a):
                original = TrackInfo(
                    file_path=Path("/a.flac"), artist=artist_a, title=title_a
                )
                original_snapshot = dataclasses.asdict(original)
                registry = SingleDeduplicator()
                registry.register(original)
                self.assertFalse(
                    registry.is_duplicate(
                        TrackInfo(
                            file_path=Path("/b.flac"),
                            artist=artist_b,
                            title=title_b,
                        )
                    )
                )
                self.assertEqual(dataclasses.asdict(original), original_snapshot)
                other_edition = TrackInfo(
                    file_path=Path("/b.flac"), artist=artist_b, title=title_b
                )
                self.assertNotEqual(
                    _resolve_single_target_file(
                        original.file_path, original, Path("/Singles")
                    ),
                    _resolve_single_target_file(
                        other_edition.file_path, other_edition, Path("/Singles")
                    ),
                )

    def test_equivalent_feature_encodings_are_duplicates(self) -> None:
        registry = SingleDeduplicator()
        registry.register(
            TrackInfo(
                file_path=Path("/a.flac"),
                artist="Artist",
                title="Song (feat. Guest)",
            )
        )
        self.assertTrue(
            registry.is_duplicate(
                TrackInfo(
                    file_path=Path("/b.flac"),
                    artist="Artist",
                    title="Song",
                    featured_artists="Guest",
                )
            )
        )

    def test_real_flac_preview_preserves_distinct_guests(self) -> None:
        source = get_api_cache_dir().parent / "fixtures"
        for index, guest in enumerate(("Alice", "Bob")):
            folder = source / f"Edition {index}"
            folder.mkdir(parents=True)
            audio_path = folder / "01 - Song.flac"
            sf.write(audio_path, np.zeros(4410), 44100, subtype="PCM_16")
            with taglib.File(str(audio_path)) as audio_tags:
                audio_tags.tags = {
                    "ARTIST": ["Artist"],
                    "TITLE": [f"Song (feat. {guest})"],
                }
                audio_tags.save()
            self.assertIsNone(read_track_metadata(audio_path).featured_artists)
        originals = {path: path.read_bytes() for path in source.rglob("*.flac")}
        self.assertEqual(
            organize_library_singles(
                source, source / "Singles", dry_run=True, max_threads=1
            ),
            2,
        )
        self.assertEqual({path: path.read_bytes() for path in originals}, originals)

    def test_layout_is_independent_of_library_size(self) -> None:
        source = get_api_cache_dir().parent / "library"
        source.mkdir(parents=True)
        track = TrackInfo(file_path=source / "Song.flac", artist="Artist", title="Song")
        destination = source / "Singles"
        expected = destination / "Artist - Song" / "01 - Song.flac"
        for count in range(8):
            (source / f"Artist {count}").mkdir()
            self.assertEqual(
                _resolve_single_target_file(track.file_path, track, destination),
                expected,
            )
