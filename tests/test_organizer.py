import dataclasses
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import soundfile as sf
import taglib
from PIL import Image

from sonora.audio import art
from sonora.audio import art as art_module
from sonora.audio import metadata as metadata_module
from sonora.audio import replaygain as replaygain_module
from sonora.audio.cuesheet import read_cuesheet_content
from sonora.audio.key import key_to_camelot
from sonora.audio.metadata import (
    clear_metadata_cache,
    read_track_metadata,
    write_track_metadata,
)
from sonora.audio.replaygain import LoudnessMeasurement, _album_loudness
from sonora.core import sidecars
from sonora.core.cache import get_api_cache_dir
from sonora.core.config import SonoraConfig
from sonora.core.files import (
    atomic_audio_edit,
    atomic_write_bytes,
    move_file_exclusively,
)
from sonora.core.library_paths import get_single_release_title
from sonora.core.logger import LOG
from sonora.core.models import RenameReport, TrackInfo
from sonora.core.playlists import update_local_playlists
from sonora.core.sidecars import find_companion_lyrics, relocate_companion_lyrics
from sonora.core.text import normalize_date
from sonora.core.titles import score_artist_title_match
from sonora.modules import (
    backup,
    organizer,
    renamer,
    tagger,
)
from sonora.modules import track_processing as track_processing_module
from sonora.modules.checker import check_library
from sonora.modules.normalization import normalize_track_metadata
from sonora.modules.organizer import (
    SingleDeduplicator,
    _resolve_single_target_file,
    consolidate_duplicate_artist_dirs,
    organize_library_singles,
)
from sonora.modules.renamer import (
    rename_album_folder,
    rename_directory_files,
    rename_single_folder,
    rename_track_file,
    sync_lrc_metadata,
)
from sonora.modules.tagger import _AlbumMetadataContext, process_single_track
from sonora.modules.track_processing import (
    _update_lyrics,
    normalize_library,
    normalize_single_track,
)
from sonora.services import lyrics
from tests.fixtures import image_bytes, write_silent_wav
from tests.module_support import ModuleTestCase
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
            return (path, TrackInfo(path, artist="Alpha", title="Song"))

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
                organize_library_singles(source, target, dry_run=True, max_threads=1), 3
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
                            file_path=Path("/b.flac"), artist=artist_b, title=title_b
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
                file_path=Path("/a.flac"), artist="Artist", title="Song (feat. Guest)"
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


class TestOrganizationWorkflows(ModuleTestCase):
    def test_duplicate_artist_dirs_with_distinct_albums_are_consolidated(self) -> None:
        source = self.tmp_path / "library"
        recordings = {
            source / "Artist" / "First Album" / "01 - First.flac": "first-release",
            source / "ARTIST" / "Second Album" / "01 - Second.flac": "second-release",
        }
        for path in recordings:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"audio")

        def read_file(path: Path) -> tuple[Path, TrackInfo]:
            return (
                path,
                TrackInfo(
                    path,
                    artist="Artist",
                    album=path.parent.name,
                    musicbrainz_albumid=recordings[path],
                ),
            )

        with patch("sonora.modules.organizer._read_file_info", side_effect=read_file):
            consolidated = consolidate_duplicate_artist_dirs(source, dry_run=True)
        self.assertEqual(consolidated, 1)
        self.assertTrue(all(path.exists() for path in recordings))
        with patch("sonora.modules.organizer._read_file_info", side_effect=read_file):
            consolidated = consolidate_duplicate_artist_dirs(source)
        self.assertEqual(consolidated, 1)
        self.assertTrue(
            (source / "Artist" / "First Album" / "01 - First.flac").exists()
        )
        self.assertTrue(
            (source / "Artist" / "Second Album" / "01 - Second.flac").exists()
        )
        self.assertFalse((source / "ARTIST").exists())

    def test_artist_consolidation_preserves_distinct_musicbrainz_releases(self) -> None:
        source = self.tmp_path / "library"
        standard = source / "Artist" / "Album"
        expanded = source / "Artist" / "ALBUM"
        standard_track = standard / "01 - Song.flac"
        expanded_track = expanded / "01 - Song.flac"
        for path in (standard_track, expanded_track):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"audio")

        def read_file(path: Path) -> tuple[Path, TrackInfo]:
            release_id = (
                "standard-release" if path == standard_track else "expanded-release"
            )
            return (
                path,
                TrackInfo(
                    path,
                    artist="Artist",
                    album="Album",
                    title="Song",
                    musicbrainz_albumid=release_id,
                    musicbrainz_releasegroupid="same-release-group",
                    track_number=1,
                    total_tracks=1,
                ),
            )

        with patch("sonora.modules.organizer._read_file_info", side_effect=read_file):
            consolidated = consolidate_duplicate_artist_dirs(source, dry_run=True)
        self.assertEqual(consolidated, 0)
        self.assertTrue(standard_track.exists())
        self.assertTrue(expanded_track.exists())

    def test_duplicate_artist_dirs_with_distinct_singles_releases_consolidate(
        self,
    ) -> None:
        source = self.tmp_path / "library"
        recordings = {
            source / "Artist" / "Singles" / "01 - First.flac": "first-release",
            source / "ARTIST" / "Singles" / "01 - Second.flac": "second-release",
        }
        for path in recordings:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"audio")

        def read_file(path: Path) -> tuple[Path, TrackInfo]:
            return (
                path,
                TrackInfo(
                    path,
                    artist="Artist",
                    title=path.stem,
                    musicbrainz_albumid=recordings[path],
                ),
            )

        with patch("sonora.modules.organizer._read_file_info", side_effect=read_file):
            consolidated = consolidate_duplicate_artist_dirs(source, dry_run=True)
        self.assertEqual(consolidated, 1)
        self.assertTrue(all(path.exists() for path in recordings))

    def test_organize_reads_one_track_folders_in_parallel(self) -> None:
        source = self.tmp_path / "parallel-library"
        files = [source / f"Folder {index}" / "01 - Song.wav" for index in range(4)]
        for file_path in files:
            file_path.parent.mkdir(parents=True)
            file_path.write_bytes(b"audio")
        active = 0
        maximum = 0
        lock = threading.Lock()

        def read_file(file_path: Path) -> tuple[Path, TrackInfo]:
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return (
                file_path,
                TrackInfo(file_path, artist="Artist", title=file_path.parent.name),
            )

        with patch("sonora.modules.organizer._read_file_info", side_effect=read_file):
            organize_library_singles(
                source, source / "Singles", dry_run=True, max_threads=4
            )
        self.assertGreater(maximum, 1)

    def test_artist_consolidation_preserves_root_and_moved_playlist_references(
        self,
    ) -> None:
        source = self.tmp_path / "library"
        canonical = source / "Primary" / "Album" / "01 - Song.wav"
        incoming = source / "PRIMARY!" / "Album" / "01 - Song.wav"
        for path, body in (
            (canonical, b"first recording"),
            (incoming, b"second recording"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        root_playlist = source / "library.m3u"
        root_playlist.write_text("PRIMARY!/Album/01 - Song.wav\n")
        moved_playlist = incoming.parent / "album.m3u"
        moved_playlist.write_text("01 - Song.wav\n")
        incoming.with_suffix(".lrc").write_text("[00:01.00] Second recording lyrics\n")
        updated: list[Path] = []
        with patch(
            "sonora.modules.organizer.read_track_metadata",
            side_effect=lambda path: TrackInfo(
                path,
                artist="Primary",
                title="Song",
                album="Album",
                track_number=1,
                total_tracks=3,
            ),
        ):
            organize_library_singles(
                source,
                source / "Singles",
                dry_run=True,
                max_threads=1,
                updated_playlists=updated,
            )
            expected_playlists = {root_playlist, canonical.parent / "album.m3u"}
            self.assertEqual(set(updated), expected_playlists)
            self.assertTrue(incoming.exists())
            self.assertTrue(incoming.with_suffix(".lrc").exists())
            self.assertFalse((source / "Singles").exists())
            updated.clear()
            organize_library_singles(
                source, source / "Singles", max_threads=1, updated_playlists=updated
            )
        for playlist in (root_playlist, canonical.parent / "album.m3u"):
            reference = playlist.read_text().strip()
            self.assertEqual(
                (playlist.parent / reference).read_bytes(), b"second recording"
            )
            self.assertIn(playlist, updated)
        self.assertEqual(canonical.read_bytes(), b"first recording")
        self.assertEqual(
            (canonical.parent / "01 - Song (2).lrc").read_text(),
            "[00:01.00] Second recording lyrics\n",
        )
        self.assertFalse(canonical.with_suffix(".lrc").exists())

    @patch("sonora.modules.organizer.read_track_metadata")
    def test_organize_library_singles(self, mock_read: Any) -> None:
        src_dir = self.tmp_path / "source"
        target_dir = self.tmp_path / "Singles"
        audio_file_1 = src_dir / "single.wav"
        write_silent_wav(audio_file_1)
        mock_read.return_value = TrackInfo(
            file_path=audio_file_1, artist="Single Artist", title="Single Song"
        )
        moved = organize_library_singles(src_dir, target_dir)
        self.assertEqual(moved, 1)
        self.assertTrue(
            (
                target_dir / "Single Artist - Single Song" / "01 - Single Song.wav"
            ).exists()
        )

    @patch("sonora.modules.organizer.read_track_metadata")
    def test_organize_library_singles_with_lrc(self, mock_read: Any) -> None:
        src_dir = self.tmp_path / "source_lrc"
        target_dir = self.tmp_path / "Singles"
        audio_file_1 = src_dir / "single.wav"
        lrc = src_dir / "single.lrc"
        write_silent_wav(audio_file_1)
        lrc.write_text("[00:01.00] lyrics", encoding="utf-8")
        mock_read.return_value = TrackInfo(
            file_path=audio_file_1, artist="Single Artist", title="Single Song"
        )
        moved = organize_library_singles(src_dir, target_dir)
        self.assertEqual(moved, 1)
        self.assertTrue(
            (
                target_dir / "Single Artist - Single Song" / "01 - Single Song.wav"
            ).exists()
        )
        self.assertTrue(
            (
                target_dir / "Single Artist - Single Song" / "01 - Single Song.lrc"
            ).exists()
        )

    @patch("sonora.modules.organizer.read_track_metadata")
    def test_organize_library_singles_quarantines_duplicates(
        self, mock_read: Any
    ) -> None:
        src_dir = self.tmp_path / "Singles" / "source"
        target_dir = self.tmp_path / "OrganizedSingles"
        audio1 = src_dir / "track1.wav"
        audio2 = src_dir / "track2.wav"
        audio3 = src_dir / "track3.wav"
        lrc2 = src_dir / "track2.lrc"
        write_silent_wav(audio1)
        write_silent_wav(audio2)
        write_silent_wav(audio3)
        lrc2.write_text("[00:01.00] dup lyrics", encoding="utf-8")

        def mock_read_meta(path: Path) -> TrackInfo:
            if path == audio1:
                return TrackInfo(
                    file_path=audio1,
                    artist="Dup Artist",
                    title="Dup Song",
                    isrc="US1234567890",
                )
            if path == audio2:
                return TrackInfo(
                    file_path=audio2,
                    artist="Dup Artist",
                    title="Dup Song",
                    isrc="US1234567890",
                )
            return TrackInfo(
                file_path=audio3,
                artist="Dup Artist",
                title="Dup Song",
                isrc="US1234567890",
            )

        mock_read.side_effect = mock_read_meta
        moved_dry = organize_library_singles(
            src_dir, target_dir, dry_run=True, max_threads=1
        )
        self.assertEqual(moved_dry, 1)
        self.assertTrue(audio1.exists())
        self.assertTrue(audio2.exists())
        self.assertTrue(audio3.exists())
        self.assertTrue(lrc2.exists())
        moved = organize_library_singles(
            src_dir, target_dir, dry_run=False, max_threads=1
        )
        self.assertEqual(moved, 1)
        self.assertTrue(
            (target_dir / "Dup Artist - Dup Song" / "01 - Dup Song.wav").exists()
        )
        quarantine_dir = target_dir / ".duplicates"
        self.assertTrue(quarantine_dir.exists())
        self.assertTrue((quarantine_dir / "track2.wav").exists())
        self.assertTrue((quarantine_dir / "track2.lrc").exists())
        self.assertTrue((quarantine_dir / "track3.wav").exists())

    @patch("sonora.modules.organizer.get_primary_artist", side_effect=lambda a: a)
    @patch("sonora.modules.organizer.read_track_metadata")
    def test_organize_multiple_folders_and_albums(
        self, mock_read: Any, mock_primary: Any
    ) -> None:
        src_dir = self.tmp_path / "multi_source"
        folder1 = src_dir / "folder1"
        folder2 = src_dir / "album_folder"
        folder3 = src_dir / "folder3"
        f1_track = folder1 / "f1.wav"
        f2_track1 = folder2 / "f2_1.wav"
        f2_track2 = folder2 / "f2_2.wav"
        f2_track3 = folder2 / "f2_3.wav"
        f3_track = folder3 / "f3.wav"
        for f in [f1_track, f2_track1, f2_track2, f2_track3, f3_track]:
            write_silent_wav(f)

        def mock_read_metadata(path: Path) -> TrackInfo:
            if path == f1_track:
                return TrackInfo(
                    file_path=f1_track,
                    artist="Artist One",
                    title="Track One",
                    album="Single",
                )
            if path == f2_track1:
                return TrackInfo(
                    file_path=f2_track1,
                    artist="Album Artist",
                    title="Song A",
                    album="Full Album",
                )
            if path == f2_track2:
                return TrackInfo(
                    file_path=f2_track2,
                    artist="Album Artist",
                    title="Song B",
                    album="Full Album",
                )
            if path == f2_track3:
                return TrackInfo(
                    file_path=f2_track3,
                    artist="Album Artist",
                    title="Song C",
                    album="Full Album",
                )
            if path == f3_track:
                return TrackInfo(
                    file_path=f3_track,
                    artist="Artist Three",
                    title="Track Three",
                    album="Single",
                )
            return TrackInfo(file_path=path)

        mock_read.side_effect = mock_read_metadata
        target_dir = self.tmp_path / "Singles"
        moved = organize_library_singles(src_dir, target_dir)
        self.assertEqual(moved, 2)
        self.assertTrue(
            (target_dir / "Artist One - Track One" / "01 - Track One.wav").exists()
        )
        self.assertTrue(
            (
                target_dir / "Artist Three - Track Three" / "01 - Track Three.wav"
            ).exists()
        )
        self.assertTrue(f2_track1.exists())
        self.assertTrue(f2_track2.exists())
        self.assertTrue(f2_track3.exists())

    def test_single_deduplicator_duration_safety(self) -> None:
        deduplicator = SingleDeduplicator()
        t1 = TrackInfo(
            file_path=Path("/tmp/t1.flac"),
            artist="Artist",
            title="Song",
            duration=200.0,
            bits_per_sample=16,
        )
        deduplicator.register(t1)
        t2_remix = TrackInfo(
            file_path=Path("/tmp/t2.flac"),
            artist="Artist",
            title="Song",
            duration=260.0,
            bits_per_sample=24,
        )
        self.assertFalse(deduplicator.is_duplicate(t2_remix))
        t3_dup = TrackInfo(
            file_path=Path("/tmp/t3.flac"),
            artist="Artist",
            title="Song",
            duration=203.0,
            bits_per_sample=16,
        )
        self.assertFalse(deduplicator.is_duplicate(t3_dup))
        t3_dup.duration = 202.0
        self.assertTrue(deduplicator.is_duplicate(t3_dup))

    def test_get_single_release_title(self) -> None:
        t1 = TrackInfo(
            file_path=Path("/tmp/t1.flac"),
            artist="DJ Sava",
            title="I Loved You",
            album="I Loved You (Monoir Remix) (feat. Irina Rimes)",
        )
        self.assertEqual(get_single_release_title(t1), "I Loved You (Monoir Remix)")
        t2 = TrackInfo(
            file_path=Path("/tmp/t2.flac"),
            artist="DJ Sava",
            title="I Loved You (Monoir Remix)",
            album="I Loved You (Monoir Remix)",
        )
        self.assertEqual(get_single_release_title(t2), "I Loved You (Monoir Remix)")
        t3 = TrackInfo(
            file_path=Path("/tmp/t3.flac"),
            artist="Dennis Lloyd",
            title="Alien",
            album="Alien",
        )
        self.assertEqual(get_single_release_title(t3), "Alien")

    @patch("sonora.modules.checker.read_track_metadata")
    def test_check_library_singles_folder_sequence_preservation(
        self, mock_read: Any
    ) -> None:
        singles_dir = self.tmp_path / "Singles" / "Artist - Song"
        singles_dir.mkdir(parents=True, exist_ok=True)
        file1 = singles_dir / "12 - Song.wav"
        write_silent_wav(file1)
        mock_read.return_value = TrackInfo(
            file_path=file1,
            artist="Artist",
            title="Song",
            album="Song",
            album_artist="Artist",
            track_number=12,
            disc_number=1,
            total_tracks=1,
        )
        report = check_library(self.tmp_path)
        folder_issues = report.issues.get(str(singles_dir), [])
        self.assertFalse(
            any("Missing track numbers in sequence" in issue for issue in folder_issues)
        )

    def test_organize_library_singles_skips_album_folders(self) -> None:
        album_dir = self.tmp_path / "AlbumFolder"
        album_dir.mkdir()
        audio_file_1 = album_dir / "01.wav"
        audio_file_2 = album_dir / "02.wav"
        audio_file_3 = album_dir / "03.wav"
        write_silent_wav(audio_file_1)
        write_silent_wav(audio_file_2)
        write_silent_wav(audio_file_3)
        target_dir = self.tmp_path / "Singles"
        with patch("sonora.modules.organizer.read_track_metadata") as mock_read:
            mock_read.return_value = TrackInfo(
                file_path=audio_file_1, artist="Artist", album="Full Album"
            )
            moved = organize_library_singles(self.tmp_path, target_dir)
            self.assertEqual(moved, 0)
            self.assertTrue(audio_file_1.exists())
            self.assertTrue(audio_file_2.exists())
            self.assertTrue(audio_file_3.exists())

    def test_organize_does_not_move_album_when_source_is_album_folder(self) -> None:
        album_dir = self.tmp_path / "Artist - Full Album"
        album_dir.mkdir()
        audio_files = [album_dir / f"{number:02d}.wav" for number in range(1, 4)]
        for audio_file in audio_files:
            write_silent_wav(audio_file)

        def read_track(path: Path) -> TrackInfo:
            return TrackInfo(
                file_path=path,
                artist="Artist",
                album="Full Album",
                title=path.stem,
                track_number=int(path.stem),
                total_tracks=12,
            )

        with patch(
            "sonora.modules.organizer.read_track_metadata", side_effect=read_track
        ):
            moved = organize_library_singles(album_dir, self.tmp_path / "Singles")
        self.assertEqual(moved, 0)
        self.assertTrue(all(path.exists() for path in audio_files))

    def test_organize_library_in_flac_container_does_not_dismantle_albums(self) -> None:
        flac_root = self.tmp_path / "Music" / "FLAC"
        album_dir = flac_root / "21 Savage" / "21 Savage - Savage Mode"
        album_dir.mkdir(parents=True)
        track_1 = album_dir / "01 - No Advance.wav"
        track_2 = album_dir / "02 - No Heart.wav"
        track_3 = album_dir / "03 - X.wav"
        write_silent_wav(track_1)
        write_silent_wav(track_2)
        write_silent_wav(track_3)
        target_dir = flac_root / "Singles"
        with patch("sonora.modules.organizer.read_track_metadata") as mock_read:

            def side_effect(path: Path) -> TrackInfo:
                return TrackInfo(
                    file_path=path,
                    artist="21 Savage",
                    album="Savage Mode",
                    total_tracks=3,
                    track_number=1,
                    title=path.stem,
                )

            mock_read.side_effect = side_effect
            moved = organize_library_singles(flac_root, target_dir)
            self.assertEqual(moved, 0)
            self.assertTrue(track_1.exists())
            self.assertTrue(track_2.exists())
            self.assertTrue(track_3.exists())

    def test_organize_incomplete_album_download_not_moved(self) -> None:
        album_dir = self.tmp_path / "Artist - Incomplete Album"
        album_dir.mkdir()
        track_1 = album_dir / "05 - Song Five.wav"
        track_2 = album_dir / "06 - Song Six.wav"
        write_silent_wav(track_1)
        write_silent_wav(track_2)
        target_dir = self.tmp_path / "Singles"
        with patch("sonora.modules.organizer.read_track_metadata") as mock_read:

            def side_effect(path: Path) -> TrackInfo:
                return TrackInfo(
                    file_path=path,
                    artist="Artist",
                    album="Incomplete Album",
                    total_tracks=12,
                    track_number=5,
                    title=path.stem,
                )

            mock_read.side_effect = side_effect
            moved = organize_library_singles(self.tmp_path, target_dir)
            self.assertEqual(moved, 0)
            self.assertTrue(track_1.exists())
            self.assertTrue(track_2.exists())

    def test_organize_already_organized_single_skipped_cleanly(self) -> None:
        single_dir = self.tmp_path / "Singles" / "Adele - Hit Song"
        single_dir.mkdir(parents=True)
        track_file = single_dir / "01 - Hit Song.wav"
        write_silent_wav(track_file)
        with patch("sonora.modules.organizer.read_track_metadata") as mock_read:
            mock_read.return_value = TrackInfo(
                file_path=track_file,
                artist="Adele",
                title="Hit Song",
                album="Hit Song",
                total_tracks=1,
                track_number=1,
            )
            moved = organize_library_singles(self.tmp_path, self.tmp_path / "Singles")
            self.assertEqual(moved, 0)
            self.assertTrue(track_file.exists())

    def test_organize_bonus_edition_tracks_do_not_dismantle_album(self) -> None:
        album_dir = self.tmp_path / "Avicii" / "Avicii - True"
        album_dir.mkdir(parents=True)
        files: list[Path] = []
        for i in range(1, 14):
            f = album_dir / f"{i:02d} - Track {i}.wav"
            write_silent_wav(f)
            files.append(f)
        target_dir = self.tmp_path / "Singles"
        with patch("sonora.modules.organizer.read_track_metadata") as mock_read:

            def side_effect(path: Path) -> TrackInfo:
                track_num = int(path.stem.split(" - ")[0])
                album_title = "True (Bonus Edition)" if track_num == 13 else "True"
                return TrackInfo(
                    file_path=path,
                    artist="Avicii",
                    album=album_title,
                    title=f"Track {track_num}",
                    track_number=track_num,
                    total_tracks=13,
                )

            mock_read.side_effect = side_effect
            moved = organize_library_singles(self.tmp_path, target_dir)
            self.assertEqual(moved, 0)
            for f in files:
                self.assertTrue(f.exists())

    def test_organize_deluxe_version_variations_do_not_dismantle_album(self) -> None:
        album_dir = self.tmp_path / "Lil Peep" / "Lil Peep - COWYS2"
        album_dir.mkdir(parents=True)
        files: list[Path] = []
        for i in range(1, 11):
            f = album_dir / f"{i:02d} - Song {i}.wav"
            write_silent_wav(f)
            files.append(f)
        target_dir = self.tmp_path / "Singles"
        with patch("sonora.modules.organizer.read_track_metadata") as mock_read:

            def side_effect(path: Path) -> TrackInfo:
                track_num = int(path.stem.split(" - ")[0])
                album_title = (
                    "Come Over When You're Sober, Pt. 2 (og version)"
                    if track_num > 7
                    else "Come Over When You're Sober, Pt. 2"
                )
                return TrackInfo(
                    file_path=path,
                    artist="Lil Peep",
                    album=album_title,
                    title=f"Song {track_num}",
                    track_number=track_num,
                    total_tracks=10,
                )

            mock_read.side_effect = side_effect
            moved = organize_library_singles(self.tmp_path, target_dir)
            self.assertEqual(moved, 0)
            for f in files:
                self.assertTrue(f.exists())

    def test_single_deduplicator_isrc_collision_prevention(self) -> None:
        deduplicator = SingleDeduplicator()
        album_track = TrackInfo(
            file_path=Path("/music/album/03 - The Walls.flac"),
            artist="Chase Atlantic",
            title="The Walls",
            isrc="USWB11701708",
        )
        deduplicator.register(album_track)
        colliding_single = TrackInfo(
            file_path=Path("/music/singles/01 - Tidal Wave.flac"),
            artist="Chase Atlantic",
            title="Tidal Wave",
            isrc="USWB11701708",
        )
        self.assertFalse(deduplicator.is_duplicate(colliding_single))

    def test_single_deduplicator_similar_title_isrc_match(self) -> None:
        deduplicator = SingleDeduplicator()
        album_track = TrackInfo(
            file_path=Path("/music/album/02 - Missing You.flac"),
            artist="Cheyanne",
            title="Missing You",
            isrc="QZDA52149872",
        )
        deduplicator.register(album_track)
        duplicate_single = TrackInfo(
            file_path=Path("/music/singles/01 - Missin' You.flac"),
            artist="Cheyanne",
            title="Missin' You",
            isrc="QZDA52149872",
        )
        self.assertTrue(deduplicator.is_duplicate(duplicate_single))

    def test_single_deduplicator_featured_artists_distinction(self) -> None:
        deduplicator = SingleDeduplicator()
        track_lil_tjay = TrackInfo(
            file_path=Path(
                "/music/singles/Lithe - Fall Back (feat. Lil Tjay)/01 - Fall Back.flac"
            ),
            artist="Lithe",
            title="Fall Back",
            featured_artists="Lil Tjay",
            isrc="USUG12406285",
            duration=160.0,
        )
        deduplicator.register(track_lil_tjay)
        track_nav = TrackInfo(
            file_path=Path(
                "/music/singles/Lithe - Fall Back (feat. NAV)/01 - Fall Back.flac"
            ),
            artist="Lithe",
            title="Fall Back",
            featured_artists="NAV",
            isrc="AUGBT2416821",
            duration=160.0,
        )
        self.assertFalse(deduplicator.is_duplicate(track_nav))

    def test_single_deduplicator_different_isrc_not_duplicate(self) -> None:
        deduplicator = SingleDeduplicator()
        track_version_one = TrackInfo(
            file_path=Path("/music/track1.flac"),
            artist="Artist",
            title="Song",
            isrc="ISRC11111111",
            duration=200.0,
        )
        deduplicator.register(track_version_one)
        track_version_two = TrackInfo(
            file_path=Path("/music/track2.flac"),
            artist="Artist",
            title="Song",
            isrc="ISRC22222222",
            duration=200.0,
        )
        self.assertFalse(deduplicator.is_duplicate(track_version_two))

    def test_organize_mixed_artist_folder_moves_singles_and_preserves_albums(
        self,
    ) -> None:
        artist_dir = self.tmp_path / "Artist"
        artist_dir.mkdir(parents=True)
        single_1 = artist_dir / "Single One.wav"
        single_2 = artist_dir / "Single Two.wav"
        album_t1 = artist_dir / "01 - Album Intro.wav"
        album_t2 = artist_dir / "02 - Album Song.wav"
        album_t3 = artist_dir / "03 - Album Outro.wav"
        for f in [single_1, single_2, album_t1, album_t2, album_t3]:
            write_silent_wav(f)

        def mock_read_metadata(path: Path) -> TrackInfo:
            if path == single_1:
                return TrackInfo(
                    file_path=single_1,
                    artist="Artist",
                    title="Single One",
                    album="Single One",
                    track_number=1,
                    total_tracks=1,
                )
            if path == single_2:
                return TrackInfo(
                    file_path=single_2,
                    artist="Artist",
                    title="Single Two",
                    album="Single Two",
                    track_number=1,
                    total_tracks=1,
                )
            if path == album_t1:
                return TrackInfo(
                    file_path=album_t1,
                    artist="Artist",
                    title="Album Intro",
                    album="Full Album",
                    track_number=1,
                    total_tracks=3,
                )
            if path == album_t2:
                return TrackInfo(
                    file_path=album_t2,
                    artist="Artist",
                    title="Album Song",
                    album="Full Album",
                    track_number=2,
                    total_tracks=3,
                )
            if path == album_t3:
                return TrackInfo(
                    file_path=album_t3,
                    artist="Artist",
                    title="Album Outro",
                    album="Full Album",
                    track_number=3,
                    total_tracks=3,
                )
            return TrackInfo(file_path=path)

        target_singles = self.tmp_path / "Singles"
        with patch(
            "sonora.modules.organizer.read_track_metadata",
            side_effect=mock_read_metadata,
        ):
            moved = organize_library_singles(self.tmp_path, target_singles)
        self.assertEqual(moved, 2)
        self.assertTrue(
            (target_singles / "Artist - Single One" / "01 - Single One.wav").exists()
        )
        self.assertTrue(
            (target_singles / "Artist - Single Two" / "01 - Single Two.wav").exists()
        )
        self.assertTrue(album_t1.exists())
        self.assertTrue(album_t2.exists())
        self.assertTrue(album_t3.exists())

    def test_organize_cli_destination_scoped_to_singles_directory(self) -> None:
        from sonora.cli.library import organize

        singles_dir = self.tmp_path / "Singles"
        singles_dir.mkdir(parents=True)
        track_file = singles_dir / "Hit Song.wav"
        write_silent_wav(track_file)
        with (
            patch(
                "sonora.modules.organizer.read_track_metadata",
                return_value=TrackInfo(
                    file_path=track_file,
                    artist="Artist",
                    title="Hit Song",
                    album="Hit Song",
                    track_number=1,
                    total_tracks=1,
                ),
            ),
            patch("sonora.cli.library.organize_library_singles") as mock_organize,
        ):
            mock_organize.return_value = 1
            ret = organize(path=singles_dir, dry_run=True)
            self.assertEqual(ret, 0)
            mock_organize.assert_called_once()
            called_source, called_dest = mock_organize.call_args[0][:2]
            self.assertEqual(called_source, singles_dir)
            self.assertEqual(called_dest, singles_dir)

    def test_cleanup_empty_dirs_prunes_abandoned_singles_folders_with_only_artwork(
        self,
    ) -> None:
        from sonora.modules.organizer import cleanup_empty_dirs

        singles_dir = self.tmp_path / "Singles"
        abandoned_folder = singles_dir / "Artist - Abandoned Single"
        abandoned_folder.mkdir(parents=True)
        (abandoned_folder / "artist.jpg").write_bytes(b"image")
        (abandoned_folder / "label.png").write_bytes(b"image")
        active_folder = singles_dir / "Artist - Active Single"
        active_folder.mkdir(parents=True)
        active_audio = active_folder / "01 - Active Single.wav"
        write_silent_wav(active_audio)
        (active_folder / "cover.jpg").write_bytes(b"image")
        removed = cleanup_empty_dirs(self.tmp_path)
        self.assertEqual(removed, 1)
        self.assertFalse(abandoned_folder.exists())
        self.assertTrue(active_folder.exists())
        self.assertTrue(active_audio.exists())


class TestRenameWorkflows(ModuleTestCase):
    def test_sync_lrc_metadata(self) -> None:
        lrc_file = self.tmp_path / "test.lrc"
        lrc_file.write_text("[00:10.00] Line 1\n", encoding="utf-8")
        success = sync_lrc_metadata(lrc_file, "Artist Name", "Track Title")
        self.assertTrue(success)
        content = lrc_file.read_text(encoding="utf-8")
        self.assertIn("[ar:Artist Name]", content)
        self.assertIn("[ti:Track Title]", content)
        self.assertIn("[00:10.00] Line 1", content)

    def test_sync_lrc_metadata_preserves_conflicting_headers(self) -> None:
        lrc_file = self.tmp_path / "existing.lrc"
        original = "[ar:Old Artist]\n[ti:Old Title]\n[00:15.00] Line 2\n"
        lrc_file.write_text(original, encoding="utf-8")
        changed = sync_lrc_metadata(lrc_file, "New Artist", "New Title")
        self.assertFalse(changed)
        self.assertEqual(lrc_file.read_text(encoding="utf-8"), original)

    @patch("sonora.modules.renamer.read_track_metadata")
    def test_rename_track_file(self, mock_read: Any) -> None:
        wav_file = self.tmp_path / "old_name.wav"
        write_silent_wav(wav_file)
        mock_read.return_value = TrackInfo(
            file_path=wav_file, artist="Beyoncé", title="Halo", track_number=1
        )
        new_path = rename_track_file(wav_file)
        self.assertTrue(new_path.exists())
        self.assertEqual(new_path.name, "01 - Halo.wav")

    @patch("sonora.modules.renamer.read_track_metadata")
    def test_rename_directory_files(self, mock_read: Any) -> None:
        audio_file_1 = self.tmp_path / "song1.wav"
        audio_file_2 = self.tmp_path / "song2.wav"
        write_silent_wav(audio_file_1)
        write_silent_wav(audio_file_2)
        mock_read.side_effect = [
            TrackInfo(
                file_path=audio_file_1, artist="Artist", title="Title 1", track_number=1
            ),
            TrackInfo(
                file_path=audio_file_2, artist="Artist", title="Title 2", track_number=2
            ),
        ]
        renamed = rename_directory_files(self.tmp_path)
        self.assertEqual(len(renamed), 2)

    @patch("sonora.modules.renamer.read_track_metadata")
    def test_rename_multiple_album_directories(self, mock_read: Any) -> None:
        album1 = self.tmp_path / "folder_one"
        album2 = self.tmp_path / "folder_two"
        album1.mkdir()
        album2.mkdir()
        t1 = album1 / "track1.wav"
        t2 = album2 / "track2.wav"
        write_silent_wav(t1)
        write_silent_wav(t2)

        def mock_meta(path: Path) -> TrackInfo:
            if path == t1:
                return TrackInfo(
                    file_path=t1,
                    artist="Artist One",
                    album="Album One",
                    title="Song 1",
                    track_number=1,
                )
            return TrackInfo(
                file_path=t2,
                artist="Artist Two",
                album="Album Two",
                title="Song 2",
                track_number=1,
            )

        mock_read.side_effect = mock_meta
        renamed = rename_directory_files(self.tmp_path)
        self.assertEqual(len(renamed), 2)
        self.assertTrue((self.tmp_path / "Artist One - Album One").exists())
        self.assertTrue((self.tmp_path / "Artist Two - Album Two").exists())

    def test_rename_album_folder(self) -> None:
        folder = self.tmp_path / "old_folder"
        folder.mkdir()
        new_folder = rename_album_folder(folder, "21 Savage", "Issa Album")
        self.assertTrue(new_folder.exists())
        self.assertEqual(new_folder.name, "21 Savage - Issa Album")

    def test_rename_album_folder_preserves_artist_container(self) -> None:
        artist_folder = self.tmp_path / "3 Doors Down"
        artist_folder.mkdir()
        result = rename_album_folder(artist_folder, "3 Doors Down", "Away From The Sun")
        self.assertEqual(result, artist_folder)
        self.assertTrue(artist_folder.exists())
        album_folder = self.tmp_path / "Album With Subdirs"
        album_folder.mkdir()
        (album_folder / "Singles").mkdir()
        result2 = rename_album_folder(album_folder, "Artist", "Album With Subdirs")
        self.assertEqual(result2, album_folder)

    def test_rename_single_folder(self) -> None:
        singles_dir = self.tmp_path / "Singles"
        singles_dir.mkdir()
        single_folder = singles_dir / "Alex Velea - Track 2"
        single_folder.mkdir()
        track_info = TrackInfo(
            file_path=single_folder / "01 - Minim Doi.wav",
            artist="Alex Velea",
            title="Minim Doi",
            album="Track 2",
        )
        renamed_folder = rename_single_folder(single_folder, track_info)
        self.assertTrue(renamed_folder.exists())
        self.assertEqual(renamed_folder.name, "Alex Velea - Minim Doi")
        preserved = rename_album_folder(renamed_folder, "Alex Velea", "Track 2")
        self.assertEqual(preserved, renamed_folder)
        self.assertEqual(preserved.name, "Alex Velea - Minim Doi")

    def test_rename_single_folder_preserves_collaborative_artists(self) -> None:
        singles_dir = self.tmp_path / "Singles"
        singles_dir.mkdir(parents=True, exist_ok=True)
        single_folder = singles_dir / "Temporary Folder"
        single_folder.mkdir()
        track_info = TrackInfo(
            file_path=single_folder / "01 - The Monster.wav",
            artist="Eminem & Rihanna",
            title="The Monster",
            album="The Monster - Single",
        )
        renamed = rename_single_folder(single_folder, track_info)
        self.assertEqual(renamed.name, "Eminem & Rihanna - The Monster")

    @patch("sonora.modules.renamer.read_track_metadata")
    def test_rename_track_file_in_singles_forces_01(self, mock_read: Any) -> None:
        singles_folder = self.tmp_path / "Artist" / "Singles" / "Artist - Song"
        singles_folder.mkdir(parents=True)
        audio_file = singles_folder / "03 - Song.wav"
        write_silent_wav(audio_file)
        lrc_file = singles_folder / "03 - Song.lrc"
        lrc_file.write_text("[00:01.00] Test\n", encoding="utf-8")
        mock_read.return_value = TrackInfo(
            file_path=audio_file,
            artist="Artist",
            title="Song",
            track_number=3,
            disc_number=2,
            total_discs=2,
        )
        renamed_audio = rename_track_file(audio_file)
        self.assertTrue(renamed_audio.exists())
        self.assertEqual(renamed_audio.name, "01 - Song.wav")
        self.assertTrue((singles_folder / "01 - Song.lrc").exists())

    @patch("sonora.modules.renamer.read_track_metadata")
    def test_rename_directory_files_updates_corrupted_singles(
        self, mock_read: Any
    ) -> None:
        single_dir = (
            self.tmp_path
            / "Connect-R"
            / "Singles"
            / "Connect-R - Track 2 (feat. Joyce Meyer)"
        )
        single_dir.mkdir(parents=True)
        audio_file = single_dir / "03 - Tren De Noapte.wav"
        write_silent_wav(audio_file)
        lrc_file = single_dir / "03 - Tren De Noapte.lrc"
        lrc_file.write_text("[00:01.00] Lyrics\n", encoding="utf-8")
        mock_read.return_value = TrackInfo(
            file_path=audio_file,
            artist="Connect-R",
            title="Tren De Noapte",
            album="Track 2 (feat. Joyce Meyer)",
            track_number=3,
        )
        renamed = rename_directory_files(self.tmp_path)
        self.assertEqual(len(renamed), 1)
        updated_folder = (
            self.tmp_path / "Connect-R" / "Singles" / "Connect-R - Tren De Noapte"
        )
        self.assertTrue(updated_folder.exists())
        self.assertTrue((updated_folder / "01 - Tren De Noapte.wav").exists())
        self.assertTrue((updated_folder / "01 - Tren De Noapte.lrc").exists())

    @patch("sonora.audio.metadata.write_track_metadata")
    @patch("sonora.audio.metadata.read_track_metadata")
    def test_process_single_track_filename_alignment(
        self, mock_read: Any, mock_write: Any
    ) -> None:
        track_file = self.tmp_path / "02 - Track Title.wav"
        write_silent_wav(track_file)
        mock_read.return_value = TrackInfo(
            file_path=track_file,
            artist="Artist",
            title="Track Title",
            track_number=1,
            album="Great Album",
        )
        mbids = {(1, 1): "uuid-1", (1, 2): "uuid-2"}
        with (
            patch("sonora.modules.tagger._update_musicbrainz"),
            patch(
                "sonora.services.itunes.fetch_itunes_track_metadata", return_value=None
            ),
        ):
            track_metadata = process_single_track(
                track_file,
                album_context=_AlbumMetadataContext(track_mbids=mbids),
                force=True,
                fetch_bpm=False,
                fetch_itunes_art=False,
            )
            self.assertEqual(track_metadata.track_number, 2)

    @patch("sonora.modules.renamer.read_track_metadata")
    def test_rename_track_file_collision_handling(self, mock_read: Any) -> None:
        audio_file_1 = self.tmp_path / "song1.wav"
        audio_file_2 = self.tmp_path / "song2.wav"
        audio_file_3 = self.tmp_path / "song3.wav"
        write_silent_wav(audio_file_1)
        write_silent_wav(audio_file_2)
        write_silent_wav(audio_file_3)
        mock_read.side_effect = [
            TrackInfo(
                file_path=audio_file_1, artist="Artist", title="Title", track_number=1
            ),
            TrackInfo(
                file_path=audio_file_2, artist="Artist", title="Title", track_number=1
            ),
            TrackInfo(
                file_path=audio_file_3, artist="Artist", title="Title", track_number=1
            ),
        ]
        renamed_path_1 = rename_track_file(audio_file_1)
        renamed_path_2 = rename_track_file(audio_file_2)
        renamed_path_3 = rename_track_file(audio_file_3)
        self.assertEqual(renamed_path_1.name, "01 - Title.wav")
        self.assertEqual(renamed_path_2.name, "01 - Title (2).wav")
        self.assertEqual(renamed_path_3.name, "01 - Title (3).wav")


class TestAssetPreservationRegressions(AudioRegressionTestCase):
    def test_timing_upgrade_requires_all_existing_words_to_agree(self) -> None:
        original = "\n".join(
            f"Line {number} has ordinary words" for number in range(80)
        )
        for replacement in (
            original.replace(
                "Line 40 has ordinary words", "Line 40 has different words"
            ),
            original + "\nAn extra verse was not in the existing lyrics",
        ):
            audio_path = self.audio("comparison.wav")
            sidecar = audio_path.with_suffix(".lrc")
            sidecar.write_text(original)
            returned = "[00:00.10]" + replacement
            conflicts: dict[str, str] = {}
            details: dict[str, object] = {}
            with patch.object(lyrics, "fetch_synced_lyrics", return_value=returned):
                text, status = track_processing_module.process_track_lyrics(
                    audio_path,
                    "Alpha",
                    "Song",
                    duration=10.0,
                    conflicts=conflicts,
                    details=details,
                )
            self.assertEqual((text, status), (original, "existing"))
            self.assertEqual(sidecar.read_text(), original)
            self.assertIn("disagree", conflicts["lyrics"])
            self.assertEqual(details["action"], "review")

    def test_sidecar_and_embedded_word_disagreement_is_not_a_timing_upgrade(
        self,
    ) -> None:
        audio_path = self.audio("two-sources.wav")
        sidecar = audio_path.with_suffix(".lrc")
        original = "The original lyric words"
        sidecar.write_text(original)
        conflicts: dict[str, str] = {}
        with patch.object(lyrics, "fetch_synced_lyrics", return_value=None):
            text, status = track_processing_module.process_track_lyrics(
                audio_path,
                "Alpha",
                "Song",
                duration=10,
                existing_lyrics="[00:00.10]<00:00.10>Other lyric words",
                conflicts=conflicts,
            )
        self.assertEqual((text, status), (original, "existing"))
        self.assertEqual(sidecar.read_text(), original)
        self.assertIn("Sidecar and embedded", conflicts["lyrics"])

    def test_punctuation_case_and_stanza_labels_allow_matching_timing_upgrade(
        self,
    ) -> None:
        audio_path = self.audio("matching.wav")
        sidecar = audio_path.with_suffix(".lrc")
        sidecar.write_text("[Verse 1]\nHello, world!\n[Chorus: Alpha]\nStay with me.")
        returned = "[00:00.10]HELLO WORLD\n[00:00.20]stay with me"
        conflicts: dict[str, str] = {}
        with patch.object(lyrics, "fetch_synced_lyrics", return_value=returned):
            text, status = track_processing_module.process_track_lyrics(
                audio_path, "Alpha", "Song", duration=10, conflicts=conflicts
            )
        self.assertEqual((text, status), (returned, "synced"))
        self.assertEqual(sidecar.read_text(), returned)
        self.assertEqual(conflicts, {})

    def test_word_disagreement_is_reported_even_if_returned_timing_is_lower_quality(
        self,
    ) -> None:
        audio_path = self.audio("downgrade.wav")
        sidecar = audio_path.with_suffix(".lrc")
        original = "[00:00.10]Keep the original words"
        sidecar.write_text(original)
        conflicts: dict[str, str] = {}
        with patch.object(
            lyrics, "fetch_synced_lyrics", return_value="Other words entirely"
        ):
            text, status = track_processing_module.process_track_lyrics(
                audio_path,
                "Alpha",
                "Song",
                duration=10,
                conflicts=conflicts,
                force=True,
            )
        self.assertEqual((text, status), (original, "existing"))
        self.assertIn("disagree", conflicts["lyrics"])

    def test_no_art_disables_final_asset_stage(self) -> None:
        track = TrackInfo(self.root / "track.flac")
        with (
            patch.object(art_module, "process_artist_artwork") as artist_art,
            patch.object(art_module, "process_label_artwork") as label_art,
        ):
            tagger._finalize_album_assets_and_replaygain(
                self.root,
                [track],
                [track],
                "Alpha",
                None,
                False,
                False,
                True,
                1,
                fetch_artwork=False,
            )
            artist_art.assert_not_called()
            label_art.assert_not_called()

    def test_mixed_downloads_receive_separate_album_replaygain(self) -> None:
        tracks = [
            TrackInfo(self.root / "a.flac", artist="Alpha", album="Album"),
            TrackInfo(self.root / "b.flac", artist="Alpha", album="Album"),
            TrackInfo(self.root / "single.flac", artist="Alpha", album="Single"),
        ]
        with patch.object(replaygain_module, "calculate_album_replaygain") as gain:
            tagger._finalize_album_assets_and_replaygain(
                self.root,
                tracks,
                [],
                "Alpha",
                None,
                True,
                False,
                True,
                2,
                fetch_artwork=False,
            )
        self.assertEqual(
            [call.args[0] for call in gain.call_args_list],
            [[tracks[0].file_path, tracks[1].file_path], [tracks[2].file_path]],
        )

    def test_lyrics_preserve_unowned_text(self) -> None:
        audio_path = self.audio("notes.wav")
        notes = audio_path.with_suffix(".txt")
        notes.write_text("Recording notes")
        with patch.object(
            lyrics, "fetch_synced_lyrics", return_value="[00:00.50] line"
        ):
            track_processing_module.process_track_lyrics(
                audio_path, "Alpha", "Song", duration=1.0
            )
        self.assertEqual(notes.read_text(), "Recording notes")

    def test_lyrics_collision_preserves_source(self) -> None:
        source = self.root / "source.flac"
        destination = self.root / "target.flac"
        source.with_suffix(".lrc").write_text("<00:01.00>enhanced")
        destination.with_suffix(".lrc").write_text("plain")
        relocate_companion_lyrics(source, destination)
        self.assertTrue(source.with_suffix(".lrc").exists())

    def test_track_prefix_is_exact(self) -> None:
        (self.root / "10 - Other.lrc").touch()
        self.assertEqual(
            find_companion_lyrics(self.root / "track.flac", track_number=1), []
        )
        (self.root / "01 - Correct.lrc").touch()
        self.assertEqual(
            find_companion_lyrics(self.root / "track.flac", track_number=1),
            [self.root / "01 - Correct.lrc"],
        )

    def test_enhanced_timestamps_validate_duration(self) -> None:
        self.assertFalse(
            sidecars.is_lyrics_duration_valid("[00:00.00]<09:59.00>word", 60)
        )
        self.assertTrue(sidecars.is_lyrics_duration_valid("<00:59.00>word", 60))
        self.assertEqual(sidecars.get_lyrics_quality("[00:59]line"), 2)

    def test_plain_cache_cannot_satisfy_synced_only(self) -> None:
        with (
            patch.object(lyrics, "get_cached_api", return_value="plain"),
            patch.object(
                lyrics, "_query_syncedlyrics", return_value="[00:01.00]synced"
            ),
        ):
            self.assertEqual(
                lyrics.fetch_synced_lyrics("Alpha", "Song", synced_only=True),
                "[00:01.00]synced",
            )

    def test_album_gate_excludes_quiet_blocks(self) -> None:
        measurements = [
            LoudnessMeasurement(self.root / "loud.flac", 0, 1, (-17,) * 40),
            LoudnessMeasurement(self.root / "quiet.flac", 0, 0.1, (-57,) * 40),
        ]
        self.assertAlmostEqual(_album_loudness(measurements) or 0.0, -17.0, places=4)
        self.assertIsNone(_album_loudness([]))
        self.assertIsNone(
            _album_loudness(
                [dataclasses.replace(measurements[0], block_loudness=(-80,))]
            )
        )

    def test_embedded_assets_keep_back_cover_and_higher_resolution(self) -> None:
        audio_path = self.audio("art.flac")
        high = io.BytesIO()
        Image.new("RGB", (1000, 1000)).save(high, format="PNG")
        low_cover = self.root / "cover.jpg"
        Image.new("RGB", (500, 500)).save(low_cover)
        with taglib.File(str(audio_path)) as audio_tags:
            audio_tags.pictures = [
                taglib.Picture(
                    data=high.getvalue(), mime_type="image/png", picture_type=kind
                )
                for kind in ("Front Cover", "Back Cover")
            ]
            audio_tags.save()
        clear_metadata_cache()
        write_track_metadata(read_track_metadata(audio_path), low_cover)
        with taglib.File(str(audio_path)) as audio_tags:
            self.assertEqual(len(audio_tags.pictures), 2)
            self.assertTrue(
                all(picture.data == high.getvalue() for picture in audio_tags.pictures)
            )

    def test_key_and_date_validators_reject_invalid_values(self) -> None:
        for invalid in ("garbage", "Banana", "Cat"):
            with self.subTest(invalid=invalid):
                self.assertIsNone(key_to_camelot(invalid))
        self.assertIsNone(normalize_date("2024-99-99"))
        self.assertIsNone(normalize_date("2023-02-29"))
        self.assertEqual(normalize_date("2024-02-29"), "2024-02-29")

    def test_alien_never_downloads_art_into_shared_album_folder(self) -> None:
        track = TrackInfo(
            self.root / "track.flac",
            artist="Foreign",
            title="Song",
            album="Other",
            is_alien=True,
        )
        original = dataclasses.replace(track)
        with patch.object(
            track_processing_module, "_update_artwork"
        ) as artwork_provider:
            track_processing_module._process_audio_and_save_tags(
                track,
                original,
                track.file_path,
                None,
                None,
                False,
                False,
                False,
                True,
                False,
                True,
            )
        artwork_provider.assert_not_called()

    def test_larger_cover_is_embedded_without_force_above_minimum(self) -> None:
        audio_path = self.audio("artwork.flac")
        cover_path = self.root / "cover.png"
        Image.new("RGB", (600, 600), "blue").save(cover_path)
        initial = TrackInfo(
            audio_path,
            artist="Alpha",
            title="Song",
            album="Album",
            album_artist="Alpha",
        )
        write_track_metadata(initial, cover_art_path=cover_path)
        original = read_track_metadata(audio_path)
        Image.new("RGB", (900, 900), "red").save(cover_path)
        updated = dataclasses.replace(original)
        track_processing_module._process_audio_and_save_tags(
            updated,
            original,
            audio_path,
            None,
            cover_path,
            False,
            False,
            False,
            True,
            False,
            False,
        )
        persisted = read_track_metadata(audio_path)
        self.assertEqual((persisted.art_width, persisted.art_height), (900, 900))
        self.assertTrue(
            any("cover_art:" in change for change in updated._diff_descriptions)
        )

    def test_same_or_smaller_cover_does_not_trigger_tag_write(self) -> None:
        for dimensions in ((600, 600), (300, 300)):
            with self.subTest(dimensions=dimensions):
                cover_path = self.root / "cover.png"
                Image.new("RGB", dimensions, "red").save(cover_path)
                original = TrackInfo(
                    self.root / "track.flac",
                    artist="Alpha",
                    title="Song",
                    album="Album",
                    album_artist="Alpha",
                    art_width=600,
                    art_height=600,
                )
                with patch.object(
                    metadata_module, "write_track_metadata"
                ) as tag_writer:
                    track_processing_module._process_audio_and_save_tags(
                        dataclasses.replace(original),
                        original,
                        original.file_path,
                        None,
                        cover_path,
                        False,
                        False,
                        False,
                        True,
                        False,
                        False,
                    )
                tag_writer.assert_not_called()


class TestAssetUpdates(IsolatedTestCase):
    def test_playlist_recovers_catalog_album_version_suffix(self) -> None:
        target = self.test_directory / "01 - Careless World.flac"
        target.touch()
        playlist = self.test_directory / "Album.m3u"
        playlist.write_text(
            "01 - Tyga - Careless World (Album Version (Explicit)).flac\n"
        )
        track = TrackInfo(target, artist="Tyga", title="Careless World")
        update_local_playlists(self.test_directory, {}, [track])
        self.assertEqual(playlist.read_text(), "01 - Careless World.flac\n")

    def test_playlist_cannot_substitute_a_different_featured_performer(self) -> None:
        target = self.test_directory / "01 - Song.flac"
        target.touch()
        playlist = self.test_directory / "Album.m3u"
        original = b"01 - Performer - Song (feat. Guest A).flac\n"
        playlist.write_bytes(original)
        track = TrackInfo(
            target, artist="Performer", title="Song", featured_artists="Guest B"
        )
        self.assertEqual(update_local_playlists(self.test_directory, {}, [track]), [])
        self.assertEqual(playlist.read_bytes(), original)
        track.featured_artists = "Guest A"
        update_local_playlists(self.test_directory, {}, [track])
        self.assertEqual(playlist.read_bytes(), b"01 - Song.flac\n")

    def test_invalid_album_artist_copies_repair_from_valid_artist_root(self) -> None:
        artist_dir = self.test_directory / "Performer"
        album = artist_dir / "Performer - Album"
        album.mkdir(parents=True)
        for filename in ("artist.jpg", "banner.jpg"):
            (artist_dir / filename).write_bytes(image_bytes())
            (album / filename).write_bytes(b"fakeimage")
        (album / "folder.jpg").write_bytes(b"invalid album cover")
        with patch.object(art, "fetch_artist_images") as fetch:
            art.process_artist_artwork(album, "Performer", dry_run=True)
            self.assertEqual((album / "artist.jpg").read_bytes(), b"fakeimage")
            art.process_artist_artwork(album, "Performer")
        fetch.assert_not_called()
        for filename in ("artist.jpg", "banner.jpg"):
            self.assertEqual((album / filename).read_bytes(), image_bytes())
        self.assertEqual((album / "folder.jpg").read_bytes(), b"invalid album cover")

    def test_invalid_downloads_cannot_replace_existing_invalid_or_valid_assets(
        self,
    ) -> None:
        (self.test_directory / "artist.jpg").write_bytes(b"fakeimage")
        banner = self.test_directory / "banner.jpg"
        banner.write_bytes(image_bytes())
        with patch.object(
            art, "fetch_artist_images", return_value=(b"fakeimage", b"fakeimage")
        ):
            art.process_artist_artwork(self.test_directory, "Performer")
        self.assertEqual(
            (self.test_directory / "artist.jpg").read_bytes(), b"fakeimage"
        )
        self.assertEqual(banner.read_bytes(), image_bytes())

    def test_failed_cover_fetch_does_not_return_corrupt_existing_image(self) -> None:
        cover = self.test_directory / "cover.jpg"
        cover.write_bytes(b"fakeimage")
        with patch.object(art, "_fetch_candidate_artwork", return_value=None):
            self.assertIsNone(
                art.process_album_cover_art(self.test_directory, "Performer", "Album")
            )
        self.assertEqual(cover.read_bytes(), b"fakeimage")

    def test_playlist_repair_preserves_bom_comments_line_endings_and_remote_entries(
        self,
    ) -> None:
        target = self.test_directory / "01 - Song.flac"
        target.touch()
        track = TrackInfo(target, artist="Performer", title="Song")
        playlist = self.test_directory / "Album.m3u8"
        original = b"\xef\xbb\xbf#EXTM3U\r\n#EXTINF:123,Custom description\r\n01 - Performer - Song.flac\r\nhttps://example.com/live\r\n"
        playlist.write_bytes(original)
        self.assertEqual(
            update_local_playlists(self.test_directory, {}, [track], dry_run=True),
            [playlist],
        )
        self.assertEqual(playlist.read_bytes(), original)
        update_local_playlists(self.test_directory, {}, [track])
        self.assertEqual(
            playlist.read_bytes(),
            original.replace(b"01 - Performer - Song.flac", b"01 - Song.flac"),
        )

    def test_playlist_ambiguous_or_different_version_entries_remain_unchanged(
        self,
    ) -> None:
        tracks = []
        for name in ("01 - Song.flac", "02 - Song.flac"):
            path = self.test_directory / name
            path.touch()
            tracks.append(TrackInfo(path, artist="Performer", title="Song"))
        playlist = self.test_directory / "Album.m3u"
        original = b"00 - Performer - Song.flac\n00 - Performer - Song (live).flac\n../outside.flac\n"
        playlist.write_bytes(original)
        self.assertEqual(update_local_playlists(self.test_directory, {}, tracks), [])
        self.assertEqual(playlist.read_bytes(), original)

    def test_playlist_exact_move_updates_relative_location(self) -> None:
        old = self.test_directory / "old.flac"
        target = self.test_directory / "nested" / "new.flac"
        target.parent.mkdir()
        target.touch()
        playlist = self.test_directory / "Album.m3u"
        playlist.write_bytes(b"old.flac\n")
        update_local_playlists(self.test_directory, {old: target})
        self.assertEqual(playlist.read_bytes(), b"nested/new.flac\n")

    def test_playlist_failed_move_does_not_publish_nonexistent_target(self) -> None:
        old = self.test_directory / "old.flac"
        old.touch()
        missing = self.test_directory / "missing.flac"
        playlist = self.test_directory / "Album.m3u"
        playlist.write_bytes(b"old.flac\n")
        self.assertEqual(
            update_local_playlists(self.test_directory, {old: missing}), []
        )
        self.assertEqual(playlist.read_bytes(), b"old.flac\n")

    def test_playlist_non_utf8_and_atomic_failure_preserve_original(self) -> None:
        target = self.test_directory / "new.flac"
        target.touch()
        playlist = self.test_directory / "Album.m3u"
        original = b"#\xff\nold.flac\n"
        playlist.write_bytes(original)
        self.assertEqual(update_local_playlists(self.test_directory, {}), [])
        self.assertEqual(playlist.read_bytes(), original)
        playlist.write_bytes(b"old.flac\n")
        with (
            patch("sonora.core.files.os.fsync", side_effect=OSError("disk full")),
            self.assertRaises(OSError),
        ):
            update_local_playlists(
                self.test_directory, {self.test_directory / "old.flac": target}
            )
        self.assertEqual(playlist.read_bytes(), b"old.flac\n")


class TestAudioWorkflows(ModuleTestCase):
    def test_normalize_track_metadata_unification(self) -> None:
        track = TrackInfo(
            file_path=Path("/music/singles/Lithe - Fall Back/01 - Fall Back.flac"),
            artist="D’banj",
            album_artist="D’banj",
            title="Fall Back (feat. Lil Tjay) [Prod: Metro Boomin]",
            album="Tomorrow‘s Midnight",
            release_country="CA",
            language="eng",
            script="Latn",
            genre="hip hop",
            track_number=5,
            total_tracks=1,
            release_type="Single",
        )
        normalized = normalize_track_metadata(track)
        self.assertEqual(normalized.artist, "D'banj")
        self.assertEqual(normalized.album_artist, "D'banj")
        self.assertEqual(normalized.title, "Fall Back")
        self.assertEqual(normalized.featured_artists, "Lil Tjay")
        self.assertEqual(normalized.producers, "Metro Boomin")
        self.assertEqual(normalized.album, "Tomorrow's Midnight")
        self.assertEqual(normalized.release_country, "Canada")
        self.assertEqual(normalized.language, "English")
        self.assertEqual(normalized.script, "Latin")
        self.assertEqual(normalized.track_number, 1)
        self.assertEqual(normalized.total_tracks, 1)

    def test_cuesheet_parsing(self) -> None:
        cue_path = self.tmp_path / "test.cue"
        cue_path.write_text(
            'REM GENRE "Hip-Hop"\nREM DATE 2021\nREM DISCNUMBER 1\nREM TOTALDISCS 2\nPERFORMER "Album Artist"\nTITLE "Album Title"\nTRACK 01 AUDIO\n  TITLE "Track One"\n  PERFORMER "Track Artist"\n  SONGWRITER "Composer Name"\n  ISRC USUM71805166\n  INDEX 00 00:00:00\n  INDEX 01 00:02:00\n',
            encoding="latin-1",
        )
        content = read_cuesheet_content(cue_path)
        self.assertIsNotNone(content)
        self.assertIn('TITLE "Track One"', content or "")

    def test_normalize_single_track_and_library(self) -> None:
        audio_file = self.tmp_path / "song.wav"
        write_silent_wav(audio_file)
        with (
            patch("sonora.audio.metadata.read_track_metadata") as mock_read,
            patch("sonora.audio.metadata.write_track_metadata") as mock_write,
        ):
            mock_read.return_value = TrackInfo(
                file_path=audio_file,
                artist="Armin (ROU)",
                title="Melodie (feat. Nane)",
                album="Album [Deluxe Edition]",
                genre="hip hop",
                date="2022-05-10",
                release_country="US",
                language="eng",
                script="Latn",
                bpm=None,
            )
            result = normalize_single_track(audio_file, fetch_bpm=False, dry_run=False)
            assert result is not None
            self.assertEqual(result.artist, "Armin")
            self.assertEqual(result.title, "Melodie")
            self.assertEqual(result.featured_artists, "Nane")
            self.assertEqual(result.genre, "Hip-Hop/Rap")
            self.assertEqual(result.release_country, "United States")
            self.assertEqual(result.language, "English")
            self.assertEqual(result.script, "Latin")
            mock_write.assert_called_once()
            mock_read.return_value = TrackInfo(
                file_path=audio_file, artist="ALEX", album_artist="Alex", title="Song"
            )
            casing_result = normalize_single_track(audio_file, dry_run=False)
            assert casing_result is not None
            self.assertEqual(casing_result.artist, "Alex")
            self.assertEqual(casing_result.album_artist, "Alex")
            mock_read.return_value = TrackInfo(
                file_path=audio_file,
                artist="21 Savage & Metro Boomin feat. Drake",
                title="Mr. Right Now",
                album="Savage Mode II",
            )
            feat_result = normalize_single_track(audio_file, dry_run=False)
            assert feat_result is not None
            self.assertEqual(feat_result.artist, "21 Savage & Metro Boomin")
            self.assertEqual(feat_result.title, "Mr. Right Now")
            self.assertEqual(feat_result.featured_artists, "Drake")
            mock_read.return_value = TrackInfo(
                file_path=audio_file,
                artist="Deliric",
                title="Din 94 feat. Mike Diamondz [Prod: Motzu]",
                producers=None,
            )
            prod_result = normalize_single_track(audio_file, dry_run=False)
            assert prod_result is not None
            self.assertEqual(prod_result.title, "Din 94")
            self.assertEqual(prod_result.featured_artists, "Mike Diamondz")
            self.assertEqual(prod_result.producers, "Motzu")
            with patch("sonora.audio.key.detect_musical_key", return_value="Am"):
                mock_read.return_value = TrackInfo(
                    file_path=audio_file,
                    artist="B.U.G. Mafia",
                    title="40 km/h",
                    initial_key="m",
                )
                key_result = normalize_single_track(
                    audio_file, fetch_key=True, dry_run=False
                )
                assert key_result is not None
                self.assertEqual(key_result.initial_key, "Am")
            mock_read.return_value = TrackInfo(
                file_path=audio_file, artist="Artist (USA)", title="Clean Song"
            )
            library_results = normalize_library(
                self.tmp_path, fetch_bpm=False, fetch_replaygain=False
            )
            self.assertEqual(mock_write.call_args.args[0].artist, "Artist")
            self.assertEqual(library_results.total_scanned, 1)
            self.assertEqual(library_results.total_modified, 1)
            self.assertEqual(library_results.total_unchanged, 0)
            self.assertIn(str(audio_file), library_results.modified_files)
            mock_write.reset_mock()
            mock_read.return_value = TrackInfo(
                file_path=audio_file, artist="Artist", title="Clean Song"
            )
            already_clean_results = normalize_library(
                self.tmp_path, fetch_bpm=False, fetch_replaygain=False
            )
            self.assertEqual(already_clean_results.total_scanned, 1)
            self.assertEqual(already_clean_results.total_modified, 0)
            self.assertEqual(already_clean_results.total_unchanged, 1)
            self.assertEqual(len(already_clean_results.modified_files), 0)
            mock_write.assert_not_called()


class TestLyricsUpdates(IsolatedTestCase):
    def test_known_track_duration_skips_another_audio_read(self) -> None:
        audio_path = self.test_directory / "recording.flac"
        audio_path.touch()
        track = TrackInfo(audio_path, artist="Alpha", title="Song", duration=60)
        with (
            patch("sonora.audio.metadata.get_audio_duration") as read_duration,
            patch(
                "sonora.modules.track_processing.process_track_lyrics",
                return_value=(None, None),
            ),
        ):
            _update_lyrics(track, audio_path, True, dry_run=True)
        read_duration.assert_not_called()

    def test_sidecar_is_read_once_for_validation(self) -> None:
        audio_path = self.test_directory / "recording.flac"
        audio_path.touch()
        lrc_path = audio_path.with_suffix(".lrc")
        lrc_path.write_text("[ar:Alpha]\n[ti:Song]\n[00:01.00] words", encoding="utf-8")
        original_read_text = Path.read_text
        reads = 0

        def count_reads(
            path: Path, encoding: str | None = None, errors: str | None = None
        ) -> str:
            nonlocal reads
            if path == lrc_path:
                reads += 1
            return original_read_text(path, encoding, errors)

        with (
            patch("sonora.audio.metadata.get_audio_duration", return_value=60),
            patch.object(Path, "read_text", count_reads),
        ):
            _update_lyrics(
                TrackInfo(audio_path, artist="Alpha", title="Song"),
                audio_path,
                True,
                dry_run=True,
            )
        self.assertEqual(reads, 1)

    def test_unresolved_local_lyrics_conflicts_remain_visible_in_tagging_results(
        self,
    ) -> None:
        audio_path = self.test_directory / "recording.flac"
        for raw_lyrics in (
            b"[ar:Unrelated Artist]\n[ti:Song]\n[00:01.00] words",
            b"[ar:Alpha]\n[ti:Song]\n[02:00.00] words",
            b"\xff\xfe",
        ):
            with self.subTest(lyrics=raw_lyrics):
                lrc_path = audio_path.with_suffix(".lrc")
                lrc_path.write_bytes(raw_lyrics)
                track = TrackInfo(audio_path, artist="Alpha", title="Song")
                with (
                    patch("sonora.audio.metadata.get_audio_duration", return_value=60),
                    patch(
                        "sonora.services.lyrics.fetch_synced_lyrics", return_value=None
                    ),
                ):
                    _update_lyrics(track, audio_path, True, dry_run=True)
                self.assertIn("lyrics", track.metadata_conflicts)
                self.assertEqual(lrc_path.read_bytes(), raw_lyrics)

    def test_accepted_replacement_clears_only_the_lyrics_conflict(self) -> None:
        audio_path = self.test_directory / "recording.flac"
        track = TrackInfo(
            audio_path,
            artist="Alpha",
            title="Song",
            lyrics="[ti:Wrong Song]\n[00:01.00] words",
            metadata_conflicts={"genre": "Providers disagree"},
        )
        replacement = "[ar:Alpha]\n[ti:Song]\n[00:01.00] words"
        with (
            patch("sonora.audio.metadata.get_audio_duration", return_value=60),
            patch(
                "sonora.modules.track_processing.process_track_lyrics",
                return_value=(replacement, "synced"),
            ),
        ):
            _update_lyrics(track, audio_path, True, dry_run=True)
        self.assertEqual(track.lyrics, replacement)
        self.assertEqual(track.metadata_conflicts, {"genre": "Providers disagree"})

    def test_headerless_lyrics_are_not_claimed_to_be_a_conflict(self) -> None:
        audio_path = self.test_directory / "recording.flac"
        track = TrackInfo(
            audio_path, artist="Alpha", title="Song", lyrics="[00:01.00] words"
        )
        with (
            patch("sonora.audio.metadata.get_audio_duration", return_value=60),
            patch(
                "sonora.modules.track_processing.process_track_lyrics",
                return_value=(None, None),
            ),
        ):
            _update_lyrics(track, audio_path, True, dry_run=True)
        self.assertNotIn("lyrics", track.metadata_conflicts)


class TestDryRunReports(IsolatedTestCase):
    def test_bpm_reports_computation_and_actual_writes_separately(self) -> None:
        from sonora.cli import audio as audio_cli

        audio_path = Path("track.flac")
        for dry_run in (True, False):
            with (
                self.subTest(dry_run=dry_run),
                patch(
                    "sonora.cli.audio._run_parallel_audio_task",
                    return_value=([(audio_path, 70.0, True)], False),
                ),
                patch("sonora.cli.audio._write_json_report") as report,
            ):
                self.assertEqual(
                    audio_cli.bpm(
                        audio_path, dry_run=dry_run, json_report=Path("report.json")
                    ),
                    0,
                )
                payload = report.call_args.args[1]
                self.assertEqual(payload["summary"]["bpm_calculated"], 1)
                self.assertEqual(payload["summary"]["bpm_tagged"], 0 if dry_run else 1)
                self.assertEqual(payload["details"][0]["modified"], not dry_run)

    def test_key_reports_detection_and_actual_writes_separately(self) -> None:
        from sonora.cli import audio as audio_cli

        audio_path = Path("track.flac")
        for dry_run in (True, False):
            with (
                self.subTest(dry_run=dry_run),
                patch(
                    "sonora.cli.audio._run_parallel_audio_task",
                    return_value=([(audio_path, "A minor (8A)", True)], False),
                ),
                patch("sonora.cli.audio._write_json_report") as report,
            ):
                self.assertEqual(
                    audio_cli.key(
                        audio_path, dry_run=dry_run, json_report=Path("report.json")
                    ),
                    0,
                )
                payload = report.call_args.args[1]
                self.assertEqual(payload["summary"]["key_detected"], 1)
                self.assertEqual(payload["summary"]["key_tagged"], 0 if dry_run else 1)
                self.assertEqual(payload["details"][0]["modified"], not dry_run)

    def test_failed_or_skipped_bpm_is_not_reported_as_modified(self) -> None:
        from sonora.cli import audio as audio_cli

        audio_path = Path("track.flac")
        with (
            patch(
                "sonora.cli.audio._run_parallel_audio_task",
                return_value=(
                    [(audio_path, None, True), (audio_path, 70.0, False)],
                    False,
                ),
            ),
            patch("sonora.cli.audio._write_json_report") as report,
        ):
            audio_cli.bpm(audio_path, json_report=Path("report.json"))
            payload = report.call_args.args[1]
            self.assertEqual(payload["summary"]["bpm_tagged"], 0)
            self.assertFalse(any(track["modified"] for track in payload["details"]))

    def test_dsp_workers_do_not_write_during_dry_run(self) -> None:
        from sonora.cli import audio as audio_cli

        audio_path = Path("track.flac")
        with (
            patch(
                "sonora.cli.audio.read_track_metadata",
                return_value=TrackInfo(audio_path),
            ),
            patch("sonora.cli.audio.calculate_bpm", return_value=70.0),
            patch(
                "sonora.cli.audio.detect_key_details",
                return_value=("A minor", "8A", 0.8),
            ),
            patch("sonora.cli.audio.write_track_metadata") as write_tags,
        ):
            self.assertEqual(
                audio_cli._process_bpm_file(audio_path, force=True, dry_run=True)[1], 70
            )
            self.assertEqual(
                audio_cli._process_key_file(audio_path, force=True, dry_run=True)[1],
                "A minor (8A)",
            )
            write_tags.assert_not_called()

    def test_exact_identity_survives_maximum_threshold(self) -> None:
        with patch(
            "sonora.core.titles.get_config",
            return_value=SonoraConfig(
                title_match_threshold=100, short_title_match_threshold=100
            ),
        ):
            self.assertEqual(
                score_artist_title_match("Artist", "Intro", "Artist", "Intro"), 100
            )
            self.assertEqual(
                score_artist_title_match("Artist", "Intro", "Artist", "Outro"), 0
            )


class TestFilePreservationRegressions(AudioRegressionTestCase):
    def test_case_collision_keeps_both_recordings(self) -> None:
        source = self.root / "01 - song.flac"
        target = self.root / "01 - Song.flac"
        source.write_bytes(b"first recording")
        if target.exists():
            self.skipTest(
                "Separate case-variant files require a case-sensitive filesystem"
            )
        target.write_bytes(b"second recording")
        renamed = renamer.rename_track_file(
            source, track_info=TrackInfo(source, title="Song", track_number=1)
        )
        self.assertEqual(target.read_bytes(), b"second recording")
        self.assertEqual(renamed.read_bytes(), b"first recording")
        self.assertNotEqual(renamed, target)

    def test_concurrent_moves_never_overwrite(self) -> None:
        sources = [self.root / "a.flac", self.root / "b.flac"]
        for index, source in enumerate(sources):
            source.write_text(str(index))
        target = self.root / "destination.flac"
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(move_file_exclusively, source, target) for source in sources
            ]
            successes = 0
            for future in futures:
                try:
                    future.result()
                    successes += 1
                except FileExistsError:
                    pass
        self.assertEqual(successes, 1)
        self.assertEqual(sum(source.exists() for source in sources), 1)
        self.assertIn(target.read_text(), ("0", "1"))

    def test_atomic_write_failure_preserves_original(self) -> None:
        destination = self.root / "lyrics.lrc"
        destination.write_bytes(b"original")
        with (
            patch.object(Path, "replace", side_effect=PermissionError),
            self.assertRaises(PermissionError),
        ):
            atomic_write_bytes(destination, b"new")
        self.assertEqual(destination.read_bytes(), b"original")
        self.assertEqual(list(self.root.iterdir()), [destination])

    def test_directory_merge_preserves_collisions(self) -> None:
        for folder, payload in (("Artist", b"original"), ("ARTIST!", b"different")):
            recording = self.root / folder / "Album" / "01.flac"
            recording.parent.mkdir(parents=True)
            recording.write_bytes(payload)
        with patch.object(organizer, "_read_file_info", return_value=(self.root, None)):
            organizer._consolidate_directory_entries(self.root)
        self.assertEqual(
            {recording.read_bytes() for recording in self.root.rglob("*.flac")},
            {b"original", b"different"},
        )

    def test_raw_backup_roundtrip_and_relocation(self) -> None:
        source = self.audio("old/Album/01.flac")
        picture_bytes = io.BytesIO()
        Image.new("RGB", (600, 600)).save(picture_bytes, format="PNG")
        original_tags = {
            "ARTIST": ["Alpha", "Beta"],
            "GENRE": ["MPB"],
            "CUSTOM_FIELD": ["one", "two"],
        }
        with taglib.File(str(source)) as audio_tags:
            audio_tags.tags.update(original_tags)
            audio_tags.pictures = [
                taglib.Picture(
                    data=picture_bytes.getvalue(),
                    mime_type="image/png",
                    picture_type="Back Cover",
                )
            ]
            audio_tags.save()
        archive = backup.backup_library_tags(
            self.root / "old", self.root / "archive.json.gz"
        )
        target = self.root / "new/Album/01.flac"
        target.parent.mkdir(parents=True)
        source.rename(target)
        with taglib.File(str(target)) as audio_tags:
            audio_tags.tags.clear()
            audio_tags.pictures = []
            audio_tags.save()
        self.assertEqual(backup.restore_library_tags(archive, self.root / "new"), 1)
        with taglib.File(str(target)) as audio_tags:
            for key, values in original_tags.items():
                self.assertEqual(audio_tags.tags[key], values)
            self.assertEqual(audio_tags.pictures[0].data, picture_bytes.getvalue())

    def test_restore_rejects_paths_outside_root(self) -> None:
        archive = self.root / "unsafe.json"
        for name in ("../track.flac", "/track.flac", "Album/../../track.flac"):
            with self.subTest(name=name):
                archive.write_text(
                    json.dumps(
                        {
                            "schema": "sonora_backup_v2",
                            "root": str(self.root),
                            "tracks": {name: {"tags": {}, "complex_tags": {}}},
                        }
                    )
                )
                with self.assertRaises(ValueError):
                    backup.restore_library_tags(archive)

    def test_restore_never_falls_back_to_basename(self) -> None:
        archive = self.root / "archive.json"
        self.audio("relocated/01.wav")
        archive.write_text(
            json.dumps(
                {
                    "schema": "sonora_backup_v2",
                    "root": str(self.root),
                    "tracks": {
                        "Album A/01.wav": {"tags": {}, "complex_tags": {}},
                        "Album B/01.wav": {"tags": {}, "complex_tags": {}},
                    },
                }
            )
        )
        self.assertEqual(
            backup.restore_library_tags(archive, self.root / "relocated"), 0
        )

    def test_atomic_audio_edit_commits_complete_payload(self) -> None:
        audio_path = self.root / "track.flac"
        audio_path.write_bytes(b"original")
        with atomic_audio_edit(audio_path) as staging_path:
            staging_path.write_bytes(b"updated")
            self.assertEqual(audio_path.read_bytes(), b"original")
        self.assertEqual(audio_path.read_bytes(), b"updated")
        self.assertEqual(list(self.root.iterdir()), [audio_path])

    def test_atomic_audio_edit_aborts_failed_tag_save(self) -> None:
        audio_path = self.root / "track.flac"
        audio_path.write_bytes(b"original")
        with self.assertRaises(OSError), atomic_audio_edit(audio_path) as staging_path:
            staging_path.write_bytes(b"incomplete")
            raise OSError("TagLib save failed")
        self.assertEqual(audio_path.read_bytes(), b"original")
        self.assertEqual(list(self.root.iterdir()), [audio_path])

    def test_atomic_audio_edit_rejects_concurrent_changes(self) -> None:
        audio_path = self.root / "track.flac"
        audio_path.write_bytes(b"original")
        with self.assertRaises(OSError), atomic_audio_edit(audio_path) as staging_path:
            staging_path.write_bytes(b"updated")
            audio_path.write_bytes(b"another process changed this recording")
        self.assertEqual(
            audio_path.read_bytes(), b"another process changed this recording"
        )
        self.assertEqual(list(self.root.iterdir()), [audio_path])

    def test_restore_validates_every_binary_property_before_writing(self) -> None:
        for invalid_property in ({"base64": "invalid!"}, {"unknown": "value"}, None):
            with self.subTest(invalid_property=invalid_property):
                archive = self.root / "invalid.json"
                archive.write_text(
                    json.dumps(
                        {
                            "schema": "sonora_backup_v2",
                            "root": str(self.root),
                            "tracks": {
                                "valid.wav": {"tags": {}, "complex_tags": {}},
                                "invalid.wav": {
                                    "tags": {},
                                    "complex_tags": {
                                        "PICTURE": [{"data": invalid_property}]
                                    },
                                },
                            },
                        }
                    )
                )
                with patch.object(backup, "_restore_single_track") as restore_worker:
                    with self.assertRaises((ValueError, TypeError)):
                        backup.restore_library_tags(archive)
                    restore_worker.assert_not_called()

    def test_restore_reports_missing_paths(self) -> None:
        archive = self.root / "missing.json"
        archive.write_text(
            json.dumps(
                {
                    "schema": "sonora_backup_v2",
                    "root": str(self.root),
                    "tracks": {"absent.flac": {"tags": {}, "complex_tags": {}}},
                }
            )
        )
        failures: list[str] = []
        self.assertEqual(backup.restore_library_tags(archive, failures=failures), 0)
        self.assertEqual(failures, ["absent.flac"])

    def test_rename_reports_failed_audio_reads(self) -> None:
        from sonora.core.models import RenameReport

        audio_path = self.root / "broken.flac"
        audio_path.write_bytes(b"not audio")
        report = RenameReport()
        self.assertEqual(renamer.rename_directory_files(self.root, report=report), [])
        self.assertEqual(report.failed_files, 1)
        self.assertEqual(report.unchanged_files, 0)


class TestRenameFailureContracts(IsolatedTestCase):
    def test_preview_returns_audio_destination_and_preserves_companion_content(
        self,
    ) -> None:
        path = self.test_directory / "recording.flac"
        path.touch()
        companion = path.with_suffix(".lrc")
        companion.write_text("[00:01] Verified lyrics")
        relocated: list[Path] = []
        renamed = renamer.rename_track_file(
            path,
            track_info=TrackInfo(path, title="Recording", track_number=1),
            dry_run=True,
            relocated_lrc_collector=relocated,
        )
        self.assertEqual(renamed.name, "01 - Recording.flac")
        self.assertEqual(relocated, [])
        self.assertFalse(renamed.with_suffix(".lrc").exists())
        self.assertEqual(companion.read_text(), "[00:01] Verified lyrics")

    def test_multidisc_album_preserves_disc_prefix_for_disc_one_tracks(self) -> None:
        album_dir = self.test_directory / "Artist - MultiDisc Album"
        album_dir.mkdir(parents=True)
        disc1_track = album_dir / "1-05 - Song.flac"
        disc1_track.touch()
        disc2_track = album_dir / "2-01 - Another.flac"
        disc2_track.touch()
        renamed = renamer.rename_track_file(
            disc1_track,
            track_info=TrackInfo(
                disc1_track, title="Song", track_number=5, disc_number=1, total_discs=1
            ),
            dry_run=True,
        )
        self.assertEqual(renamed.name, "1-05 - Song.flac")

    def test_single_hierarchy_failures_and_canonical_folder_identity_are_reported(
        self,
    ) -> None:
        folder = self.test_directory / "Singles" / "Primary - Recording"
        folder.mkdir(parents=True)
        path = folder / "01 - Recording.flac"
        path.touch()
        failed = RenameReport()
        with patch.object(
            renamer, "read_track_metadata", side_effect=OSError("broken tags")
        ):
            self.assertEqual(
                renamer.rename_directory_files(self.test_directory, report=failed), []
            )
        self.assertEqual((failed.failed_files, failed.folders_renamed), (1, 0))
        report = RenameReport()
        with patch.object(
            renamer,
            "read_track_metadata",
            return_value=TrackInfo(
                path, artist="Primary", title="Recording", track_number=1
            ),
        ):
            self.assertEqual(
                renamer.rename_directory_files(self.test_directory, report=report),
                [path],
            )
        self.assertEqual((report.unchanged_files, report.folders_renamed), (1, 0))

    def test_generic_containers_and_disc_folders_are_not_renamed_as_single_releases(
        self,
    ) -> None:
        for name in ("Singles", "CD 2", "Downloads"):
            folder = self.test_directory / name
            folder.mkdir()
            recording = TrackInfo(
                folder / "recording.flac", artist="Primary", title="Recording"
            )
            with self.subTest(name=name):
                self.assertEqual(
                    renamer.rename_single_folder(folder, recording), folder
                )

    def test_single_folder_preview_returns_canonical_path_without_moving_assets(
        self,
    ) -> None:
        folder = self.test_directory / "Record"
        folder.mkdir()
        recording = TrackInfo(
            folder / "recording.flac", artist="Primary", title="Recording"
        )
        self.assertEqual(
            renamer.rename_single_folder(folder, recording, dry_run=True),
            self.test_directory / "Primary - Recording",
        )
        self.assertTrue(folder.exists())

    def test_unreadable_single_directory_is_preserved(self) -> None:
        folder = self.test_directory / "Record"
        folder.mkdir()
        with patch.object(Path, "iterdir", side_effect=OSError("locked directory")):
            self.assertEqual(
                renamer.rename_single_folder(
                    folder,
                    TrackInfo(
                        folder / "recording.flac", artist="Primary", title="Recording"
                    ),
                ),
                folder,
            )

    def test_nonexistent_directory_and_programming_failure_preserve_domain_errors(
        self,
    ) -> None:
        with self.assertRaises(FileNotFoundError):
            renamer.rename_directory_files(self.test_directory / "absent")
        path = self.test_directory / "recording.flac"
        path.touch()
        with (
            patch.object(
                renamer,
                "read_track_metadata",
                side_effect=RuntimeError("tag reader bug"),
            ),
            self.assertRaisesRegex(RuntimeError, "tag reader bug"),
        ):
            renamer.rename_directory_files(self.test_directory)

    def test_single_folders_containing_subdirectories_are_not_renamed(self) -> None:
        folder = self.test_directory / "Record"
        (folder / "Nested").mkdir(parents=True)
        self.assertEqual(
            renamer.rename_single_folder(
                folder,
                TrackInfo(
                    folder / "recording.flac", artist="Primary", title="Recording"
                ),
            ),
            folder,
        )

    def test_single_folder_is_renamed_after_track_success(self) -> None:
        folder = self.test_directory / "Singles" / "Record"
        folder.mkdir(parents=True)
        path = folder / "recording.flac"
        path.touch()
        recording = TrackInfo(path, artist="Primary", title="Recording", total_tracks=1)
        with patch.object(renamer, "read_track_metadata", return_value=recording):
            renamed = renamer.rename_directory_files(self.test_directory)
        self.assertEqual(
            renamed,
            [
                self.test_directory
                / "Singles"
                / "Primary - Recording"
                / "01 - Recording.flac"
            ],
        )
        self.assertTrue(renamed[0].exists())

    def test_multidisc_and_numberless_filenames_follow_declared_structure(self) -> None:
        self.assertEqual(
            renamer.build_new_filename(
                1, "Recording", ".flac", disc_number=1, total_discs=2
            ),
            "1-01 - Recording.flac",
        )
        self.assertEqual(
            renamer.build_new_filename(None, "Recording", ".flac", disc_number=2),
            "2-Recording.flac",
        )
        self.assertEqual(
            renamer.build_new_filename(3, "Recording", ".flac"), "03 - Recording.flac"
        )

    def test_multiple_singles_preserve_their_individual_track_numbers(self) -> None:
        folder = self.test_directory / "Singles"
        folder.mkdir()
        path = folder / "recording.flac"
        path.touch()
        (folder / "other.flac").touch()
        recording = TrackInfo(path, artist="Primary", title="Recording", track_number=2)
        self.assertEqual(
            renamer.rename_track_file(path, track_info=recording, dry_run=True),
            folder / "02 - Recording.flac",
        )

    def test_canonical_and_colliding_single_folders_are_preserved(self) -> None:
        canonical = self.test_directory / "Primary - Recording"
        canonical.mkdir()
        recording = TrackInfo(
            canonical / "recording.flac",
            artist="Primary",
            title="Recording",
            featured_artists="Guest",
        )
        self.assertEqual(renamer.rename_single_folder(canonical, recording), canonical)
        (self.test_directory / "Primary - Recording (feat. Guest)").mkdir()
        source = self.test_directory / "Original"
        source.mkdir()
        self.assertEqual(
            renamer.rename_single_folder(
                source,
                dataclasses.replace(recording, file_path=source / "recording.flac"),
            ),
            source,
        )

    def test_album_folder_identity_collision_and_preview_are_explicit(self) -> None:
        canonical = self.test_directory / "Primary - Record"
        canonical.mkdir()
        self.assertEqual(
            renamer.rename_album_folder(canonical, "Primary", "Record"), canonical
        )
        source = self.test_directory / "Original"
        source.mkdir()
        self.assertEqual(
            renamer.rename_album_folder(source, "Primary", "Record"), source
        )
        self.assertEqual(
            renamer.rename_album_folder(source, "Primary", "Other", dry_run=True),
            self.test_directory / "Primary - Other",
        )
        self.assertTrue(source.exists())

    def test_directory_report_records_failed_and_unchanged_files(self) -> None:
        path = self.test_directory / "01 - Recording.flac"
        path.touch()
        report = RenameReport()
        with patch.object(
            renamer, "read_track_metadata", side_effect=OSError("unreadable")
        ):
            self.assertEqual(
                renamer.rename_directory_files(self.test_directory, report=report), []
            )
        self.assertEqual((report.total_files, report.failed_files), (1, 1))
        recording = TrackInfo(
            path,
            artist="Unknown Artist",
            album="Unknown Album",
            title="Recording",
            track_number=1,
        )
        report = RenameReport()
        with patch.object(renamer, "read_track_metadata", return_value=recording):
            self.assertEqual(
                renamer.rename_directory_files(self.test_directory, report=report),
                [path],
            )
        self.assertEqual(
            (report.unchanged_files, report.files_renamed, report.failed_files),
            (1, 0, 0),
        )

    def test_collaborative_album_counts_and_mixed_release_directory(self) -> None:
        paths = [
            self.test_directory / f"{position:02} - Recording {position}.flac"
            for position in range(1, 4)
        ]
        for path in paths:
            path.touch()
        metadata_by_path = {
            path: TrackInfo(
                path,
                artist=f"Performer {position}",
                album="Record",
                title=f"Recording {position}",
                track_number=position,
            )
            for position, path in enumerate(paths, 1)
        }
        with (
            patch.object(
                renamer, "read_track_metadata", side_effect=metadata_by_path.__getitem__
            ),
            patch.object(
                renamer, "rename_album_folder", return_value=self.test_directory
            ) as rename_folder,
        ):
            self.assertEqual(
                len(
                    renamer.rename_directory_files(
                        self.test_directory, dry_run=True, max_threads=2
                    )
                ),
                3,
            )
        rename_folder.assert_called_once_with(
            self.test_directory, "Various Artists", "Record", dry_run=True
        )
        for position, recording in enumerate(metadata_by_path.values(), 1):
            recording.album = f"Record {position}"
        with (
            patch.object(
                renamer, "read_track_metadata", side_effect=metadata_by_path.__getitem__
            ),
            patch.object(renamer, "rename_album_folder") as rename_folder,
        ):
            self.assertEqual(
                len(
                    renamer.rename_directory_files(
                        self.test_directory, dry_run=True, max_threads=2
                    )
                ),
                3,
            )
        rename_folder.assert_not_called()

    def test_metadata_read_failure_preserves_the_original_io_error(self) -> None:
        path = self.test_directory / "recording.flac"
        path.touch()
        failure = OSError("unreadable tags")
        with (
            patch.object(renamer, "read_track_metadata", side_effect=failure),
            self.assertRaises(OSError) as raised,
        ):
            renamer.rename_track_file(path)
        self.assertIs(raised.exception, failure)
        self.assertTrue(path.exists())

    def test_empty_title_and_missing_audio_have_explicit_outcomes(self) -> None:
        self.assertIsNone(renamer.build_new_filename(1, "", ".flac"))
        with self.assertRaises(FileNotFoundError):
            renamer.rename_track_file(self.test_directory / "absent.flac")
        path = self.test_directory / "recording.flac"
        path.touch()
        self.assertEqual(
            renamer.rename_track_file(path, track_info=TrackInfo(path, title="")), path
        )
        self.assertEqual(
            renamer.rename_track_file(
                path,
                format_pattern="{artist} - {title}",
                track_info=TrackInfo(path, artist="Primary", title="Recording"),
                dry_run=True,
            ),
            self.test_directory / "Primary - Recording.flac",
        )
        self.assertTrue(path.exists())

    def test_lyric_header_sync_missing_file_and_write_failure(self) -> None:
        path = self.test_directory / "recording.lrc"
        self.assertFalse(renamer.sync_lrc_metadata(path, "Primary", "Recording"))
        path.write_text("[00:01] Lyrics")
        self.assertTrue(renamer.sync_lrc_metadata(path, "Primary", "Recording"))
        self.assertEqual(
            path.read_text(), "[ar:Primary]\n[ti:Recording]\n[00:01] Lyrics"
        )
        path.write_text("[ar:primary]\n[ti:recording]\n[00:01] Lyrics")
        with patch.object(
            renamer, "atomic_write_bytes", side_effect=OSError("disk full")
        ):
            self.assertFalse(renamer.sync_lrc_metadata(path, "Primary", "Recording"))
        self.assertIn("[ar:primary]", path.read_text())

    def test_rename_failure_and_collision_retry_preserve_source_recording(self) -> None:
        path = self.test_directory / "recording.flac"
        path.write_bytes(b"audio")
        recording = TrackInfo(path, artist="Primary", title="Recording", track_number=1)
        with (
            patch.object(
                renamer,
                "safe_case_rename",
                side_effect=PermissionError("read-only directory"),
            ),
            self.assertRaises(PermissionError),
        ):
            renamer.rename_track_file(path, track_info=recording)
        self.assertEqual(path.read_bytes(), b"audio")
        with patch.object(
            renamer,
            "safe_case_rename",
            side_effect=[FileExistsError("racing creator"), None],
        ) as rename:
            result = renamer.rename_track_file(path, track_info=recording)
        self.assertEqual(result.name, "01 - Recording.flac")
        self.assertEqual(rename.call_count, 2)

    def test_folder_rename_protects_containers_and_handles_io_failure(self) -> None:
        folder = self.test_directory / "Original"
        folder.mkdir()
        recording = TrackInfo(
            folder / "recording.flac", artist="Primary", title="Recording"
        )
        with patch.object(
            Path, "iterdir", side_effect=PermissionError("directory locked")
        ):
            self.assertEqual(renamer.rename_single_folder(folder, recording), folder)
            self.assertEqual(
                renamer.rename_album_folder(folder, "Primary", "Record"), folder
            )
        with patch.object(
            renamer, "safe_case_rename", side_effect=PermissionError("directory locked")
        ):
            self.assertEqual(renamer.rename_single_folder(folder, recording), folder)
            self.assertEqual(
                renamer.rename_album_folder(folder, "Primary", "Record"), folder
            )
        (folder / "Nested").mkdir()
        self.assertEqual(renamer.rename_single_folder(folder, recording), folder)
        self.assertEqual(
            renamer.rename_album_folder(folder, "Primary", "Record"), folder
        )


class TestOrganizationContracts(IsolatedTestCase):
    def test_single_folder_classification_respects_album_sequence_and_agreement(
        self,
    ) -> None:
        folder = self.test_directory / "Collection"
        folder.mkdir()
        paths = [folder / f"recording-{index}.flac" for index in range(3)]
        cases = (
            (
                [
                    TrackInfo(path, album=f"Record {index}", track_number=1)
                    for index, path in enumerate(paths)
                ],
                True,
            ),
            (
                [
                    TrackInfo(path, album="", track_number=index + 1)
                    for index, path in enumerate(paths)
                ],
                False,
            ),
            ([TrackInfo(path, album="", total_tracks=10) for path in paths], False),
            ([TrackInfo(path, album="Shared Record") for path in paths], False),
            ([TrackInfo(paths[0], album="", track_number=3)], False),
            ([TrackInfo(paths[0], album="Collection", total_tracks=2)], False),
        )
        for recordings, expected in cases:
            with self.subTest(recordings=recordings):
                self.assertEqual(
                    organizer._is_single_folder(
                        folder,
                        [recording.file_path for recording in recordings],
                        self.test_directory,
                        [(recording.file_path, recording) for recording in recordings],
                    ),
                    expected,
                )
        self.assertFalse(
            organizer._is_single_folder(folder, paths, self.test_directory, [])
        )

    def test_cleanup_protects_target_hierarchy_and_removes_known_junk(self) -> None:
        protected = self.test_directory / "Primary" / "Singles"
        protected.mkdir(parents=True)
        junk_directory = self.test_directory / "Empty"
        junk_directory.mkdir()
        (junk_directory / ".DS_Store").touch()
        self.assertEqual(
            organizer.cleanup_empty_dirs(
                self.test_directory, target_singles_dir=protected
            ),
            1,
        )
        self.assertTrue(protected.exists())
        self.assertFalse(junk_directory.exists())
        self.assertEqual(
            organizer.cleanup_empty_dirs(self.test_directory / "missing"), 0
        )
        self.assertEqual(
            organizer.consolidate_duplicate_artist_dirs(
                self.test_directory / "missing"
            ),
            0,
        )
