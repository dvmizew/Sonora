import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import taglib
from PIL import Image

from sonora.core.files import (
    atomic_audio_edit,
    atomic_write_bytes,
    move_file_exclusively,
)
from sonora.core.models import TrackInfo
from sonora.modules import backup, organizer, renamer
from tests.regression_support import AudioRegressionTestCase


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
        for folder, payload in (("Artist", b"original"), ("ARTIST", b"different")):
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

    def test_empty_compressed_backup_is_valid(self) -> None:
        archive = backup.backup_library_tags(self.root, self.root / "empty.json.gz")
        self.assertEqual(backup.restore_library_tags(archive), 0)

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
