import gzip
import json
from typing import Any
from unittest.mock import MagicMock, patch

import taglib

from sonora.modules import backup
from tests.support import IsolatedTestCase


class TestBackupManifestContracts(IsolatedTestCase):
    def setUp(self) -> None:
        self.archive = self.test_directory / "backup.json"

    def test_backup_requires_a_directory(self) -> None:
        with self.assertRaises(ValueError):
            backup.backup_library_tags(self.archive)
        self.archive.touch()
        with self.assertRaises(ValueError):
            backup.backup_library_tags(self.archive)

    def test_missing_corrupt_and_wrong_schema_archives_fail_before_writing(
        self,
    ) -> None:
        with self.assertRaises(FileNotFoundError):
            backup.restore_library_tags(self.archive)
        for content, error in (
            (b"broken json", ValueError),
            (gzip.compress(b"broken json"), ValueError),
            (b"[]", ValueError),
            (b'{"schema": "other"}', ValueError),
        ):
            self.archive.write_bytes(content)
            with self.subTest(content=content), self.assertRaises(error):
                backup.restore_library_tags(self.archive)

    def test_invalid_manifest_records_and_binary_properties_are_rejected(self) -> None:
        track_properties: tuple[object, ...] = (
            [],
            {},
            {"tags": {}, "complex_tags": []},
            {"tags": {"TITLE": "not a list"}, "complex_tags": {}},
            {"tags": {"TITLE": [12]}, "complex_tags": {}},
            {"tags": {}, "complex_tags": {"PICTURE": "not a list"}},
            {"tags": {}, "complex_tags": {"PICTURE": ["not an object"]}},
            {"tags": {}, "complex_tags": {"PICTURE": [{"data": {"base64": "!!!"}}]}},
            {"tags": {}, "complex_tags": {"PICTURE": [{"data": {"other": "value"}}]}},
            {"tags": {}, "complex_tags": {"PICTURE": [{"data": None}]}},
        )
        for properties in track_properties:
            manifest = {
                "schema": "sonora_backup_v2",
                "root": str(self.test_directory),
                "tracks": {"recording.flac": properties},
            }
            self.archive.write_text(json.dumps(manifest))
            with (
                self.subTest(properties=properties),
                patch.object(backup, "_restore_single_track") as restore,
                self.assertRaises((ValueError, TypeError)),
            ):
                backup.restore_library_tags(self.archive)
            restore.assert_not_called()
        self.archive.write_text(
            json.dumps({"schema": "sonora_backup_v2", "root": None, "tracks": {}})
        )
        with self.assertRaisesRegex(TypeError, "Invalid backup manifest"):
            backup.restore_library_tags(self.archive)

    def test_restore_reports_missing_and_failed_tracks_separately(self) -> None:
        manifest = {
            "schema": "sonora_backup_v2",
            "root": str(self.test_directory),
            "tracks": {
                "missing.flac": {"tags": {}, "complex_tags": {}},
                "failed.flac": {"tags": {}, "complex_tags": {}},
            },
        }
        self.archive.write_text(json.dumps(manifest))
        failures: list[str] = []
        with patch.object(
            backup, "_restore_single_track", side_effect=[(False, True), (False, False)]
        ):
            self.assertEqual(
                backup.restore_library_tags(self.archive, failures=failures), 0
            )
        self.assertCountEqual(failures, ["missing.flac", "failed.flac"])

    def test_backup_of_unreadable_audio_saves_partial_manifest_and_reports_failure(
        self,
    ) -> None:
        recording = self.test_directory / "recording.flac"
        recording.write_bytes(b"corrupt recording")
        with self.assertRaisesRegex(OSError, "Partial backup saved"):
            backup.backup_library_tags(self.test_directory, self.archive)
        manifest = json.loads(self.archive.read_text())
        self.assertEqual(manifest["schema"], "sonora_backup_v2")
        self.assertEqual(manifest["tracks"], {})
        self.assertEqual(manifest["root"], str(self.test_directory))

    def test_programming_failures_in_backup_and_restore_are_not_masked(self) -> None:
        recording = self.test_directory / "recording.flac"
        recording.touch()
        with (
            patch.object(
                backup, "_read_track_for_backup", side_effect=RuntimeError("reader bug")
            ),
            self.assertRaisesRegex(RuntimeError, "reader bug"),
        ):
            backup.backup_library_tags(self.test_directory, self.archive)
        self.archive.write_text(
            json.dumps(
                {
                    "schema": "sonora_backup_v2",
                    "root": str(self.test_directory),
                    "tracks": {recording.name: {"tags": {}, "complex_tags": {}}},
                }
            )
        )
        with (
            patch.object(
                backup, "_restore_single_track", side_effect=RuntimeError("writer bug")
            ),
            self.assertRaisesRegex(RuntimeError, "writer bug"),
        ):
            backup.restore_library_tags(self.archive)


class TestBinaryPropertyContracts(IsolatedTestCase):
    def test_raw_binary_properties_use_valid_base64_and_preserve_other_fields(
        self,
    ) -> None:
        audio_tags = MagicMock(spec=taglib.File)
        audio_tags.tags = {"TITLE": ["Recording"]}
        audio_tags.complex_property_keys = ["PICTURE"]
        audio_tags.complex_properties.return_value = [
            {"data": b"binary image", "description": "Front", "type": 3}
        ]
        with patch("taglib.File") as open_audio:
            open_audio.return_value.__enter__.return_value = audio_tags
            path, properties = backup._read_track_for_backup(
                self.test_directory / "recording.flac"
            )
        self.assertTrue(path.endswith("recording.flac"))
        self.assertEqual(
            properties,
            {
                "tags": {"TITLE": ["Recording"]},
                "complex_tags": {
                    "PICTURE": [
                        {
                            "data": {"base64": "YmluYXJ5IGltYWdl"},
                            "description": "Front",
                            "type": 3,
                        }
                    ]
                },
            },
        )

    def test_restore_binary_property_and_save_failures_leave_original_untouched(
        self,
    ) -> None:
        path = self.test_directory / "recording.flac"
        path.write_bytes(b"original audio")
        properties: dict[str, Any] = {
            "tags": {"TITLE": ["Recording"]},
            "complex_tags": {
                "PICTURE": [
                    {"data": {"base64": "YmluYXJ5IGltYWdl"}, "description": "Front"}
                ]
            },
        }
        for property_success, save_result, expected in (
            (False, [], False),
            (True, ["TITLE"], False),
            (True, [], True),
        ):
            audio_tags = MagicMock(spec=taglib.File)
            audio_tags.tags = {"TITLE": ["Old"]}
            audio_tags.complex_property_keys = []
            audio_tags.set_complex_properties.return_value = property_success
            audio_tags.save.return_value = save_result
            with (
                self.subTest(
                    property_success=property_success, save_result=save_result
                ),
                patch("taglib.File") as open_audio,
            ):
                open_audio.return_value.__enter__.return_value = audio_tags
                self.assertEqual(
                    backup._restore_single_track(
                        path.name, properties, self.test_directory
                    ),
                    (expected, False),
                )
            self.assertEqual(path.read_bytes(), b"original audio")
            audio_tags.set_complex_properties.assert_called_once_with(
                "PICTURE", [{"data": b"binary image", "description": "Front"}]
            )
            self.assertEqual(list(self.test_directory.glob(".*recording*")), [])

    def test_restore_read_only_audio_does_not_open_tag_writer(self) -> None:
        path = self.test_directory / "recording.flac"
        path.touch()
        with (
            patch("sonora.modules.backup.os.access", return_value=False),
            patch("taglib.File") as open_audio,
        ):
            self.assertEqual(
                backup._restore_single_track(
                    path.name, {"tags": {}, "complex_tags": {}}, self.test_directory
                ),
                (False, False),
            )
        open_audio.assert_not_called()
