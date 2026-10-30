import gzip
import json
import tracemalloc
from pathlib import Path
from typing import Any
from unittest.mock import patch

from sonora.core.models import RenameReport, TrackInfo
from sonora.modules import backup, renamer
from tests.support import IsolatedTestCase


def _accept_restored_tags(
    relative_path: str, backup_tags: dict[str, Any], restore_root: Path
) -> tuple[bool, bool]:
    return True, False


class TestRestoreMemoryAndPreflight(IsolatedTestCase):
    def test_large_artwork_restore_has_bounded_peak_memory(self) -> None:
        archive = self.test_directory / "backup.json.gz"
        with gzip.open(archive, "wb") as archive_stream:
            archive_stream.write(
                b'{"schema":"sonora_backup_v2","root":'
                + json.dumps(str(self.test_directory)).encode()
                + b',"tracks":{'
            )
            for index in range(48):
                if index:
                    archive_stream.write(b",")
                archive_stream.write(json.dumps(f"recording-{index}.flac").encode())
                archive_stream.write(b":")
                archive_stream.write(
                    json.dumps(
                        {
                            "tags": {"TITLE": [f"Recording {index}"]},
                            "complex_tags": {
                                "PICTURE": [{"data": {"base64": "A" * (512 * 1024)}}]
                            },
                        }
                    ).encode()
                )
            archive_stream.write(b"}}")
        tracemalloc.start()
        try:
            with patch.object(
                backup, "_restore_single_track", new=_accept_restored_tags
            ):
                self.assertEqual(
                    backup.restore_library_tags(archive, max_threads=2), 48
                )
            _, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak_bytes, 12 * 1024 * 1024)

    def test_invalid_final_entry_and_truncated_json_never_start_restoring(self) -> None:
        archive = self.test_directory / "backup.json"
        valid_tags = {"tags": {"TITLE": ["Recording"]}, "complex_tags": {}}
        manifest = {
            "schema": "sonora_backup_v2",
            "root": str(self.test_directory),
            "tracks": {"first.flac": valid_tags, "last.flac": {"tags": []}},
        }
        invalid_entry_payload = json.dumps(manifest)
        manifest["tracks"] = {"first.flac": valid_tags, "last.flac": valid_tags}
        for payload in (invalid_entry_payload, json.dumps(manifest)[:-1]):
            archive.write_text(payload)
            with (
                self.subTest(truncated=not payload.endswith("}")),
                patch.object(backup, "_restore_single_track") as restore,
                self.assertRaises((TypeError, ValueError)),
            ):
                backup.restore_library_tags(archive)
            restore.assert_not_called()

    def test_truncated_and_invalid_deflate_archives_fail_before_writes(self) -> None:
        archive = self.test_directory / "backup.json.gz"
        manifest = {
            "schema": "sonora_backup_v2",
            "root": str(self.test_directory),
            "tracks": {"recording.flac": {"tags": {}, "complex_tags": {}}},
        }
        compressed_archive = gzip.compress(json.dumps(manifest).encode())
        corrupted_archive = bytearray(compressed_archive)
        # The first DEFLATE block uses the reserved block type (RFC 1951 section 3.2.3).
        corrupted_archive[10] = 7
        for payload in (compressed_archive[:-8], bytes(corrupted_archive)):
            archive.write_bytes(payload)
            with (
                self.subTest(truncated=payload == compressed_archive[:-8]),
                patch.object(backup, "_restore_single_track") as restore,
                self.assertRaisesRegex(ValueError, "Invalid backup archive"),
            ):
                backup.restore_library_tags(archive)
            restore.assert_not_called()

    def test_empty_tracks_and_manifest_fields_after_tracks_are_supported(self) -> None:
        archive = self.test_directory / "backup.json"
        track_cases: tuple[dict[str, dict[str, Any]], ...] = (
            {},
            {"recording.flac": {"tags": {}, "complex_tags": {}}},
        )
        for tracks in track_cases:
            archive.write_text(
                json.dumps(
                    {
                        "tracks": tracks,
                        "root": str(self.test_directory),
                        "schema": "sonora_backup_v2",
                    }
                )
            )
            with (
                self.subTest(track_count=len(tracks)),
                patch.object(
                    backup, "_restore_single_track", side_effect=_accept_restored_tags
                ),
            ):
                self.assertEqual(backup.restore_library_tags(archive), len(tracks))

    def test_duplicate_manifest_keys_and_tracks_are_rejected_before_writes(
        self,
    ) -> None:
        archive = self.test_directory / "backup.json"
        prefix = '{"schema":"sonora_backup_v2","root":' + json.dumps(
            str(self.test_directory)
        )
        tags = '{"tags":{},"complex_tags":{}}'
        payloads = (
            prefix + ',"root":"other","tracks":{}}',
            prefix + ',"tracks":{"same.flac":' + tags + ',"same.flac":' + tags + "}}",
        )
        for payload in payloads:
            archive.write_text(payload)
            with (
                self.subTest(payload=payload),
                patch.object(backup, "_restore_single_track") as restore,
                self.assertRaisesRegex(ValueError, "Duplicate backup"),
            ):
                backup.restore_library_tags(archive)
            restore.assert_not_called()


class TestRenameConsensusAndLyrics(IsolatedTestCase):
    def test_invalid_utf8_lyrics_remain_unchanged(self) -> None:
        lyrics_path = self.test_directory / "recording.lrc"
        for payload in (b"[00:01.00]caf\xe9\n", b"[ar:Primary]\n[00:01.00]\xff\xfe\n"):
            with self.subTest(payload=payload):
                lyrics_path.write_bytes(payload)
                self.assertFalse(
                    renamer.sync_lrc_metadata(lyrics_path, "Artist", "Song")
                )
                self.assertEqual(lyrics_path.read_bytes(), payload)

    def test_utf8_lyrics_keep_timestamps_and_non_ascii_text(self) -> None:
        lyrics_path = self.test_directory / "recording.lrc"
        lyrics_path.write_text("[00:01.00]café\n", encoding="utf-8")
        self.assertTrue(renamer.sync_lrc_metadata(lyrics_path, "Artist", "Song"))
        self.assertIn("[00:01.00]café\n", lyrics_path.read_text(encoding="utf-8"))

    def test_matching_headers_are_a_noop_including_crlf_files(self) -> None:
        lyrics_path = self.test_directory / "recording.lrc"
        for newline in (b"\n", b"\r\n"):
            payload = newline.join((b"[ar:Artist]", b"[ti:Song]", b"[00:01.00] Lyrics"))
            lyrics_path.write_bytes(payload)
            with (
                self.subTest(newline=newline),
                patch.object(renamer, "atomic_write_bytes") as write,
            ):
                self.assertFalse(
                    renamer.sync_lrc_metadata(lyrics_path, "Artist", "Song")
                )
            write.assert_not_called()
            self.assertEqual(lyrics_path.read_bytes(), payload)

    def test_rename_report_counts_header_changes_without_audio_rename(self) -> None:
        audio_path = self.test_directory / "01 - Song.flac"
        audio_path.touch()
        lyrics_path = audio_path.with_suffix(".lrc")
        lyrics_path.write_text("[ar:Old]\n[ti:Old]\n[00:01.00] Lyrics\n")
        recording = TrackInfo(
            audio_path,
            artist="Artist",
            title="Song",
            track_number=1,
            album="Unknown Album",
        )
        with patch.object(renamer, "read_track_metadata", return_value=recording):
            first_report = RenameReport()
            renamer.rename_directory_files(self.test_directory, report=first_report)
            second_report = RenameReport()
            renamer.rename_directory_files(self.test_directory, report=second_report)
        self.assertEqual((first_report.files_renamed, first_report.lrc_synced), (0, 1))
        self.assertEqual(
            (second_report.files_renamed, second_report.lrc_synced), (0, 0)
        )

    def test_rename_report_excludes_malformed_lyrics_and_dry_run_updates(self) -> None:
        audio_path = self.test_directory / "original.flac"
        audio_path.touch()
        lyrics_path = audio_path.with_suffix(".lrc")
        malformed_lyrics = b"[00:01.00]caf\xe9\n"
        lyrics_path.write_bytes(malformed_lyrics)
        recording = TrackInfo(
            audio_path,
            artist="Artist",
            title="Song",
            track_number=1,
            album="Unknown Album",
        )
        with patch.object(renamer, "read_track_metadata", return_value=recording):
            dry_report = RenameReport()
            renamer.rename_directory_files(
                self.test_directory, dry_run=True, report=dry_report
            )
            write_report = RenameReport()
            renamer.rename_directory_files(self.test_directory, report=write_report)
        self.assertEqual(dry_report.lrc_synced, 0)
        self.assertEqual((write_report.files_renamed, write_report.lrc_synced), (1, 0))
        self.assertEqual(
            (self.test_directory / "01 - Song.lrc").read_bytes(), malformed_lyrics
        )

    def test_successful_header_sync_is_reported_when_audio_rename_fails(self) -> None:
        audio_path = self.test_directory / "original.flac"
        audio_path.touch()
        audio_path.with_suffix(".lrc").write_text("[00:01.00] Lyrics\n")
        recording = TrackInfo(
            audio_path,
            artist="Artist",
            title="Song",
            track_number=1,
            album="Unknown Album",
        )
        report = RenameReport()
        with (
            patch.object(renamer, "read_track_metadata", return_value=recording),
            patch.object(
                renamer, "safe_case_rename", side_effect=PermissionError("locked")
            ),
        ):
            renamer.rename_directory_files(self.test_directory, report=report)
        self.assertEqual((report.failed_files, report.lrc_synced), (1, 1))

    def test_tied_distinct_albums_do_not_rename_folder_in_either_order(self) -> None:
        paths = [
            self.test_directory / "first.flac",
            self.test_directory / "second.flac",
        ]
        for path in paths:
            path.touch()
        tracks = {
            path: TrackInfo(
                path,
                artist=f"Artist {index}",
                album=f"Album {index}",
                title=f"Song {index}",
            )
            for index, path in enumerate(paths)
        }
        for ordered_paths in (paths, paths[::-1]):
            with (
                self.subTest(order=ordered_paths),
                patch.object(renamer, "find_audio_files", return_value=ordered_paths),
                patch.object(
                    renamer, "read_track_metadata", side_effect=tracks.__getitem__
                ),
                patch.object(renamer, "rename_album_folder") as rename_folder,
            ):
                renamer.rename_directory_files(
                    self.test_directory, dry_run=True, max_threads=1
                )
            rename_folder.assert_not_called()

    def test_tied_artists_on_same_album_use_compilation_credit(self) -> None:
        paths = [
            self.test_directory / "first.flac",
            self.test_directory / "second.flac",
        ]
        for path in paths:
            path.touch()
        tracks = {
            path: TrackInfo(
                path,
                artist=f"Artist {index}",
                album="Compilation",
                title=f"Song {index}",
            )
            for index, path in enumerate(paths)
        }
        with (
            patch.object(
                renamer, "read_track_metadata", side_effect=tracks.__getitem__
            ),
            patch.object(
                renamer, "rename_album_folder", return_value=self.test_directory
            ) as rename_folder,
        ):
            renamer.rename_directory_files(
                self.test_directory, dry_run=True, max_threads=1
            )
        rename_folder.assert_called_once_with(
            self.test_directory, "Various Artists", "Compilation", dry_run=True
        )

    def test_incomplete_album_evidence_does_not_become_compilation(self) -> None:
        paths = [
            self.test_directory / "first.flac",
            self.test_directory / "second.flac",
        ]
        for path in paths:
            path.touch()
        tracks = {
            paths[0]: TrackInfo(
                paths[0], artist="Primary", album="Record", title="Song"
            ),
            paths[1]: TrackInfo(
                paths[1],
                artist="Unknown Artist",
                album="Unknown Album",
                title="Unknown",
            ),
        }
        with (
            patch.object(
                renamer, "read_track_metadata", side_effect=tracks.__getitem__
            ),
            patch.object(renamer, "rename_album_folder") as rename_folder,
        ):
            renamer.rename_directory_files(
                self.test_directory, dry_run=True, max_threads=1
            )
        rename_folder.assert_not_called()
