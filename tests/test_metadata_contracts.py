import dataclasses
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import taglib
from mutagen.mp4 import MP4

from sonora.audio import metadata
from sonora.core.models import NormalizeReport, TrackInfo
from sonora.modules.backup import backup_library_tags, restore_library_tags
from sonora.modules.metadata_merge import apply_metadata_mapping
from sonora.modules.track_processing import normalize_single_track
from tests.fixtures import image_bytes
from tests.regression_support import AudioRegressionTestCase
from tests.support import IsolatedTestCase


class TestMetadataContracts(IsolatedTestCase):
    def test_provider_album_capitalization_preserves_consistent_local_spelling(
        self,
    ) -> None:
        track = TrackInfo(Path("recording.flac"), album="Un Film De Oscar")
        apply_metadata_mapping(
            track, {"album": "Un film de oscar"}, {"album": "album"}, force=True
        )
        self.assertEqual(track.album, "Un Film De Oscar")
        apply_metadata_mapping(
            track,
            {"album": "Un Film De Oscar (Deluxe)"},
            {"album": "album"},
            force=True,
        )
        self.assertEqual(track.album, "Un Film De Oscar (Deluxe)")

    def test_report_export_owns_provenance_and_excludes_internal_changes(self) -> None:
        track = TrackInfo(
            Path("recording.flac"),
            metadata_sources={"title": "MusicBrainz"},
            _diff_descriptions=["changed"],
        )
        exported = track.to_dict()
        sources = exported["metadata_sources"]
        assert isinstance(sources, dict)
        sources.clear()
        self.assertEqual(track.metadata_sources, {"title": "MusicBrainz"})
        self.assertNotIn("_diff_descriptions", exported)
        self.assertEqual(exported["file_path"], "recording.flac")

    def test_failed_atomic_commit_keeps_pending_advisory_normalization(self) -> None:
        path = self.test_directory / "recording.flac"
        path.write_bytes(b"original audio")
        recording = TrackInfo(path, _advisory_needs_normalization=True)
        song = MagicMock(spec=taglib.File)
        song.tags = {}
        song.save.return_value = []
        with (
            patch("taglib.File") as open_audio,
            patch.object(Path, "replace", side_effect=OSError("disk full")),
            self.assertRaisesRegex(OSError, "disk full"),
        ):
            open_audio.return_value.__enter__.return_value = song
            metadata.write_track_metadata(recording)
        self.assertTrue(recording._advisory_needs_normalization)
        self.assertEqual(path.read_bytes(), b"original audio")

    def test_track_copy_owns_provenance_and_change_collections(self) -> None:
        original = TrackInfo(
            Path("recording.flac"),
            metadata_sources={"title": "MusicBrainz"},
            metadata_conflicts={"isrc": "Catalog disagreement"},
            _diff_descriptions=["Title normalized"],
        )
        copied = original.copy()
        self.assertEqual(copied, original)
        copied.metadata_sources.clear()
        copied.metadata_conflicts.clear()
        copied._diff_descriptions.clear()
        self.assertEqual(original.metadata_sources, {"title": "MusicBrainz"})
        self.assertEqual(original.metadata_conflicts, {"isrc": "Catalog disagreement"})
        self.assertEqual(original._diff_descriptions, ["Title normalized"])
        empty = TrackInfo(Path("empty.flac"))
        empty.copy().metadata_conflicts["title"] = "Review required"
        self.assertEqual(empty.metadata_conflicts, {})

    def test_normalization_report_counts_include_failures(self) -> None:
        report = NormalizeReport()
        self.assertEqual((report.total_scanned, report.total_unchanged), (0, 0))
        report.total_scanned += 1
        self.assertEqual((report.total_scanned, report.total_unchanged), (1, 1))
        report.modified_files["recording.flac"] = ["Title normalized"]
        self.assertEqual((report.total_modified, report.total_unchanged), (1, 0))
        report.total_scanned += 1
        report.failed_files += 1
        self.assertEqual((report.total_scanned, report.total_unchanged), (2, 0))

    def test_embedded_dimensions_and_explicit_feature_tags_need_no_fallback(
        self,
    ) -> None:
        path = self.test_directory / "recording.wav"
        path.touch()
        song = MagicMock(spec=taglib.File)
        song.tags = {
            "TITLE": ["Recording"],
            "ARTIST": ["Primary"],
            "FEATURED_ARTISTS": ["Guest"],
        }
        song.pictures = [
            taglib.Picture(data=b"", mime_type="image/png", width=600, height=600)
        ]
        song.sampleRate = 44100
        song.bitrate = 128
        song.channels = 2
        song.length = 1
        with patch("taglib.File") as open_audio:
            open_audio.return_value.__enter__.return_value = song
            recording = metadata.read_track_metadata(path)
        self.assertEqual((recording.art_width, recording.art_height), (600, 600))
        self.assertEqual(recording.featured_artists, "Guest")

    def test_missing_audio_cannot_be_written(self) -> None:
        with self.assertRaises(FileNotFoundError):
            metadata.write_track_metadata(
                TrackInfo(self.test_directory / "missing.flac")
            )

    def test_numbering_and_date_writes_remove_absent_disc_numbers(self) -> None:
        tags = {"DISCNUMBER": ["2"]}
        metadata._write_numbering_and_dates(
            tags, TrackInfo(Path("recording.flac"), disc_number=None, date="2020")
        )
        self.assertNotIn("DISCNUMBER", tags)
        self.assertEqual(tags["DATE"], ["2020"])

    def test_read_failure_from_concurrent_removal_preserves_file_not_found(
        self,
    ) -> None:
        path = self.test_directory / "recording.wav"
        path.touch()
        with (
            patch("taglib.File", side_effect=FileNotFoundError("removed concurrently")),
            self.assertRaises(FileNotFoundError),
        ):
            metadata.read_track_metadata(path)

    def test_artist_lists_deduplicate_overlapping_features_and_empty_album_credits(
        self,
    ) -> None:
        tags: dict[str, list[str]] = {}
        metadata._write_artist_tags(
            tags,
            TrackInfo(
                Path("recording.flac"),
                artist="Primary & Guest",
                featured_artists="Guest",
                album_artist=" ",
            ),
        )
        self.assertEqual(tags["ARTISTS"], ["Primary", "Guest"])
        self.assertNotIn("ALBUMARTISTS", tags)

    def test_filename_and_disc_folder_recover_missing_numbering(self) -> None:
        for relative_path, expected_disc, expected_track in (
            ("CD 2/recording.wav", 2, None),
            ("2-03 - Recording.wav", 2, 3),
        ):
            path = self.test_directory / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            song = MagicMock(spec=taglib.File)
            song.tags = {
                "TITLE": ["Recording"],
                "ARTIST": ["Primary"],
                "ARTISTS": ["Primary", "Guest"],
                "ADVISORY": ["1"],
            }
            song.pictures = [
                taglib.Picture(
                    data=image_bytes(), mime_type="image/png", picture_type="Back Cover"
                )
            ]
            song.length = 1
            song.bitrate = 128
            song.sampleRate = 44100
            song.channels = 2
            with (
                self.subTest(relative_path=relative_path),
                patch("taglib.File") as open_audio,
            ):
                open_audio.return_value.__enter__.return_value = song
                recording = metadata.read_track_metadata(path)
            self.assertEqual(
                (recording.disc_number, recording.track_number),
                (expected_disc, expected_track),
            )
            self.assertEqual(recording.featured_artists, "Guest")
            self.assertEqual(recording.advisory, "Explicit")

    def test_read_permission_and_cache_capacity_are_explicit(self) -> None:
        path = self.test_directory / "recording.wav"
        path.touch()
        with (
            patch.object(os, "access", return_value=False),
            self.assertRaises(PermissionError),
        ):
            metadata.write_track_metadata(TrackInfo(path))
        with (
            patch.object(metadata, "_MAX_METADATA_CACHE_SIZE", 0),
            patch("taglib.File") as open_audio,
        ):
            song = open_audio.return_value.__enter__.return_value
            song.tags = {}
            song.pictures = []
            song.length = 1
            song.bitrate = 128
            song.sampleRate = 44100
            song.channels = 2
            metadata.read_track_metadata(path)
        self.assertEqual(metadata.get_metadata_cache_size(), 1)

    def test_numbering_without_totals_and_original_year_are_written(self) -> None:
        tags: dict[str, list[str]] = {}
        metadata._write_numbering_and_dates(
            tags,
            TrackInfo(
                Path("recording.flac"),
                disc_number=2,
                track_number=3,
                original_date="1999-01-02",
            ),
        )
        self.assertEqual(tags["DISCNUMBER"], ["2"])
        self.assertEqual(tags["TRACKNUMBER"], ["3"])
        self.assertEqual(tags["ORIGINALYEAR"], ["1999"])
        self.assertNotIn("TRACKTOTAL", tags)
        self.assertNotIn("DISCTOTAL", tags)

    def test_mutagen_duration_and_corrupt_embedded_cover(self) -> None:
        with patch(
            "mutagen.File",
            return_value=SimpleNamespace(info=SimpleNamespace(length=12.5)),
        ):
            self.assertEqual(metadata.get_audio_duration(Path("recording.flac")), 12.5)
        path = self.test_directory / "recording.wav"
        path.write_bytes(b"audio fixture")
        cover = self.test_directory / "cover.png"
        cover.write_bytes(image_bytes())
        with patch("taglib.File") as open_audio:
            song = open_audio.return_value.__enter__.return_value
            song.tags = {}
            song.pictures = [
                taglib.Picture(
                    data=b"corrupt", mime_type="image/png", picture_type="Front Cover"
                )
            ]
            song.save.return_value = []
            metadata.write_track_metadata(TrackInfo(path), cover)
            self.assertEqual(len(song.pictures), 1)
            self.assertEqual(song.pictures[0].data, cover.read_bytes())

    def test_cover_replacement_cannot_reduce_either_image_dimension(self) -> None:
        path = self.test_directory / "recording.wav"
        path.write_bytes(b"audio fixture")
        replacement = self.test_directory / "cover.png"
        replacement.write_bytes(image_bytes((1000, 1000)))
        for dimensions in ((1400, 600), (600, 1400)):
            embedded = taglib.Picture(
                data=image_bytes(dimensions),
                mime_type="image/png",
                picture_type="Front Cover",
            )
            with (
                self.subTest(dimensions=dimensions),
                patch("taglib.File") as open_audio,
            ):
                song = open_audio.return_value.__enter__.return_value
                song.tags = {}
                song.pictures = [embedded]
                song.save.return_value = []
                metadata.write_track_metadata(TrackInfo(path), replacement)
                self.assertEqual(song.pictures[0].data, embedded.data)

    def test_unchanged_album_artist_preserves_original_multivalue_credits(self) -> None:
        original = {"ALBUMARTIST": ["Primary"], "ALBUMARTISTS": ["Primary", "Guest"]}
        updated = {"ALBUMARTIST": ["Primary"]}
        metadata._preserve_unchanged_multivalues(original, updated)
        self.assertEqual(updated["ALBUMARTISTS"], ["Primary", "Guest"])

    def test_mp4_codec_distinguishes_lossless_from_lossy(self) -> None:
        for codec, expected in (("alac", True), ("aac", False)):
            with (
                self.subTest(codec=codec),
                patch.object(
                    metadata,
                    "MP4",
                    return_value=SimpleNamespace(info=SimpleNamespace(codec=codec)),
                ),
            ):
                self.assertEqual(
                    metadata._read_audio_encoding(Path("recording.m4a")),
                    (expected, None),
                )
        with patch.object(metadata, "MP4", side_effect=OSError("broken mp4")):
            self.assertEqual(
                metadata._read_audio_encoding(Path("recording.m4a")), (False, None)
            )
        self.assertEqual(
            metadata._read_audio_encoding(Path("recording.unknown")), (True, None)
        )

    def test_audio_properties_roundtrip_optional_values_and_compilation_policy(
        self,
    ) -> None:
        recording = TrackInfo(
            Path("recording.flac"), bpm=123.45, rating=8.5, compilation=True
        )
        tags: dict[str, list[str]] = {}
        metadata._write_audio_properties_tags(tags, recording)
        self.assertEqual(tags["BPM"], ["123.5"])
        self.assertEqual(tags["RATING"], ["8.5"])
        self.assertEqual(tags["COMPILATION"], ["1"])
        metadata._write_audio_properties_tags(
            tags, dataclasses.replace(recording, compilation=False)
        )
        self.assertEqual(tags["COMPILATION"], ["0"])
        metadata._write_audio_properties_tags(
            tags, TrackInfo(recording.file_path, album_artist="Various Artists")
        )
        self.assertEqual(tags["COMPILATION"], ["1"])
        self.assertNotIn("BPM", tags)
        metadata._write_audio_properties_tags(tags, TrackInfo(recording.file_path))
        self.assertNotIn("COMPILATION", tags)

    def test_schema_writer_removes_invalid_uuid_and_clean_advisory_aliases(
        self,
    ) -> None:
        tags = {
            "MUSICBRAINZ_TRACKID": ["old"],
            "ITUNESADVISORY": ["1"],
            "ADVISORY": ["Explicit"],
        }
        recording = TrackInfo(
            Path("recording.flac"),
            musicbrainz_trackid="invalid",
            advisory="Clean",
            disc_subtitle="Second movement",
        )
        metadata._write_schema_tags(tags, recording)
        self.assertNotIn("MUSICBRAINZ_TRACKID", tags)
        self.assertNotIn("ADVISORY", tags)
        self.assertNotIn("ITUNESADVISORY", tags)
        self.assertEqual(tags["DISCSUBTITLE"], ["Second movement"])
        self.assertEqual(tags["TSST"], ["Second movement"])

    def test_picture_dimensions_are_decoded_when_taglib_does_not_supply_them(
        self,
    ) -> None:
        path = self.test_directory / "recording.wav"
        path.write_bytes(b"audio fixture")
        song = MagicMock(spec=taglib.File)
        song.tags = {
            "TITLE": ["Recording"],
            "ARTIST": ["Primary"],
            "ADVISORY": ["clean"],
        }
        song.pictures = [
            taglib.Picture(
                data=image_bytes((640, 640)),
                mime_type="image/png",
                picture_type="Front Cover",
            )
        ]
        song.length = 1
        song.bitrate = 128
        song.sampleRate = 44100
        song.channels = 2
        with patch("taglib.File") as open_audio:
            open_audio.return_value.__enter__.return_value = song
            recording = metadata.read_track_metadata(path)
        self.assertEqual((recording.art_width, recording.art_height), (640, 640))
        self.assertIsNone(recording.advisory)
        metadata.clear_metadata_cache()
        song.pictures = [
            taglib.Picture(
                data=b"broken", mime_type="image/png", picture_type="Front Cover"
            )
        ]
        with patch("taglib.File") as open_audio:
            open_audio.return_value.__enter__.return_value = song
            recording = metadata.read_track_metadata(path)
        self.assertIsNone(recording.art_width)
        self.assertIsNone(recording.art_height)

    def test_failed_save_and_concurrent_file_removal_leave_audio_untouched(
        self,
    ) -> None:
        path = self.test_directory / "recording.flac"
        path.write_bytes(b"original audio")
        recording = TrackInfo(path)
        song = MagicMock(spec=taglib.File)
        song.tags = {}
        song.pictures = []
        song.save.return_value = ["TITLE"]
        with (
            patch("taglib.File") as open_audio,
            self.assertRaisesRegex(OSError, "Unsupported tags"),
        ):
            open_audio.return_value.__enter__.return_value = song
            metadata.write_track_metadata(recording)
        self.assertEqual(path.read_bytes(), b"original audio")
        with (
            patch("taglib.File", side_effect=FileNotFoundError("removed concurrently")),
            self.assertRaises(FileNotFoundError),
        ):
            metadata.write_track_metadata(recording)
        self.assertEqual(path.read_bytes(), b"original audio")


class TestAdvisoryPersistence(AudioRegressionTestCase):
    def test_lyrics_only_write_preserves_every_other_raw_tag(self) -> None:
        path = self.audio("lyrics.flac")
        with taglib.File(str(path)) as recording:
            recording.tags.update(
                {
                    "ARTIST": ["Artist feat. Guest"],
                    "BPM": ["123.456"],
                    "TOTALTRACKS": ["99"],
                    "CUSTOM": ["Keep"],
                }
            )
            recording.save()
        with taglib.File(str(path)) as recording:
            before = dict(recording.tags)
        track = metadata.read_track_metadata(path)
        track.lyrics = "[00:00.00] Correct words"
        metadata.write_track_metadata(track, lyrics_only=True)
        with taglib.File(str(path)) as recording:
            after = dict(recording.tags)
        self.assertEqual(
            {k: v for k, v in after.items() if k not in {"LYRICS", "UNSYNCEDLYRICS"}},
            before,
        )
        self.assertEqual(after["LYRICS"], [track.lyrics])

    def test_embedded_catalog_isrc_respects_explicit_edition(self) -> None:
        path = self.audio("edition.flac")
        metadata.write_track_metadata(
            TrackInfo(
                path,
                artist="Artist",
                title="Song",
                album="Album",
                advisory="Explicit",
                isrc="USABC2100001",
            )
        )
        for explicit, expected in ((False, "USABC2100001"), (True, "USABC2100002")):
            with self.subTest(explicit=explicit):
                with taglib.File(str(path)) as recording:
                    recording.tags["TIDAL_DATA"] = [
                        json.dumps(
                            {
                                "artist": {"name": "Artist"},
                                "title": "Song",
                                "album": {"title": "Album"},
                                "duration": 1,
                                "explicit": explicit,
                                "isrc": "USABC2100002",
                            }
                        )
                    ]
                    recording.save()
                metadata.clear_metadata_cache()
                result = metadata.read_track_metadata(path)
                self.assertEqual(result.isrc, expected)
                self.assertEqual(result.advisory, "Explicit")

    def test_tag_write_invalidates_only_the_changed_file(self) -> None:
        first = self.audio("first.flac")
        second = self.audio("second.flac")
        metadata.read_track_metadata(first)
        metadata.read_track_metadata(second)
        self.assertEqual(metadata.get_metadata_cache_size(), 2)
        metadata.write_track_metadata(TrackInfo(first, title="Changed"))
        self.assertEqual(metadata.get_metadata_cache_size(), 1)
        with patch("taglib.File", side_effect=AssertionError("unrelated cache miss")):
            self.assertEqual(metadata.read_track_metadata(second).file_path, second)

    def test_clean_native_tags_are_removed_without_force_and_dry_run_is_read_only(
        self,
    ) -> None:
        path = self.audio("recording.flac")
        metadata.write_track_metadata(TrackInfo(path, artist="Alpha", title="Song"))
        with taglib.File(str(path)) as recording:
            recording.tags["ITUNESADVISORY"] = ["2"]
            recording.tags["ADVISORY"] = ["Clean"]
            recording.save()
        metadata.clear_metadata_cache()
        before = path.read_bytes()
        preview = normalize_single_track(path, dry_run=True)
        assert preview is not None
        self.assertTrue(
            any("advisory" in change for change in preview._diff_descriptions)
        )
        self.assertEqual(path.read_bytes(), before)
        result = normalize_single_track(path)
        assert result is not None
        self.assertIsNone(result.advisory)
        with taglib.File(str(path)) as recording:
            self.assertNotIn("ITUNESADVISORY", recording.tags)
            self.assertNotIn("ADVISORY", recording.tags)
        unchanged = path.read_bytes()
        result = normalize_single_track(path)
        assert result is not None
        self.assertFalse(result._diff_descriptions)
        self.assertEqual(path.read_bytes(), unchanged)

    def test_explicit_advisory_round_trips_native_flac_mp3_and_mp4_tags(self) -> None:
        source = self.audio("source.wav")
        for suffix, codec in (("flac", "flac"), ("mp3", "libmp3lame"), ("m4a", "aac")):
            with self.subTest(format=suffix):
                path = self.root / f"recording.{suffix}"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-i",
                        str(source),
                        "-c:a",
                        codec,
                        str(path),
                    ],
                    check=True,
                    capture_output=True,
                )
                for value in (" explicit ", "1", "Clean", None):
                    expected = "Explicit" if value in (" explicit ", "1") else None
                    metadata.write_track_metadata(
                        TrackInfo(path, artist="Alpha", title="Song", advisory=value)
                    )
                    self.assertEqual(
                        metadata.read_track_metadata(path).advisory, expected
                    )
                    if suffix == "m4a":
                        native = MP4(path)
                        self.assertEqual(native.get("rtng"), [4] if expected else None)
                        self.assertNotIn("----:com.apple.iTunes:ITUNESADVISORY", native)
                        self.assertNotIn("----:com.apple.iTunes:ADVISORY", native)
                    else:
                        with taglib.File(str(path)) as native_tags:
                            self.assertEqual(
                                native_tags.tags.get("ITUNESADVISORY"),
                                ["1"] if expected else None,
                            )

    def test_native_mp4_rating_takes_precedence_over_conflicting_freeform_tags(
        self,
    ) -> None:
        source = self.audio("source.wav")
        path = self.root / "recording.m4a"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(source), "-c:a", "aac", str(path)],
            check=True,
            capture_output=True,
        )
        for rating, expected in ((4, "Explicit"), (2, None), (0, None)):
            with self.subTest(rating=rating):
                native = MP4(path)
                native["rtng"] = [rating]
                native["----:com.apple.iTunes:ADVISORY"] = [b"Explicit"]
                native.save()
                metadata.clear_metadata_cache()
                recording = metadata.read_track_metadata(path)
                self.assertEqual(recording.advisory, expected)
                self.assertTrue(recording._advisory_needs_normalization)
                metadata.write_track_metadata(recording)
                self.assertEqual(MP4(path).get("rtng"), [4] if expected else None)

    def test_backup_restores_native_mp4_rating_and_its_absence(self) -> None:
        source = self.audio("source.wav")
        path = self.root / "recording.m4a"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(source), "-c:a", "alac", str(path)],
            check=True,
            capture_output=True,
        )
        archive = self.root / "backup.json.gz"
        for rating in ([4], None):
            with self.subTest(rating=rating):
                native = MP4(path)
                if rating is None:
                    native.pop("rtng", None)
                else:
                    native["rtng"] = rating
                native.save()
                backup_library_tags(self.root, archive, max_threads=1)
                native = MP4(path)
                native["rtng"] = [2]
                native.save()
                self.assertEqual(restore_library_tags(archive, max_threads=1), 2)
                self.assertEqual(MP4(path).get("rtng"), rating)
