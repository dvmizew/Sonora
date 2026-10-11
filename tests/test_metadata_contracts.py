import dataclasses
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import taglib

from sonora.audio import metadata
from sonora.core.models import TrackInfo
from tests.fixtures import image_bytes
from tests.support import IsolatedTestCase


class TestMetadataContracts(IsolatedTestCase):
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
