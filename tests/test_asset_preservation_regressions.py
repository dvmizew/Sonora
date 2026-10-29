import dataclasses
import io
from unittest.mock import patch

import taglib
from PIL import Image

from sonora.audio import art as art_module
from sonora.audio import metadata as metadata_module
from sonora.audio.key import key_to_camelot
from sonora.audio.metadata import (
    clear_metadata_cache,
    read_track_metadata,
    write_track_metadata,
)
from sonora.audio.replaygain import LoudnessMeasurement, _album_loudness
from sonora.core import sidecars
from sonora.core.models import TrackInfo
from sonora.core.sidecars import find_companion_lyrics, relocate_companion_lyrics
from sonora.core.text import normalize_date
from sonora.modules import tagger
from sonora.modules import track_processing as track_processing_module
from sonora.services import lyrics
from tests.regression_support import AudioRegressionTestCase


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
            details: dict = {}
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
                audio_path, "Alpha", "Song", duration=10, conflicts=conflicts
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
            LoudnessMeasurement(self.root / "loud.flac", 0, 1, -17, 4, (-17,) * 40),
            LoudnessMeasurement(self.root / "quiet.flac", 0, 0.1, -57, 4, (-57,) * 40),
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
