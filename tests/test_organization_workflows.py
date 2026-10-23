from pathlib import Path
from typing import Any
from unittest.mock import patch

from sonora.core.library_paths import get_single_release_title
from sonora.core.models import TrackInfo
from sonora.modules.checker import (
    check_library,
)
from sonora.modules.organizer import (
    SingleDeduplicator,
    organize_library_singles,
)
from tests.fixtures import write_silent_wav
from tests.module_support import ModuleTestCase


class TestOrganizationWorkflows(ModuleTestCase):
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

        # Dry-run first: files should NOT be moved or unlinked
        moved_dry = organize_library_singles(
            src_dir, target_dir, dry_run=True, max_threads=1
        )
        self.assertEqual(moved_dry, 1)
        self.assertTrue(audio1.exists())
        self.assertTrue(audio2.exists())
        self.assertTrue(audio3.exists())
        self.assertTrue(lrc2.exists())

        # Real run: first single moved, remaining duplicates quarantined to .duplicates/
        moved = organize_library_singles(
            src_dir, target_dir, dry_run=False, max_threads=1
        )
        self.assertEqual(moved, 1)

        # First track organized
        self.assertTrue(
            (target_dir / "Dup Artist - Dup Song" / "01 - Dup Song.wav").exists()
        )

        # Quarantined duplicates exist in .duplicates/ (NOT unlinked/destroyed)
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
        # Album folder should remain intact
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
        # Album has version not present in title -> appended
        t1 = TrackInfo(
            file_path=Path("/tmp/t1.flac"),
            artist="DJ Sava",
            title="I Loved You",
            album="I Loved You (Monoir Remix) (feat. Irina Rimes)",
        )
        self.assertEqual(get_single_release_title(t1), "I Loved You (Monoir Remix)")

        # Title already has version -> not duplicated
        t2 = TrackInfo(
            file_path=Path("/tmp/t2.flac"),
            artist="DJ Sava",
            title="I Loved You (Monoir Remix)",
            album="I Loved You (Monoir Remix)",
        )
        self.assertEqual(get_single_release_title(t2), "I Loved You (Monoir Remix)")

        # Generic album title -> title unmodified
        t3 = TrackInfo(
            file_path=Path("/tmp/t3.flac"),
            artist="Dennis Lloyd",
            title="Alien",
            album="Alien",
        )
        self.assertEqual(get_single_release_title(t3), "Alien")

    @patch("sonora.modules.checker.read_track_metadata")
    def test_check_library_singles_folder_sequence_shielding(
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

        # Different collaboration / feature on same track title -> NOT a duplicate
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
