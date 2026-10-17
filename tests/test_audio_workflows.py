from pathlib import Path
from unittest.mock import patch

from sonora.audio.cuesheet import read_cuesheet_content
from sonora.core.models import TrackInfo
from sonora.modules.normalization import normalize_track_metadata
from sonora.modules.track_processing import (
    _enrich_lyrics,
    normalize_library,
    normalize_single_track,
)
from tests.fixtures import write_silent_wav
from tests.module_support import ModuleTestCase
from tests.support import IsolatedTestCase


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
            'REM GENRE "Hip-Hop"\n'
            "REM DATE 2021\n"
            "REM DISCNUMBER 1\n"
            "REM TOTALDISCS 2\n"
            'PERFORMER "Album Artist"\n'
            'TITLE "Album Title"\n'
            "TRACK 01 AUDIO\n"
            '  TITLE "Track One"\n'
            '  PERFORMER "Track Artist"\n'
            '  SONGWRITER "Composer Name"\n'
            "  ISRC USUM71805166\n"
            "  INDEX 00 00:00:00\n"
            "  INDEX 01 00:02:00\n",
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
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.artist, "Armin")
            self.assertEqual(result.title, "Melodie")
            self.assertEqual(result.featured_artists, "Nane")
            self.assertEqual(result.genre, "Hip-Hop/Rap")
            self.assertEqual(result.release_country, "United States")
            self.assertEqual(result.language, "English")
            self.assertEqual(result.script, "Latin")
            mock_write.assert_called_once()

            # Test artist and album_artist casing reconciliation
            mock_read.return_value = TrackInfo(
                file_path=audio_file,
                artist="ALEX",
                album_artist="Alex",
                title="Song",
            )
            casing_result = normalize_single_track(audio_file, dry_run=False)
            self.assertIsNotNone(casing_result)
            assert casing_result is not None
            self.assertEqual(casing_result.artist, "Alex")
            self.assertEqual(casing_result.album_artist, "Alex")

            # Test extracting feat from artist in normalize_single_track
            mock_read.return_value = TrackInfo(
                file_path=audio_file,
                artist="21 Savage & Metro Boomin feat. Drake",
                title="Mr. Right Now",
                album="Savage Mode II",
            )
            feat_result = normalize_single_track(audio_file, dry_run=False)
            self.assertIsNotNone(feat_result)
            assert feat_result is not None
            self.assertEqual(feat_result.artist, "21 Savage & Metro Boomin")
            self.assertEqual(feat_result.title, "Mr. Right Now")
            self.assertEqual(feat_result.featured_artists, "Drake")

            # Test extracting producer bracket and cleaning title
            mock_read.return_value = TrackInfo(
                file_path=audio_file,
                artist="Deliric",
                title="Din 94 feat. Mike Diamondz [Prod: Motzu]",
                producers=None,
            )
            prod_result = normalize_single_track(audio_file, dry_run=False)
            self.assertIsNotNone(prod_result)
            assert prod_result is not None
            self.assertEqual(prod_result.title, "Din 94")
            self.assertEqual(prod_result.featured_artists, "Mike Diamondz")
            self.assertEqual(prod_result.producers, "Motzu")

            # Test invalid initial_key re-detection ('m' -> 'Am')
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
                self.assertIsNotNone(key_result)
                assert key_result is not None
                self.assertEqual(key_result.initial_key, "Am")

            # Test normalize_library
            mock_read.return_value = TrackInfo(
                file_path=audio_file,
                artist="Artist (USA)",
                title="Clean Song",
            )
            library_results = normalize_library(
                self.tmp_path, fetch_bpm=False, fetch_replaygain=False
            )
            self.assertEqual(len(library_results), 1)
            self.assertEqual(library_results[0].artist, "Artist")
            self.assertEqual(library_results.total_scanned, 1)
            self.assertEqual(library_results.total_modified, 1)
            self.assertEqual(library_results.total_unchanged, 0)
            self.assertIn(str(audio_file), library_results.modified_files)

            # Test already normalized (skip disk write)
            mock_write.reset_mock()
            mock_read.return_value = TrackInfo(
                file_path=audio_file,
                artist="Artist",
                title="Clean Song",
            )
            already_clean_results = normalize_library(
                self.tmp_path, fetch_bpm=False, fetch_replaygain=False
            )
            self.assertEqual(already_clean_results.total_scanned, 1)
            self.assertEqual(already_clean_results.total_modified, 0)
            self.assertEqual(already_clean_results.total_unchanged, 1)
            self.assertEqual(len(already_clean_results.modified_files), 0)
            mock_write.assert_not_called()


class TestLyricsEnrichment(IsolatedTestCase):
    def test_unresolved_local_lyrics_conflicts_remain_visible_in_tagging_results(
        self,
    ) -> None:
        audio_path = self.test_directory / "recording.flac"
        for lyrics in (
            b"[ar:Unrelated Artist]\n[ti:Song]\n[00:01.00] words",
            b"[ar:Alpha]\n[ti:Song]\n[02:00.00] words",
            b"\xff\xfe",
        ):
            with self.subTest(lyrics=lyrics):
                lrc_path = audio_path.with_suffix(".lrc")
                lrc_path.write_bytes(lyrics)
                track = TrackInfo(audio_path, artist="Alpha", title="Song")
                with (
                    patch("sonora.audio.metadata.get_audio_duration", return_value=60),
                    patch(
                        "sonora.services.lyrics.process_track_lyrics",
                        return_value=(None, None),
                    ),
                ):
                    _enrich_lyrics(track, audio_path, True, dry_run=True)
                self.assertIn("lyrics", track.metadata_conflicts)
                self.assertEqual(lrc_path.read_bytes(), lyrics)

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
                "sonora.services.lyrics.process_track_lyrics",
                return_value=(replacement, "synced"),
            ),
        ):
            _enrich_lyrics(track, audio_path, True, dry_run=True)
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
                "sonora.services.lyrics.process_track_lyrics", return_value=(None, None)
            ),
        ):
            _enrich_lyrics(track, audio_path, True, dry_run=True)
        self.assertNotIn("lyrics", track.metadata_conflicts)
