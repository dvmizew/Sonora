"""
Unit and integration tests for Sonora audio engine modules:
audio decoding, metadata I/O, checksum verification, and cuesheet processing.
"""

import subprocess
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile

from sonora.audio.checksum import verify_flac_checksum
from sonora.audio.cuesheet import find_companion_cuesheet, read_cuesheet_content
from sonora.audio.decode import load_audio
from sonora.audio.metadata import (
    clear_metadata_cache,
    read_track_metadata,
    write_track_metadata,
)
from sonora.core.models import TrackInfo
from tests.support import IsolatedTestCase


def create_dummy_wav_file(dest_path: Path) -> Path:
    """Create a temporary 1-second WAV audio file with standard audio properties."""
    sample_rate = 44100
    time_axis = np.linspace(0, 1, sample_rate, endpoint=False)
    samples = (np.sin(2 * np.pi * 440 * time_axis) * 16000).astype(np.int16)
    stereo = np.column_stack([samples, samples]).flatten()
    with wave.open(str(dest_path), "wb") as wave_file:
        wave_file.setnchannels(2)
        wave_file.setsampwidth(2)
        wave_file.setframerate(sample_rate)
        wave_file.writeframes(stereo.tobytes())
    return dest_path


class TestAudioDecoding(IsolatedTestCase):
    def test_seek_near_or_beyond_end_never_restarts_the_track(self) -> None:
        path = self.test_directory / "short.wav"
        signal = np.arange(2000, dtype=np.float32) / 2000
        soundfile.write(path, signal, 1000, subtype="FLOAT")
        for offset, expected in (
            (1.5, signal[1500:]),
            (2, signal[:0]),
            (3, signal[:0]),
        ):
            with self.subTest(offset=offset):
                loaded = load_audio(path, mono=True, offset_seconds=offset)
                assert loaded is not None
                samples, _ = loaded
                np.testing.assert_array_equal(samples, expected)

    def test_real_seek_and_duration_limit(self) -> None:
        audio_path = self.test_directory / "album.wav"
        signal = np.zeros(5000, dtype=np.float32)
        signal[1000:2000] = 0.5
        soundfile.write(audio_path, signal, 1000, subtype="FLOAT")
        loaded = load_audio(audio_path, mono=True, offset_seconds=1, max_seconds=1)
        assert loaded is not None
        audio, rate = loaded
        self.assertEqual(rate, 1000)
        self.assertEqual(audio.shape, (1000,))
        np.testing.assert_array_equal(audio, np.full(1000, 0.5))

    def test_ffmpeg_fallback_respects_channels_seek_and_duration(self) -> None:
        audio_path = self.test_directory / "track.m4a"
        raw_audio = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        for mono, expected_shape in ((True, (4,)), (False, (2, 2))):
            completed = subprocess.CompletedProcess([], 0, stdout=raw_audio.tobytes())
            with (
                self.subTest(mono=mono),
                patch("soundfile.SoundFile", side_effect=OSError("unsupported format")),
                patch("subprocess.run", return_value=completed) as run,
            ):
                loaded = load_audio(
                    audio_path, mono=mono, offset_seconds=2, max_seconds=5
                )
            assert loaded is not None
            audio, rate = loaded
            self.assertEqual(rate, 44100)
            self.assertEqual(audio.shape, expected_shape)
            np.testing.assert_array_equal(audio.ravel(), raw_audio)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("-ss") + 1], "2")
            self.assertEqual(command[command.index("-t") + 1], "5")
            self.assertEqual(command[command.index("-ac") + 1], "1" if mono else "2")

    def test_ffmpeg_failure_and_empty_output_return_no_audio(self) -> None:
        for outcome in (
            subprocess.CompletedProcess([], 0, stdout=b""),
            subprocess.TimeoutExpired("ffmpeg", 15),
        ):
            with (
                self.subTest(outcome=outcome),
                patch("soundfile.SoundFile", side_effect=OSError("decode failed")),
                patch(
                    "subprocess.run",
                    side_effect=outcome if isinstance(outcome, Exception) else None,
                    return_value=outcome,
                ),
            ):
                self.assertIsNone(load_audio(self.test_directory / "damaged.wav"))

    def test_format_lossless_and_lossy_classification(self) -> None:
        lossless_exts = [".flac", ".wav", ".aiff", ".alac", ".ape", ".wv"]
        lossy_exts = [".mp3", ".ogg", ".opus", ".mpc", ".wma"]

        for ext in lossless_exts:
            p = self.test_directory / f"track{ext}"
            p.write_bytes(b"dummy")
            with patch("taglib.File") as mock_taglib:
                mock_song = MagicMock()
                mock_song.save.return_value = {}
                mock_song.tags = {"ARTIST": ["A"], "TITLE": ["T"]}
                mock_song.sampleRate = 44100
                mock_song.bitrate = 1411
                mock_song.channels = 2
                mock_song.pictures = []
                mock_taglib.return_value.__enter__.return_value = mock_song
                track_metadata = read_track_metadata(p)
                self.assertTrue(
                    track_metadata.is_lossless,
                    f"Expected {ext} to be classified as lossless",
                )

        for ext in lossy_exts:
            p = self.test_directory / f"track{ext}"
            p.write_bytes(b"dummy")
            with patch("taglib.File") as mock_taglib:
                mock_song = MagicMock()
                mock_song.save.return_value = {}
                mock_song.tags = {"ARTIST": ["A"], "TITLE": ["T"]}
                mock_song.sampleRate = 44100
                mock_song.bitrate = 320
                mock_song.channels = 2
                mock_song.pictures = []
                mock_taglib.return_value.__enter__.return_value = mock_song
                track_metadata = read_track_metadata(p)
                self.assertFalse(
                    track_metadata.is_lossless,
                    f"Expected {ext} to be classified as lossy",
                )


class TestTrackMetadataIO(IsolatedTestCase):
    def setUp(self) -> None:
        self.dummy_audio_path = create_dummy_wav_file(
            self.test_directory / "dummy_audio.wav"
        )

    def test_read_real_audio_metadata(self) -> None:
        track_info = read_track_metadata(self.dummy_audio_path)
        self.assertIsInstance(track_info, TrackInfo)
        self.assertEqual(track_info.sample_rate, 44100)
        self.assertEqual(track_info.channels, 2)

    @patch("taglib.File")
    def test_write_flac_audio_metadata(self, mock_taglib_cls: MagicMock) -> None:
        mock_file_instance = MagicMock()
        mock_file_instance.save.return_value = {}
        mock_file_instance.tags = {}
        mock_taglib_cls.return_value.__enter__.return_value = mock_file_instance

        flac_path = self.test_directory / "test_track.flac"
        flac_path.write_bytes(b"dummy flac data")

        track_info = TrackInfo(
            file_path=flac_path,
            artist="Test Artist",
            title="Test Track",
            album="Test Album",
            replaygain_track_gain=-4.25,
            replaygain_track_peak=0.951234,
        )
        write_track_metadata(track_info)

        mock_file_instance.save.assert_called_once()
        self.assertEqual(mock_file_instance.tags["ARTIST"], ["Test Artist"])
        self.assertEqual(mock_file_instance.tags["TITLE"], ["Test Track"])
        self.assertEqual(mock_file_instance.tags["REPLAYGAIN_TRACK_GAIN"], ["-4.25 dB"])
        self.assertEqual(mock_file_instance.tags["REPLAYGAIN_TRACK_PEAK"], ["0.951234"])

    @patch("taglib.File")
    def test_write_track_metadata_purges_none_and_cleared_tags(
        self, mock_taglib_cls: MagicMock
    ) -> None:
        mock_file_instance = MagicMock()
        mock_file_instance.save.return_value = {}
        mock_file_instance.tags = {
            "ARTIST": ["Old Artist"],
            "TITLE": ["Old Title"],
            "ALBUM": ["Old Album"],
            "ALBUMARTIST": ["Old Album Artist"],
            "GENRE": ["Old Genre"],
            "MUSICBRAINZ_TRACKID": ["5a9fc94b-ec0c-4619-acc8-388a022630d0"],
            "MUSICBRAINZ TRACK ID": ["5a9fc94b-ec0c-4619-acc8-388a022630d0"],
            "ARTISTSORT": ["Old Sort"],
            "BPM": ["120.0"],
        }
        mock_taglib_cls.return_value.__enter__.return_value = mock_file_instance

        flac_path = self.test_directory / "test_purge.flac"
        flac_path.write_bytes(b"dummy flac data")

        track_info = TrackInfo(
            file_path=flac_path,
            artist="New Artist",
            title="New Title",
            album="New Album",
            album_artist=None,
            genre=None,
            musicbrainz_trackid=None,
            artist_sort=None,
            bpm=None,
        )
        write_track_metadata(track_info)

        mock_file_instance.save.assert_called_once()
        self.assertEqual(mock_file_instance.tags["ARTIST"], ["New Artist"])
        self.assertEqual(mock_file_instance.tags["TITLE"], ["New Title"])
        self.assertEqual(mock_file_instance.tags["ALBUM"], ["New Album"])
        self.assertNotIn("ALBUMARTIST", mock_file_instance.tags)
        self.assertNotIn("GENRE", mock_file_instance.tags)
        self.assertNotIn("MUSICBRAINZ_TRACKID", mock_file_instance.tags)
        self.assertNotIn("MUSICBRAINZ TRACK ID", mock_file_instance.tags)
        self.assertNotIn("ARTISTSORT", mock_file_instance.tags)
        self.assertNotIn("BPM", mock_file_instance.tags)

    @patch("os.access")
    def test_write_track_metadata_permission_denied_raises_oserror(
        self, mock_access: MagicMock
    ) -> None:
        mock_access.return_value = False
        flac_path = self.test_directory / "test_readonly.flac"
        flac_path.write_bytes(b"dummy flac data")
        track_info = TrackInfo(
            file_path=flac_path,
            artist="Readonly Artist",
            title="Readonly Title",
            album="Readonly Album",
        )
        with self.assertRaises(PermissionError):
            write_track_metadata(track_info)

    def test_read_nonexistent_file_raises_metadata_error(self) -> None:
        bogus_path = self.test_directory / "nonexistent_audio_track_9999.flac"
        with self.assertRaises(FileNotFoundError):
            read_track_metadata(bogus_path)

    @patch("taglib.File")
    def test_read_metadata_unsupported_format(self, mock_file: MagicMock) -> None:
        mock_file.return_value = None
        dummy_path = self.test_directory / "song.xyz"
        dummy_path.write_bytes(b"dummy")

        with self.assertRaises(ValueError):
            read_track_metadata(dummy_path)

    def test_metadata_in_memory_cache(self) -> None:
        wav_path = self.test_directory / "cached_track.wav"
        create_dummy_wav_file(wav_path)

        with patch("taglib.File") as mock_taglib_file:
            mock_song = MagicMock()
            mock_song.save.return_value = {}
            mock_song.tags = {"ARTIST": ["Cached Artist"], "TITLE": ["Cached Title"]}
            mock_song.sampleRate = 44100
            mock_song.bitrate = 1411
            mock_song.channels = 2
            mock_song.pictures = []
            mock_taglib_file.return_value.__enter__.return_value = mock_song

            info1 = read_track_metadata(wav_path)
            self.assertEqual(info1.artist, "Cached Artist")
            self.assertEqual(mock_taglib_file.call_count, 1)

            info2 = read_track_metadata(wav_path)
            self.assertEqual(info2.artist, "Cached Artist")
            self.assertEqual(mock_taglib_file.call_count, 1)

    def test_metadata_safe_numeric_parsing(self) -> None:
        wav_path = self.test_directory / "safe_numeric.wav"
        create_dummy_wav_file(wav_path)

        with patch("taglib.File") as mock_taglib_file:
            mock_song = MagicMock()
            mock_song.save.return_value = {}
            mock_song.tags = {
                "ARTIST": ["Artist"],
                "TITLE": ["Title"],
                "TRACKTOTAL": ["12/12"],
                "DISCTOTAL": ["2/2"],
                "RATING": ["not_a_float"],
            }
            mock_song.sampleRate = 44100
            mock_song.bitrate = 1411
            mock_song.channels = 2
            mock_song.pictures = []
            mock_taglib_file.return_value.__enter__.return_value = mock_song

            track_metadata = read_track_metadata(wav_path)
            self.assertEqual(track_metadata.total_tracks, 12)
            self.assertEqual(track_metadata.total_discs, 2)
            self.assertIsNone(track_metadata.rating)

    @patch("taglib.File")
    def test_advisory_metadata_explicit_and_clean_purging(
        self, mock_taglib_cls: MagicMock
    ) -> None:
        mock_file_instance = MagicMock()
        mock_file_instance.save.return_value = {}
        mock_file_instance.tags = {
            "ITUNESADVISORY": ["2"],
            "ADVISORY": ["Clean"],
        }
        mock_taglib_cls.return_value.__enter__.return_value = mock_file_instance

        flac_path = self.test_directory / "test_advisory.flac"
        flac_path.write_bytes(b"dummy flac data")

        clean_track = TrackInfo(
            file_path=flac_path,
            artist="Artist",
            title="Clean Song",
            advisory=None,
        )
        write_track_metadata(clean_track)
        self.assertNotIn("ITUNESADVISORY", mock_file_instance.tags)
        self.assertNotIn("ADVISORY", mock_file_instance.tags)

        explicit_track = TrackInfo(
            file_path=flac_path,
            artist="Artist",
            title="Explicit Song",
            advisory="Explicit",
        )
        write_track_metadata(explicit_track)
        self.assertEqual(mock_file_instance.tags["ITUNESADVISORY"], ["1"])
        self.assertEqual(mock_file_instance.tags["ADVISORY"], ["Explicit"])

    @patch("taglib.File")
    def test_featured_artists_and_multi_artists_tag(
        self, mock_taglib_cls: MagicMock
    ) -> None:
        mock_file_instance = MagicMock()
        mock_file_instance.save.return_value = {}
        mock_file_instance.tags = {}
        mock_taglib_cls.return_value.__enter__.return_value = mock_file_instance

        flac_path = self.test_directory / "test_featured.flac"
        flac_path.write_bytes(b"dummy flac data")

        track_with_feat = TrackInfo(
            file_path=flac_path,
            artist="Armin",
            title="Melodie",
            featured_artists="Nane, Super ED",
        )
        write_track_metadata(track_with_feat)
        self.assertEqual(
            mock_file_instance.tags["ARTISTS"], ["Armin", "Nane", "Super ED"]
        )

        track_without_feat = TrackInfo(
            file_path=flac_path,
            artist="Armin",
            title="Melodie",
            featured_artists=None,
        )
        write_track_metadata(track_without_feat)
        self.assertNotIn("ARTISTS", mock_file_instance.tags)

    @patch("taglib.File")
    def test_collaborative_release_multi_artists_and_album_artists(
        self, mock_taglib_cls: MagicMock
    ) -> None:
        mock_file_instance = MagicMock()
        mock_file_instance.save.return_value = {}
        mock_file_instance.tags = {}
        mock_taglib_cls.return_value.__enter__.return_value = mock_file_instance

        flac_path = self.test_directory / "test_collab.flac"
        flac_path.write_bytes(b"dummy flac data")

        collab_track = TrackInfo(
            file_path=flac_path,
            artist="21 Savage & Metro Boomin",
            album_artist="21 Savage & Metro Boomin",
            title="No Heart",
            album="Savage Mode",
            featured_artists=None,
            lyrics="Sample lyrics text",
        )
        write_track_metadata(collab_track)
        self.assertEqual(
            mock_file_instance.tags["ARTIST"], ["21 Savage & Metro Boomin"]
        )
        self.assertEqual(
            mock_file_instance.tags["ARTISTS"], ["21 Savage", "Metro Boomin"]
        )
        self.assertEqual(
            mock_file_instance.tags["ALBUMARTIST"], ["21 Savage & Metro Boomin"]
        )
        self.assertEqual(
            mock_file_instance.tags["ALBUMARTISTS"], ["21 Savage", "Metro Boomin"]
        )
        self.assertEqual(mock_file_instance.tags["LYRICS"], ["Sample lyrics text"])
        self.assertEqual(
            mock_file_instance.tags["UNSYNCEDLYRICS"], ["Sample lyrics text"]
        )

        collab_with_feat = TrackInfo(
            file_path=flac_path,
            artist="21 Savage & Metro Boomin",
            album_artist="21 Savage & Metro Boomin",
            title="X",
            album="Savage Mode",
            featured_artists="Future",
        )
        write_track_metadata(collab_with_feat)
        self.assertEqual(
            mock_file_instance.tags["ARTISTS"],
            ["21 Savage", "Metro Boomin", "Future"],
        )

        compilation_track = TrackInfo(
            file_path=flac_path,
            artist="Individual Artist",
            album_artist="Various Artists",
            title="Hits Track",
            album="Top Hits 2026",
        )
        write_track_metadata(compilation_track)
        self.assertEqual(mock_file_instance.tags["COMPILATION"], ["1"])

    @patch("taglib.File")
    def test_read_track_metadata_filters_primary_artists_from_artists_tag(
        self, mock_taglib_cls: MagicMock
    ) -> None:
        mock_file_instance = MagicMock()
        mock_file_instance.save.return_value = {}
        mock_file_instance.tags = {
            "ARTIST": ["21 Savage & Metro Boomin"],
            "TITLE": ["No Heart"],
            "ALBUM": ["Savage Mode"],
            "ARTISTS": ["21 Savage", "Metro Boomin"],
        }
        mock_file_instance.sampleRate = 44100
        mock_file_instance.channels = 2
        mock_file_instance.bitrate = 1411
        mock_file_instance.length = 180
        mock_file_instance.pictures = []
        mock_taglib_cls.return_value.__enter__.return_value = mock_file_instance

        flac_path = self.test_directory / "test_read_collab.flac"
        flac_path.write_bytes(b"dummy flac data")

        track_info = read_track_metadata(flac_path)
        self.assertIsNone(track_info.featured_artists)

        clear_metadata_cache()
        mock_file_instance.tags["ARTISTS"] = ["21 Savage", "Metro Boomin", "Future"]
        track_info_with_feat = read_track_metadata(flac_path)
        self.assertEqual(track_info_with_feat.featured_artists, "Future")


class TestCueSheet(IsolatedTestCase):
    def test_missing_unreadable_and_absent_cuesheets(self) -> None:
        cue_path = self.test_directory / "album.cue"
        self.assertIsNone(read_cuesheet_content(cue_path))
        self.assertIsNone(find_companion_cuesheet(cue_path))
        self.assertIsNone(find_companion_cuesheet(self.test_directory))
        cue_path.write_text('TITLE "Album"')
        self.assertEqual(find_companion_cuesheet(self.test_directory), cue_path)
        with patch.object(
            Path, "read_bytes", side_effect=PermissionError("locked cue")
        ):
            self.assertIsNone(read_cuesheet_content(cue_path))

    def test_cuesheet_encodings_preserve_metadata(self) -> None:
        for encoded, expected in (
            (b'\xef\xbb\xbfTITLE "Caf\xc3\xa9"', 'TITLE "Café"'),
            (b'TITLE "Prix \x80"', 'TITLE "Prix €"'),
            (b'TITLE "\x81"', 'TITLE "\x81"'),
        ):
            cue_path = self.test_directory / "album.cue"
            cue_path.write_bytes(encoded)
            with self.subTest(encoded=encoded):
                self.assertEqual(read_cuesheet_content(cue_path), expected)


class TestChecksum(IsolatedTestCase):
    def test_verify_nonexistent_file_raises_audio_error(self) -> None:
        bogus_path = self.test_directory / "nonexistent_audio_track_9999.flac"
        with self.assertRaises(FileNotFoundError):
            verify_flac_checksum(bogus_path)

    def test_verify_non_flac_returns_true(self) -> None:
        dummy_wav = create_dummy_wav_file(self.test_directory / "dummy.wav")
        self.assertTrue(verify_flac_checksum(dummy_wav))

    @patch("subprocess.run")
    def test_checksum_binary_not_found(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = FileNotFoundError()
        flac_path = self.test_directory / "dummy_flac_check_99.flac"
        flac_path.write_bytes(b"dummy")
        with self.assertRaises(OSError):
            verify_flac_checksum(flac_path)
