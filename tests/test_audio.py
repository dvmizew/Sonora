"""
Unit and integration tests for Sonora audio engine modules.
"""

import io
import sys
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile
from PIL import Image

# Guarantee src/ is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import unittest

from sonora.audio.art import _find_artist_directory, process_album_cover_art
from sonora.audio.bpm import calculate_bpm
from sonora.audio.checksum import verify_flac_checksum
from sonora.audio.metadata import (
    clear_metadata_cache,
    read_track_metadata,
    write_track_metadata,
)
from sonora.audio.replaygain import calculate_album_replaygain
from sonora.audio.spectral import detect_fake_lossless
from sonora.core.models import TrackInfo


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


class TestAudioEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        self.dummy_audio_path = create_dummy_wav_file(self.tmp_path / "dummy_audio.wav")

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_read_real_audio_metadata(self) -> None:
        track_info = read_track_metadata(self.dummy_audio_path)
        self.assertIsInstance(track_info, TrackInfo)
        self.assertEqual(track_info.sample_rate, 44100)
        self.assertEqual(track_info.channels, 2)

    @patch("taglib.File")
    def test_write_flac_audio_metadata(self, mock_taglib_cls: MagicMock) -> None:
        mock_file_instance = MagicMock()
        mock_file_instance.tags = {}
        mock_taglib_cls.return_value.__enter__.return_value = mock_file_instance

        flac_path = self.tmp_path / "test_track.flac"
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

        flac_path = self.tmp_path / "test_purge.flac"
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
        flac_path = self.tmp_path / "test_readonly.flac"
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
        bogus_path = self.tmp_path / "nonexistent_audio_track_9999.flac"
        with self.assertRaises(FileNotFoundError):
            read_track_metadata(bogus_path)

    def test_verify_nonexistent_file_raises_audio_error(self) -> None:
        bogus_path = self.tmp_path / "nonexistent_audio_track_9999.flac"
        with self.assertRaises(FileNotFoundError):
            verify_flac_checksum(bogus_path)

    def test_verify_non_flac_returns_true(self) -> None:
        self.assertTrue(verify_flac_checksum(self.dummy_audio_path))

    def test_calculate_track_replaygain_success(self) -> None:
        replaygain_success = calculate_album_replaygain(
            [self.dummy_audio_path], force=True
        )
        self.assertTrue(replaygain_success)
        reloaded_info = read_track_metadata(self.dummy_audio_path)
        self.assertIsNotNone(reloaded_info.replaygain_track_gain)

    def test_calculate_album_replaygain_success(self) -> None:
        track1_path = self.tmp_path / "1.wav"
        track2_path = self.tmp_path / "2.wav"
        create_dummy_wav_file(track1_path)
        create_dummy_wav_file(track2_path)

        result = calculate_album_replaygain([track1_path, track2_path], force=True)
        self.assertTrue(result)

        info1 = read_track_metadata(track1_path)
        self.assertIsNotNone(info1.replaygain_track_gain)
        self.assertIsNotNone(info1.replaygain_album_gain)
        self.assertIsNotNone(info1.replaygain_track_peak)
        self.assertIsNotNone(info1.replaygain_album_peak)

    def test_calculate_bpm_with_scipy(self) -> None:
        with (
            patch(
                "sonora.audio.bpm.load_audio",
                return_value=(np.random.rand(44100 * 10).astype(np.float32), 44100),
            ),
            patch(
                "scipy.signal.spectrogram",
                return_value=(None, None, np.tile(np.linspace(1, 10, 100), (10, 1))),
            ),
        ):
            bpm = calculate_bpm(self.dummy_audio_path)
            self.assertIsNotNone(bpm)
            self.assertIsInstance(bpm, float)
            if bpm:
                self.assertTrue(40.0 <= bpm <= 220.0)

    @patch("taglib.File")
    def test_read_metadata_unsupported_format(self, mock_file: MagicMock) -> None:
        mock_file.return_value = None
        dummy_path = self.tmp_path / "song.xyz"
        dummy_path.write_bytes(b"dummy")

        with self.assertRaises(ValueError):
            read_track_metadata(dummy_path)

    @patch("subprocess.run")
    def test_checksum_binary_not_found(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = FileNotFoundError()
        flac_path = self.tmp_path / "dummy_flac_check_99.flac"
        flac_path.write_bytes(b"dummy")
        with self.assertRaises(RuntimeError):
            verify_flac_checksum(flac_path)

    def test_calculate_album_replaygain_failure_empty(self) -> None:
        result = calculate_album_replaygain([])
        self.assertFalse(result)

    def test_calculate_album_replaygain_corrupted_file(self) -> None:
        fail_wav = self.tmp_path / "corrupt.wav"
        fail_wav.write_bytes(b"not an audio file")
        result = calculate_album_replaygain([fail_wav])
        self.assertFalse(result)

    def test_metadata_in_memory_cache(self) -> None:
        wav_path = self.tmp_path / "cached_track.wav"
        create_dummy_wav_file(wav_path)

        with patch("taglib.File") as mock_taglib_file:
            mock_song = MagicMock()
            mock_song.tags = {"ARTIST": ["Cached Artist"], "TITLE": ["Cached Title"]}
            mock_song.sampleRate = 44100
            mock_song.bitrate = 1411
            mock_song.channels = 2
            mock_song.pictures = []
            mock_taglib_file.return_value.__enter__.return_value = mock_song

            info1 = read_track_metadata(wav_path)
            self.assertEqual(info1.artist, "Cached Artist")
            self.assertEqual(mock_taglib_file.call_count, 1)

            # Second read from same unmodified file returns from in-memory cache without taglib.File call
            info2 = read_track_metadata(wav_path)
            self.assertEqual(info2.artist, "Cached Artist")
            self.assertEqual(mock_taglib_file.call_count, 1)

    def test_find_artist_directory_singles_hierarchy(self) -> None:
        artist_dir = self.tmp_path / "3 Doors Down"
        singles_dir = artist_dir / "Singles"
        track_folder = singles_dir / "3 Doors Down - Here Without You"
        track_folder.mkdir(parents=True)

        found = _find_artist_directory(track_folder, "3 Doors Down")
        self.assertEqual(found.resolve(), artist_dir.resolve())

    def test_find_artist_directory_fallback(self) -> None:
        folder = self.tmp_path / "Other Artist" / "Album"
        folder.mkdir(parents=True)
        found = _find_artist_directory(folder, "Unknown Artist")
        self.assertEqual(found.resolve(), folder.resolve())

    def test_metadata_safe_numeric_parsing(self) -> None:
        wav_path = self.tmp_path / "safe_numeric.wav"
        create_dummy_wav_file(wav_path)

        with patch("taglib.File") as mock_taglib_file:
            mock_song = MagicMock()
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

            info = read_track_metadata(wav_path)
            self.assertEqual(info.total_tracks, 12)
            self.assertEqual(info.total_discs, 2)
            self.assertIsNone(info.rating)

    def test_process_album_cover_art_disc_folder(self) -> None:
        album_dir = self.tmp_path / "The Wall"
        disc_dir = album_dir / "CD1"
        disc_dir.mkdir(parents=True)
        cover_path = album_dir / "cover.jpg"
        cover_path.write_bytes(b"dummy image data")

        # Looking up art for CD1 should discover existing cover.jpg in the parent album folder
        found = process_album_cover_art(disc_dir, "Pink Floyd", "The Wall")
        self.assertIsNotNone(found)
        if found:
            self.assertEqual(found.resolve(), cover_path.resolve())

    def test_process_album_cover_art_low_resolution_upgrade(self) -> None:
        album_dir = self.tmp_path / "LowResAlbum"
        album_dir.mkdir(parents=True)
        cover_path = album_dir / "cover.jpg"

        low_res = Image.new("RGB", (300, 300), color="blue")
        low_res_buf = io.BytesIO()
        low_res.save(low_res_buf, format="JPEG")
        cover_path.write_bytes(low_res_buf.getvalue())

        hi_res = Image.new("RGB", (1400, 1400), color="blue")
        hi_res_buf = io.BytesIO()
        hi_res.save(hi_res_buf, format="JPEG")
        hi_res_bytes = hi_res_buf.getvalue()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = hi_res_bytes
        mock_resp.raise_for_status = MagicMock()

        with (
            patch(
                "sonora.audio.art.fetch_itunes_cover_art_url",
                return_value="https://itunes.com/art.jpg",
            ),
            patch("sonora.core.http.SESSION.get", return_value=mock_resp),
        ):
            found = process_album_cover_art(album_dir, "Artist", "Album", force=False)
            self.assertIsNotNone(found)
            assert found is not None
            with Image.open(found) as img:
                self.assertEqual(img.size, (1400, 1400))

    def test_process_album_cover_art_caa_low_res_falls_back_to_itunes_hi_res(
        self,
    ) -> None:
        album_dir = self.tmp_path / "CaaLowResAlbum"
        album_dir.mkdir(parents=True)

        caa_low = Image.new("RGB", (400, 400), color="red")
        caa_buf = io.BytesIO()
        caa_low.save(caa_buf, format="JPEG")
        caa_bytes = caa_buf.getvalue()

        itunes_hi = Image.new("RGB", (1400, 1400), color="blue")
        itunes_buf = io.BytesIO()
        itunes_hi.save(itunes_buf, format="JPEG")
        itunes_bytes = itunes_buf.getvalue()

        def mock_get(url: str, **kwargs: object) -> MagicMock:
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            if "coverartarchive" in url:
                resp.content = caa_bytes
            else:
                resp.content = itunes_bytes
            return resp

        with (
            patch(
                "sonora.audio.art.fetch_cover_art_archive_url",
                return_value="https://coverartarchive.org/release/123/front",
            ),
            patch(
                "sonora.audio.art.fetch_itunes_cover_art_url",
                return_value="https://itunes.com/art_hi.jpg",
            ),
            patch("sonora.core.http.SESSION.get", side_effect=mock_get),
        ):
            found = process_album_cover_art(
                album_dir,
                "Artist",
                "Album",
                musicbrainz_album_id="123",
                force=True,
            )
            self.assertIsNotNone(found)
            assert found is not None
            with Image.open(found) as img:
                self.assertEqual(img.size, (1400, 1400))

    def test_process_album_cover_art_caa_low_res_preserved_when_no_higher_source(
        self,
    ) -> None:
        album_dir = self.tmp_path / "CaaOnlyAlbum"
        album_dir.mkdir(parents=True)

        caa_low = Image.new("RGB", (400, 400), color="red")
        caa_buf = io.BytesIO()
        caa_low.save(caa_buf, format="JPEG")
        caa_bytes = caa_buf.getvalue()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = caa_bytes
        mock_resp.raise_for_status = MagicMock()

        with (
            patch(
                "sonora.audio.art.fetch_cover_art_archive_url",
                return_value="https://coverartarchive.org/release/456/front",
            ),
            patch("sonora.audio.art.fetch_itunes_cover_art_url", return_value=None),
            patch("sonora.audio.art.fetch_deezer_cover_art_url", return_value=None),
            patch("sonora.core.http.SESSION.get", return_value=mock_resp),
        ):
            found = process_album_cover_art(
                album_dir,
                "Artist",
                "Album",
                musicbrainz_album_id="456",
                force=True,
            )
            self.assertIsNotNone(found)
            assert found is not None
            with Image.open(found) as img:
                self.assertEqual(img.size, (400, 400))

    def test_process_album_cover_art_quality_downgrade_protection(self) -> None:
        album_dir = self.tmp_path / "HighResAlbum"
        album_dir.mkdir(parents=True)
        cover_path = album_dir / "cover.jpg"

        hi_res = Image.new("RGB", (1400, 1400), color="green")
        hi_res_buf = io.BytesIO()
        hi_res.save(hi_res_buf, format="JPEG")
        cover_path.write_bytes(hi_res_buf.getvalue())

        low_res = Image.new("RGB", (500, 500), color="green")
        low_res_buf = io.BytesIO()
        low_res.save(low_res_buf, format="JPEG")
        low_res_bytes = low_res_buf.getvalue()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = low_res_bytes
        mock_resp.raise_for_status = MagicMock()

        with (
            patch(
                "sonora.audio.art.fetch_itunes_cover_art_url",
                return_value="https://itunes.com/art_small.jpg",
            ),
            patch("sonora.core.http.SESSION.get", return_value=mock_resp),
        ):
            found = process_album_cover_art(album_dir, "Artist", "Album", force=True)
            self.assertIsNotNone(found)
            assert found is not None
            with Image.open(found) as img:
                self.assertEqual(img.size, (1400, 1400))

    def test_format_lossless_and_lossy_classification(self) -> None:
        lossless_exts = [".flac", ".wav", ".aiff", ".alac", ".ape", ".wv"]
        lossy_exts = [".mp3", ".ogg", ".opus", ".mpc", ".wma"]

        for ext in lossless_exts:
            p = self.tmp_path / f"track{ext}"
            p.write_bytes(b"dummy")
            with patch("taglib.File") as mock_taglib:
                mock_song = MagicMock()
                mock_song.tags = {"ARTIST": ["A"], "TITLE": ["T"]}
                mock_song.sampleRate = 44100
                mock_song.bitrate = 1411
                mock_song.channels = 2
                mock_song.pictures = []
                mock_taglib.return_value.__enter__.return_value = mock_song
                info = read_track_metadata(p)
                self.assertTrue(
                    info.is_lossless, f"Expected {ext} to be classified as lossless"
                )

        for ext in lossy_exts:
            p = self.tmp_path / f"track{ext}"
            p.write_bytes(b"dummy")
            with patch("taglib.File") as mock_taglib:
                mock_song = MagicMock()
                mock_song.tags = {"ARTIST": ["A"], "TITLE": ["T"]}
                mock_song.sampleRate = 44100
                mock_song.bitrate = 320
                mock_song.channels = 2
                mock_song.pictures = []
                mock_taglib.return_value.__enter__.return_value = mock_song
                info = read_track_metadata(p)
                self.assertFalse(
                    info.is_lossless, f"Expected {ext} to be classified as lossy"
                )

    def test_detect_fake_lossless_authentic_audio(self) -> None:
        authentic_path = self.tmp_path / "authentic.wav"
        rng = np.random.default_rng(42)
        samples = rng.normal(0.0, 0.15, int(44100 * 25.0)).astype(np.float32)
        soundfile.write(str(authentic_path), samples, 44100)

        is_fake, cutoff_khz, description = detect_fake_lossless(authentic_path)
        self.assertFalse(is_fake)
        self.assertEqual(cutoff_khz, 0.0)
        self.assertIsNone(description)

    def test_detect_fake_lossless_brickwall_transcode(self) -> None:
        fake_path = self.tmp_path / "transcode_128k.wav"
        rng = np.random.default_rng(42)
        time_axis = np.linspace(0, 25.0, int(44100 * 25.0), endpoint=False)
        audio_signal = (
            np.sin(2 * np.pi * 100 * time_axis) * 0.5
            + np.sin(2 * np.pi * 1000 * time_axis) * 0.2
            + rng.normal(0.0, 0.05, len(time_axis))
        )
        fft_coefficients = np.fft.rfft(audio_signal)
        fft_frequencies = np.fft.rfftfreq(len(audio_signal), d=1 / 44100)
        fft_coefficients[fft_frequencies > 16000] = 1e-6
        brickwall_audio = np.fft.irfft(fft_coefficients, n=len(audio_signal)).astype(
            np.float32
        )
        soundfile.write(str(fake_path), brickwall_audio, 44100)

        is_fake, cutoff_khz, description = detect_fake_lossless(fake_path)
        self.assertTrue(is_fake)
        self.assertGreaterEqual(cutoff_khz, 14.5)
        self.assertLessEqual(cutoff_khz, 16.5)
        assert description is not None
        self.assertIn("128kbps", description)

    def test_detect_fake_lossless_nonexistent_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            detect_fake_lossless(self.tmp_path / "nonexistent.flac")

    def test_detect_fake_lossless_short_audio(self) -> None:
        short_path = self.tmp_path / "short.wav"
        rng = np.random.default_rng(42)
        samples = rng.normal(0.0, 0.15, int(44100 * 1.0)).astype(np.float32)
        soundfile.write(str(short_path), samples, 44100)

        is_fake, cutoff_khz, description = detect_fake_lossless(short_path)
        self.assertFalse(is_fake)
        self.assertEqual(cutoff_khz, 0.0)
        self.assertIsNone(description)

    @patch("taglib.File")
    def test_advisory_metadata_explicit_and_clean_purging(
        self, mock_taglib_cls: MagicMock
    ) -> None:
        mock_file_instance = MagicMock()
        mock_file_instance.tags = {
            "ITUNESADVISORY": ["2"],
            "ADVISORY": ["Clean"],
        }
        mock_taglib_cls.return_value.__enter__.return_value = mock_file_instance

        flac_path = self.tmp_path / "test_advisory.flac"
        flac_path.write_bytes(b"dummy flac data")

        # 1. Non-explicit track must purge ITUNESADVISORY and ADVISORY (no Clean spam)
        clean_track = TrackInfo(
            file_path=flac_path,
            artist="Artist",
            title="Clean Song",
            advisory=None,
        )
        write_track_metadata(clean_track)
        self.assertNotIn("ITUNESADVISORY", mock_file_instance.tags)
        self.assertNotIn("ADVISORY", mock_file_instance.tags)

        # 2. Explicit track must write ITUNESADVISORY="1" and ADVISORY="Explicit"
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
        mock_file_instance.tags = {}
        mock_taglib_cls.return_value.__enter__.return_value = mock_file_instance

        flac_path = self.tmp_path / "test_featured.flac"
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
        mock_file_instance.tags = {}
        mock_taglib_cls.return_value.__enter__.return_value = mock_file_instance

        flac_path = self.tmp_path / "test_collab.flac"
        flac_path.write_bytes(b"dummy flac data")

        # 1. Collaborative primary artist and album artist (e.g. 21 Savage & Metro Boomin)
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

        # 2. Collaborative artist with additional featured artist
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

        # 3. Various Artists compilation auto-flag
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
        # Tags has collaborative ARTIST and multi-valued ARTISTS
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

        flac_path = self.tmp_path / "test_read_collab.flac"
        flac_path.write_bytes(b"dummy flac data")

        track_info = read_track_metadata(flac_path)
        # Must NOT classify primary artists 21 Savage or Metro Boomin as featured artists
        self.assertIsNone(track_info.featured_artists)

        # Case with genuine extra artist in ARTISTS tag
        clear_metadata_cache()
        mock_file_instance.tags["ARTISTS"] = ["21 Savage", "Metro Boomin", "Future"]
        track_info_with_feat = read_track_metadata(flac_path)
        self.assertEqual(track_info_with_feat.featured_artists, "Future")


if __name__ == "__main__":
    unittest.main()
