"""
Unit and integration tests for Musical Key and Camelot wheel tonality detection.
"""

import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

# Guarantee src/ is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import unittest

from sonora.audio.key import (
    calculate_chroma,
    detect_key_details,
    detect_key_from_chroma,
    detect_musical_key,
    get_chroma_filterbank,
    key_to_camelot,
)
from sonora.audio.metadata import read_track_metadata
from sonora.modules.tagger import normalize_library, normalize_single_track


def create_chord_wav(
    dest_path: Path, freqs: list[float], duration: float = 2.0, sample_rate: int = 22050
) -> Path:
    """Create a temporary WAV audio file containing specified chord frequencies."""
    time_axis = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.zeros_like(time_axis)
    for f in freqs:
        signal += np.sin(2 * np.pi * f * time_axis)

    # Normalize amplitude to prevent clipping
    signal = signal / (len(freqs) * 1.2)
    samples = (signal * 32767).astype(np.int16)
    stereo = np.column_stack([samples, samples]).flatten()

    with wave.open(str(dest_path), "wb") as wave_file:
        wave_file.setnchannels(2)
        wave_file.setsampwidth(2)
        wave_file.setframerate(sample_rate)
        wave_file.writeframes(stereo.tobytes())
    return dest_path


class TestMusicalKey(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_key_to_camelot_major_keys(self) -> None:
        self.assertEqual(key_to_camelot("C"), "8B")
        self.assertEqual(key_to_camelot("C#"), "3B")
        self.assertEqual(key_to_camelot("Db"), "3B")
        self.assertEqual(key_to_camelot("D"), "10B")
        self.assertEqual(key_to_camelot("D#"), "5B")
        self.assertEqual(key_to_camelot("Eb"), "5B")
        self.assertEqual(key_to_camelot("E"), "12B")
        self.assertEqual(key_to_camelot("F"), "7B")
        self.assertEqual(key_to_camelot("F#"), "2B")
        self.assertEqual(key_to_camelot("Gb"), "2B")
        self.assertEqual(key_to_camelot("G"), "9B")
        self.assertEqual(key_to_camelot("G#"), "4B")
        self.assertEqual(key_to_camelot("Ab"), "4B")
        self.assertEqual(key_to_camelot("A"), "11B")
        self.assertEqual(key_to_camelot("A#"), "6B")
        self.assertEqual(key_to_camelot("Bb"), "6B")
        self.assertEqual(key_to_camelot("B"), "1B")

    def test_key_to_camelot_minor_keys(self) -> None:
        self.assertEqual(key_to_camelot("Am"), "8A")
        self.assertEqual(key_to_camelot("A#m"), "3A")
        self.assertEqual(key_to_camelot("Bbm"), "3A")
        self.assertEqual(key_to_camelot("Bm"), "10A")
        self.assertEqual(key_to_camelot("Cm"), "5A")
        self.assertEqual(key_to_camelot("C#m"), "12A")
        self.assertEqual(key_to_camelot("Dbm"), "12A")
        self.assertEqual(key_to_camelot("Dm"), "7A")
        self.assertEqual(key_to_camelot("D#m"), "2A")
        self.assertEqual(key_to_camelot("Ebm"), "2A")
        self.assertEqual(key_to_camelot("Em"), "9A")
        self.assertEqual(key_to_camelot("Fm"), "4A")
        self.assertEqual(key_to_camelot("F#m"), "11A")
        self.assertEqual(key_to_camelot("Gbm"), "11A")
        self.assertEqual(key_to_camelot("Gm"), "6A")
        self.assertEqual(key_to_camelot("G#m"), "1A")
        self.assertEqual(key_to_camelot("Abm"), "1A")

    def test_key_to_camelot_extended_formats(self) -> None:
        self.assertEqual(key_to_camelot("C# minor"), "12A")
        self.assertEqual(key_to_camelot("A minor"), "8A")
        self.assertEqual(key_to_camelot("F# min"), "11A")
        self.assertEqual(key_to_camelot("Bb Major"), "6B")
        self.assertEqual(key_to_camelot("C# major"), "3B")
        self.assertEqual(key_to_camelot("Eb maj"), "5B")
        self.assertEqual(key_to_camelot("Am (8A)"), "8A")
        self.assertEqual(key_to_camelot("Bbm (3A)"), "3A")
        self.assertEqual(key_to_camelot("8A - Am"), "8A")
        self.assertEqual(key_to_camelot("D♯"), "5B")
        self.assertEqual(key_to_camelot("B♭m"), "3A")
        self.assertEqual(key_to_camelot("8A"), "8A")
        self.assertEqual(key_to_camelot("12b"), "12B")
        self.assertEqual(key_to_camelot("1A"), "1A")
        self.assertIsNone(key_to_camelot(None))
        self.assertIsNone(key_to_camelot(""))
        self.assertIsNone(key_to_camelot("Unknown123"))

    def test_chroma_filterbank_shape_and_properties(self) -> None:
        fb = get_chroma_filterbank(sr=22050, n_fft=4096)
        self.assertEqual(fb.shape, (12, 2049))
        self.assertTrue(np.all(fb >= 0))
        self.assertEqual(fb.dtype, np.float32)

    def test_detect_synthetic_c_major(self) -> None:
        # C4 (261.63 Hz), E4 (329.63 Hz), G4 (392.00 Hz)
        c_chord_path = self.tmp_path / "c_major.wav"
        create_chord_wav(c_chord_path, [261.63, 329.63, 392.00])

        details = detect_key_details(c_chord_path)
        self.assertIsNotNone(details)
        if details:
            key_name, camelot, conf = details
            self.assertEqual(key_name, "C")
            self.assertEqual(camelot, "8B")
            self.assertGreater(conf, 0.5)

        detected_key = detect_musical_key(c_chord_path)
        self.assertEqual(detected_key, "C")

    def test_detect_synthetic_a_minor(self) -> None:
        # A4 (440.00 Hz), C5 (523.25 Hz), E5 (659.25 Hz)
        am_chord_path = self.tmp_path / "a_minor.wav"
        create_chord_wav(am_chord_path, [440.00, 523.25, 659.25])

        details = detect_key_details(am_chord_path)
        self.assertIsNotNone(details)
        if details:
            key_name, camelot, conf = details
            self.assertEqual(key_name, "Am")
            self.assertEqual(camelot, "8A")
            self.assertGreater(conf, 0.5)

    def test_empty_and_silent_audio(self) -> None:
        silent_audio = np.zeros(22050 * 2, dtype=np.float32)
        chroma = calculate_chroma(silent_audio, 22050)
        self.assertIsNone(chroma)

        empty_audio = np.array([], dtype=np.float32)
        self.assertIsNone(calculate_chroma(empty_audio, 22050))

        short_audio = np.ones(500, dtype=np.float32)
        self.assertIsNone(calculate_chroma(short_audio, 22050))

    def test_detect_from_invalid_chroma(self) -> None:
        invalid_chroma = np.ones(5, dtype=np.float32)
        self.assertIsNone(detect_key_from_chroma(invalid_chroma))

    def test_detect_nonexistent_file_raises(self) -> None:
        bogus = self.tmp_path / "does_not_exist.wav"
        with self.assertRaises(FileNotFoundError):
            detect_musical_key(bogus)

    def test_normalize_single_track_with_key(self) -> None:
        # Create C major audio file
        c_chord_path = self.tmp_path / "c_chord.wav"
        create_chord_wav(c_chord_path, [261.63, 329.63, 392.00])

        track_info = read_track_metadata(c_chord_path)
        self.assertIsNone(track_info.initial_key)

        updated = normalize_single_track(c_chord_path, fetch_key=True)
        self.assertIsNotNone(updated)
        if updated:
            self.assertEqual(updated.initial_key, "C")

        # Verify saved to disk
        reloaded = read_track_metadata(c_chord_path)
        self.assertEqual(reloaded.initial_key, "C")

    def test_normalize_library_with_key(self) -> None:
        album_dir = self.tmp_path / "Artist - Album"
        album_dir.mkdir()
        create_chord_wav(album_dir / "01 - Song.wav", [261.63, 329.63, 392.00])

        results = normalize_library(album_dir, fetch_key=True)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].initial_key, "C")


if __name__ == "__main__":
    unittest.main()
