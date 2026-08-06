"""
BPM calculation module for audio files using Librosa.
"""

from pathlib import Path
from sonora.core.exceptions import AudioProcessingError

try:
    import librosa
except ImportError:
    librosa = None


def calculate_bpm(file_path: Path) -> float | None:
    """
    Calculate the BPM (beats per minute) of an audio file using Librosa.
    Returns float BPM rounded to 1 decimal place, or None if calculation fails.
    """
    if not file_path.exists():
        raise AudioProcessingError(f"File not found: {file_path}")

    if librosa is None:
        raise AudioProcessingError("Librosa library is not installed.")

    try:
        y, sr = librosa.load(str(file_path), sr=None, duration=120)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if hasattr(tempo, "__getitem__"):
            tempo = tempo[0]
        return round(float(tempo), 1)
    except Exception as e:
        raise AudioProcessingError(f"BPM calculation failed for {file_path}: {e}") from e
