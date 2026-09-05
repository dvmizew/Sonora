import functools
import re
from pathlib import Path

import numpy as np
import scipy.signal

from sonora.audio.bpm import load_audio
from sonora.core.logger import LOG

PITCH_CLASSES: tuple[str, ...] = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)

# Empirically optimized Shaath profiles (KeyFinder / Essentia standard for modern music)
SHAATH_MAJOR: np.ndarray = np.array(
    [6.6, 2.0, 3.5, 2.3, 4.6, 4.0, 2.5, 5.2, 2.4, 3.7, 2.3, 3.4], dtype=np.float32
)
SHAATH_MINOR: np.ndarray = np.array(
    [6.5, 2.7, 3.5, 5.4, 2.6, 3.5, 2.5, 5.2, 4.0, 2.7, 4.3, 3.2], dtype=np.float32
)

# Camelot wheel tonality mappings (Universal DJ harmonic mixing standard)
CAMELOT_MAP: dict[str, str] = {
    # Major keys (B suffix)
    "B": "1B",
    "F#": "2B",
    "Gb": "2B",
    "C#": "3B",
    "Db": "3B",
    "G#": "4B",
    "Ab": "4B",
    "D#": "5B",
    "Eb": "5B",
    "A#": "6B",
    "Bb": "6B",
    "F": "7B",
    "C": "8B",
    "G": "9B",
    "D": "10B",
    "A": "11B",
    "E": "12B",
    # Minor keys (A suffix)
    "G#m": "1A",
    "Abm": "1A",
    "D#m": "2A",
    "Ebm": "2A",
    "A#m": "3A",
    "Bbm": "3A",
    "Fm": "4A",
    "Cm": "5A",
    "Gm": "6A",
    "Dm": "7A",
    "Am": "8A",
    "Em": "9A",
    "Bm": "10A",
    "F#m": "11A",
    "Gbm": "11A",
    "C#m": "12A",
    "Dbm": "12A",
}

_CAMELOT_REGEX: re.Pattern[str] = re.compile(r"^(?:1[0-2]|[1-9])[ABab]$")


def key_to_camelot(key_str: str | None) -> str | None:
    """
    Convert a musical key notation (e.g. 'Am', 'C# minor', 'Eb') to its Camelot wheel code.
    If the string is already a valid Camelot code (e.g. '8A', '12B'), returns it normalized.
    """
    if not key_str:
        return None

    cleaned = key_str.strip()
    if _CAMELOT_REGEX.match(cleaned):
        return cleaned.upper()

    # Clean punctuation and normalize symbols (musical sharp / flat)
    cleaned = cleaned.replace("♯", "#").replace("♭", "b")
    # Normalize descriptors
    is_minor = bool(
        re.search(r"(?:minor|min|\bm\b)", cleaned, flags=re.IGNORECASE)
        or (len(cleaned) >= 2 and cleaned.endswith("m") and not cleaned.endswith("pm"))
    )

    # Extract base note (A-G with optional # or b)
    match = re.match(r"^([A-Ga-g][#b]?)", cleaned)
    if not match:
        return None

    base_note = match.group(1)
    base_note = base_note[0].upper() + (
        base_note[1].lower() if len(base_note) > 1 else ""
    )

    lookup_key = f"{base_note}m" if is_minor else base_note
    return CAMELOT_MAP.get(lookup_key)


def _hz_to_octs(
    frequencies: np.ndarray,
    tuning: float = 0.0,
    bins_per_octave: int = 12,
) -> np.ndarray:
    """Convert frequencies (Hz) to fractional octave coordinates relative to A0 (27.5 Hz)."""
    a440 = 440.0
    a0 = a440 / 16.0  # 27.5 Hz
    freq_tuned = frequencies * (2.0 ** (-tuning / bins_per_octave))
    safe_freq = np.maximum(freq_tuned / a0, 1e-10)
    octs: np.ndarray = np.log2(safe_freq)
    return octs


@functools.lru_cache(maxsize=4)
def get_chroma_filterbank(
    sr: int = 22050,
    n_fft: int = 4096,
    n_chroma: int = 12,
    tuning: float = 0.0,
    ctroct: float = 4.0,
    octwidth: float = 1.5,
) -> np.ndarray:
    """
    Construct a librosa-compatible Gaussian chroma filterbank matrix.
    Projects linear FFT magnitude bins onto 12 pitch classes (C, C#, ..., B).
    """
    wts = np.zeros((n_chroma, n_fft), dtype=np.float64)
    frequencies = np.linspace(0, sr, n_fft, endpoint=False)[1:]

    frqbins = n_chroma * _hz_to_octs(
        frequencies, tuning=tuning, bins_per_octave=n_chroma
    )
    frqbins = np.concatenate(([frqbins[0] - 1.5 * n_chroma], frqbins))
    binwidthbins = np.concatenate((np.maximum(frqbins[1:] - frqbins[:-1], 1.0), [1.0]))

    diff_matrix = np.subtract.outer(frqbins, np.arange(0, n_chroma, dtype=np.float64)).T
    n_chroma2 = np.round(float(n_chroma) / 2.0)
    diff_matrix = (
        np.remainder(diff_matrix + n_chroma2 + 10.0 * n_chroma, n_chroma) - n_chroma2
    )

    wts = np.exp(-0.5 * (2.0 * diff_matrix / np.tile(binwidthbins, (n_chroma, 1))) ** 2)

    # Normalize each frequency column
    col_norms = np.sqrt(np.sum(wts**2, axis=0, keepdims=True))
    col_norms[col_norms == 0] = 1.0
    wts = wts / col_norms

    # Octave Gaussian weighting window centered at ctroct (440 Hz)
    oct_weight = np.exp(-0.5 * (((frqbins / n_chroma - ctroct) / octwidth) ** 2))[
        np.newaxis, :
    ]
    wts *= oct_weight

    # Roll from A-based to C-based (A is index 0 -> roll -3 places C at index 0)
    wts = np.roll(wts, -3 * (n_chroma // 12), axis=0)
    cutoff = int(1 + n_fft // 2)
    return np.ascontiguousarray(wts[:, :cutoff], dtype=np.float32)


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient between two 1D vectors."""
    x_zm = x - np.mean(x)
    y_zm = y - np.mean(y)
    denom = float(np.sqrt(np.sum(x_zm**2) * np.sum(y_zm**2)))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(x_zm * y_zm) / denom)


def calculate_chroma(audio: np.ndarray, sample_rate: int) -> np.ndarray | None:
    """
    Calculate the 12-dimensional mean chroma vector from mono audio signal.
    """
    if len(audio) == 0 or np.all(audio == 0):
        return None

    # Downsample to ~22,050 Hz for efficient spectral analysis
    target_sr = 22050
    if sample_rate > 24000:
        decimation = sample_rate // target_sr
        if decimation > 1:
            audio = audio[::decimation]
            sample_rate = sample_rate // decimation

    nperseg = 4096
    noverlap = nperseg // 2
    if len(audio) < nperseg:
        return None

    _, _, spectrogram = scipy.signal.spectrogram(
        audio,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        window="hann",
    )

    if spectrogram.size == 0:
        return None

    # Logarithmic magnitude compression to balance dynamic range against percussion
    spec_log = np.log1p(1000.0 * spectrogram)

    # Project linear frequencies into 12 chroma pitch classes
    filterbank = get_chroma_filterbank(sr=sample_rate, n_fft=nperseg)
    chroma = filterbank @ spec_log

    # L2-normalize each time frame
    frame_norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    chroma_norm = chroma / np.maximum(frame_norms, 1e-6)

    # Compute mean energy across all analyzed time frames
    chroma_mean: np.ndarray = np.mean(chroma_norm, axis=1)
    vector_norm = np.linalg.norm(chroma_mean)
    if vector_norm <= 0:
        return None

    normalized: np.ndarray = np.asarray(chroma_mean / vector_norm, dtype=np.float32)
    return normalized


def detect_key_from_chroma(
    chroma_vector: np.ndarray,
) -> tuple[str, str, float] | None:
    """
    Match a 12-element chroma vector against Shaath key profiles using Pearson correlation.
    Returns (key_name, camelot_code, confidence) or None if correlation is negligible.
    """
    if len(chroma_vector) != 12:
        return None

    best_key: str | None = None
    best_score: float = -1.0

    for i in range(12):
        base_name = PITCH_CLASSES[i]

        # Major correlation
        score_maj = _pearson_r(chroma_vector, np.roll(SHAATH_MAJOR, i))
        if score_maj > best_score:
            best_score = score_maj
            best_key = base_name

        # Minor correlation
        score_min = _pearson_r(chroma_vector, np.roll(SHAATH_MINOR, i))
        if score_min > best_score:
            best_score = score_min
            best_key = f"{base_name}m"

    if best_key is None or best_score < 0.20:
        return None

    camelot = CAMELOT_MAP.get(best_key, "")
    return best_key, camelot, round(best_score, 3)


def detect_key_details(
    file_path: Path,
) -> tuple[str, str, float] | None:
    """
    Analyze audio file and return (key_name, camelot_code, confidence) or None.
    Skips the first 10 seconds of audio when duration permits to bypass ambient intros.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        loaded = load_audio(file_path, mono=True, max_seconds=60.0, offset_seconds=10.0)
        if loaded is None or len(loaded[0]) < 4096:
            # Fallback to beginning of file if offset seek yielded insufficient samples (e.g. short track)
            loaded = load_audio(
                file_path, mono=True, max_seconds=60.0, offset_seconds=0.0
            )

        if loaded is None:
            return None

        audio_mono, sample_rate = loaded
        chroma = calculate_chroma(audio_mono, sample_rate)
        if chroma is None:
            return None

        return detect_key_from_chroma(chroma)
    except (OSError, ValueError, RuntimeError, IndexError, TypeError) as error:
        LOG.debug(f"Key detection failed for {file_path}: {error}")
        return None


def detect_musical_key(file_path: Path) -> str | None:
    """
    Detect canonical musical key (e.g. 'C#m', 'Am', 'C') for an audio file.
    """
    result = detect_key_details(file_path)
    if result is None:
        return None
    return result[0]
