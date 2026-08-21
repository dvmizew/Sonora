import re
import shutil
import subprocess
from pathlib import Path

from sonora.core.constants import BPM_TAG_CMD
from sonora.core.exceptions import AudioProcessingError
from sonora.core.logger import LOG


def calculate_bpm(file_path: Path) -> float | None:
    """
    Calculate the BPM (beats per minute) of an audio file.
    Fast path: Uses C-based `bpm-tag` for maximum speed.
    Fallback path: Uses optimized Librosa (low sample rate, truncated duration).
    """
    if not file_path.exists():
        raise AudioProcessingError(f"File not found: {file_path}")
    if shutil.which(BPM_TAG_CMD):
        try:
            # -f forces analysis ignoring existing tags, -n prints to stderr
            cmd = [BPM_TAG_CMD, "-f", "-n", str(file_path)]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
            combined_output = r.stderr + r.stdout
            m = re.search(r"([\d.]+)\s*BPM", combined_output)
            if m:
                bpm = float(m.group(1))
                if bpm > 0:
                    return round(bpm, 1)
        except (subprocess.SubprocessError, ValueError, OSError) as e:
            LOG.debug(f"bpm-tag failed for {file_path}: {e}")
            
    try:
        import librosa
    except ImportError:
        raise AudioProcessingError("Librosa and bpm-tools are both missing. Cannot calculate BPM.")

    try:
        # Skip intro (30s), read only 60s, and force downsample to 22050 Hz (sufficient for beat detection)
        y, sr = librosa.load(str(file_path), sr=22050, offset=30, duration=60)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, (list, tuple)):
            bpm_val = float(tempo[0])
        elif isinstance(tempo, (int, float)):
            bpm_val = float(tempo)
        elif hasattr(tempo, "item"):
            item_func = tempo.item
            bpm_val = float(item_func())
        else:
            bpm_val = float(str(tempo))
        return round(bpm_val, 1)
    except (OSError, ValueError, RuntimeError, AudioProcessingError) as e:
        raise AudioProcessingError(f"Librosa BPM calculation failed for {file_path}: {e}") from e
