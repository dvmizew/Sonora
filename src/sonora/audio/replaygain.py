from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyloudnorm

from sonora.audio.bpm import load_audio
from sonora.audio.metadata import read_track_metadata, write_track_metadata
from sonora.core.constants import SUPPORTED_EXTS
from sonora.core.logger import LOG


def calculate_track_replaygain(
    file_path: Path, target_lufs: float = -18.0
) -> tuple[float, float] | None:
    """
    Calculate ReplayGain for a single track.
    Returns (gain_db, peak_amplitude) or None if measurement fails.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    loaded = load_audio(file_path)
    if loaded is None or len(loaded[0]) == 0 or loaded[1] <= 0:
        return None

    audio_data, sample_rate = loaded
    try:
        meter = pyloudnorm.Meter(sample_rate)
        loudness = meter.integrated_loudness(audio_data)
        gain_db = (
            target_lufs - loudness
            if not (np.isnan(loudness) or np.isinf(loudness))
            else 0.0
        )
        peak_amp = float(np.max(np.abs(audio_data)))
        return float(gain_db), float(peak_amp)
    except (ValueError, RuntimeError) as error:
        LOG.debug(f"pyloudnorm measurement failed for {file_path}: {error}")
        return None


def calculate_album_replaygain(
    files: Sequence[Path],
    force: bool = False,
    dry_run: bool = False,
    target_lufs: float = -18.0,
) -> bool:
    """
    Calculate ReplayGain (Track and Album Mode) for all audio files.
    Writes REPLAYGAIN_TRACK_GAIN, REPLAYGAIN_TRACK_PEAK,
    REPLAYGAIN_ALBUM_GAIN, and REPLAYGAIN_ALBUM_PEAK tags.
    """
    valid_files = [
        file_path
        for file_path in files
        if file_path.exists() and file_path.suffix.lower() in SUPPORTED_EXTS
    ]
    if not valid_files:
        return False

    # 1. Check if ReplayGain is already calculated on all files
    if not force:
        already_tagged = True
        for audio_path in valid_files:
            try:
                info = read_track_metadata(audio_path)
                if (
                    info.replaygain_track_gain is None
                    or info.replaygain_album_gain is None
                ):
                    already_tagged = False
                    break
            except (OSError, ValueError, RuntimeError):
                already_tagged = False
                break
        if already_tagged:
            LOG.debug("Files already contain ReplayGain tags. Skipping.")
            return False

    LOG.info(f"🔊 Calculating ReplayGain for {len(valid_files)} track(s)...")

    track_results: list[tuple[Path, float, float, np.ndarray, int]] = []
    max_album_peak = 0.0

    # 2. Compute individual track loudness
    for audio_path in valid_files:
        loaded = load_audio(audio_path)
        if loaded is None or len(loaded[0]) == 0 or loaded[1] <= 0:
            continue
        audio_data, sample_rate = loaded

        try:
            meter = pyloudnorm.Meter(sample_rate)
            track_loudness = meter.integrated_loudness(audio_data)
            track_gain = (
                target_lufs - track_loudness
                if not (np.isnan(track_loudness) or np.isinf(track_loudness))
                else 0.0
            )
            track_peak = float(np.max(np.abs(audio_data)))
            max_album_peak = max(max_album_peak, track_peak)
            track_results.append(
                (audio_path, track_gain, track_peak, audio_data, sample_rate)
            )
        except (ValueError, RuntimeError) as error:
            LOG.debug(f"Failed to measure loudness for {audio_path}: {error}")

    if not track_results:
        LOG.warning("Could not calculate loudness for any audio files.")
        return False

    # 3. Compute album loudness across all tracks
    sample_rates = {result[4] for result in track_results}
    if len(sample_rates) == 1:
        common_sample_rate = track_results[0][4]
        aligned_arrays = []
        for _, _, _, audio_data, _ in track_results:
            if audio_data.ndim == 1:
                aligned_arrays.append(np.column_stack([audio_data, audio_data]))
            elif audio_data.shape[1] == 1:
                aligned_arrays.append(
                    np.column_stack([audio_data[:, 0], audio_data[:, 0]])
                )
            else:
                aligned_arrays.append(audio_data)
        try:
            concatenated = np.concatenate(aligned_arrays, axis=0)
            album_meter = pyloudnorm.Meter(common_sample_rate)
            album_loudness = album_meter.integrated_loudness(concatenated)
            album_gain = (
                target_lufs - album_loudness
                if not (np.isnan(album_loudness) or np.isinf(album_loudness))
                else float(np.mean([result[1] for result in track_results]))
            )
        except (ValueError, RuntimeError):
            album_gain = float(np.mean([result[1] for result in track_results]))
    else:
        album_gain = float(np.mean([result[1] for result in track_results]))

    # 4. Write ReplayGain metadata tags to each file
    tagged_count = 0
    for file_path, track_gain, track_peak, _, _ in track_results:
        if dry_run:
            LOG.info(
                f"[DRY-RUN] Would tag {file_path.name}: Track Gain={track_gain:+.2f} dB, "
                f"Album Gain={album_gain:+.2f} dB, Track Peak={track_peak:.6f}, Album Peak={max_album_peak:.6f}"
            )
            tagged_count += 1
            continue

        try:
            info = read_track_metadata(file_path)
            info.replaygain_track_gain = round(track_gain, 2)
            info.replaygain_track_peak = round(track_peak, 6)
            info.replaygain_album_gain = round(album_gain, 2)
            info.replaygain_album_peak = round(max_album_peak, 6)
            write_track_metadata(info)
            tagged_count += 1
        except (OSError, ValueError, RuntimeError) as error:
            LOG.debug(f"Failed to write ReplayGain tags to {file_path}: {error}")

    LOG.info(f"✅ Applied ReplayGain to {tagged_count}/{len(valid_files)} track(s).")
    return tagged_count > 0
