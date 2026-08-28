from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyloudnorm

from sonora.audio.bpm import load_audio
from sonora.audio.metadata import read_track_metadata, write_track_metadata
from sonora.core.constants import SUPPORTED_EXTS
from sonora.core.logger import LOG


def _measure_track_loudness(
    audio_path: Path, target_lufs: float = -18.0
) -> tuple[Path, float, float, float, float] | None:
    """
    Measures loudness and peak amplitude of a single track.
    Returns (audio_path, track_gain, track_peak, track_loudness, duration) or None.
    """
    try:
        loaded = load_audio(audio_path)
        if loaded is None or len(loaded[0]) == 0 or loaded[1] <= 0:
            return None
        audio_data, sample_rate = loaded

        meter = pyloudnorm.Meter(sample_rate)
        track_loudness = float(meter.integrated_loudness(audio_data))
        duration = float(len(audio_data) / sample_rate)
        track_peak = float(np.max(np.abs(audio_data)))

        track_gain = (
            target_lufs - track_loudness
            if not (np.isnan(track_loudness) or np.isinf(track_loudness))
            else 0.0
        )
        return (audio_path, track_gain, track_peak, track_loudness, duration)
    except (ValueError, RuntimeError, OSError) as error:
        LOG.debug(f"Failed to measure loudness for {audio_path}: {error}")
        return None


def calculate_track_replaygain(
    file_path: Path, target_lufs: float = -18.0
) -> tuple[float, float] | None:
    """
    Calculate ReplayGain for a single track.
    Returns (gain_db, peak_amplitude) or None if measurement fails.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    res = _measure_track_loudness(file_path, target_lufs=target_lufs)
    if res is None:
        return None
    _, track_gain, track_peak, _, _ = res
    return float(track_gain), float(track_peak)


def calculate_album_replaygain(
    files: Sequence[Path],
    force: bool = False,
    dry_run: bool = False,
    target_lufs: float = -18.0,
    max_threads: int = 4,
) -> bool:
    """
    Calculate ReplayGain (Track and Album Mode) for all audio files in parallel.
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

    # 2. Compute individual track loudness in parallel across threads
    track_results: list[tuple[Path, float, float, float, float]] = []
    max_album_peak = 0.0

    executor = ThreadPoolExecutor(max_workers=max_threads)
    try:
        futures = [
            executor.submit(_measure_track_loudness, audio_path, target_lufs)
            for audio_path in valid_files
        ]
        for future in futures:
            try:
                res = future.result()
                if res is not None:
                    track_results.append(res)
                    max_album_peak = max(max_album_peak, res[2])
            except (
                OSError,
                ValueError,
                RuntimeError,
                TypeError,
                KeyError,
                AttributeError,
            ) as error:
                LOG.debug(f"Track loudness measurement failed: {error}")
    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    if not track_results:
        LOG.warning("Could not calculate loudness for any audio files.")
        return False

    # 3. Compute album loudness via ITU-R BS.1770 duration-weighted linear energy integration
    total_energy = sum(
        (10.0 ** (loudness / 10.0)) * duration
        for _, _, _, loudness, duration in track_results
        if not (np.isnan(loudness) or np.isinf(loudness))
    )
    total_duration = sum(
        duration
        for _, _, _, loudness, duration in track_results
        if not (np.isnan(loudness) or np.isinf(loudness))
    )

    if total_duration > 0 and total_energy > 0:
        album_loudness = 10.0 * float(np.log10(total_energy / total_duration))
        album_gain = (
            target_lufs - album_loudness
            if not (np.isnan(album_loudness) or np.isinf(album_loudness))
            else float(np.mean([result[1] for result in track_results]))
        )
    else:
        album_gain = float(np.mean([result[1] for result in track_results]))

    # 4. Write ReplayGain metadata tags to each file in parallel
    def _write_tags(entry: tuple[Path, float, float, float, float]) -> bool:
        file_path, track_gain, track_peak, _, _ = entry
        if dry_run:
            LOG.info(
                f"[DRY-RUN] Would tag {file_path.name}: Track Gain={track_gain:+.2f} dB, "
                f"Album Gain={album_gain:+.2f} dB, Track Peak={track_peak:.6f}, Album Peak={max_album_peak:.6f}"
            )
            return True
        try:
            info = read_track_metadata(file_path)
            info.replaygain_track_gain = round(track_gain, 2)
            info.replaygain_track_peak = round(track_peak, 6)
            info.replaygain_album_gain = round(album_gain, 2)
            info.replaygain_album_peak = round(max_album_peak, 6)
            write_track_metadata(info)
            return True
        except (OSError, ValueError, RuntimeError) as err:
            LOG.debug(f"Failed to write ReplayGain tags to {file_path}: {err}")
            return False

    executor_write = ThreadPoolExecutor(max_workers=max_threads)
    try:
        write_results = list(executor_write.map(_write_tags, track_results))
    except KeyboardInterrupt:
        executor_write.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        executor_write.shutdown(wait=False, cancel_futures=True)

    tagged_count = sum(1 for success in write_results if success)
    LOG.info(f"✅ Applied ReplayGain to {tagged_count}/{len(valid_files)} track(s).")
    return tagged_count > 0
