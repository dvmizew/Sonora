import functools
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyloudnorm
from rich.markup import escape

from sonora.audio.bpm import load_audio
from sonora.audio.metadata import read_track_metadata, write_track_metadata
from sonora.core.constants import SUPPORTED_EXTS
from sonora.core.logger import LOG
from sonora.core.utils import is_interruption


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
    except (ValueError, OSError) as error:
        LOG.debug(f"Failed to measure loudness for {audio_path}: {error}")
        return None


def _write_album_replaygain_track(
    entry: tuple[Path, float, float, float, float],
    album_gain: float,
    max_album_peak: float,
    dry_run: bool = False,
) -> bool:
    file_path, track_gain, track_peak, _, _ = entry
    if dry_run:
        LOG.info(
            f"[DRY-RUN] Would tag {escape(file_path.name)}: Track Gain={track_gain:+.2f} dB, "
            f"Album Gain={album_gain:+.2f} dB, Track Peak={track_peak:.6f}, Album Peak={max_album_peak:.6f}"
        )
        return True
    try:
        track_info = read_track_metadata(file_path)
        track_info.replaygain_track_gain = round(track_gain, 2)
        track_info.replaygain_track_peak = round(track_peak, 6)
        track_info.replaygain_album_gain = round(album_gain, 2)
        track_info.replaygain_album_peak = round(max_album_peak, 6)
        write_track_metadata(track_info)
        return True
    except OSError as err:
        LOG.debug(f"Failed to write ReplayGain tags to {file_path}: {err}")
        return False


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

    if not force:
        already_tagged = True
        for audio_path in valid_files:
            try:
                track_info = read_track_metadata(audio_path)
                if (
                    track_info.replaygain_track_gain is None
                    or track_info.replaygain_album_gain is None
                ):
                    already_tagged = False
                    break
            except OSError:
                already_tagged = False
                break
        if already_tagged:
            LOG.debug("Files already contain ReplayGain tags. Skipping.")
            return False

    LOG.info(f"🔊 Calculating ReplayGain for {len(valid_files)} track(s)...")

    track_results: list[tuple[Path, float, float, float, float]] = []
    max_album_peak = 0.0

    with ThreadPoolExecutor(max_workers=min(max_threads, 4)) as executor:
        try:
            futures = [
                executor.submit(_measure_track_loudness, audio_path, target_lufs)
                for audio_path in valid_files
            ]
            for future in futures:
                try:
                    loudness_metrics = future.result()
                    if loudness_metrics is not None:
                        track_results.append(loudness_metrics)
                        max_album_peak = max(max_album_peak, loudness_metrics[2])
                except OSError as error:
                    LOG.debug(f"Track loudness measurement failed: {error}")
        except (KeyboardInterrupt, RuntimeError) as exc:
            if not is_interruption(exc):
                raise
            executor.shutdown(wait=True, cancel_futures=True)
            raise

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
            else float(np.mean([metrics[1] for metrics in track_results]))
        )
    else:
        album_gain = float(np.mean([metrics[1] for metrics in track_results]))

    writer = functools.partial(
        _write_album_replaygain_track,
        album_gain=album_gain,
        max_album_peak=max_album_peak,
        dry_run=dry_run,
    )
    with ThreadPoolExecutor(max_workers=max_threads) as executor_write:
        try:
            write_results = list(executor_write.map(writer, track_results))
        except (KeyboardInterrupt, RuntimeError) as exc:
            if not is_interruption(exc):
                raise
            executor_write.shutdown(wait=True, cancel_futures=True)
            raise

    tagged_count = sum(1 for success in write_results if success)
    LOG.info(f"✅ Applied ReplayGain to {tagged_count}/{len(valid_files)} track(s).")
    return tagged_count > 0
