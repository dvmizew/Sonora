import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from mutagen.flac import FLAC
from mutagen.flac import error as FLACError

from sonora.core.constants import METAFLAC_CMD
from sonora.core.logger import LOG


def calculate_album_replaygain(
    files: Sequence[Path], options: dict | None = None
) -> bool:
    """
    Use metaflac to calculate both Track and Album ReplayGain for a list of FLAC files.
    Skips if REPLAYGAIN_ALBUM_GAIN is already present.
    Falls back to track-mode if audio properties (sample rate, channels, bit depth) are mixed.
    Returns True if ReplayGain was calculated and added, False otherwise.
    """
    flac_files = [str(f) for f in files if f.exists() and f.suffix.lower() == ".flac"]
    if not flac_files:
        return False

    if not shutil.which(METAFLAC_CMD):
        LOG.warning(f"'{METAFLAC_CMD}' not found in PATH! ReplayGain disabled.")
        return False

    options = options or {}
    dry_run = options.get("dry_run", False)

    properties = set()
    has_album_gain = False

    try:
        for f in flac_files:
            a = FLAC(f)
            # Sample Rate, Channels, Bits must match for metaflac album mode
            props = (a.info.sample_rate, a.info.channels, a.info.bits_per_sample)
            properties.add(props)
            if not has_album_gain and "REPLAYGAIN_ALBUM_GAIN" in a:
                has_album_gain = True
    except (FLACError, OSError, ValueError, KeyError) as e:
        LOG.debug(f"Failed to read properties for ReplayGain: {e}")
        return False

    if has_album_gain:
        LOG.debug("Album already has ReplayGain tags. Skipping.")
        return False

    try:
        is_uniform = len(properties) <= 1
        if is_uniform:
            LOG.info("🔊 Calculating ReplayGain (Album Mode)...")
            if not dry_run:
                cmd = [METAFLAC_CMD, "--add-replay-gain"] + flac_files
                r = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300, check=False
                )
                if r.returncode != 0:
                    LOG.error(f"metaflac failed: {r.stderr}")
                    return False
            else:
                LOG.info(
                    f"[DRY-RUN] Would calculate ReplayGain (Album Mode) for {len(flac_files)} files"
                )
        else:
            LOG.warning(
                "⚠️  Mixed audio properties detected. Falling back to Track-only ReplayGain."
            )
            any_success = False
            for f in flac_files:
                if not dry_run:
                    cmd = [METAFLAC_CMD, "--add-replay-gain", f]
                    r = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=60, check=False
                    )
                    if r.returncode != 0:
                        LOG.error(f"metaflac failed for {f}: {r.stderr}")
                    else:
                        any_success = True
                else:
                    LOG.info(f"[DRY-RUN] Would calculate Track-only ReplayGain for {f}")
                    any_success = True
            return any_success
        return True
    except subprocess.TimeoutExpired:
        LOG.error("metaflac timed out while calculating ReplayGain.")
        return False
    except (subprocess.SubprocessError, OSError) as e:
        LOG.error(f"Failed to calculate ReplayGain: {e}")
        return False
