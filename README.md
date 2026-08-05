# Sonora

Sonora is a CLI music management tool for FLAC, MP3, M4A, OGG, and WAV collections. It handles automatic metadata tagging, audio stream integrity auditing, loudness analysis, synchronized lyrics fetching, and library organization.

## Features

- **Autotagging**: Query MusicBrainz, AcoustID, Discogs, Last.fm, and iTunes Search API for metadata, release dates, mood tags, and high-res cover art.
- **Audio Processing**: 
  - Librosa BPM detection.
  - FFmpeg EBU R128 loudness analysis & ReplayGain 2.0 calculations.
  - Optional SoX 16kHz spectral analysis for fake lossless detection.
- **Synchronized Lyrics**: Fetch `.lrc` lyrics via Lrclib, Musixmatch, Genius, and NetEase (`syncedlyrics`).
- **Integrity Auditing**: Verify native FLAC audio stream MD5 checksums (`flac -t`), detect corrupt bracket metadata (`[HQ]`, `[FLAC]`), and identify missing tags.
- **Renaming & Organization**: Rename tracks from metadata while updating `.lrc` header tags, and automatically separate standalone singles from full album folders.

## System Dependencies

Sonora uses system binaries for low-level audio analysis and verification:

```bash
# Arch Linux
sudo pacman -S ffmpeg flac sox

# Debian / Ubuntu
sudo apt install ffmpeg flac sox

# Fedora
sudo dnf install ffmpeg flac sox

# macOS
brew install ffmpeg flac sox
```

## Installation

```bash
git clone https://github.com/dvmizew/Sonora.git
cd Sonora
pip install -e .
```

## CLI Usage

### `sonora tag` — Autotag audio files
Tags files in a directory using parallel worker threads:

```bash
# Tag a folder (all processing engines enabled by default)
sonora tag /path/to/album

# With API keys and custom thread count
sonora tag /path/to/album -w 8 --lastfm-key KEY --acoustid-key KEY --discogs-token TOKEN

# Disable specific features
sonora tag /path/to/album --no-bpm --no-replaygain --no-art
```

### `sonora audit` — Audit library integrity
Scans folders recursively for corrupted FLACs, bad tags, or missing `.lrc` files:

```bash
# Basic audit
sonora audit /path/to/library

# Export JSON report
sonora audit /path/to/library --json report.json

# Include spectral cutoff check (slow)
sonora audit /path/to/library --spectral
```

### `sonora rename` — Rename files & sync LRC headers
Renames files to `01 - Artist - Title.ext` and keeps `.lrc` header tags (`[ar:]`, `[ti:]`) in sync:

```bash
sonora rename /path/to/album
```

### `sonora organize` — Separate single tracks
Moves single tracks (folders with <= 2 tracks or mixed album tags) to a dedicated Singles directory:

```bash
sonora organize /path/to/music --target-singles /path/to/Singles
```

## Development & Testing

Run unit tests:

```bash
python3 -m unittest discover -s tests
```