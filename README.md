# Sonora

CLI tool for managing local music libraries. Fully supports FLAC, MP3, M4A, MP4, ALAC, OGG, OPUS, WAV, AIFF, WMA, APE, WV, and MPC.

Features:
- **Tagging**: Fetches metadata, cover art, and mood tags from MusicBrainz, AcoustID, Discogs, Last.fm, and iTunes.
- **Processing**: 
  - BPM detection (`bpm-tools` / `librosa`).
  - EBU R128 loudness & ReplayGain 2.0 (`ffmpeg`).
  - 16kHz spectral analysis for fake lossless detection (`sox`).
- **Lyrics**: Downloads synchronized `.lrc` lyrics (Lrclib, Musixmatch, Genius, NetEase).
- **Auditing**: Validates FLAC MD5 checksums and flags missing/corrupted metadata.
- **Organization**: Renames files based on metadata and isolates singles into a dedicated directory.

## Dependencies

Requires system packages for audio processing:

```bash
# Arch Linux
sudo pacman -S ffmpeg flac sox bpm-tools

# Debian / Ubuntu
sudo apt install ffmpeg flac sox bpm-tools

# macOS
brew install ffmpeg flac sox bpm-tools
```

## Build

Compile the portable binary via PyInstaller:

```bash
git clone https://github.com/dvmizew/Sonora.git
cd Sonora
pip install -r requirements.txt
pyinstaller --name sonora --onefile --clean src/sonora/cli/main.py

./dist/sonora --help
```

## Usage

### `tag`
Tags files concurrently.

```bash
sonora tag /path/to/album

# Dry run (preview only)
sonora tag /path/to/album --dry-run

# Force refresh (ignore MBID cache)
sonora tag /path/to/album --force

# With API keys and custom threads
sonora tag /path/to/album -w 8 --lastfm-key KEY --acoustid-key KEY --discogs-token TOKEN
```

### `audit`
Scans for corrupt files and missing tags.

```bash
sonora audit /path/to/library

# Export JSON
sonora audit /path/to/library --json report.json

# Enable spectral cutoff check (slow)
sonora audit /path/to/library --spectral
```

### `rename` & `organize`
```bash
# Rename files & sync LRC headers
sonora rename /path/to/album

# Move 1-2 track folders to a Singles directory
sonora organize /path/to/music --target-singles /path/to/Singles
```