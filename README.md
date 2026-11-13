# Sonora

Sonora is a CLI tool for music library tagging, audio analysis, metadata normalization, and file organization.

Supports **FLAC, MP3, M4A, ALAC, OPUS, OGG, WAV, AIFF, WMA, APE, WV, and MPC**.

[![Release](https://img.shields.io/github/v/release/dvmizew/Sonora)](https://github.com/dvmizew/Sonora/releases)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## Features

- **Metadata Tagging**: Multi-service tag lookup across MusicBrainz, Discogs, Apple Music, Deezer, Genius, Shazam, and AcoustID fingerprinting.
- **Audio Analysis**: Offline calculation of BPM, Musical Key / Camelot tonality, and ITU-R BS.1770 / EBU R128 ReplayGain 2.0.
- **Transcode Detection**: FFT spectral cutoff analysis to identify lossy transcodes and upsampled files.
- **Artwork & Lyrics**: Downloads front cover art (up to 3000px) and synchronized (`.lrc`) lyrics with audio duration validation.
- **File Organization**: Renames files to standard patterns (`NN - Title.ext`), organizes singles, and isolates duplicates.
- **Backup & Restore**: Exports and restores tags and embedded artwork using JSON or `.json.gz` archives.

---

## Requirements

System tools required for audio decoding and fingerprinting:

- `ffmpeg`, `flac`, and `chromaprint` (`fpcalc`)

```bash
# Arch Linux
sudo pacman -S ffmpeg flac chromaprint

# Debian / Ubuntu
sudo apt install ffmpeg flac chromaprint

# Fedora
sudo dnf install ffmpeg flac chromaprint

# macOS
brew install ffmpeg flac chromaprint
```

---

## Installation

### Option 1: Standalone Binary

Pre-built executables for Linux and Windows are available on the [Releases](https://github.com/dvmizew/Sonora/releases) page:

```bash
# Linux
chmod +x sonora-linux-x86_64
./sonora-linux-x86_64 tag /path/to/music

# Windows (PowerShell / Command Prompt)
.\sonora-windows-x86_64.exe tag C:\path\to\music
```

### Option 2: From Source (Python 3.10+)

```bash
git clone https://github.com/dvmizew/Sonora.git
cd Sonora
pip install -e .
```

### Option 3: Compile with PyInstaller

```bash
uv sync --locked --group build --python 3.12 --no-editable
uv run --no-sync pyinstaller sonora-linux-x86_64.spec --clean -y

# Executable is generated in ./dist/
```

---

## Quick Start

```bash
# Tag an album with metadata, cover art, lyrics, and audio analysis
sonora tag /path/to/music

# Tag metadata only (skipping audio DSP analysis)
sonora tag /path/to/music --no-dsp

# Check library health and detect corrupt or transcoded files
sonora check /path/to/music --spectral

# Standardize track file names (e.g. "01 - Track Title.flac")
sonora rename /path/to/album

# Clean tags offline and calculate BPM, Key, and ReplayGain
sonora normalize /path/to/music

# Back up metadata and embedded artwork to an archive
sonora backup /path/to/library --out backup.json.gz
```

---

## Commands

### `sonora tag`

Tags audio files and albums with metadata, cover art, synchronized lyrics, BPM, musical key, and ReplayGain.

```bash
sonora tag /path/to/music [OPTIONS]
```

**Options:**
- `-t, --threads N`: Number of worker threads (default: `4`).
- `--dry-run`: Preview proposed tag changes without modifying files.
- `--force`: Reprocess files even if already marked as tagged.
- `--no-dsp`: Skip audio analysis (BPM, Key, ReplayGain) to tag metadata only.
- `--no-bpm` / `--no-key` / `--no-replaygain`: Skip specific audio analysis steps.
- `--no-art`: Skip downloading album and artist artwork.
- `--no-lyrics`: Skip downloading synchronized `.lrc` lyrics.
- `--no-shazam`: Skip Shazam acoustic waveform recognition fallback.
- `--ignore-cache`: Bypass local API cache and force remote fetching.
- `-j, --json PATH`: Save detailed tagging report to a JSON file.

---

### `sonora check`

Checks audio integrity, missing tags, title bracket clutter, and spectral signs of lossy transcoding.

```bash
sonora check /path/to/music [OPTIONS]
```

**Options:**
- `-t, --threads N`: Number of worker threads (default: `8`).
- `--spectral`: Check for spectral signs of lossy transcoding using FFT cutoff analysis.
- `-j, --json PATH`: Save check findings to a JSON file.

---

### `sonora rename`

Renames audio files to `NN - Title.ext` (or `Disc-NN - Title.ext` for multi-disc releases) and updates `.lrc` metadata headers.

```bash
sonora rename /path/to/album [OPTIONS]
```

**Options:**
- `-t, --threads N`: Number of worker threads (default: `4`).
- `--dry-run`: Preview file renames without modifying files on disk.
- `-j, --json PATH`: Save rename report to a JSON file.

---

### `sonora organize`

Moves standalone singles into `<destination>/<artist> - <release>/01 - <title>.<ext>` and moves duplicates into `.duplicates`.

```bash
sonora organize /path/to/music [OPTIONS]
```

**Options:**
- `--target-singles PATH`: Destination directory for singles (default: `<path>/Singles`).
- `-t, --threads N`: Number of worker threads (default: `4`).
- `--dry-run`: Preview movements without relocating files.
- `-j, --json PATH`: Save organization report to a JSON file.

---

### `sonora normalize`

Offline tag cleanup: removes bracket noise (e.g. `[Official Video]`, `(Explicit)`), normalizes artist/title, and computes BPM, Key, and ReplayGain locally without network requests.

```bash
sonora normalize /path/to/music [OPTIONS]
```

**Options:**
- `-t, --threads N`: Number of worker threads (default: `4`).
- `--no-dsp`: Skip audio analysis (BPM, Key, ReplayGain).
- `--force`: Force reprocessing of already compliant files.
- `--dry-run`: Preview normalization without modifying files.
- `-j, --json PATH`: Save normalization report to a JSON file.

---

### `sonora lyrics`

Fetches synchronized lyrics, cleans existing `.lrc` files, and upgrades plain text lyrics with audio duration validation.

```bash
sonora lyrics /path/to/music [OPTIONS]
```

**Options:**
- `-t, --threads N`: Number of worker threads (default: `4`).
- `--force`: Refresh existing synchronized lyrics.
- `--dry-run`: Preview changes without changing audio or `.lrc` files.
- `-j, --json PATH`: Save lyrics report to a JSON file.

---

### Audio Analysis: `bpm`, `key`, `replaygain`

Individual offline audio analysis tools:

```bash
sonora bpm /path/to/music         # Calculates and embeds BPM tempo tags
sonora key /path/to/music         # Detects and embeds musical key (INITIALKEY) & Camelot tonality
sonora replaygain /path/to/music  # Calculates ITU-R BS.1770 / EBU R128 track & album loudness
```

*All three commands accept `-t, --threads N`, `--force`, `--dry-run`, and `-j, --json PATH`.*

---

### `sonora backup` & `sonora restore`

Backs up and restores exact TagLib tags, custom properties, and embedded artwork snapshots:

```bash
# Create a backup archive
sonora backup /path/to/library --out my_backup.json.gz

# Restore tags from an archive (supports relocated library root)
sonora restore my_backup.json.gz --target /path/to/new/location
```

---

### `sonora cache`

Inspects and manages local cache layers and persistent scan state:

```bash
sonora cache                  # Overview of cache directories
sonora cache stats            # Cache statistics table
sonora cache clear            # Clear transient API response cache
sonora cache clear --all      # Reset all caches and library state
sonora cache clear --all --purge  # Delete cache files and databases from disk
```

---

### Global Options

- `--verbose`: Show per-file changes and provider decisions.
- `-v, --version`: Print Sonora version.
- `-h, --help`: Display help for Sonora or any subcommand.

---

## Configuration

Sonora works without configuration keys for basic tagging. To enable additional providers or customize matching thresholds, copy `.env.example` to `.env` or edit `~/.config/sonora/config.toml`:

### Environment Variables (`.env`)

```env
DISCOGS_TOKEN="your_token"          # https://www.discogs.com/settings/developers
ACOUSTID_API_KEY="your_api_key"     # https://acoustid.org/new-application
GENIUS_API_TOKEN="your_token"       # https://genius.com/api-clients
LASTFM_API_KEY="your_api_key"       # https://www.lastfm.com/api/account/create
FANART_API_KEY="your_api_key"       # https://fanart.tv/get-an-api-key
MUSIXMATCH_TOKEN="your_token"       # (Optional) Token for synced lyrics
```

### Config File (`~/.config/sonora/config.toml`)

```toml
[sonora]
artist_match_threshold = 85.0
album_match_threshold = 75.0
title_match_threshold = 85.0
short_title_match_threshold = 95.0
acoustid_confidence_threshold = 0.8
allow_unverified_lyrics = false
import_genius_descriptions = false
enable_shazam = true
```

---

## Supported Services

| Service | Data Retrieved | Authentication |
| :--- | :--- | :--- |
| **MusicBrainz** | Track & release metadata, MBIDs, credits, labels, releases | None |
| **Cover Art Archive** | High-resolution front release artwork | None |
| **Apple Music / iTunes** | High-resolution cover art (up to 3000px), genre, advisory | None |
| **Deezer** | ISRC, BPM, track gain, cover art, contributors | None |
| **Shazam** | Acoustic audio recognition directly from sound waveform | None |
| **TheAudioDB** | Artist avatars (`artist.jpg`), banners (`banner.jpg`) | None |
| **LRCLIB** | Synchronized `.lrc` lyrics with duration verification | None |
| **AcoustID** | Fingerprint-based MBID lookup using Chromaprint (`fpcalc`) | API Key |
| **Discogs** | Record labels, catalog numbers, release year, formats | Token |
| **Genius** | Song IDs, lyricists, writers, and song descriptions | Token |
| **Last.fm** | Top tags (genres/moods) and popularity counts | API Key |
| **Fanart.tv** | CD disc art (`cdart.png`), logos (`logo.png`), backgrounds | API Key |
