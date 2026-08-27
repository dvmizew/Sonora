# Sonora

Sonora is a CLI tool for music tagging, metadata enrichment, tag backup/restore, and file organization.

Supports FLAC, MP3, M4A, MP4, ALAC, OGG, OPUS, WAV, AIFF, WMA, APE, WV, and MPC.

---

## Features

- **Metadata & Tagging**: Multi-service tag lookup with AcoustID audio fingerprinting and MusicBrainz matching.
- **Audio Processing**: Calculates BPM and ReplayGain 2.0 (ITU-R BS.1770-4 / EBU R128) for tracks and albums.
- **Spectral Analysis**: Detects fake lossless files (e.g. MP3 transcodes upscaled to FLAC) via FFT spectrogram cutoff checks.
- **Artwork & Lyrics**: Downloads high-resolution cover art (up to 3000x3000px) and synchronized `.lrc` lyrics.
- **Library Tools**: Validates audio file integrity, standardizes file names (`NN - Title.ext`), organizes singles, and backs up tags to JSON / `.json.gz`.

### Supported Metadata Services

| Service | Data Retrieved |
| :--- | :--- |
| **MusicBrainz** | Track & release metadata, MBIDs, release groups, composers, lyricists, producers |
| **Discogs** | Record labels, catalog numbers, release year, barcodes, media formats |
| **Apple Music / iTunes** | Cover art (up to 3000px), primary genre, parental advisory |
| **Deezer** | ISRC, BPM, track gain, cover art, featured artists, producers |
| **Genius** | Song descriptions, featured artists, producers |
| **Last.fm** | Top tags (genres/moods), listener counts, play counts |
| **TheAudioDB** | Artist avatars (`artist.jpg`), banners (`banner.jpg`), music video URLs |
| **AcoustID** | Fingerprint-based MBID lookup using Chromaprint (`fpcalc`) |
| **Syncedlyrics** | Synced `.lrc` and plain lyrics from Musixmatch, Lrclib, and NetEase |
| **Cover Art Archive** | Front cover art for MusicBrainz releases |

---

## Requirements

- Python 3.10+
- System tools: `ffmpeg`, `flac`, and `chromaprint` (`fpcalc`)

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

```bash
git clone https://github.com/dvmizew/Sonora.git
cd Sonora
pip install -e .
```

---

## Configuration

Optional API keys can be provided via environment variables or a `.env` file:

```env
LASTFM_API_KEY=your_lastfm_key
ACOUSTID_API_KEY=your_acoustid_key
DISCOGS_TOKEN=your_discogs_token
GENIUS_API_TOKEN=your_genius_token
```

MusicBrainz, Cover Art Archive, iTunes, and Deezer work without an API key.

---

## Commands

### `sonora tag`

Tags audio files with metadata, cover art, lyrics, BPM, and ReplayGain.

```bash
# Tag a folder
sonora tag /path/to/music

# Dry run (no files modified)
sonora tag /path/to/music --dry-run

# Run with 8 threads and skip cache
sonora tag /path/to/music -t 8 --force

# Save report to JSON
sonora tag /path/to/music --json tag_report.json
```

**Options:**
- `path`: Directory to tag (required).
- `-t, --threads N`: Number of parallel threads (default: `4`).
- `--force`: Ignore disk cache and retag from scratch.
- `--dry-run`: Preview actions without modifying files.
- `--no-bpm`: Skip BPM calculation.
- `--no-replaygain`: Skip ReplayGain calculation.
- `--no-lyrics`: Skip `.lrc` lyrics download.
- `--no-art`: Skip cover art download.
- `--json PATH`: Save report to a JSON file.
- `--lastfm-key KEY`: Last.fm API key.
- `--acoustid-key KEY`: AcoustID API key.
- `--discogs-token TOKEN`: Discogs user token.
- `--genius-token TOKEN`: Genius API token.

---

### `sonora check`

Checks audio files for corruption, missing tags, bracket clutter in titles, and fake lossless audio.

```bash
# Basic library check
sonora check /path/to/library

# Check with 16 threads and run spectral analysis
sonora check /path/to/library -t 16 --spectral

# Export issues to JSON
sonora check /path/to/library --json report.json
```

**Options:**
- `path`: Directory to check (required).
- `-t, --threads N`: Number of parallel threads (default: `8`).
- `--spectral`: Check for fake lossless files via FFT spectral cutoff analysis.
- `--json PATH`: Save check results to a JSON file.

---

### `sonora rename`

Renames audio files to `NN - Title.ext` (or `Disc-NN - Title.ext` for multi-disc albums) and updates `.lrc` metadata headers.

```bash
sonora rename /path/to/album
sonora rename /path/to/album --dry-run
```

**Options:**
- `path`: Directory to rename (required).
- `--dry-run`: Preview new file names without renaming.

---

### `sonora organize`

Moves standalone 1–2 track single releases into `Singles/Artist/` and removes duplicates already present in full albums.

```bash
sonora organize /path/to/music
sonora organize /path/to/music --target-singles /path/to/Singles
sonora organize /path/to/music --dry-run
```

**Options:**
- `path`: Source directory (required).
- `--target-singles PATH`: Destination for singles (default: `<path>/Singles`).
- `--dry-run`: Preview file movements.

---

### `sonora backup`

Backs up all audio tags in a library to a JSON or compressed `.json.gz` file.

```bash
sonora backup /path/to/library
sonora backup /path/to/library --out my_backup.json.gz
```

**Options:**
- `path`: Music directory to back up (required).
- `--out PATH`: Output path (`.json` or `.json.gz`).

---

### `sonora restore`

Restores audio tags from a JSON or `.json.gz` backup file.

```bash
sonora restore my_backup.json.gz
```

**Options:**
- `backup_file`: Path to the backup file (required).