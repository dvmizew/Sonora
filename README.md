# Sonora

Sonora is a CLI tool for music tagging, metadata enrichment, tag backup/restore, and file organization.

Supports FLAC, MP3, M4A, MP4, ALAC, OGG, OPUS, WAV, AIFF, WMA, APE, WV, and MPC.

---

## Features

- **Metadata & Tagging**:
  - AcoustID (Chromaprint) audio fingerprinting and MusicBrainz lookup.
  - Secondary enrichment from Discogs (release date, label, catalog number, barcode), Last.fm (genres, listener stats), Deezer (ISRC, barcode, label, release date), and Genius (producer credits, featured artists, song descriptions).
  - Synchronized and plain lyrics (`.lrc`) via `syncedlyrics` (Musixmatch, Lrclib, NetEase).
  - Extended tags for Symfonium, Navidrome, and Plex (`ARTISTSORT`, `ALBUMARTISTSORT`, `RELEASETYPE`, `BARCODE`, `CATALOGNUMBER`, `LABEL`, `ORIGINALDATE`, `ISRC`).
- **Cover Art Engine**:
  - Multi-source fallback: MusicBrainz Cover Art Archive -> iTunes (up to 3000x3000px) -> Deezer (1000x1000px) -> TheAudioDB (`artist.jpg` and `banner.jpg`).
  - Perceptual hashing (`pHash`) check via `imagehash` with EXIF transposition to prevent overwriting custom covers unless visually matched.
- **Audio Engines**:
  - Tempo calculation (STFT onset envelope autocorrelation via SciPy).
  - ReplayGain 2.0 Album Mode (`metaflac`).
  - 16kHz spectral cutoff detection (SciPy FFT spectrogram) to flag fake-lossless files.
  - Bit-exact audio stream MD5 checksum verification (`flac -t`).
- **Library Tools**:
  - Library checker for corrupt FLACs, missing tags, bracket clutter, and missing lyrics.
  - File renamer and folder structure standardizer (`NN - Title.ext` or `Disc-NN - Title.ext`).
  - Single track organizer (moves 1-2 track releases to `Singles/` and deduplicates against albums).
  - Streaming JSON tag backup and restore.
  - SQLite disk caching (`~/.cache/sonora`) with 30-day TTL to minimize API requests.

---

## Requirements

- **Python**: 3.10 or higher
- **System Audio Tools**:
  - `flac` (includes `metaflac` for ReplayGain 2.0 and checksum validation)
  - `ffmpeg` (audio decoding and metadata extraction)
  - `chromaprint` (provides `fpcalc` for AcoustID audio fingerprinting)

### System Package Installation

```bash
# Arch Linux
sudo pacman -S python ffmpeg flac chromaprint

# Debian / Ubuntu
sudo apt install python3 ffmpeg flac chromaprint

# macOS
brew install python ffmpeg flac chromaprint
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

Optional API keys can be set via environment variables or a `.env` file in your working directory:

```env
LASTFM_API_KEY=your_lastfm_key
ACOUSTID_API_KEY=your_acoustid_key
DISCOGS_USER_TOKEN=your_discogs_token
GENIUS_API_TOKEN=your_genius_token
```

MusicBrainz, Cover Art Archive, iTunes, and Deezer lookups work automatically without API keys.

---

## Usage

### Global Options

- `-v, --version`: Show program version.
- `--dry-run`: Simulate operations without modifying files or directories.

---

### `sonora tag`
Tag an album or directory automatically with metadata, artwork, lyrics, BPM, and ReplayGain:

```bash
sonora tag /path/to/music
sonora tag /path/to/music --dry-run
sonora tag /path/to/music --force
sonora tag /path/to/music -t 8
sonora tag /path/to/music --json tag_report.json
```

**Options:**
- `path`: Directory containing audio files to tag (required).
- `-t, --threads N`: Number of parallel worker threads (default: `4`).
- `--force`: Force retagging by ignoring disk cache and existing MBIDs.
- `--no-bpm`: Disable BPM calculation.
- `--no-replaygain`: Disable ReplayGain 2.0 calculation.
- `--no-lyrics`: Disable `.lrc` lyrics fetching.
- `--no-art`: Disable cover art downloading.
- `--json PATH`: Output path to save JSON report.
- `--lastfm-key KEY`: Last.fm API key (overrides `LASTFM_API_KEY` env).
- `--acoustid-key KEY`: AcoustID API key (overrides `ACOUSTID_API_KEY` env).
- `--discogs-token TOKEN`: Discogs personal token (overrides `DISCOGS_TOKEN` env).
- `--genius-token TOKEN`: Genius API token (overrides `GENIUS_API_TOKEN` env).

---

### `sonora check`
Check music library for FLAC integrity, bracket corruption, missing tags, and fake-lossless audio:

```bash
sonora check /path/to/library
sonora check /path/to/library -t 16
sonora check /path/to/library --spectral
sonora check /path/to/library --json check_report.json
```

**Options:**
- `path`: Directory containing music library to check (required).
- `-t, --threads N`: Number of parallel worker threads (default: `8`).
- `--spectral`: Enable deep spectral cutoff analysis to flag fake-lossless audio (MP3 upscaled to FLAC).
- `--json PATH`: Output path to save check JSON report.

---

### `sonora rename`
Standardize file names (`NN - Artist - Title.ext` or `Disc-NN - Title.ext`), rename album directories, and synchronize `.lrc` metadata headers:

```bash
sonora rename /path/to/album
sonora rename /path/to/library --dry-run
```

**Options:**
- `path`: Directory containing audio files to rename (required).

---

### `sonora organize`
Organize standalone single releases into `Singles/Primary Artist/`, deduplicate against full albums, and clean empty directories:

```bash
sonora organize /path/to/music
sonora organize /path/to/music --target-singles /path/to/Singles
```

**Options:**
- `path`: Source music directory (required).
- `--target-singles PATH`: Destination directory for single tracks (default: `<path>/Singles`).

---

### `sonora backup`
Create a high-speed, streaming JSON backup of audio tags across your library:

```bash
sonora backup /path/to/library
sonora backup /path/to/library --out my_backup.json
```

**Options:**
- `path`: Music directory to back up (required).
- `--out PATH`: Output JSON backup file path (default: `backup_YYYY-MM-DD_HH-MM-SS.json`).

---

### `sonora restore`
Restore audio tags from a streaming JSON backup file:

```bash
sonora restore my_backup.json
```

**Options:**
- `backup_file`: Path to JSON backup file (required).