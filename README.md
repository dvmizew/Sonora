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
| **Shazam** | Acoustic recognition (identifies un-tagged/corrupted audio directly from sound waveform) |
| **Fanart.tv** | Audiophile artwork: transparent CD disc art (`cdart.png`), artist logos (`logo.png`), 4K fanart |
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

### Option 1: Standalone Binary (Linux x86_64)

Run Sonora directly without installing Python or dependencies:

```bash
chmod +x sonora-linux-x86_64
./sonora-linux-x86_64 tag /path/to/music
```

### Option 2: From Source (Python 3.10+)

```bash
git clone https://github.com/dvmizew/Sonora.git
cd Sonora
pip install -e .
```

### Option 3: Compile Standalone Executable (PyInstaller)

To compile your own single-file portable binary for Linux x86_64:

```bash
# Install PyInstaller
pip install pyinstaller

# Build standalone binary with all collected audio/service hooks
pyinstaller sonora-linux-x86_64.spec --clean -y

# Output executable is located at:
./dist/sonora-linux-x86_64
```

---

## Configuration (API Keys)

Sonora works out of the box without keys. To unlock full metadata and higher rate limits, copy `.env.example` to `.env` in your music folder, working directory, or `~/.config/sonora/.env`:

```env
DISCOGS_TOKEN="your_token"          # https://www.discogs.com/settings/developers -> Generate token
ACOUSTID_API_KEY="your_api_key"     # https://acoustid.org/api-key -> Get API key
GENIUS_API_TOKEN="your_token"       # https://genius.com/api-clients -> Create client -> Generate Access Token
LASTFM_API_KEY="your_api_key"       # https://www.lastfm.com/api/account/create -> Create API account
FANART_API_KEY="your_api_key"       # https://fanart.tv/get-an-api-key -> Get Personal API Key
FANART_CLIENT_KEY="your_client_key" # (Optional) Personal VIP client key for Fanart.tv
MUSIXMATCH_TOKEN="your_token"       # (Optional) Desktop app/web token for word-synced lyrics
```

You can also use a configuration file at `~/.config/sonora/config.toml`:

```toml
[sonora]
artist_match_threshold = 85.0
album_match_threshold = 75.0
fanart_api_key = "your_api_key"
fanart_client_key = "your_client_key"
enable_shazam = true
```

| Key | Where to get it | What it unlocks |
| :--- | :--- | :--- |
| **`DISCOGS_TOKEN`** | [Discogs Developers](https://www.discogs.com/settings/developers) | 60 req/min limit, labels, catalog numbers, barcodes |
| **`ACOUSTID_API_KEY`** | [AcoustID API](https://acoustid.org/api-key) | Audio fingerprinting (identifies tracks with 0 initial tags) |
| **`GENIUS_API_TOKEN`** | [Genius API Clients](https://genius.com/api-clients) | Song descriptions and background stories in comment tag |
| **`LASTFM_API_KEY`** | [Last.fm API](https://www.last.fm/api) | Dynamic mood & style tags |
| **`FANART_API_KEY`** | [Fanart.tv API](https://fanart.tv/get-an-api-key) | CD art (`cdart.png`), transparent artist logos (`logo.png`), 4K fanart |
| **`FANART_CLIENT_KEY`** | [Fanart.tv VIP](https://fanart.tv) | VIP access to newly added and exclusive high-res artwork |
| **`MUSIXMATCH_TOKEN`** | Desktop app / web dump | Word-by-word synced lyrics (`<mm:ss.xx>`) |

*MusicBrainz, Cover Art Archive, Apple Music/iTunes, TheAudioDB, Deezer, **Shazam**, and **PyCountry** require no keys.*

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
- `--no-key`: Skip musical key detection.
- `--no-replaygain`: Skip ReplayGain calculation.
- `--no-lyrics`: Skip `.lrc` lyrics download.
- `--no-art`: Skip cover art download.
- `--no-shazam`: Skip Shazam audio recognition fallback.
- `--json PATH`: Save report to a JSON file.
- `--lastfm-key KEY`: Last.fm API key.
- `--acoustid-key KEY`: AcoustID API key.
- `--discogs-token TOKEN`: Discogs user token.
- `--genius-token TOKEN`: Genius API token.
- `--fanart-key KEY`: Fanart.tv project API key.
- `--fanart-client-key KEY`: Fanart.tv VIP client key.
- `--musixmatch-token TOKEN`: Musixmatch user token for synced lyrics.

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

---

### `sonora bpm`

Calculates and embeds audio tempo (BPM) tags locally using spectral audio analysis (100% offline).

```bash
sonora bpm /path/to/music
sonora bpm /path/to/music --dry-run
```

**Options:**
- `path`: Directory containing audio files (required).
- `-t, --threads N`: Number of parallel threads (default: `4`).
- `--dry-run`: Preview calculated BPM without modifying files.

---

### `sonora key`

Detects and embeds musical key (`INITIALKEY`) and Camelot wheel notation locally (100% offline).

```bash
sonora key /path/to/music
sonora key /path/to/music --dry-run
```

**Options:**
- `path`: Directory containing audio files (required).
- `-t, --threads N`: Number of parallel threads (default: `4`).
- `--dry-run`: Preview detected keys without modifying files.

---

### `sonora replaygain`

Calculates and embeds ITU-R BS.1770 / EBU R128 ReplayGain 2.0 loudness normalization tags for tracks and albums.

```bash
sonora replaygain /path/to/album
sonora replaygain /path/to/album --dry-run
```

**Options:**
- `path`: Directory containing audio files (required).
- `-t, --threads N`: Number of parallel threads (default: `4`).
- `--dry-run`: Preview loudness gains without modifying files.

---

### `sonora normalize`

100% offline tag cleanup: removes bracket noise (e.g. `[Official Video]`, `(Explicit)`), normalizes artist/title, and computes BPM, Key, and ReplayGain locally without making network calls.

```bash
sonora normalize /path/to/music
sonora normalize /path/to/music --dry-run
```

**Options:**
- `path`: Directory containing audio files (required).
- `-t, --threads N`: Number of parallel threads (default: `4`).
- `--dry-run`: Preview normalization without modifying files.

---

### `sonora lyrics`

Fetches and saves synchronized `.lrc` lyrics files and embedded lyrics from syncedlyrics providers.

```bash
sonora lyrics /path/to/music
sonora lyrics /path/to/music --force
```

**Options:**
- `path`: Directory containing audio files (required).
- `-t, --threads N`: Number of parallel threads (default: `4`).
- `--force`: Refetch lyrics even if `.lrc` file already exists.
- `--musixmatch-token TOKEN`: Musixmatch user token for enhanced synced lyrics.

---

### `sonora cache` & `sonora clear-cache`

Inspects and manages Sonora cache layers and persistent scan state.

```bash
# Display cache statistics
sonora cache stats
sonora cache stats --all

# Clear transient API cache and memory caches
sonora clear-cache

# Purge all caches and SQLite state from disk
sonora clear-cache --all --purge
```

**Options for `clear-cache`:**
- `--api`: Clear API response disk cache.
- `--state`: Clear persistent library state database.
- `--memory`: Clear in-memory caches.
- `-a, --all`: Target all cache layers.
- `-p, --purge`: Delete SQLite database files and disk cache directory completely.
- `--dry-run`: Preview cache clearing without deleting files.
