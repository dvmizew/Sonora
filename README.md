# Sonora

Sonora is a CLI tool for music tagging, library auditing, metadata enrichment, tag backup/restore, and file organization.

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
  - Grayscale correlation check (NumPy) to avoid overwriting custom covers unless visually similar (≥ 0.82 correlation).
- **Audio Engines**:
  - Tempo calculation (STFT onset envelope autocorrelation via SciPy).
  - ReplayGain 2.0 Album Mode (`metaflac`).
  - 16kHz spectral cutoff detection (`sox`) to flag fake-lossless files.
  - Bit-exact audio stream MD5 checksum verification (`flac -t`).
- **Library Tools**:
  - Library auditor for corrupt FLACs, missing tags, bracket clutter, and missing lyrics.
  - File renamer and folder structure standardizer (`NN - Artist - Title.ext`).
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
  - `sox` (spectral cutoff analysis)

### System Package Installation

```bash
# Arch Linux
sudo pacman -S python ffmpeg flac sox chromaprint

# Debian / Ubuntu
sudo apt install python3 ffmpeg flac sox chromaprint

# macOS
brew install python ffmpeg flac sox chromaprint
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

### `tag`
Tag an album or directory with metadata, artwork, lyrics, BPM, and ReplayGain:

```bash
sonora tag /path/to/album
sonora tag /path/to/album --dry-run
sonora tag /path/to/album --force
sonora tag /path/to/album -w 8
```

### `audit`
Audit library integrity, missing metadata, corrupt FLACs, and fake-lossless audio:

```bash
sonora audit /path/to/library
sonora audit /path/to/library --spectral
sonora audit /path/to/library --json report.json
```

### `rename`
Rename audio files (`NN - Artist - Title.ext`), update album folder names, and keep `.lrc` lyrics in sync:

```bash
sonora rename /path/to/album
```

### `organize`
Move standalone single releases to `Singles/Primary Artist/`, deduplicate against full albums, and clean empty directories:

```bash
sonora organize /path/to/music
```

### `backup` & `restore`
Backup audio metadata tags to a JSON file and restore them:

```bash
sonora backup /path/to/library -o backup.json
sonora restore backup.json
```