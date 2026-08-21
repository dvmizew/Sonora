# Sonora

CLI tool for music tagging, library auditing, tag backup/restore, and file organization.

Supports **FLAC, MP3, M4A, MP4, ALAC, OGG, OPUS, WAV, AIFF, WMA, APE, WV, MPC**.

---

## ⚡ Features & APIs

- **APIs & Services**: MusicBrainz, AcoustID (Chromaprint), Discogs, Last.fm, Genius, iTunes HD Cover Art, TheAudioDB, syncedlyrics (Musixmatch, Lrclib, NetEase).
- **Symfonium & Navidrome Extended Tags**: `ARTISTSORT`, `ALBUMARTISTSORT`, `TOTALTRACKS`, `TOTALDISCS`, `RELEASETYPE`, `RELEASESTATUS`, `RELEASECOUNTRY`, `BARCODE`, `CATALOGNUMBER`, `LABEL`, `ORIGINALDATE`, `CUESHEET`.
- **Audio Engines**:
  - BPM detection (`bpm-tools` / `librosa`).
  - ReplayGain 2.0 Album Mode (`metaflac`).
  - 16kHz spectral cutoff analysis (`sox`).
  - Bit-exact audio stream MD5 verification (`flac -t`).
- **Cover Art Safeguard**: PIL Pearson correlation prevents replacing custom cover art unless visually matching (correlation ≥ 0.82).
- **Artist Artwork**: Downloads `artist.jpg` (thumb) and `banner.jpg` via TheAudioDB.
- **Tag Safety**: Memory-safe streaming JSON backup and restore (`backup` / `restore`).

---

## 🛠 Dependencies

Requires system packages for external audio tools:

```bash
# Arch Linux
sudo pacman -S ffmpeg flac sox bpm-tools

# Debian / Ubuntu
sudo apt install ffmpeg flac sox bpm-tools

# macOS
brew install ffmpeg flac sox bpm-tools
```

---

## 🚀 Installation

```bash
git clone https://github.com/dvmizew/Sonora.git
cd Sonora
pip install -e .
```

### Environment Variables (`.env`)

```env
LASTFM_API_KEY=your_lastfm_key
ACOUSTID_API_KEY=your_acoustid_key
DISCOGS_USER_TOKEN=your_discogs_token
GENIUS_API_TOKEN=your_genius_token
```

---

## 💻 Commands

### `tag`
Tags files with metadata, artwork, lyrics, BPM, ReplayGain, and extended tags.

```bash
sonora tag /path/to/album
sonora tag /path/to/album --dry-run
sonora tag /path/to/album --force
sonora tag /path/to/album -w 8
```

### `audit`
Checks library for corrupted FLACs, missing tags, bracket clutter, and fake lossless.

```bash
sonora audit /path/to/library
sonora audit /path/to/library --spectral
sonora audit /path/to/library --json report.json
```

### `rename`
Renames tracks (`NN - Artist - Title.ext`), syncs `.lrc` headers, and renames album folders based on consensus.

```bash
sonora rename /path/to/album
```

### `organize`
Moves 1-2 track single releases to `Singles/Primary Artist/`, deduplicates single tracks against albums, and cleans empty directories.

```bash
sonora organize /path/to/music
```

### `backup` & `restore`
Streaming JSON tag backup and restore.

```bash
sonora backup /path/to/library -o backup.json
sonora restore backup.json
```

---

## 🧪 Tests

```bash
python3 -m unittest discover -s tests
```