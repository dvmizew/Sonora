# Sonora

Sonora is a CLI tool for music tagging, metadata updates, tag backup/restore, and file organization.

Supports FLAC, MP3, M4A, MP4, ALAC, OGG, OPUS, WAV, AIFF, WMA, APE, WV, and MPC.

---

## Features

- **Metadata & Tagging**: Multi-service tag lookup with AcoustID audio fingerprinting and MusicBrainz matching.
- **Audio Processing**: Calculates BPM, Musical Key & Camelot wheel tonality (`INITIALKEY`), and ReplayGain 2.0 (ITU-R BS.1770-4 / EBU R128) for tracks and albums.
- **High-Performance Architecture**: Compound album release queries, offline string distance matching, thread-safe persistent cache, and smart library state tracking.
- **Spectral Analysis**: Checks for spectral signs of lossy transcoding using FFT cutoff analysis.
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

Run Sonora without installing Python packages. The external tools listed in Requirements must still be available:

```bash
chmod +x sonora-linux-x86_64
./sonora-linux-x86_64 tag /path/to/music
```

### Option 2: From Source (Python 3.10+)

Source installation requires Git because ShazamIO is pinned to upstream commit `8afb07c0ff31c663daa895ebc1d5db17a74fc4ed`. Its unreleased 0.9.0 package supports `shazamio-core==1.2.0`, which provides the native wheel used on Python 3.14. The released 0.8.1 SDK requires an older native core without an official Python 3.14 wheel. Replace the source pin with a released version once a compatible release is available.

```bash
git clone https://github.com/dvmizew/Sonora.git
cd Sonora
pip install -e .
```

### Option 3: Compile Standalone Executable (PyInstaller)

Build in an isolated environment using the checked-in `uv.lock`. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) version `0.13.0`,
then run these commands from a clean checkout:

```bash
uv sync --locked --group build --python 3.12.15 --no-editable
uv run --no-sync python -W error -m unittest discover -s tests
uv run --no-sync pyinstaller sonora-linux-x86_64.spec --clean -y

# Output executable is located at:
./dist/sonora-linux-x86_64
```

Release CI uses this same dependency lock on Linux and Windows. The spec keeps
all Sonora features and excludes only pyacoustid's unused `audioread` decoder;
Sonora fingerprints through `fpcalc`. UPX is disabled so builds do not depend on
whether a compression tool happens to be installed. Platform libraries and the
Python toolchain still affect executable bytes; the lock guarantees dependency
selection, not bit-for-bit identical binaries.

When rebuilding an existing working tree, remove the generated `build/lib/`
directory and run `uv cache clean sonora` before syncing. Setuptools can otherwise
carry deleted modules from an earlier build into the new wheel.

For development, use `uv sync --locked --extra dev`. Ruff includes the Pyflakes
checks; mypy is the primary CI type checker, with Pyright retained for editor or
manual complementary checks. To update dependencies intentionally, run
`uv lock --upgrade`, review the lock diff, and rerun the supported Python CI matrix and
packaged-feature checks before releasing.

---

## Configuration (API Keys)

Sonora works out of the box without keys. To unlock full metadata and higher rate limits, copy `.env.example` to `.env` in your music folder, working directory, or `~/.config/sonora/.env`:

```env
DISCOGS_TOKEN="your_token"          # https://www.discogs.com/settings/developers -> Generate token
ACOUSTID_API_KEY="your_api_key"     # https://acoustid.org/new-application -> Register application
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
title_match_threshold = 85.0
short_title_match_threshold = 95.0
short_title_length = 8
acoustid_confidence_threshold = 0.8
acoustid_match_margin = 0.02
allow_unverified_lyrics = false
import_genius_descriptions = false
key_confidence_threshold = 0.2
bpm_search_min = 60.0
bpm_search_max = 200.0
fanart_api_key = "your_api_key"
fanart_client_key = "your_client_key"
enable_shazam = true
```

Title matching retains strict scores above 85, or above 95 for titles shorter than eight characters; configuration can strengthen these safeguards. Exact normalized matches remain eligible. BPM estimates are no longer automatically doubled or halved into a preferred tempo range. Set both `bpm_octave_min` and `bpm_octave_max` to opt into octave adjustment; the interval must span at least one octave. Other application policies and structural patterns are listed in [`defaults.json`](src/sonora/defaults.json).

Genius descriptions are community prose and can contain unrelated or vandalized text. They are not copied into comments by default; set `import_genius_descriptions = true` to opt in. Existing comments are preserved. Catalog agreement alone never clears an existing Explicit advisory, because providers may agree on a different clean edition.

Independent metadata providers receive separate copies of one resolved recording identity. Their proposals merge in a fixed order: conflicting recording identities or ISRCs prevent tag and asset writes, and conflicting individual fields preserve existing values. JSON reports include `metadata_conflicts` keyed by the disputed field. Tracks with unresolved conflicts are not recorded as complete. Provider agreement is evidence, not a correctness guarantee.

Tagging uses bounded worker pools for tracks and provider queries, and prefetches at most two album contexts while retaining album finalization order. `--threads` controls each main pool; total worker count can therefore exceed that number. API quotas and identical cache-backed lookups are coordinated between Sonora processes using the same cache directory. The portable `filelock` dependency supplies native process locks; coordination files remain outside DiskCache's managed directory. Confirmed catalog misses expire after 30 minutes, successful metadata normally after 28 days, and transient network failures are not stored as misses. Fingerprints persist with file path, size, modification time, change time and decoder-setting invalidation. `--ignore-cache` deliberately bypasses cache reads; `--force` does not.

Identification can abstain when recordings or releases are ambiguous. Stored recording IDs must agree with meaningful artist/title tags and available duration evidence; an ID alone cannot overwrite a conflicting identity. AcoustID uses whole-file duration, validates candidate metadata, and rejects near ties within `acoustid_match_margin`. Cached fingerprint evidence is validated again against current metadata and settings.

Lyrics sources that return only text cannot establish recording identity. They are disabled by default; `allow_unverified_lyrics = true` explicitly enables those fallbacks with a warning. LRCLIB responses still require matching artist/title and duration. These safeguards reduce coverage when evidence is insufficient; they do not guarantee universal identification accuracy.

ShazamIO and syncedlyrics are community integrations, not official Shazam or Musixmatch service APIs. Their versions are locked and provider-specific failures are isolated. Supplied Musixmatch credentials belong to each SDK client; Sonora does not write the SDK token cache or fabricate credential expiry. Logs retain dependency warnings and errors rather than globally silencing them.

Matching penalties and ranking weights can also be configured with `version_equivalence_penalty`, `version_mismatch_penalty`, `remix_modifier_mismatch_penalty`, and `title_score_weight`. Single-edition comparisons use `single_duration_tolerance_seconds` (2 seconds by default) and `single_identifier_title_threshold`. The duration limit favors withholding uncertain matches when recordings differ in length; it does not verify clean versus explicit editions. JSON/TOML boolean settings require actual booleans, and structural lists require strings; invalid settings produce an error.

| Key | Where to get it | What it unlocks |
| :--- | :--- | :--- |
| **`DISCOGS_TOKEN`** | [Discogs Developers](https://www.discogs.com/settings/developers) | 60 req/min limit, labels, catalog numbers, barcodes |
| **`ACOUSTID_API_KEY`** | [AcoustID application registration](https://acoustid.org/new-application) | Audio fingerprinting with an application/client key; user submission keys do not authorize lookups |
| **`GENIUS_API_TOKEN`** | [Genius API Clients](https://genius.com/api-clients) | Song descriptions and background stories in comment tag |
| **`LASTFM_API_KEY`** | [Last.fm API](https://www.last.fm/api) | Dynamic mood & style tags |
| **`FANART_API_KEY`** | [Fanart.tv API](https://fanart.tv/get-an-api-key) | CD art (`cdart.png`), transparent artist logos (`logo.png`), 4K fanart |
| **`FANART_CLIENT_KEY`** | [Fanart.tv VIP](https://fanart.tv) | VIP access to newly added and exclusive high-res artwork |
| **`MUSIXMATCH_TOKEN`** | Desktop app / web dump | Word-by-word synced lyrics (`<mm:ss.xx>`) |

*MusicBrainz, Cover Art Archive, Apple Music/iTunes, TheAudioDB, Deezer, **Shazam**, and **PyCountry** require no keys.*

---

## Commands

### `sonora tag`

Tags audio files and albums automatically with metadata, cover art, lyrics, BPM, musical key, and ReplayGain.

```bash
# Tag an album folder
sonora tag /path/to/music

# Ultra-fast metadata, lyrics & artwork tagging (skipping heavy audio DSP)
sonora tag /path/to/music --no-dsp

# Dry run (simulate actions without modifying files)
sonora tag /path/to/music --dry-run

# Run with 8 threads and force re-processing
sonora tag /path/to/music -t 8 --force

# Force remote re-fetching bypassing the local API cache
sonora tag /path/to/music --ignore-cache

# Save tagging report to JSON
sonora tag /path/to/music --json tag_report.json
```

**Options:**
- `path`: Directory containing audio files (required).
- `-t, --threads N`: Number of parallel threads (default: `4`).
- `--no-dsp`: Disable all heavy audio DSP analysis (BPM, Musical Key, and ReplayGain) simultaneously for ultra-fast metadata tagging.
- `--force`: Force reprocessing even if tracks were already marked as valid/tagged in the library state database.
- `--ignore-cache`: Bypass local API metadata cache and force re-fetching from remote services.
- `--dry-run`: Preview actions without modifying files on disk.
- `--no-bpm`: Skip BPM calculation.
- `--no-key`: Skip musical key and Camelot tonality detection.
- `--no-replaygain`: Skip ReplayGain loudness normalization tags.
- `--no-lyrics`: Skip synchronized (`.lrc`) lyrics download.
- `--no-art`: Skip album and artist artwork download.
- `--no-shazam`: Skip Shazam acoustic recognition fallback.
- `-j, --json PATH`: Save tagging report to a JSON file.
- `--lastfm-key KEY`: Last.fm API key for genre and mood lookup.
- `--acoustid-key KEY`: AcoustID API key for acoustic fingerprinting.
- `--discogs-token TOKEN`: Discogs user token for catalog, label, and credit data.
- `--genius-token TOKEN`: Genius API token for song descriptions.
- `--fanart-key KEY`: Fanart.tv project API key for logos, CD art, and fanart.
- `--fanart-client-key KEY`: Fanart.tv personal VIP client key.
- `--musixmatch-token TOKEN`: Musixmatch user token for word-synced lyrics.

---

### `sonora check`

Checks audio integrity, missing tags, bracket clutter in titles, and spectral signs of lossy transcoding.

Stored MusicBrainz, ISRC and iTunes identifiers are compared with available cached
recording responses. Conflicting artists, titles and durations are reported for
review without fetching online or rewriting tags. Missing cache evidence cannot
verify an identity. The check also reports invalid positions, filename/track-number
differences, and missing trailing positions when a disc's stored totals agree.

The command exits with status `1` when corrupt audio is found, `130` when interrupted,
and `0` after a completed check without corruption. Optional metadata warnings remain
available in the JSON report. Zero or negative BPM tags are treated as unknown and
can be recalculated with `sonora bpm` without `--force`.

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
- `--spectral`: Check for spectral signs of lossy transcoding using FFT cutoff analysis.
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

Moves standalone releases into `<destination>/<artist credit> - <release>/01 - <title>.<extension>`. The default destination is `<path>/Singles`, independent of library size. Featured artists, co-billed credits, and distinct recording identifiers preserve separate editions; confirmed duplicates are quarantined in `.duplicates` with audio quality taken into account.

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

Backs up exact TagLib text values and complex properties (including embedded pictures) to a canonical `sonora_backup_v2` JSON or compressed `.json.gz` archive. Relative paths are preserved; ambiguous basename matching is never used.

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
sonora restore my_backup.json.gz --target /path/to/relocated/library
```

**Options:**
- `backup_file`: Path to a canonical `sonora_backup_v2` archive (required).
- `--target PATH`: Restore to a relocated library root, preserving relative paths.

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

### Console output

Commands show progress, failures and a final summary by default. Use `--verbose`
for per-file changes, provider decisions and diagnostic messages:

```sh
sonora tag /path/to/music --dry-run --verbose
sonora --verbose check /path/to/music --json report.json
```

The option works before or after a command. `-v` still prints the version.
JSON reports always include full results, regardless of verbosity. Metadata
conflicts remain visible in the final summary; uncertain fields are preserved.

### `sonora lyrics`

Fetches lyrics, upgrades plain text when a matching synchronized version is available, and cleans existing LRC files. Recognized web markup and footer debris are removed while timing markers, LRC metadata and lyric words are preserved.

The command reports conflicting lyrics for review instead of silently replacing different words. It preserves existing synchronized lyrics when only a lower-quality result is available. Matching provider metadata does not independently prove that every lyric word or timestamp is correct.

Verified cached results are reused, including plain or line-synced results from an earlier enhanced-lyrics lookup. After three consecutive temporary provider failures, network lookups stop for that run; local cleanup and valid cached results continue. The JSON report includes each file's source, proposed action and review reason, and marks incomplete lookups separately from unavailable lyrics. An incomplete run exits with status `1` and can be retried later.

```bash
sonora lyrics /path/to/music
sonora lyrics /path/to/music --dry-run --json lyrics-preview.json
sonora lyrics /path/to/music --force
```

**Options:**
- `path`: Directory containing audio files (required).
- `-t, --threads N`: Number of parallel threads (default: `4`).
- `--force`: Refetch lyrics even if `.lrc` file already exists.
- `--dry-run`: Preview cleanup and replacements without changing audio or LRC files.
- `--json PATH`: Save changes, review cases and failures to a report.
- `--musixmatch-token TOKEN`: Musixmatch user token for enhanced synced lyrics.

---

### `sonora cache` & `sonora clear-cache`

Inspects and manages Sonora cache layers and persistent scan state. Complies with the XDG Base Directory specification (`$XDG_CACHE_HOME/sonora/` or `~/.cache/sonora/`).

- **API Cache**: `$XDG_CACHE_HOME/sonora/api-msgpack-v1/` (isolated SQLite index and MessagePack metadata/binary-asset shards)
- **Library State**: `$XDG_CACHE_HOME/sonora/library_state.db` (Persistent SQLite tracker for processed files and tags)

API metadata uses MessagePack with an explicit tuple extension for catalog indexes and artwork pairs. Cache reads never deserialize Python objects. The current format starts with an empty `api-msgpack-v1` directory; previous transient caches are not read or migrated, and `library_state.db` remains separate. MessagePack is an explicit dependency because it preserves binary assets and tuple-keyed indexes while removing executable-object deserialization.

```bash
# Display cache overview
sonora cache

# Display detailed cache statistics table
sonora cache stats
sonora cache stats --json

# Clear transient API cache and memory caches (safely preserves library scan history)
sonora cache clear
sonora clear-cache

# Preview cache clearing without touching files
sonora cache clear --dry-run
sonora cache clear --dry-run --json

# Clear specific cache layers
sonora cache clear --api      # Clear only API metadata cache
sonora cache clear --state    # Clear only library scan state database
sonora cache clear --memory   # Clear in-memory metadata caches

# Complete reset: clear all cache layers and library state
sonora cache clear --all

# Purge: completely delete cache files and directories from disk
sonora cache clear --all --purge
```

**Options for `cache stats`:**
- `--json`: Output cache statistics in machine-readable JSON format.

**Options for `cache clear` / `clear-cache`:**
- `--api`: Clear API response disk cache.
- `--state`: Clear persistent library tracking state database.
- `--memory`: Clear in-memory metadata and normalization caches.
- `-a, --all`: Target all cache layers (API metadata, library state, and in-memory caches).
- `-p, --purge`: Delete SQLite database files and disk cache directory completely.
- `--dry-run`: Simulate cache clearing without modifying or deleting files.
- `--json`: Output clearing results in machine-readable JSON format.
