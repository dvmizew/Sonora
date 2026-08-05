"""
Domain data models for Sonora tracks, albums, and audit reports.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrackInfo:
    """Represents a single audio track and its associated metadata."""

    file_path: Path
    artist: str = "Unknown Artist"
    title: str = "Unknown Title"
    album: str = "Unknown Album"
    album_artist: str | None = None
    track_number: int | None = None
    disc_number: int | None = 1
    date: str | None = None
    genre: str | None = None
    isrc: str | None = None
    bpm: float | None = None
    replaygain_track_gain: float | None = None
    replaygain_track_peak: float | None = None
    lyrics: str | None = None
    synced_lyrics: str | None = None
    musicbrainz_trackid: str | None = None
    musicbrainz_albumid: str | None = None
    acoustid_fingerprint: str | None = None
    sample_rate: int | None = None
    bitrate: int | None = None
    channels: int | None = None
    is_lossless: bool = True

    def to_dict(self) -> dict[str, str]:
        """Convert metadata attributes to a dictionary."""
        return {
            "artist": self.artist,
            "title": self.title,
            "album": self.album,
            "album_artist": self.album_artist or self.artist,
            "track_number": str(self.track_number) if self.track_number else "",
            "date": self.date or "",
            "genre": self.genre or "",
            "isrc": self.isrc or "",
            "bpm": f"{self.bpm:.1f}" if self.bpm else "",
        }


@dataclass
class AlbumInfo:
    """Represents a collection of audio tracks belonging to an album."""

    title: str
    artist: str
    tracks: list[TrackInfo] = field(default_factory=list)
    year: str | None = None
    genre: str | None = None
    cover_art_path: Path | None = None
    musicbrainz_albumid: str | None = None

    @property
    def track_count(self) -> int:
        return len(self.tracks)


@dataclass
class AuditReport:
    """Represents the audit results for a track or directory."""

    file_path: Path | None = None
    total_files: int = 0
    corrupt_files: int = 0
    missing_metadata: int = 0
    missing_lrc: int = 0
    issues: dict[str, list[str]] = field(default_factory=dict)
    is_valid: bool = True
    missing_tags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    is_fake_lossless: bool = False
    md5_verified: bool = False
