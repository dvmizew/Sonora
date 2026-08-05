"""
Domain data models for Sonora tracks, albums, and audit reports.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class TrackInfo:
    """Represents a single audio track and its associated metadata."""

    file_path: Path
    artist: str = "Unknown Artist"
    title: str = "Unknown Title"
    album: str = "Unknown Album"
    album_artist: Optional[str] = None
    track_number: Optional[int] = None
    disc_number: Optional[int] = 1
    date: Optional[str] = None
    genre: Optional[str] = None
    isrc: Optional[str] = None
    bpm: Optional[float] = None
    replaygain_track_gain: Optional[float] = None
    replaygain_track_peak: Optional[float] = None
    lyrics: Optional[str] = None
    synced_lyrics: Optional[str] = None
    musicbrainz_trackid: Optional[str] = None
    musicbrainz_albumid: Optional[str] = None
    acoustid_fingerprint: Optional[str] = None
    sample_rate: Optional[int] = None
    bitrate: Optional[int] = None
    channels: Optional[int] = None
    is_lossless: bool = True

    def to_dict(self) -> Dict[str, str]:
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
    tracks: List[TrackInfo] = field(default_factory=list)
    year: Optional[str] = None
    genre: Optional[str] = None
    cover_art_path: Optional[Path] = None
    musicbrainz_albumid: Optional[str] = None

    @property
    def track_count(self) -> int:
        return len(self.tracks)


@dataclass
class AuditReport:
    """Represents the audit results for a track or directory."""

    file_path: Path
    is_valid: bool = True
    missing_tags: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    is_fake_lossless: bool = False
    md5_verified: bool = False
