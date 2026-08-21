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
    musicbrainz_releasegroupid: str | None = None
    musicbrainz_artistid: str | None = None
    musicbrainz_workid: str | None = None
    acoustid_id: str | None = None
    acoustid_fingerprint: str | None = None
    discogs_release_id: str | None = None
    itunes_trackid: str | None = None
    itunes_collectionid: str | None = None
    itunes_artistid: str | None = None
    album_artist_sort: str | None = None
    artist_sort: str | None = None
    total_tracks: int | None = None
    total_discs: int | None = None
    release_type: str | None = None
    release_status: str | None = None
    release_country: str | None = None
    label: str | None = None
    catalog_number: str | None = None
    barcode: str | None = None
    media: str | None = None
    comment: str | None = None
    advisory: str | None = None
    original_date: str | None = None
    cuesheet: str | None = None
    sample_rate: int | None = None
    bitrate: int | None = None
    channels: int | None = None
    is_lossless: bool = True
    art_width: int | None = None
    art_height: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert metadata attributes to a complete dictionary representation."""
        return {
            "file_path": str(self.file_path),
            "file_name": self.file_path.name,
            "artist": self.artist,
            "title": self.title,
            "album": self.album,
            "album_artist": self.album_artist or self.artist,
            "track_number": self.track_number,
            "total_tracks": self.total_tracks,
            "disc_number": self.disc_number,
            "total_discs": self.total_discs,
            "date": self.date,
            "genre": self.genre,
            "isrc": self.isrc,
            "bpm": self.bpm,
            "replaygain_track_gain": self.replaygain_track_gain,
            "replaygain_track_peak": self.replaygain_track_peak,
            "musicbrainz_trackid": self.musicbrainz_trackid,
            "musicbrainz_albumid": self.musicbrainz_albumid,
            "musicbrainz_releasegroupid": self.musicbrainz_releasegroupid,
            "musicbrainz_artistid": self.musicbrainz_artistid,
            "discogs_release_id": self.discogs_release_id,
            "release_type": self.release_type,
            "release_country": self.release_country,
            "label": self.label,
            "barcode": self.barcode,
            "is_lossless": self.is_lossless,
            "sample_rate": self.sample_rate,
            "bitrate": self.bitrate,
        }


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
