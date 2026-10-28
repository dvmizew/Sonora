from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Literal

MetadataSource = Literal[
    "MusicBrainz",
    "Discogs",
    "iTunes",
    "Deezer",
    "Last.fm",
    "Genius",
    "TheAudioDB",
    "Shazam",
    "AcoustID",
    "Tidal",
]


@dataclass(slots=True)
class TrackInfo:
    """Represents a single audio track and its associated metadata."""

    file_path: Path
    artist: str = "Unknown Artist"
    title: str = "Unknown Title"
    album: str = "Unknown Album"
    album_artist: str | None = None
    track_number: int | None = None
    disc_number: int | None = 1
    disc_subtitle: str | None = None
    date: str | None = None
    genre: str | None = None
    isrc: str | None = None
    bpm: float | None = None
    replaygain_track_gain: float | None = None
    replaygain_track_peak: float | None = None
    replaygain_album_gain: float | None = None
    replaygain_album_peak: float | None = None
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
    composer: str | None = None
    lyricist: str | None = None
    remixer: str | None = None
    initial_key: str | None = None
    copyright: str | None = None
    compilation: bool | None = None
    spotify_trackid: str | None = None
    musicbrainz_albumartistid: str | None = None
    discogs_artist_id: str | None = None
    language: str | None = None
    script: str | None = None
    mood: str | None = None
    style: str | None = None
    disambiguation: str | None = None
    rating: float | None = None
    featured_artists: str | None = None
    producers: str | None = None
    genius_song_id: str | None = None
    music_video_url: str | None = None
    sample_rate: int | None = None
    bitrate: int | None = None
    channels: int | None = None
    duration: float | None = None
    bits_per_sample: int | None = None
    is_lossless: bool = True
    art_width: int | None = None
    art_height: int | None = None
    is_alien: bool = False
    metadata_sources: dict[str, MetadataSource] = field(default_factory=dict)
    metadata_conflicts: dict[str, str] = field(default_factory=dict)
    _advisory_needs_normalization: bool = field(
        default=False, repr=False, compare=False
    )
    _numbering_needs_normalization: bool = field(
        default=False, repr=False, compare=False
    )
    _diff_descriptions: list[str] = field(
        default_factory=list, repr=False, compare=False
    )

    def copy(self) -> "TrackInfo":
        """Copy a track without sharing mutable provenance or proposed changes."""
        return replace(
            self,
            metadata_sources=dict(self.metadata_sources),
            metadata_conflicts=dict(self.metadata_conflicts),
            _diff_descriptions=list(self._diff_descriptions),
        )

    def to_dict(self) -> dict[str, object]:
        """Convert metadata attributes to a complete dictionary representation."""
        track_dict: dict[str, object] = {
            attribute.name: getattr(self, attribute.name)
            for attribute in fields(self)
            if not attribute.name.startswith("_")
        }
        track_dict["metadata_sources"] = dict(self.metadata_sources)
        track_dict["metadata_conflicts"] = dict(self.metadata_conflicts)
        track_dict["file_path"] = str(self.file_path)
        track_dict["file_name"] = self.file_path.name
        if not track_dict.get("album_artist"):
            track_dict["album_artist"] = self.album_artist or self.artist
        return track_dict


@dataclass
class CheckReport:
    """Represents the aggregate validation/check results for an audio directory."""

    total_files: int = 0
    corrupt_files: int = 0
    missing_metadata: int = 0
    missing_lrc: int = 0

    @property
    def files_with_issues(self) -> int:
        from sonora.core.constants import SUPPORTED_EXTS

        return sum(Path(path).suffix.lower() in SUPPORTED_EXTS for path in self.issues)

    issues: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class RenameReport:
    """Represents the results of a file and directory renaming run."""

    total_files: int = 0
    files_renamed: int = 0
    folders_renamed: int = 0
    lrc_synced: int = 0
    playlists_updated: int = 0
    unchanged_files: int = 0
    failed_files: int = 0


@dataclass
class NormalizeReport:
    """Tracks scanned and changes produced by offline normalization."""

    total_scanned: int = 0
    failed_files: int = 0
    modified_files: dict[str, list[str]] = field(default_factory=dict)

    @property
    def total_modified(self) -> int:
        return len(self.modified_files)

    @property
    def total_unchanged(self) -> int:
        return max(0, self.total_scanned - self.failed_files - self.total_modified)
