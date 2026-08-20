class SonoraError(Exception):
    """Base exception for all Sonora errors."""


class AudioProcessingError(SonoraError):
    """Raised when low-level audio processing fails (FFmpeg, Librosa, Mutagen)."""


class MetadataError(SonoraError):
    """Raised when reading or writing track metadata fails."""


class APIServiceError(SonoraError):
    """Raised when external API client operations encounter an error."""

