"""
Custom exception hierarchy for Sonora.
"""


class SonoraError(Exception):
    """Base exception for all Sonora errors."""
    pass


class AudioProcessingError(SonoraError):
    """Raised when low-level audio processing fails (FFmpeg, Librosa, Mutagen)."""
    pass


class MetadataError(SonoraError):
    """Raised when reading or writing track metadata fails."""
    pass


class APIServiceError(SonoraError):
    """Raised when external API client operations encounter an error."""
    pass


class ValidationError(SonoraError):
    """Raised when track or file integrity audit fails."""
    pass
