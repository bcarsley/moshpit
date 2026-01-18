"""Format detection and conversion utilities."""

from pathlib import Path

# Supported input formats
SUPPORTED_FORMATS = {
    ".mp4": "h264",
    ".mov": "h264",
    ".avi": "mpeg4",
    ".mkv": "h264",
    ".webm": "vp9",
}

# Output codec mappings
OUTPUT_CODECS = {
    "mp4": "libx264",
    "mov": "libx264",
    "avi": "mpeg4",
    "mkv": "libx264",
    "webm": "libvpx-vp9",
}


def detect_format(path: Path | str) -> str | None:
    """Detect video format from file extension.

    Args:
        path: Path to video file

    Returns:
        Format string (e.g., 'mp4', 'avi') or None if unsupported
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in SUPPORTED_FORMATS:
        return ext[1:]  # Remove leading dot
    return None


def get_codec_for_format(format_str: str) -> str:
    """Get the appropriate codec for an output format.

    Args:
        format_str: Output format (e.g., 'mp4', 'avi')

    Returns:
        Codec string for ffmpeg
    """
    return OUTPUT_CODECS.get(format_str, "libx264")


def is_supported(path: Path | str) -> bool:
    """Check if a video format is supported.

    Args:
        path: Path to video file

    Returns:
        True if format is supported
    """
    return detect_format(path) is not None
