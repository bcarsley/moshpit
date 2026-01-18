"""Video I/O utilities using ffmpeg."""

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PreviewSettings:
    """Settings for preview mode rendering."""

    resolution: str = "480"  # Height in pixels
    crf: int = 28  # Quality (higher = lower quality, faster)

    @property
    def scale_filter(self) -> str:
        """Get ffmpeg scale filter string."""
        return f"scale=-2:{self.resolution}"


@dataclass
class VideoInfo:
    """Information about a video file."""

    width: int
    height: int
    fps: float
    duration: float
    codec: str
    frame_count: int

    @classmethod
    def from_ffprobe(cls, data: dict) -> "VideoInfo":
        """Create VideoInfo from ffprobe JSON output."""
        video_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        if not video_stream:
            raise ValueError("No video stream found")

        # Parse frame rate (can be "30/1" or "29.97")
        fps_str = video_stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)

        # Get duration from format or stream
        duration = float(
            data.get("format", {}).get("duration", 0) or video_stream.get("duration", 0)
        )

        # Calculate frame count
        nb_frames = video_stream.get("nb_frames")
        if nb_frames:
            frame_count = int(nb_frames)
        else:
            frame_count = int(fps * duration)

        return cls(
            width=int(video_stream.get("width", 0)),
            height=int(video_stream.get("height", 0)),
            fps=fps,
            duration=duration,
            codec=video_stream.get("codec_name", "unknown"),
            frame_count=frame_count,
        )


def get_video_info(path: Path | str) -> VideoInfo:
    """Get information about a video file using ffprobe.

    Args:
        path: Path to video file

    Returns:
        VideoInfo object with video metadata
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    return VideoInfo.from_ffprobe(data)


def convert_to_avi(
    input_path: Path | str,
    output_path: Path | str | None = None,
    preview: PreviewSettings | None = None,
) -> Path:
    """Convert a video to AVI format for datamoshing.

    Uses MPEG4 codec with specific settings optimized for datamoshing:
    - No B-frames (only I and P frames)
    - High quality for better manipulation
    - Configurable GOP size

    Args:
        input_path: Path to input video
        output_path: Path for output AVI (optional, uses temp file if not provided)
        preview: Preview settings for fast rendering

    Returns:
        Path to the output AVI file
    """
    input_path = Path(input_path)

    if output_path is None:
        # Create temp file
        fd, temp_path = tempfile.mkstemp(suffix=".avi")
        output_path = Path(temp_path)
        import os

        os.close(fd)
    else:
        output_path = Path(output_path)

    # Build ffmpeg command
    cmd = ["ffmpeg", "-y", "-i", str(input_path)]

    # Add preview scaling if requested
    if preview:
        cmd.extend(["-vf", preview.scale_filter])

    # MPEG4 encoding settings optimized for datamoshing
    cmd.extend(
        [
            "-c:v",
            "mpeg4",
            "-q:v",
            "1",  # Highest quality
            "-bf",
            "0",  # No B-frames
            "-g",
            "300",  # Large GOP (more P-frames to work with)
            "-an",  # No audio for processing (we'll re-add later)
            str(output_path),
        ]
    )

    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def convert_from_avi(
    input_path: Path | str,
    output_path: Path | str,
    codec: str = "libx264",
    crf: int = 18,
    audio_source: Path | str | None = None,
) -> Path:
    """Convert a datamoshed AVI back to a standard format.

    Args:
        input_path: Path to datamoshed AVI
        output_path: Path for output video
        codec: Output codec (default: libx264)
        crf: Quality setting (lower = better quality)
        audio_source: Optional path to original video for audio track

    Returns:
        Path to the output video
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    cmd = ["ffmpeg", "-y", "-i", str(input_path)]

    # Add audio from original if specified
    if audio_source:
        audio_source = Path(audio_source)
        cmd.extend(["-i", str(audio_source)])
        cmd.extend(["-map", "0:v:0", "-map", "1:a:0?"])
        cmd.extend(["-c:a", "aac", "-b:a", "192k"])

    # Video encoding settings
    cmd.extend(
        [
            "-c:v",
            codec,
            "-crf",
            str(crf),
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",  # Compatibility
            str(output_path),
        ]
    )

    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def encode_video(
    input_path: Path | str,
    output_path: Path | str,
    codec: str = "libx264",
    crf: int = 18,
    preview: PreviewSettings | None = None,
    extra_args: list[str] | None = None,
) -> Path:
    """General-purpose video encoding.

    Args:
        input_path: Path to input video
        output_path: Path for output video
        codec: Output codec
        crf: Quality setting
        preview: Preview settings for fast rendering
        extra_args: Additional ffmpeg arguments

    Returns:
        Path to the output video
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    cmd = ["ffmpeg", "-y", "-i", str(input_path)]

    # Add preview scaling
    if preview:
        cmd.extend(["-vf", preview.scale_filter])
        crf = preview.crf

    # Encoding settings
    cmd.extend(
        [
            "-c:v",
            codec,
            "-crf",
            str(crf),
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
        ]
    )

    if extra_args:
        cmd.extend(extra_args)

    cmd.append(str(output_path))
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def extract_frames(
    input_path: Path | str,
    output_dir: Path | str,
    start_frame: int | None = None,
    end_frame: int | None = None,
    format: str = "png",
) -> list[Path]:
    """Extract frames from a video as images.

    Args:
        input_path: Path to input video
        output_dir: Directory for output frames
        start_frame: First frame to extract (optional)
        end_frame: Last frame to extract (optional)
        format: Image format (png, jpg)

    Returns:
        List of paths to extracted frames
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    info = get_video_info(input_path)

    cmd = ["ffmpeg", "-y", "-i", str(input_path)]

    # Add frame selection
    if start_frame is not None or end_frame is not None:
        start = start_frame or 0
        end = end_frame or info.frame_count
        # Use select filter for frame range
        cmd.extend(
            [
                "-vf",
                f"select='between(n\\,{start}\\,{end})'",
                "-vsync",
                "vfr",
            ]
        )

    output_pattern = output_dir / f"frame_%06d.{format}"
    cmd.append(str(output_pattern))

    subprocess.run(cmd, capture_output=True, check=True)

    # Return list of created files
    frames = sorted(output_dir.glob(f"frame_*.{format}"))
    return frames


def frames_to_video(
    frames_dir: Path | str,
    output_path: Path | str,
    fps: float,
    codec: str = "libx264",
    crf: int = 18,
    pattern: str = "frame_%06d.png",
) -> Path:
    """Create a video from a sequence of frames.

    Args:
        frames_dir: Directory containing frames
        output_path: Path for output video
        fps: Frame rate
        codec: Output codec
        crf: Quality setting
        pattern: Frame filename pattern

    Returns:
        Path to the output video
    """
    frames_dir = Path(frames_dir)
    output_path = Path(output_path)

    input_pattern = frames_dir / pattern

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(input_pattern),
        "-c:v",
        codec,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]

    subprocess.run(cmd, capture_output=True, check=True)
    return output_path
