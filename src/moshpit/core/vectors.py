"""Motion vector manipulation using FFglitch.

This module uses ffglitch tools (ffedit, ffgac) to extract, modify,
and re-apply motion vectors for advanced datamoshing effects.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

from moshpit.io.video import PreviewSettings, get_video_info

# Type alias for vector transform functions
VectorTransform = Callable[[np.ndarray, int, dict], np.ndarray]


def check_ffglitch_installed() -> bool:
    """Check if ffglitch tools are available."""
    try:
        subprocess.run(["ffedit", "-version"], capture_output=True, check=True)
        subprocess.run(["ffgac", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def extract_motion_vectors(input_path: Path | str, output_path: Path | str) -> Path:
    """Extract motion vectors from video using ffedit.

    Args:
        input_path: Path to input video
        output_path: Path for output JSON file with motion vectors

    Returns:
        Path to the motion vector JSON file
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Use ffedit to extract motion vectors
    cmd = [
        "ffedit",
        "-i",
        str(input_path),
        "-f",
        "mv:0",  # Extract motion vectors from stream 0
        "-e",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffedit failed: {result.stderr}")

    return output_path


def apply_motion_vectors(
    input_path: Path | str,
    vectors_path: Path | str,
    output_path: Path | str,
) -> Path:
    """Apply modified motion vectors to video using ffgac.

    Args:
        input_path: Original video
        vectors_path: JSON file with motion vectors
        output_path: Path for output video

    Returns:
        Path to the output video
    """
    input_path = Path(input_path)
    vectors_path = Path(vectors_path)
    output_path = Path(output_path)

    # Use ffgac to apply motion vectors
    cmd = [
        "ffgac",
        "-i",
        str(input_path),
        "-f",
        "mv:0",
        "-s",
        str(vectors_path),
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffgac failed: {result.stderr}")

    return output_path


def load_motion_vectors(path: Path | str) -> list[np.ndarray]:
    """Load motion vectors from JSON file.

    Args:
        path: Path to motion vector JSON file

    Returns:
        List of numpy arrays, one per frame
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    frames = []
    for frame_data in data.get("frames", []):
        mv = frame_data.get("mv", [])
        if mv:
            frames.append(np.array(mv))
        else:
            frames.append(np.array([]))

    return frames


def save_motion_vectors(frames: list[np.ndarray], path: Path | str) -> None:
    """Save motion vectors to JSON file.

    Args:
        frames: List of numpy arrays with motion vectors
        path: Output path for JSON file
    """
    path = Path(path)

    data = {"frames": [{"mv": frame.tolist() if len(frame) > 0 else []} for frame in frames]}

    with open(path, "w") as f:
        json.dump(data, f)


def apply_vector_transform(
    input_path: Path | str,
    output_path: Path | str,
    transform: VectorTransform | str,
    preview: PreviewSettings | None = None,
    **transform_kwargs,
) -> Path:
    """Apply a transform to motion vectors.

    Args:
        input_path: Path to input video
        output_path: Path for output video
        transform: Transform function or name of built-in transform
        preview: Preview settings for fast rendering
        **transform_kwargs: Arguments to pass to the transform function

    Returns:
        Path to the output video
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not check_ffglitch_installed():
        raise RuntimeError(
            "FFglitch tools (ffedit, ffgac) not found. Please install from https://ffglitch.org/"
        )

    # Get transform function
    if isinstance(transform, str):
        from moshpit.transforms import get_transform

        transform_fn = get_transform(transform)
    else:
        transform_fn = transform

    video_info = get_video_info(input_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Prepare input (apply preview scaling if needed)
        if preview:
            scaled_input = tmpdir / "scaled.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(input_path),
                    "-vf",
                    preview.scale_filter,
                    "-c:v",
                    "mpeg4",
                    "-q:v",
                    "2",
                    str(scaled_input),
                ],
                capture_output=True,
                check=True,
            )
            work_input = scaled_input
        else:
            work_input = input_path

        # Extract motion vectors
        vectors_path = tmpdir / "vectors.json"
        extract_motion_vectors(work_input, vectors_path)

        # Load, transform, and save
        frames = load_motion_vectors(vectors_path)

        context = {
            "width": video_info.width,
            "height": video_info.height,
            "fps": video_info.fps,
            "total_frames": len(frames),
            **transform_kwargs,
        }

        transformed = []
        for i, frame in enumerate(frames):
            if len(frame) > 0:
                transformed.append(transform_fn(frame, i, context))
            else:
                transformed.append(frame)

        # Save transformed vectors
        transformed_path = tmpdir / "transformed.json"
        save_motion_vectors(transformed, transformed_path)

        # Apply transformed vectors
        temp_output = tmpdir / "output.mp4"
        apply_motion_vectors(work_input, transformed_path, temp_output)

        # Encode final output
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(temp_output),
                "-c:v",
                "libx264",
                "-crf",
                str(preview.crf if preview else 18),
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ],
            capture_output=True,
            check=True,
        )

    return output_path


def apply_expression(
    input_path: Path | str,
    output_path: Path | str,
    expression: str,
    preview: PreviewSettings | None = None,
) -> Path:
    """Apply an inline expression to motion vectors.

    The expression has access to:
    - frame: numpy array of motion vectors
    - i: frame index
    - np: numpy module

    Example expressions:
    - "frame[:,:,0] *= 2"  # Double horizontal motion
    - "frame[:,:,1] = 0"   # Kill vertical motion
    - "frame += np.random.randn(*frame.shape) * 0.5"  # Add noise

    Args:
        input_path: Path to input video
        output_path: Path for output video
        expression: Python expression to apply
        preview: Preview settings

    Returns:
        Path to the output video
    """

    def expr_transform(frame: np.ndarray, i: int, context: dict) -> np.ndarray:
        # Create a copy to work with
        frame = frame.copy()
        # Execute expression in controlled namespace
        exec(expression, {"frame": frame, "i": i, "np": np, "context": context})
        return frame

    return apply_vector_transform(
        input_path,
        output_path,
        transform=expr_transform,
        preview=preview,
    )


def apply_script(
    input_path: Path | str,
    output_path: Path | str,
    script_path: Path | str,
    preview: PreviewSettings | None = None,
) -> Path:
    """Apply a custom Python script to motion vectors.

    The script must define a function:
        def mosh_frames(frames: list[np.ndarray], context: dict) -> list[np.ndarray]

    Args:
        input_path: Path to input video
        output_path: Path for output video
        script_path: Path to Python script
        preview: Preview settings

    Returns:
        Path to the output video
    """
    script_path = Path(script_path)

    # Load and execute script
    import importlib.util

    spec = importlib.util.spec_from_file_location("user_script", script_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "mosh_frames"):
        raise ValueError("Script must define 'mosh_frames(frames, context)' function")

    mosh_fn = module.mosh_frames

    # Wrap as per-frame transform
    class ScriptTransform:
        def __init__(self):
            self.all_frames = None
            self.transformed = None

        def __call__(self, frame: np.ndarray, i: int, context: dict) -> np.ndarray:
            # Collect all frames first, then transform all at once
            if self.all_frames is None:
                self.all_frames = []
            self.all_frames.append(frame)

            if i == context["total_frames"] - 1:
                # Last frame, run the full transform
                self.transformed = mosh_fn(self.all_frames, context)

            if self.transformed is not None:
                return self.transformed[i]
            return frame

    return apply_vector_transform(
        input_path,
        output_path,
        transform=ScriptTransform(),
        preview=preview,
    )
