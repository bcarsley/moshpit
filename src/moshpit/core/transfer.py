"""Motion transfer between videos.

Transfer motion vectors from one video to another, allowing the motion
characteristics of one clip to be applied to different imagery.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from moshpit.core.vectors import (
    apply_motion_vectors,
    check_ffglitch_installed,
    extract_motion_vectors,
    load_motion_vectors,
    save_motion_vectors,
)
from moshpit.io.video import PreviewSettings, get_video_info


def transfer_motion(
    source_path: Path | str,
    target_path: Path | str,
    output_path: Path | str,
    blend: float = 1.0,
    preview: PreviewSettings | None = None,
) -> Path:
    """Transfer motion vectors from source video to target video.

    Takes the motion characteristics from the source video and applies
    them to the target video, creating an effect where the target imagery
    moves according to the source's motion.

    Args:
        source_path: Video to extract motion from
        target_path: Video to apply motion to
        output_path: Path for output video
        blend: Blend factor (0=target motion, 1=source motion)
        preview: Preview settings for fast rendering

    Returns:
        Path to the output video
    """
    source_path = Path(source_path)
    target_path = Path(target_path)
    output_path = Path(output_path)

    if not check_ffglitch_installed():
        raise RuntimeError(
            "FFglitch tools (ffedit, ffgac) not found. Please install from https://ffglitch.org/"
        )

    _source_info = get_video_info(source_path)
    _target_info = get_video_info(target_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Prepare videos (apply preview scaling if needed)
        if preview:
            scaled_source = tmpdir / "source_scaled.mp4"
            scaled_target = tmpdir / "target_scaled.mp4"

            for inp, out in [(source_path, scaled_source), (target_path, scaled_target)]:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(inp),
                        "-vf",
                        preview.scale_filter,
                        "-c:v",
                        "mpeg4",
                        "-q:v",
                        "2",
                        str(out),
                    ],
                    capture_output=True,
                    check=True,
                )

            work_source = scaled_source
            work_target = scaled_target
        else:
            work_source = source_path
            work_target = target_path

        # Extract motion vectors from both
        source_vectors_path = tmpdir / "source_vectors.json"
        target_vectors_path = tmpdir / "target_vectors.json"

        extract_motion_vectors(work_source, source_vectors_path)
        extract_motion_vectors(work_target, target_vectors_path)

        # Load vectors
        source_frames = load_motion_vectors(source_vectors_path)
        target_frames = load_motion_vectors(target_vectors_path)

        # Blend/transfer vectors
        min_frames = min(len(source_frames), len(target_frames))
        transferred = []

        for i in range(min_frames):
            src = source_frames[i]
            tgt = target_frames[i]

            if len(src) == 0 or len(tgt) == 0:
                # Keep target vectors if either is empty
                transferred.append(tgt if len(tgt) > 0 else src)
                continue

            # Handle different sizes by resizing source to match target
            if src.shape != tgt.shape:
                # Simple nearest-neighbor resize
                src_resized = resize_vectors(src, tgt.shape)
            else:
                src_resized = src

            # Blend vectors
            blended = (1 - blend) * tgt + blend * src_resized
            transferred.append(blended.astype(tgt.dtype))

        # Save transferred vectors
        transferred_path = tmpdir / "transferred.json"
        save_motion_vectors(transferred, transferred_path)

        # Apply to target video
        temp_output = tmpdir / "output.mp4"
        apply_motion_vectors(work_target, transferred_path, temp_output)

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


def resize_vectors(vectors: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resize motion vector array to match target shape.

    Uses simple nearest-neighbor interpolation.

    Args:
        vectors: Source motion vectors
        target_shape: Target array shape

    Returns:
        Resized motion vectors
    """
    if vectors.shape == target_shape:
        return vectors

    # Get dimensions
    src_h, src_w = vectors.shape[:2]
    tgt_h, tgt_w = target_shape[:2]

    # Create output array
    result = np.zeros(target_shape, dtype=vectors.dtype)

    # Nearest-neighbor interpolation
    for y in range(tgt_h):
        for x in range(tgt_w):
            src_y = int(y * src_h / tgt_h)
            src_x = int(x * src_w / tgt_w)
            result[y, x] = vectors[src_y, src_x]

    # Scale vector magnitudes by size ratio
    scale_x = tgt_w / src_w
    scale_y = tgt_h / src_h

    if len(target_shape) > 2:
        result[..., 0] *= scale_x  # X component
        result[..., 1] *= scale_y  # Y component

    return result
