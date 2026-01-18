"""I-frame removal for datamoshing.

I-frame removal is the classic datamoshing technique. By removing keyframes
(I-frames), the video decoder continues applying motion deltas (P-frames)
to outdated image data, creating the "pixel bleeding" effect.
"""

import tempfile
from pathlib import Path

from moshpit.core.avi import (
    parse_avi_index,
    read_avi_data,
    remove_iframes_binary,
    write_avi_data,
)
from moshpit.io.video import (
    PreviewSettings,
    convert_from_avi,
    convert_to_avi,
)


def remove_iframes(
    input_path: Path | str,
    output_path: Path | str,
    start_frame: int | None = None,
    end_frame: int | None = None,
    keep_first: bool = True,
    preview: PreviewSettings | None = None,
    audio: bool = True,
) -> Path:
    """Remove I-frames from a video to create datamosh effect.

    This is the core datamoshing operation that creates the classic
    "pixel bleeding" effect where motion from one scene bleeds into
    the imagery of a previous scene.

    Args:
        input_path: Path to input video
        output_path: Path for output video
        start_frame: First frame to process (removes I-frames from here)
        end_frame: Last frame to process
        keep_first: Keep the very first I-frame for initialization
        preview: Preview settings for fast rendering
        audio: Whether to preserve audio from original

    Returns:
        Path to the output video
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Step 1: Convert to AVI with MPEG4 codec (no B-frames)
        avi_path = tmpdir / "input.avi"
        convert_to_avi(input_path, avi_path, preview=preview)

        # Step 2: Parse AVI and remove I-frames
        data, index = read_avi_data(avi_path)

        if not index.iframe_indices:
            raise ValueError("No I-frames found in video")

        if not index.pframe_indices:
            raise ValueError("No P-frames found - cannot datamosh")

        # Remove I-frames
        moshed_data = remove_iframes_binary(
            data,
            index,
            start_frame=start_frame,
            end_frame=end_frame,
            keep_first=keep_first,
        )

        # Step 3: Write modified AVI
        moshed_avi = tmpdir / "moshed.avi"
        write_avi_data(moshed_avi, moshed_data)

        # Step 4: Convert back to output format
        convert_from_avi(
            moshed_avi,
            output_path,
            audio_source=input_path if audio else None,
        )

    return output_path


def get_iframe_info(input_path: Path | str) -> dict:
    """Get information about I-frames in a video.

    Useful for planning where to remove I-frames.

    Args:
        input_path: Path to video

    Returns:
        Dictionary with I-frame information
    """
    input_path = Path(input_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        avi_path = tmpdir / "temp.avi"
        convert_to_avi(input_path, avi_path)
        index = parse_avi_index(avi_path)

    return {
        "total_frames": len(index.frames),
        "iframe_count": len(index.iframe_indices),
        "pframe_count": len(index.pframe_indices),
        "iframe_positions": index.iframe_indices,
        "iframe_percentage": len(index.iframe_indices) / len(index.frames) * 100
        if index.frames
        else 0,
    }
