"""P-frame duplication for datamoshing.

P-frame duplication amplifies motion by repeating predicted frames,
causing motion deltas to accumulate and create a "bloom" or "smear" effect.
"""

import tempfile
from pathlib import Path

from moshpit.core.avi import (
    duplicate_pframes_binary,
    read_avi_data,
    write_avi_data,
)
from moshpit.io.video import (
    PreviewSettings,
    convert_from_avi,
    convert_to_avi,
)


def duplicate_pframes(
    input_path: Path | str,
    output_path: Path | str,
    dup_count: int = 3,
    start_frame: int | None = None,
    preview: PreviewSettings | None = None,
    audio: bool = True,
) -> Path:
    """Duplicate P-frames to create bloom/smear effect.

    This effect amplifies motion by repeating P-frames. Each P-frame
    contains motion delta information, so repeating them causes the
    motion to "build up" dramatically.

    Note: This increases video duration since frames are duplicated.

    Args:
        input_path: Path to input video
        output_path: Path for output video
        dup_count: Number of times to duplicate each P-frame (1-10)
        start_frame: Start duplicating from this frame
        preview: Preview settings for fast rendering
        audio: Whether to preserve audio (will be out of sync due to duration change)

    Returns:
        Path to the output video
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Clamp dup_count to reasonable range
    dup_count = max(1, min(10, dup_count))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Step 1: Convert to AVI
        avi_path = tmpdir / "input.avi"
        convert_to_avi(input_path, avi_path, preview=preview)

        # Step 2: Parse and duplicate P-frames
        data, index = read_avi_data(avi_path)

        if not index.pframe_indices:
            raise ValueError("No P-frames found - cannot duplicate")

        moshed_data = duplicate_pframes_binary(
            data,
            index,
            dup_count=dup_count,
            start_frame=start_frame,
        )

        # Step 3: Write modified AVI
        moshed_avi = tmpdir / "moshed.avi"
        write_avi_data(moshed_avi, moshed_data)

        # Step 4: Convert back to output format
        # Note: Audio will be out of sync since we changed duration
        convert_from_avi(
            moshed_avi,
            output_path,
            audio_source=input_path if audio else None,
        )

    return output_path
