"""AVI binary parsing utilities for datamoshing.

This module provides low-level AVI file manipulation for datamoshing effects.
AVI files use a RIFF structure with chunks that can be identified and modified.

Frame types in MPEG4 AVI:
- I-frame (Intra-coded): Complete image, starts with 0x000001B6 followed by 00
- P-frame (Predicted): Delta from previous frame, starts with 0x000001B6 followed by 10
- B-frame (Bi-directional): We disable these during encoding with -bf 0
"""

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

# MPEG4 frame type markers
MPEG4_VOP_START = b"\x00\x00\x01\xb6"

# Frame type bits after VOP start code (bits 6-7 of the byte after 0xB6)
# 00 = I-frame, 01 = P-frame, 10 = B-frame, 11 = S-frame
FRAME_TYPE_I = 0b00
FRAME_TYPE_P = 0b01
FRAME_TYPE_B = 0b10


@dataclass
class AVIFrame:
    """Represents a single video frame in an AVI file."""

    offset: int  # Byte offset in the file
    size: int  # Frame size in bytes
    frame_type: str  # 'I', 'P', 'B', or 'unknown'
    index: int  # Frame number
    data: bytes = field(default=b"", repr=False)


@dataclass
class AVIIndex:
    """Index of frames in an AVI file."""

    frames: list[AVIFrame]
    movi_offset: int  # Offset of the 'movi' chunk

    @property
    def iframe_indices(self) -> list[int]:
        """Get indices of all I-frames."""
        return [f.index for f in self.frames if f.frame_type == "I"]

    @property
    def pframe_indices(self) -> list[int]:
        """Get indices of all P-frames."""
        return [f.index for f in self.frames if f.frame_type == "P"]


def detect_frame_type(data: bytes) -> str:
    """Detect the frame type from MPEG4 frame data.

    Args:
        data: Raw frame bytes

    Returns:
        'I', 'P', 'B', or 'unknown'
    """
    # Look for MPEG4 VOP (Video Object Plane) start code
    vop_pos = data.find(MPEG4_VOP_START)
    if vop_pos == -1:
        return "unknown"

    # The frame type is in bits 6-7 of the byte after the start code
    if vop_pos + 4 >= len(data):
        return "unknown"

    type_byte = data[vop_pos + 4]
    frame_type_bits = (type_byte >> 6) & 0b11

    if frame_type_bits == FRAME_TYPE_I:
        return "I"
    elif frame_type_bits == FRAME_TYPE_P:
        return "P"
    elif frame_type_bits == FRAME_TYPE_B:
        return "B"
    else:
        return "unknown"


def read_chunk_header(f: BinaryIO) -> tuple[bytes, int]:
    """Read a RIFF chunk header.

    Args:
        f: File handle

    Returns:
        Tuple of (fourcc, size)
    """
    fourcc = f.read(4)
    if len(fourcc) < 4:
        return b"", 0
    size_bytes = f.read(4)
    if len(size_bytes) < 4:
        return fourcc, 0
    size = struct.unpack("<I", size_bytes)[0]
    return fourcc, size


def parse_avi_index(path: Path | str) -> AVIIndex:
    """Parse an AVI file and build a frame index.

    This function scans the AVI file structure to locate all video frames
    and determine their types (I-frame or P-frame).

    Args:
        path: Path to AVI file

    Returns:
        AVIIndex with frame information
    """
    path = Path(path)
    frames = []
    movi_offset = 0

    with open(path, "rb") as f:
        # Check RIFF header
        riff = f.read(4)
        if riff != b"RIFF":
            raise ValueError(f"Not a valid RIFF file: {path}")

        file_size = struct.unpack("<I", f.read(4))[0]
        avi_magic = f.read(4)
        if avi_magic != b"AVI ":
            raise ValueError(f"Not a valid AVI file: {path}")

        frame_index = 0

        # Scan for movi chunk which contains the actual frame data
        while f.tell() < file_size:
            chunk_start = f.tell()
            fourcc, size = read_chunk_header(f)

            if not fourcc:
                break

            if fourcc == b"LIST":
                list_type = f.read(4)
                if list_type == b"movi":
                    movi_offset = chunk_start
                    # Parse frames within movi
                    movi_end = chunk_start + size + 8
                    while f.tell() < movi_end - 8:
                        frame_start = f.tell()
                        frame_fourcc, frame_size = read_chunk_header(f)

                        if not frame_fourcc or frame_size == 0:
                            break

                        # Video frames are marked with ##dc or ##db
                        if frame_fourcc[2:4] in (b"dc", b"db"):
                            # Read frame data to detect type
                            data = f.read(frame_size)
                            frame_type = detect_frame_type(data)

                            frames.append(
                                AVIFrame(
                                    offset=frame_start,
                                    size=frame_size,
                                    frame_type=frame_type,
                                    index=frame_index,
                                    data=data,
                                )
                            )
                            frame_index += 1

                            # Align to word boundary
                            if frame_size % 2 == 1:
                                f.read(1)
                        else:
                            # Skip non-video chunks (like audio)
                            f.seek(frame_size, 1)
                            if frame_size % 2 == 1:
                                f.read(1)
                else:
                    # Skip other LIST chunks
                    f.seek(size - 4, 1)
            else:
                # Skip other chunks
                f.seek(size, 1)

            # Align to word boundary
            if size % 2 == 1:
                f.seek(1, 1)

    return AVIIndex(frames=frames, movi_offset=movi_offset)


def read_avi_data(path: Path | str) -> tuple[bytes, AVIIndex]:
    """Read entire AVI file and parse index.

    Args:
        path: Path to AVI file

    Returns:
        Tuple of (file data, frame index)
    """
    path = Path(path)
    with open(path, "rb") as f:
        data = f.read()

    index = parse_avi_index(path)
    return data, index


def write_avi_data(path: Path | str, data: bytes) -> None:
    """Write AVI data to file.

    Args:
        path: Path for output file
        data: AVI file data
    """
    path = Path(path)
    with open(path, "wb") as f:
        f.write(data)


def remove_iframes_binary(
    data: bytes,
    index: AVIIndex,
    start_frame: int | None = None,
    end_frame: int | None = None,
    keep_first: bool = True,
) -> bytes:
    """Remove I-frames from AVI data by replacing them with P-frames.

    This is the core datamoshing operation. When I-frames are removed,
    the video decoder continues applying P-frame deltas to stale data,
    creating the characteristic "pixel bleeding" effect.

    Args:
        data: AVI file data
        index: Frame index
        start_frame: First frame to process (default: 0)
        end_frame: Last frame to process (default: last frame)
        keep_first: Keep the very first I-frame for initialization

    Returns:
        Modified AVI data
    """
    data = bytearray(data)
    start_frame = start_frame or 0
    end_frame = end_frame or len(index.frames)

    # Find reference P-frame to use as replacement
    reference_pframe = None
    for frame in index.frames:
        if frame.frame_type == "P" and frame.data:
            reference_pframe = frame
            break

    if reference_pframe is None:
        # No P-frames found, can't datamosh
        return bytes(data)

    first_iframe_found = False

    for frame in index.frames:
        if frame.frame_type != "I":
            continue

        if frame.index < start_frame or frame.index >= end_frame:
            continue

        if keep_first and not first_iframe_found:
            first_iframe_found = True
            continue

        # Replace I-frame data with P-frame data
        # We need to copy the P-frame data into the I-frame's location
        # This requires careful handling of size differences

        frame_data_offset = frame.offset + 8  # Skip chunk header

        # Create minimal P-frame by copying reference P-frame's VOP header
        # and zeroing out the rest (creates "hold" effect)
        vop_pos = reference_pframe.data.find(MPEG4_VOP_START)
        if vop_pos >= 0:
            # Copy enough of the P-frame header to make it valid
            # but minimize the actual content
            new_frame = bytearray(frame.size)

            # Copy the P-frame VOP start and type bits
            header_len = min(len(reference_pframe.data), frame.size)
            new_frame[:header_len] = reference_pframe.data[:header_len]

            # Ensure it's marked as P-frame
            if vop_pos + 4 < len(new_frame):
                # Set frame type bits to P-frame (01)
                new_frame[vop_pos + 4] = (new_frame[vop_pos + 4] & 0b00111111) | 0b01000000

            data[frame_data_offset : frame_data_offset + frame.size] = new_frame

    return bytes(data)


def duplicate_pframes_binary(
    data: bytes,
    index: AVIIndex,
    dup_count: int = 1,
    start_frame: int | None = None,
) -> bytes:
    """Duplicate P-frames to create "bloom" effect.

    This effect amplifies motion by repeating P-frames, causing
    the accumulated motion deltas to build up dramatically.

    Note: This creates a new AVI file with additional frames, so the
    output will be longer than the input.

    Args:
        data: AVI file data
        index: Frame index
        dup_count: Number of times to duplicate each P-frame
        start_frame: Start duplicating from this frame

    Returns:
        Modified AVI data (may be larger than original)
    """
    if dup_count < 1:
        return data

    start_frame = start_frame or 0

    # For P-frame duplication, we need to rebuild the movi chunk
    # This is a complex operation, so we'll use a simpler approach:
    # Create a new frame list and rebuild

    output = bytearray()

    # Copy header up to movi chunk
    output.extend(data[: index.movi_offset])

    # We'll build a new movi LIST chunk
    new_frames = []

    for frame in index.frames:
        if frame.index < start_frame or frame.frame_type != "P":
            new_frames.append(frame.data)
        else:
            # Duplicate this P-frame
            for _ in range(dup_count + 1):
                new_frames.append(frame.data)

    # Build new movi chunk
    movi_data = bytearray()
    for i, frame_data in enumerate(new_frames):
        # Write frame header (00dc for video)
        movi_data.extend(b"00dc")
        movi_data.extend(struct.pack("<I", len(frame_data)))
        movi_data.extend(frame_data)
        # Pad to word boundary
        if len(frame_data) % 2 == 1:
            movi_data.append(0)

    # Write movi LIST header
    output.extend(b"LIST")
    output.extend(struct.pack("<I", len(movi_data) + 4))
    output.extend(b"movi")
    output.extend(movi_data)

    # Update RIFF size in header
    riff_size = len(output) - 8
    output[4:8] = struct.pack("<I", riff_size)

    return bytes(output)


def get_frame_data(data: bytes, frame: AVIFrame) -> bytes:
    """Extract raw frame data from AVI.

    Args:
        data: AVI file data
        frame: Frame to extract

    Returns:
        Raw frame bytes
    """
    start = frame.offset + 8  # Skip chunk header
    return data[start : start + frame.size]


def set_frame_data(data: bytearray, frame: AVIFrame, new_data: bytes) -> None:
    """Replace frame data in AVI.

    Args:
        data: AVI file data (mutable)
        frame: Frame to replace
        new_data: New frame data (must be same size or smaller)
    """
    if len(new_data) > frame.size:
        raise ValueError(f"New frame data too large: {len(new_data)} > {frame.size}")

    start = frame.offset + 8
    # Pad with zeros if new data is smaller
    padded = new_data.ljust(frame.size, b"\x00")
    data[start : start + frame.size] = padded
