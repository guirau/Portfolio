"""Helper functions for testing oma_recipecropper."""

import mmap
import struct
from io import BytesIO


def has_valid_photoshop_path(file_bytes: bytes) -> bool:
    """
    Check if an image file contains a valid Photoshop clipping path.

    Args:
        file_bytes: Image file data as bytes

    Returns:
        bool: True if a valid path is found, False otherwise
    """
    try:
        with BytesIO(file_bytes) as bio:
            with mmap.mmap(bio.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                # Look for 8BIM markers which indicate Photoshop resources
                offset = 0
                while offset < len(mm):
                    next_offset = mm.find(b"8BIM", offset)
                    if next_offset == -1:
                        break

                    offset = next_offset

                    # Skip if we don't have enough data
                    if offset + 6 >= len(mm):
                        offset += 4
                        continue

                    # Extract resource ID
                    resource_id = struct.unpack(">H", mm[offset + 4 : offset + 6])[0]

                    # Check if it's a valid path resource ID (2000-2997) or clipping path (2999)
                    if (2000 <= resource_id <= 2997) or resource_id == 2999:
                        # Get name length
                        name_len = mm[offset + 6]

                        # Name length is padded to even length
                        name_padding = 1 if name_len % 2 == 0 else 2

                        # Get data size
                        data_size_offset = offset + 6 + 1 + name_len + name_padding

                        # Skip if we don't have enough data
                        if data_size_offset + 4 >= len(mm):
                            offset += 4
                            continue

                        data_size = struct.unpack(
                            ">I", mm[data_size_offset : data_size_offset + 4]
                        )[0]

                        # Get data block
                        data_offset = data_size_offset + 4

                        # Skip if we don't have enough data
                        if data_offset + data_size > len(mm):
                            offset += 4
                            continue

                        # If data size is a multiple of 26 (path record size), likely a valid path
                        if data_size > 0 and data_size % 26 == 0:
                            # Check first selector is valid (0-8)
                            first_selector = struct.unpack(
                                ">H", mm[data_offset : data_offset + 2]
                            )[0]
                            if 0 <= first_selector <= 8:
                                return True

                    # Move past the current 8BIM marker
                    offset += 4

        return False
    except Exception:
        return False


def get_image_dimensions(image_bytes: bytes) -> tuple[int, int] | None:
    """
    Get the dimensions of an image from its bytes.

    Args:
        image_bytes: Image file data as bytes

    Returns:
        tuple: (width, height) or None if dimensions can't be determined
    """
    try:
        with BytesIO(image_bytes) as bio:
            # Check if it's a JPEG by looking for JPEG SOI marker
            if image_bytes[0:2] == b"\xff\xd8":
                offset = 2
                while offset < len(image_bytes):
                    # Check for valid marker
                    if image_bytes[offset] != 0xFF:
                        break

                    # Skip padding
                    if image_bytes[offset] == 0xFF and image_bytes[offset + 1] == 0x00:
                        offset += 2
                        continue

                    # Found a marker
                    marker = image_bytes[offset + 1]

                    # SOF0 marker contains dimensions
                    if marker == 0xC0:
                        # Height is at offset+5, width at offset+7, both 2 bytes big-endian
                        height = struct.unpack(
                            ">H", image_bytes[offset + 5 : offset + 7]
                        )[0]
                        width = struct.unpack(
                            ">H", image_bytes[offset + 7 : offset + 9]
                        )[0]
                        return (width, height)

                    # Not SOF0, skip this segment
                    size = struct.unpack(">H", image_bytes[offset + 2 : offset + 4])[0]
                    offset += 2 + size

                return None

            # For PNG, dimensions are in the IHDR chunk
            elif image_bytes[0:8] == b"\x89PNG\r\n\x1a\n":
                # PNG width is at offset 16, height at offset 20, both 4 bytes big-endian
                width = struct.unpack(">I", image_bytes[16:20])[0]
                height = struct.unpack(">I", image_bytes[20:24])[0]
                return (width, height)

            return None
    except Exception:
        return None
