"""
Utils for handling image metadata (EXIF, XMP, IPTC, ICC)
and Photoshop path resources for segmentation.
"""

import json
import os
import struct
import tempfile

import pyexiv2
from loguru import logger
from PIL import Image


class MetadataHandler:
    """Handles image metadata operations."""

    @staticmethod
    def extract_metadata(image_bytes: bytes) -> dict:
        """
        Extract metadata from an image file path.

        Args:
            image_bytes: Image data as bytes.

        Returns:
            Dictionary with extracted metadata (EXIF, XMP, IPTC, ICC profile)
        """
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            file_path = tmp_file.name

        with open(file_path, "wb") as f:
            f.write(image_bytes)

        try:
            metadata = {}
            with pyexiv2.Image(file_path) as img_metadata:
                try:
                    metadata["EXIF"] = img_metadata.read_exif()
                except Exception as ex:
                    logger.warning(f"Failed to read EXIF metadata: {ex}")
                    metadata["EXIF"] = {}

                try:
                    metadata["XMP"] = img_metadata.read_xmp()
                except Exception as ex:
                    logger.warning(f"Failed to read XMP metadata: {ex}")
                    metadata["XMP"] = {}

                try:
                    metadata["IPTC"] = img_metadata.read_iptc()
                except Exception as ex:
                    logger.warning(f"Failed to read IPTC metadata: {ex}")
                    metadata["IPTC"] = {}

                try:
                    metadata["ICC"] = img_metadata.read_icc()
                except Exception as ex:
                    logger.warning(f"Failed to read ICC profile: {ex}")
                    metadata["ICC"] = None

            return metadata

        except Exception as ex:
            logger.error(f"Error extracting metadata: {ex}")
            return {}

        finally:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)  # Cleanup

    def apply_metadata(
        self,
        image: Image.Image,
        output_path: str,
        metadata: dict,
        additional_keywords: list = None,
        segmentation_paths: list = None,
    ) -> str:
        """
        Save an image with metadata to a path.

        Args:
            image: PIL Image.
            output_path: Path to save the image with metadata.
            metadata: Metadata dictionary from extract_metadata
            additional_keywords: List of additional keywords to add to IPTC keywords.
            segmentation_paths: List of coordinate paths for segmentation.

        Returns:
            Path to the saved image with metadata.
        """
        logger.info("Updating metadata...")

        try:
            # Save image without metadata to output path
            image.save(output_path, format="JPEG", quality=100)

            # Add metadata
            with pyexiv2.Image(output_path) as img_metadata:

                # EXIF metadata
                try:
                    exif_data = metadata.get("EXIF", {}).copy()
                    img_metadata.modify_exif(exif_data)
                except Exception as ex:
                    logger.warning(f"Failed to apply EXIF metadata: {ex}")

                # XMP metadata
                try:
                    xmp_data = metadata.get("XMP", {}).copy()
                    if segmentation_paths:
                        # Store segmentation paths if provided
                        paths_json = json.dumps(segmentation_paths)
                        xmp_data["Xmp.digiKam.SegmentationPaths"] = paths_json
                    img_metadata.modify_xmp(xmp_data)
                except Exception as ex:
                    logger.warning(f"Failed to apply XMP metadata: {ex}")

                # IPTC metadata
                try:
                    iptc_data = metadata.get("IPTC", {}).copy()
                    # Add additional keywords if provided
                    if (
                        additional_keywords
                        and "Iptc.Application2.Keywords" in iptc_data
                    ):
                        # Get existing keywords and append new ones
                        current_keywords = iptc_data.get(
                            "Iptc.Application2.Keywords", []
                        )
                        for keyword in additional_keywords:
                            if keyword not in current_keywords:
                                current_keywords.append(keyword)
                        iptc_data["Iptc.Application2.Keywords"] = current_keywords
                    elif additional_keywords:
                        # Create the keywords field if it doesn't exist
                        iptc_data["Iptc.Application2.Keywords"] = additional_keywords
                    img_metadata.modify_iptc(iptc_data)
                except Exception as ex:
                    logger.warning(f"Failed to apply IPTC metadata: {ex}")

                # ICC profile
                try:
                    icc_data = metadata.get("ICC")
                    if icc_data:
                        img_metadata.modify_icc(icc_data)
                except Exception as ex:
                    logger.warning(f"Failed to apply ICC profile: {ex}")

                # Photoshop metadata (only segmentation paths)
                # Add segmentation paths if provided
                if segmentation_paths and len(segmentation_paths) > 0:
                    logger.info(
                        f"Processing {len(segmentation_paths)} segmentation paths"
                    )

                    # Process all paths
                    normalized_paths = []
                    for path in segmentation_paths:
                        # Convert segmentation paths to Photoshop format
                        normalized_points = []
                        for point in path:
                            # Normalize coordinates to 0-1 range
                            x, y = point
                            width, height = image.size
                            x_norm = x / width
                            y_norm = y / height
                            normalized_points.append([x_norm, y_norm])

                        # Make sure path is closed (last point = first point)
                        if normalized_points and len(normalized_points) > 0:
                            if normalized_points[0] != normalized_points[-1]:
                                normalized_points.append(normalized_points[0])

                        normalized_paths.append(normalized_points)

                    # Insert path data into image file
                    output_path = self.insert_photoshop_path(
                        output_path, normalized_paths, resource_id=1025
                    )

        except Exception as ex:
            logger.error(f"Error applying metadata: {ex}")

        return output_path

    def insert_photoshop_path(
        self, file_path: str, paths: list, resource_id: int = 2000
    ) -> str:
        """
        Insert path data into a JPEG file's metadata as a Photoshop path resource.

        Args:
            file_path: Path to the JPEG file
            paths: List of paths, where each path is a list of [x, y] coordinates
            resource_id: Resource ID to use (default: 2000, standard path resource range)

        Returns:
            Path to the modified JPEG file
        """
        # Create new output file
        output_path = file_path.replace(".jpg", f"_path_{resource_id}.jpg")

        # Read original file
        with open(file_path, "rb") as f:
            jpeg_data = f.read()

        # Create path resource
        path_resource = self.create_path_resource(paths)

        # Create complete resource block
        # Format: 8BIM + resource_id (2 bytes) + name length (1 byte) + name + padding + data size (4 bytes) + data
        resource_block = b"8BIM" + struct.pack(">H", resource_id)  # Marker and ID
        resource_block += b"\0"  # Empty name (0 length)
        # Padding byte needed when name length is even
        resource_block += b"\0"  # Padding byte
        resource_block += struct.pack(">I", len(path_resource))  # Data size
        resource_block += path_resource  # The actual path data

        # Find where to insert the new resource block
        # Look for APP13 marker (Photoshop IRB)
        app13_pos = jpeg_data.find(b"\xff\xed")

        if app13_pos != -1:
            # Found APP13 marker, check if it's a Photoshop marker
            # Get segment size
            segment_size = struct.unpack(
                ">H", jpeg_data[app13_pos + 2 : app13_pos + 4]
            )[0]

            # Check for Photoshop signature
            if b"Photoshop 3.0" in jpeg_data[app13_pos + 4 : app13_pos + 20]:
                # Find the end of the current APP13 segment
                segment_end = app13_pos + 2 + segment_size

                # Create a new APP13 segment with our resource
                # Format: FF ED + size (2 bytes) + "Photoshop 3.0\0" + our resource block
                app13_header = b"\xff\xed"
                ps_signature = b"Photoshop 3.0\0"

                # Calculate new segment size (2 bytes for size field itself not included)
                new_segment_size = len(ps_signature) + len(resource_block)
                app13_header += struct.pack(
                    ">H", new_segment_size + 2
                )  # +2 for size field
                app13_header += ps_signature

                # Construct the new JPEG
                new_jpeg = (
                    jpeg_data[:app13_pos]
                    + app13_header
                    + resource_block
                    + jpeg_data[segment_end:]
                )

            else:
                # Not a Photoshop APP13, create a new one
                # Insert after this APP13 segment
                segment_end = app13_pos + 2 + segment_size

                # Create a new APP13 segment
                app13_header = b"\xff\xed"
                ps_signature = b"Photoshop 3.0\0"

                new_segment_size = len(ps_signature) + len(resource_block)
                app13_header += struct.pack(">H", new_segment_size + 2)
                app13_header += ps_signature

                # Construct the new JPEG
                new_jpeg = (
                    jpeg_data[:segment_end]
                    + app13_header
                    + resource_block
                    + jpeg_data[segment_end:]
                )

        else:
            # No APP13 found, insert after APP0 (JFIF) marker
            app0_pos = jpeg_data.find(b"\xff\xe0")

            if app0_pos != -1:
                # Get APP0 segment size
                app0_size = struct.unpack(">H", jpeg_data[app0_pos + 2 : app0_pos + 4])[
                    0
                ]
                app0_end = app0_pos + 2 + app0_size

                # Create new APP13 segment
                app13_header = b"\xff\xed"
                ps_signature = b"Photoshop 3.0\0"

                new_segment_size = len(ps_signature) + len(resource_block)
                app13_header += struct.pack(">H", new_segment_size + 2)
                app13_header += ps_signature

                # Construct the new JPEG
                new_jpeg = (
                    jpeg_data[:app0_end]
                    + app13_header
                    + resource_block
                    + jpeg_data[app0_end:]
                )

            else:
                logger.error("Cannot find APP0 marker, cannot insert Photoshop data")
                return file_path

        # Write the new file
        with open(output_path, "wb") as f:
            f.write(new_jpeg)

        logger.info(f"Segmentation paths added as Photoshop paths")
        return output_path

    def create_path_resource(self, paths: list) -> bytes:
        """
        Create a Photoshop path resource from a list of paths.

        Args:
            paths: List of paths, where each path is a list of [x, y] coordinates

        Returns:
            Path resource data in Photoshop format.
        """
        path_data = bytearray()

        # Record 0: Path fill rule record (selector 6)
        path_data.extend(struct.pack(">H", 6))  # Selector
        path_data.extend(bytes(24))  # 24 bytes of zeros

        # Record 1: Initial fill rule record (selector 8)
        path_data.extend(struct.pack(">H", 8))  # Selector
        path_data.extend(struct.pack(">H", 0))  # Fill starts with all pixels: False
        path_data.extend(bytes(22))  # 22 bytes of zeros

        # Process paths
        for path_points in paths:
            # Make sure we have enough points
            if len(path_points) < 3:
                logger.warning(f"Path has only {len(path_points)} points - skipping")
                continue

            # Record: Closed subpath length record (selector 0)
            path_data.extend(struct.pack(">H", 0))  # Selector for closed path
            path_data.extend(struct.pack(">H", len(path_points)))  # Number of knots
            path_data.extend(bytes(22))  # 22 bytes of zeros

            # Add Bezier knot records for each point in this path
            for i, point in enumerate(path_points):
                x, y = point  # Normalized coordinates (0-1)

                # Make sure coordinates are within valid range
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))

                # Convert to Photoshop's fixed-point format (8.24)
                x_fixed = int(x * 0x1000000)
                y_fixed = int(y * 0x1000000)

                # Use unlinked knots for simplicity (selector 2)
                selector = 2

                # Create the 26-byte record for this point
                point_record = bytearray()
                point_record.extend(struct.pack(">H", selector))

                # Control in point
                point_record.extend(struct.pack(">i", y_fixed))
                point_record.extend(struct.pack(">i", x_fixed))

                # Anchor point
                point_record.extend(struct.pack(">i", y_fixed))
                point_record.extend(struct.pack(">i", x_fixed))

                # Control out point
                point_record.extend(struct.pack(">i", y_fixed))
                point_record.extend(struct.pack(">i", x_fixed))

                # Make sure record is exactly 26 bytes
                assert (
                    len(point_record) == 26
                ), f"Point record is {len(point_record)} bytes, should be 26"

                # Add to path data
                path_data.extend(point_record)

        # Make sure total path data is a multiple of 26 bytes
        total_length = len(path_data)
        if total_length % 26 != 0:
            logger.error(f"Path data length {total_length} is not a multiple of 26")
            # Pad with zeros to make it a multiple of 26
            padding_needed = 26 - (total_length % 26)
            path_data.extend(bytes(padding_needed))

        return bytes(path_data)
