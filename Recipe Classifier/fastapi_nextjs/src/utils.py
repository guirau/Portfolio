import json
import os
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

    @staticmethod
    def apply_metadata(
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

        except Exception as ex:
            logger.error(f"Error applying metadata: {ex}")

        return output_path
