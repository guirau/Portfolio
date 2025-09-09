"""
Recipe Cropper App for image segmentation/background removal.

This module contains services for cropping images using different segmentation models:
- SegFormer: NVIDIA's mit-b0 model for semantic segmentation.
- RMBG-2.0: BRIA AI's model for background removal.

Images are cropped based on the segmentation mask inferred by one of the models, and
uploaded to an S3 bucket with transparent background.
"""

import json
import os
import tempfile
from abc import ABC, abstractmethod
from io import BytesIO

import aioboto3
import aiohttp
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from loguru import logger
from PIL import Image, ImageDraw
from safetensors.torch import load_file  # pylint: disable=unused-import
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

from oma_recipecropper.src.config.settings import AWS_BUCKET_NAME, AWS_DEFAULT_REGION
from oma_recipecropper.src.models.briaai import BiRefNet, BiRefNetConfig
from oma_recipecropper.src.models.model_manager import ModelManager
from oma_recipecropper.src.schemas.prediction import OutputType
from oma_recipecropper.src.utils import MetadataHandler
from utils.notifications_hooks.notifications_hook import Notifications

# pylint: disable=too-many-locals, line-too-long, unused-variable, no-member, broad-exception-caught


@Notifications(f"{os.getenv("SLACK_SCRIPT_NAME", "Recipe Cropper")}")
class BaseModelService(ABC):
    """
    Base class for model services.

    Provides common methods for image cropping, downloading, and uploading to S3.

    Attributes:
        device: Device to run the model on (CPU or GPU).
        aws_session: Async S3 session for uploading results.
        bucket_name: S3 bucket name for uploading results.
    """

    def __init__(self):
        """Initialize base model service with device setup, S3 client, and model manager."""
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Initialize S3 client
        self.aws_session = aioboto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=AWS_DEFAULT_REGION,
        )
        self.bucket_name = AWS_BUCKET_NAME
        logger.info(f"Using S3 bucket: {self.bucket_name}")

        # Initialize model manager to download model files
        self.model_manager = ModelManager()

    @abstractmethod
    def _process_image(self, image: Image.Image, metadata: dict, crop: bool) -> str:
        """
        Process an image using the specific model implementation.

        Args:
            image: Image to process.
            metadata: Dictionary with image metadata.
            crop: If True, crop the image.

        Returns:
            str: Local path of the cropped image.
        """

    def _crop_image(
        self, image: Image.Image, predicted_segmentation_map: np.ndarray
    ) -> str:
        """
        Crop an image based on the segmentation map and saves it locally with
        transparent background.

        Args:
            image: Original image to crop.
            predicted_segmentation_map: Segmentation map inferred by the model.

        Returns:
            str: Local path of the cropped image.
        """
        logger.info("Cropping...")

        # Create binary mask where the segmentation map == 1
        binary_mask = (predicted_segmentation_map == 1).astype(np.uint8)

        # Create a mask image and draw the binary mask over it
        mask = Image.new("L", (binary_mask.shape[1], binary_mask.shape[0]), 0)
        draw = ImageDraw.Draw(mask)
        for y in range(binary_mask.shape[0]):
            for x in range(binary_mask.shape[1]):
                if binary_mask[y, x] == 1:
                    draw.point((x, y), fill=255)

        # Apply mask to original image to create image with transparency
        image_with_transparency = Image.composite(
            image, Image.new("RGBA", image.size), mask
        )

        # Find bounding box of segmented region
        foreground_indices = np.where(binary_mask == 1)
        if len(foreground_indices[0]) == 0 or len(foreground_indices[1]) == 0:
            logger.error("No foreground detected in segmentation map")
            raise ValueError("No foreground detected in segmentation map")

        top_left_y = np.min(foreground_indices[0])
        bottom_right_y = np.max(foreground_indices[0])
        top_left_x = np.min(foreground_indices[1])
        bottom_right_x = np.max(foreground_indices[1])

        # Crop the image to the bounding box
        cropped_image_with_transparency = image_with_transparency.crop(
            (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
        )

        # Save the cropped image locally
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            local_path = tmp_file.name
        cropped_image_with_transparency.save(local_path, "PNG")

        logger.info("Image cropped successfully")

        return local_path

    def _generate_path_metadata(self, predicted_segmentation_map: np.ndarray) -> list:
        """
        Generate segmentation path/s from the segmentation map to embed as metadata.

        Args:
            predicted_segmentation_map: Segmentation map inferred by the model.

        Returns:
            list: List of paths with coordinates.
        """
        logger.info("Generating path metadata...")

        # Create binary mask where segmentation map == 1
        binary_mask = (predicted_segmentation_map == 1).astype(np.uint8)

        ## Mask preprocessing ##

        # Close small holes/corners
        # MORPH_CLOSE = Dilation followed by Erosion
        kernel_holes = np.ones((25, 25), np.uint8)
        binary_mask = cv2.morphologyEx(
            binary_mask, cv2.MORPH_CLOSE, kernel_holes, iterations=1
        )

        # Remove small noise
        # MORPH_OPEN = Erosion followed by Dilation
        kernel_noise = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_noise)

        ## Find paths ##

        # Find contours (paths) in binary mask (including holes)
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # If no contours found, return original image
        if not contours:
            logger.error("No contours found in segmentation map")
            return []

        # Sort contours by area (largest first)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Keep only significant contours
        top_n_contours = 3  # Keep only top 3 contours
        area_threshold = 0.1  # Keep contours with area >= 10% of the largest contour
        largest_area = cv2.contourArea(sorted_contours[0])
        significant_contours = []

        for contour in sorted_contours[:top_n_contours]:
            area = cv2.contourArea(contour)
            if area >= (largest_area * area_threshold):
                significant_contours.append(contour)

        logger.info(f"Found {len(significant_contours)} path/s")

        # Process each significant contour
        all_paths = []
        for contour in significant_contours:
            # Convert contour to path and reduce number of points
            # Adaptative epsilon based on contour perimeter
            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.0005 * perimeter
            approx_path = cv2.approxPolyDP(contour, epsilon, True)

            # Format path as list of coordinates
            path_points = approx_path.reshape(-1, 2).tolist()
            all_paths.append(path_points)

            logger.info(f"Path created with {len(path_points)} points")

        return all_paths

    async def _upload_to_s3(self, local_path: str, filename: str) -> str:
        """
        Upload a file to S3.

        Args:
            local_path: Local path of the file to upload.
            filename: Original filename to base the S3 key on.

        Returns:
            str: Public S3 URL of the uploaded file.

        Raises:
            RuntimeError: If S3 upload fails.
        """
        async with self.aws_session.client("s3") as s3:
            try:
                # Upload to S3
                result_name = os.path.basename(filename)
                s3_key = f"recipecropper/results/{result_name}"

                logger.info(f"Uploading to S3: {s3_key}")
                await s3.upload_file(local_path, self.bucket_name, s3_key)

                s3_url = f"https://{self.bucket_name}.s3.{AWS_DEFAULT_REGION}.amazonaws.com/{s3_key}"
                logger.success(f"Successfully uploaded to S3: {s3_url}")

                return s3_url

            except Exception as ex:
                logger.exception("Failed to upload to S3")
                raise RuntimeError("Failed to upload to S3") from ex

            finally:
                # Cleanup
                if os.path.exists(local_path):
                    os.remove(local_path)

    async def predict(
        self, image_url: str, output: OutputType = OutputType.S3_URL, crop: bool = True
    ) -> bytes | str:
        """
        Process image from URL, crop, and upload the result to S3.

        Args:
            image_url: URL of the image to process.
            output: Type of output to return:
                - OutputType.FILE: Return the image file with the paths embedded as metadata
                    under the key "paths".
                - OutputType.S3_URL: Return the S3 URL of the image with paths embedded as
                    metadata under the key "paths".
            crop: If True, crop the image.

        Returns:
            bytes | str: Depends on output parameter:
                - If output=OutputType.FILE: Image file with metadata.
                - If output=OutputType.S3_URL: S3 URL of the image with metadata.

        Raises:
            RuntimeError:  If image processing fails, if file reading fails, or if S3 upload fails.
            ValueError: If image download fails.
        """
        # Download image
        logger.info(f"Downloading image from URL: {image_url}")

        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                if response.status != 200:
                    logger.error(f"Failed to download image: {response.text}")
                    raise ValueError(f"Failed to download image: {response.text}")
                image_bytes = await response.read()

        return await self.predict_from_bytes(
            image_bytes, os.path.basename(image_url), output, crop
        )

    async def predict_from_bytes(
        self,
        image_bytes: bytes,
        filename: str = None,
        output: OutputType = OutputType.S3_URL,
        crop: bool = True,
    ) -> bytes | str:
        """
        Process image from bytes, crop, and upload the result to S3.

        Args:
            image_bytes: Bytes of the image to process.
            filename: Original filename of the image.
            output: Type of output to return:
                - OutputType.FILE: Return the image file with the paths embedded as metadata
                    under the key "paths".
                - OutputType.S3_URL: Return the S3 URL of the image with paths embedded as
                    metadata under the key "paths".
            crop: If True, crop the image.

        Returns:
            bytes | str: Depends on output parameter:
                - If output=OutputType.FILE: Image file with metadata.
                - If output=OutputType.S3_URL: S3 URL of the image with metadata.

        Raises:
            RuntimeError: If image processing fails, if file reading fails, or if S3 upload fails.
        """
        logger.info(f"Procesing image with filename: {filename}")

        # Extract metadata
        metadata = MetadataHandler.extract_metadata(image_bytes)

        # Load and process image
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        local_path = self._process_image(image, metadata, crop)

        if output == OutputType.FILE:
            # Return image file with metadata
            try:
                with open(local_path, "rb") as f:
                    file_data = f.read()
                return file_data
            except Exception as ex:
                logger.exception("Failed to read image file")
                raise RuntimeError("Failed to read image file") from ex
            finally:
                if os.path.exists(local_path):
                    os.remove(local_path)  # Cleanup

        if output == OutputType.S3_URL:
            # Upload to S3 and return URL
            s3_url = await self._upload_to_s3(local_path, filename)
            return s3_url


class SegFormerService(BaseModelService):
    """
    Model service for NVIDIA/SegFormer model.
    Model card: https://huggingface.co/nvidia/mit-b0

    Attributes:
        model: NVIDIA/SegFormer model for semantic segmentation.
        feature_extractor: Feature extractor for image preprocessing.
    """

    def __init__(self, model_path: str):
        """Initialize the NVIDIA/SegFormer model service.

        Args:
            model_path: Path to the model checkpoint.
        """
        super().__init__()
        logger.info(
            f"Loading NVIDIA/SegFormer model and feature extractor from: {model_path}"
        )

        # Download model files if not present
        self.model_manager.check_model_files("segformer")

        # Load model
        try:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_path
            ).to(self.device)
            self.feature_extractor = SegformerFeatureExtractor.from_pretrained(
                model_path
            )
            self.model.eval()
            logger.success("NVIDIA/SegFormer model loaded successfully")
        except Exception as ex:
            logger.exception(f"Failed to load model: {str(ex)}")
            raise RuntimeError("Failed to load model") from ex

    def _process_image(
        self, image: Image.Image, metadata: dict, crop: bool = True
    ) -> str:
        """
        Process an image using the NVIDIA/SegFormer model.

        Args:
            image: Image to process.
            metadata: Dictionary with image metadata.
            crop: If True, crop the image.

        Returns:
            str: Local path of the result image.

        Raises:
            RuntimeError: If image processing fails.
        """
        try:
            logger.info("Processing image with NVIDIA/SegFormer...")
            # Preprocess image using feature extractor
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)

            # Model inference
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)
                logits = outputs.logits.cpu()

            # Post-process segmentation_map
            predicted_segmentation_map = (
                self.feature_extractor.post_process_semantic_segmentation(
                    outputs, target_sizes=[image.size[::-1]]
                )[0]
            )
            predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()

            # Add metadata to image
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                local_path = tmp_file.name
            path_metadata = self._generate_path_metadata(predicted_segmentation_map)
            metadata_handler = MetadataHandler()
            local_path = metadata_handler.apply_metadata(
                image=image,
                output_path=local_path,
                metadata=metadata,
                segmentation_paths=path_metadata,
            )

            if crop:
                # Crop and save image
                image = Image.open(local_path).convert("RGBA")
                local_path = self._crop_image(image, predicted_segmentation_map)

            return local_path

        except Exception as ex:
            logger.exception("NVIDIA/SegFormer image processing failed")
            raise RuntimeError("NVIDIA/SegFormer image processing failed") from ex


class BriaaiService(BaseModelService):
    """
    Model service using BRIAAI/RMBG-2.0 model.
    Model card: https://huggingface.co/briaai/RMBG-2.0

    Attributes:
        model: Briaai model for background removal.
        transform_image: Image transformation pipeline.
    """

    def __init__(self, model_path: str, config_path: str):
        """
        Initialize the BRIAAI/RMBG-2.0 model service.

        Args:
            model_path: Path to the model checkpoint.
            config_path: Path to the model configuration.

        Raises:
            RuntimeError: If model or config loading fails.
        """
        super().__init__()
        logger.info(f"Loading BRIAAI/RMBG-2.0 model from: {model_path}")
        logger.info(f"Loading model configuration from: {config_path}")

        # Download model files if not present
        self.model_manager.check_model_files("briaai")

        # Load config
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as ex:
            logger.exception(f"Failed to load config: {str(ex)}")
            raise RuntimeError("Failed to load config") from ex

        # Load model
        try:
            self.model = BiRefNet(BiRefNetConfig(**config))

            # For loading base model
            # base_model_state_dict = load_file(model_path)
            # self.model.load_state_dict(base_model_state_dict)

            # For loading fine-tuned model
            best_model_checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(best_model_checkpoint["model_state_dict"])

            torch.set_float32_matmul_precision(["high", "highest"][0])
            self.model.to(self.device)
            self.model.eval()
            logger.success("BRIAAI/RMBG-2.0 model loaded successfully")

        except Exception as ex:
            logger.exception(f"Failed to load model: {str(ex)}")
            raise RuntimeError("Failed to load model") from ex

        # Setup transforms
        self.transform_image = T.Compose(
            [
                T.Resize((1024, 1024)),
                T.ToTensor(),
                T.Lambda(lambda x: x * 1 / 255),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _process_image(
        self, image: Image.Image, metadata: dict, crop: bool = True
    ) -> str:
        """
        Process an image using the BRIAAI/RMBG-2.0 model.

        Args:
            image: Image to process.
            metadata: Dictionary with image metadata.
            crop: If True, crop the image.

        Returns:
            str: Local path of the result image.

        Raises:
            RuntimeError: If image processing fails.
        """
        try:
            logger.info("Processing image with BRIAAI/RMBG-2.0...")
            original_image_size = (image.size[1], image.size[0])
            input_images = self.transform_image(image).unsqueeze(0).to(self.device)

            # Model inference
            with torch.no_grad():
                preds = self.model(input_images)
                logits = preds[0]
                logits = F.interpolate(
                    logits,
                    size=original_image_size,
                    mode="bilinear",
                    align_corners=False,
                )
                pred_mask = torch.sigmoid(logits).cpu()
                logger.info("Model inference successful")

            # Convert mask to numpy array and threshold
            predicted_segmentation_map = (pred_mask.squeeze().numpy() > 0.5).astype(
                np.uint8
            )

            # Add metadata to image
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                local_path = tmp_file.name
            path_metadata = self._generate_path_metadata(predicted_segmentation_map)
            metadata_handler = MetadataHandler()
            local_path = metadata_handler.apply_metadata(
                image=image,
                output_path=local_path,
                metadata=metadata,
                segmentation_paths=path_metadata,
            )

            if crop:
                # Crop and save image
                image = Image.open(local_path).convert("RGBA")
                local_path = self._crop_image(image, predicted_segmentation_map)

            return local_path

        except Exception as ex:
            logger.exception("BRIAAI/RMBG-2.0 image processing failed")
            raise RuntimeError("BRIAAI/RMBG-2.0 image processing failed") from ex


def model_service(model_type: str = "segformer", **kwargs) -> BaseModelService:
    """
    Factory function to create the appropriate model service.

    Args:
        model_type: Model type to use ('segformer' or 'briaai')
        **kwargs: Additional arguments needed for specific model initialization
            - model_path: Required for both model types
            - config_path: Required for 'briaai' model type

    Returns:
        BaseModelService: Model service instance of the specified type.

    Raises:
        ValueError: If an invalid model type is provided or required arguments are missing.
    """
    logger.info(f"Creating model service: {model_type}")

    try:
        if model_type == "segformer":
            if "model_path" not in kwargs:
                raise ValueError("model_path is required for SegFormer model")
            return SegFormerService(kwargs.get("model_path"))
        if model_type == "briaai":
            if "model_path" not in kwargs or "config_path" not in kwargs:
                raise ValueError(
                    "Both model_path and config_path are required for BRIAAI model"
                )
            return BriaaiService(kwargs.get("model_path"), kwargs.get("config_path"))
        raise ValueError(f"Invalid model type: {model_type}")

    except Exception:
        logger.exception(f"Failed to create model service of type {model_type}")
        raise
