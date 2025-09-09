"""
Recipe Classifier App for image classification.

This module contains a service for classyfing recipe images using a fine-tuned CLIP model.
The model identifies the class of the recipe step image and returns the top 3 predictions.
"""

import os
import tempfile
from io import BytesIO

import aioboto3
import clip
import torch
from loguru import logger
from PIL import Image

from oma_recipeclassifier.src.config.model_config import CLASSES
from oma_recipeclassifier.src.config.settings import AWS_BUCKET_NAME, AWS_DEFAULT_REGION
from oma_recipeclassifier.src.models.model_manager import ModelManager
from oma_recipeclassifier.src.schemas.prediction import OutputType
from oma_recipeclassifier.src.utils import MetadataHandler
from utils.notifications_hooks.notifications_hook import Notifications

# pylint: disable=line-too-long, too-many-locals, too-few-public-methods, broad-exception-caught


@Notifications(f"{os.getenv('SLACK_SCRIPT_NAME', 'Recipe Classifier')}")
class ModelService:
    """
    Service for step image classification using CLIP model.

    Attributes:
        device: Device to run the model on (CPU or GPU).
        model: CLIP model.
        preprocess: Image preprocessing module.
        text_features: Precomputed text features for all classes.
    """

    def __init__(self, model_path: str, model_name: str = "ViT-B/32"):
        """
        Initialize the CLIP model service with an S3 client.

        Args:
            model_path: Path to the fine-tuned model weights.
            model_name: CLIP model variant.
        """
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Download model files if not present
        self.model_manager = ModelManager()
        self.model_manager.check_model_files()

        # Load model
        self.model = None
        self.preprocess = None
        self._load_model(model_path, model_name)

        # Precompute text features
        self.text_features = self._precompute_text_features()

        # Initialize S3 client
        self.aws_session = aioboto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=AWS_DEFAULT_REGION,
        )
        self.bucket_name = AWS_BUCKET_NAME
        logger.info(f"Using S3 bucket: {self.bucket_name}")

    def _load_model(self, model_path: str, model_name: str) -> None:
        """
        Load the CLIP model and fine-tuned weights.

        Args:
            model_path: Path to the fine-tuned model weights.
            model_name: CLIP model variant.

        Raises:
            RuntimeError: If the model fails to load.
        """
        try:
            # Load CLIP model
            logger.info(f"Loading base CLIP model: {model_name}")
            self.model, self.preprocess = clip.load(
                model_name, device=self.device, jit=False
            )

            # Load fine-tuned weights
            logger.info(f"Loading fine-tuned weights from: {model_path}")
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            logger.info("Model loaded successfully")

        except Exception as ex:
            logger.exception("Failed to load model")
            raise RuntimeError("Failed to load model") from ex

    def _precompute_text_features(self) -> torch.Tensor:
        """
        Precompute text features for all classes.

        Returns:
            torch.Tensor: Normalized text features for all classes.
        """
        logger.info(f"Precomputing text features for all {len(CLASSES)} classes")

        text_prompts = [f"A photo of a {class_name}." for class_name in CLASSES]
        tokenized_text = clip.tokenize(text_prompts).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(tokenized_text)
            text_features /= text_features.norm(dim=-1, keepdim=True)  # Normalize

        return text_features

    async def _upload_to_s3(self, local_path: str, filename: str) -> str:
        """
        Upload a file to S3.

        Args:
            local_path: Local path of the file to upload.
            filename: Original filename to base the S3 key on.

        Returns:
            str: Public S3 URL of the uploaded file.

        Raises:
            Exception: If S3 upload fails.
        """
        async with self.aws_session.client("s3") as s3:
            try:
                # Upload to S3
                result_name = os.path.basename(filename)
                s3_key = f"recipeclassifier/results/{result_name}"

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
        self,
        image_data: bytes,
        filename: str = None,
        output: OutputType = OutputType.PREDICTIONS,
    ) -> list[dict[str, float]] | bytes | str:
        """
        Predict the class of an input image.

        Args:
            image_data: Image data as bytes.
            filename: Original image filename.
            output: Type of output to return:
                - OutputType.PREDICTIONS: Return top 3 classification predictions as a dict.
                - OutputType.FILE:  Return the image file with the top classification keyword
                    embedded as metadata under the key "Keywords".
                - OutputType.S3_URL: Return the S3 URL of the image with the top classification
                    keyword embedded as metadata under the key "Keywords".

        Returns:
            list[dict[str, float]] | bytes | str: Depends on output parameter:
                - If output=OutputType.PREDICTIONS: List of top 3 predictions, each containing:
                    - keyword: Predicted class name.
                    - confidence: Confidence score (0-100)
                - If output=OutputType.FILE: Image file with metadata.
                - If output=OutputType.S3_URL: S3 URL of the image with metadata.

        Raises:
            RuntimeError: If the prediction fails, if file reading fails, or if S3 upload fails.
        """
        try:
            logger.info("Predicting image class...")

            # Extract metadata
            metadata = MetadataHandler.extract_metadata(image_data)

            # Load and preprocess image
            image = Image.open(BytesIO(image_data))
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # Generate image features
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize

                # Compute cosine similarity with precomputed text features
                similarity = (100.0 * image_features @ self.text_features.T).softmax(
                    dim=-1
                )

                # Get top predictions
                probs, pred_indices = similarity[0].topk(3)

                predictions = [
                    {
                        "keyword": CLASSES[idx.item()],
                        "confidence": prob.item() * 100,
                    }
                    for prob, idx in zip(probs, pred_indices)
                ]

                if output == OutputType.PREDICTIONS:
                    # Return top 3 predictions
                    return predictions

                # Add metadata to image
                with tempfile.NamedTemporaryFile(
                    suffix=".jpg", delete=False
                ) as tmp_file:
                    local_path = tmp_file.name
                additional_keywords = [predictions[0]["keyword"]]
                local_path = MetadataHandler.apply_metadata(
                    image=image,
                    output_path=local_path,
                    metadata=metadata,
                    additional_keywords=additional_keywords,
                )
                logger.info(f"Keyword: {additional_keywords}")

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
                    # Upload image to S3 and return URL
                    s3_url = await self._upload_to_s3(local_path, filename)
                    return s3_url

        except Exception as ex:
            logger.exception("Prediction failed")
            raise RuntimeError("Prediction failed") from ex
