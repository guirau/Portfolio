"""Model management utilities for downloading model files."""

import logging
import os
from pathlib import Path

import boto3
from loguru import logger
from tqdm import tqdm

from oma_recipeclassifier.src.config.model_config import MODEL_PATH, MODEL_BUCKET_PATH
from oma_recipeclassifier.src.config.settings import AWS_BUCKET_NAME, AWS_DEFAULT_REGION

# pylint: disable=too-few-public-methods


class ModelManager:
    """Manages model files, making sure they're available locally."""

    def __init__(
        self, s3_bucket: str = AWS_BUCKET_NAME, models_s3_path: str = MODEL_BUCKET_PATH
    ):
        """Initialize the model manager."""
        # Set boto3 log level to WARNING
        logging.getLogger("botocore").setLevel(logging.WARNING)

        self.s3_bucket = s3_bucket
        self.models_s3_path = models_s3_path
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=AWS_DEFAULT_REGION,
        )

    def check_model_files(self) -> None:
        """Make sure model files are available locally, downloading them if necessary."""
        if Path(MODEL_PATH).exists():
            logger.info("CLIP model files already exist locally")
        else:
            logger.info("Downloading CLIP model files")

            # Create directory if it doesn't exist
            local_path = Path(MODEL_PATH).parent
            Path.mkdir(local_path, parents=True, exist_ok=True)

            # Download model files
            download_path = f"{self.models_s3_path}"
            self._download_model_files(download_path, local_path)

    def _download_model_files(self, download_path: str, local_path: str) -> None:
        """
        Download all files from an S3 bucket path recursively.

        Args:
            download_path: S3 bucket path to download from
            local_path: Local path to save the files

        Raises:
            RuntimeError: If the download fails
        """
        try:
            # List all files in the S3 path
            paginator = self.s3_client.get_paginator("list_objects_v2")
            all_objects = []

            # Get all objects in the S3 path
            for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=download_path):
                if "Contents" in page:
                    all_objects.extend(
                        [
                            obj
                            for obj in page["Contents"]
                            if not obj["Key"].endswith("/")  # Skip directories
                        ]
                    )

            with tqdm(total=len(all_objects), desc="Downloading model files") as pbar:
                for obj in all_objects:
                    s3_key = obj["Key"]

                    # Get relative path from prefix
                    download_path_parts = download_path.rstrip("/").split("/")
                    s3_key_parts = s3_key.split("/")
                    relative_path = "/".join(s3_key_parts[len(download_path_parts) :])

                    # Create local file path
                    local_file_path = local_path / relative_path

                    # Create parent directories if they don't exist
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)

                    # Skip if file already exists
                    if local_file_path.exists():
                        logger.debug(f"File already exists: {local_file_path}")
                        pbar.update(1)
                        continue

                    logger.info(
                        f"Downloading s3://{self.s3_bucket}/{s3_key} to {local_file_path}"
                    )
                    self.s3_client.download_file(
                        self.s3_bucket, s3_key, str(local_file_path)
                    )
                    logger.info(f"Successfully downloaded {s3_key}")
                    pbar.update(1)

        except Exception as ex:
            logger.exception(f"Failed to download files from {download_path}")
            raise RuntimeError(f"Failed to download files from {download_path}") from ex
