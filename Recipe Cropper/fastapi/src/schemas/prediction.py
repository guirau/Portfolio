"""Pydantic schemas for prediction."""

from enum import Enum

from pydantic import BaseModel, HttpUrl


class ImageInput(BaseModel):
    """Input schema for image prediction."""

    image_path: HttpUrl


class PredictionResponse(BaseModel):
    """Response schema for image prediction."""

    s3_url: HttpUrl


class ImageType(Enum):
    """Enumeration of image types."""

    STEP = "step"
    MAIN = "main"
