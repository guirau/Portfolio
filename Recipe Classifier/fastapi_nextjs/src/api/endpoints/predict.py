"""Recipe Classifier predict endpoint."""

from fastapi import APIRouter, File, HTTPException, UploadFile, Request
from loguru import logger
import aiohttp

from oma_recipeclassifier.src.app import ModelService
from oma_recipeclassifier.src.config.model_config import MODEL_NAME, MODEL_PATH
from oma_recipeclassifier.src.schemas.prediction import Prediction

router = APIRouter()

model_service = ModelService(MODEL_PATH, MODEL_NAME)


@router.post("/predict", response_model=list[Prediction])
async def predict(file: UploadFile = File(...)):
    """
    Classify an image from an upload.

    Args:
        file: Uploaded image file.

    Returns:
        list[Prediction]: Predicted classes and confidence scores.

    Raises:
        HTTPException: If the prediction fails.
    """
    if not file.content_type.startswith("image/"):
        logger.error("File must be an image")
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        logger.info(f"Processing input image: {file.filename}")
        contents = await file.read()
        logger.info("File read successfully")

        predictions = model_service.predict(contents)
        logger.success(f"Successfully classified image. Predictions: {predictions}")
        return predictions
    except Exception as ex:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(ex)) from ex


@router.get("/classify", response_model=list[Prediction])
async def classify(request: Request):
    """
    Classify an image from a URL.

    Args:
        request: Request object with image URL as query parameter.

    Returns:
        list[Prediction]: Predicted classes and confidence scores.

    Raises:
        HTTPException: If the prediction fails.
    """
    image_url = str(request.url).split("?", 1)[1] if "?" in str(request.url) else None
    if not image_url:
        logger.error("No image URL provided")
        raise HTTPException(
            status_code=400, detail="Provide image URL: /classify?<image_url>"
        )
    try:
        logger.info(f"Downloading image from URL: {image_url}")
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                if response.status != 200:
                    logger.error(f"Failed to download image: {response.status}")
                    raise HTTPException(
                        status_code=400, detail="Failed to download image"
                    )
                contents = await response.read()

        predictions = model_service.predict(contents)
        logger.success(
            f"Successfully classified image: {image_url}. Predictions: {predictions}"
        )
        return predictions
    except Exception as ex:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(ex)) from ex


@router.get("/health")
def health_check():
    """Check the health status of the service."""
    if model_service.model is None:
        logger.error("Health check failed: Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    status = {
        "status": "healthy",
        "device": model_service.device,
        "CLIP_base_model": MODEL_NAME,
        "model_weights": MODEL_PATH,
    }
    logger.success("Health check passed", extra={"status": status})
    return status
