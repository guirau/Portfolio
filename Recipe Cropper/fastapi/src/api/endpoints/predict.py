"""Recipe Cropper predict endpoint."""

from urllib.parse import urlparse

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from loguru import logger

from oma_recipecropper.src.app import model_service
from oma_recipecropper.src.config.model_config import (
    BRIAAI_BASE_MODEL_PATH,
    BRIAAI_MODEL_PATH,
    SEGFORMER_MODEL_PATH,
)
from oma_recipecropper.src.config.settings import CLOUDINARY_BASE_URL
from oma_recipecropper.src.schemas.prediction import (
    ImageInput,
    ImageType,
    PredictionResponse,
)

router = APIRouter()

# Models for different image types
MODEL_SERVICES = {
    ImageType.STEP: model_service("segformer", model_path=SEGFORMER_MODEL_PATH),
    ImageType.MAIN: model_service(
        "briaai",
        # model_path=f"{BRIAAI_BASE_MODEL_PATH}/model.safetensors",
        model_path=BRIAAI_MODEL_PATH,
        config_path=f"{BRIAAI_BASE_MODEL_PATH}/config.json",
    ),
}

MODEL_PATHS = {
    ImageType.STEP: SEGFORMER_MODEL_PATH,
    ImageType.MAIN: BRIAAI_BASE_MODEL_PATH,
}


@router.get("/{path}")
async def root_msg(path: str):
    """
    Root endpoint for base paths without image type.
    Returns help message to specify image type in the URL.
    """
    if path in ["predict", "predict_upload", "crop", "health"]:
        return {"message": "Please specify an image type in the URL: /step or /main"}
    raise HTTPException(status_code=404, detail="Not found")


@router.post("/{image_type}/predict", response_model=PredictionResponse)
async def predict(image_type: ImageType, data: ImageInput):
    """
    Crop an image from a URL.

    Args:
        image_type: Type of image (step/main)
        data: Input data with image URL.

    Returns:
        PredictionResponse: S3 URL of the processed image.

    Raises:
        HTTPException: If the prediction fails.
    """
    try:
        s3_url = await MODEL_SERVICES[image_type].predict(str(data.image_path))
        return PredictionResponse(s3_url=s3_url)
    except Exception as ex:
        logger.exception(f"[{image_type.value}] Prediction failed")
        raise HTTPException(status_code=500) from ex


@router.post("/{image_type}/predict_upload", response_model=PredictionResponse)
async def predict_upload(image_type: ImageType, file: UploadFile = File(...)):
    """
    Crop an image from an upload.

    Args:
        image_type: Type of image (step/main)
        file: Uploaded image file.

    Returns:
        PredictionResponse: S3 URL of the processed image.

    Raises:
        HTTPException: If the prediction fails.
    """
    if not file.content_type.startswith("image/"):
        logger.error("File must be an image")
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        s3_url = await MODEL_SERVICES[image_type].predict_from_bytes(
            contents, file.filename
        )
        return PredictionResponse(s3_url=s3_url)
    except Exception as ex:
        logger.exception(f"[{image_type.value}] Prediction failed")
        raise HTTPException(status_code=500) from ex


@router.get("/{image_type}/crop")
async def crop(image_type: ImageType, request: Request) -> RedirectResponse:
    """
    Crop an image from a URL and redirect to Cloudinary.

    Args:
        image_type: Type of image (step/main)
        request: FastAPI request object with image URL as query parameter.

    Returns:
        RedirectResponse: Redirect to Cloudinary URL of the processed image.

    Raises:
        HTTPException: If the prediction fails.
    """
    image_url = str(request.url).split("?", 1)[1] if "?" in str(request.url) else None
    if not image_url:
        raise HTTPException(
            status_code=400, detail="Provide image URL: /crop?<image_url>"
        )

    try:
        logger.info(f"Input image: {image_url}")
        s3_url = await MODEL_SERVICES[image_type].predict(image_url)

        parsed_s3_url = urlparse(s3_url)
        file_path = parsed_s3_url.path.lstrip("/")
        cloudinary_url = f"{CLOUDINARY_BASE_URL}{file_path}"

        logger.success(f"Successfully processed image. S3 URL: {s3_url}")
        return RedirectResponse(url=cloudinary_url)

    except Exception as ex:
        logger.exception(f"[{image_type.value}] Image processing failed")
        raise HTTPException(status_code=500) from ex


@router.get("/{image_type}/health")
def health_check(image_type: ImageType):
    """
    Check the health status of the service.

    Args:
        image_type: Type of image (step/main)
    """
    _model_service = MODEL_SERVICES[image_type]
    if _model_service.model is None:
        logger.error(f"Health check failed: [{image_type.value}] Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    status = {
        "status": "healthy",
        "device": _model_service.device,
        "model_weights": MODEL_PATHS[image_type],
    }
    logger.success(
        f"[{image_type.value}] Health check passed", extra={"status": status}
    )
    return status
