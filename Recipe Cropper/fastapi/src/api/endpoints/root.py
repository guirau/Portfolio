"""Root API endpoint."""

from fastapi import APIRouter

from oma_recipecropper.src.schemas.prediction import ImageType

router = APIRouter()


@router.get("/")
def root():
    """Root endpoint that returns service information."""
    return {
        "message": "Recipe Cropper API v2",
        "available_image_types": [t.value for t in ImageType],
        "documentation": {"swagger": "/api/v2/docs", "redoc": "/api/v2/redoc"},
    }
