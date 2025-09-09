"""Root API endpoint."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def root():
    """Root endpoint that returns service information."""
    return {
        "message": "Recipe Classifier API v1",
        "documentation": {"swagger": "/api/v1/docs", "redoc": "/api/v1/redoc"},
    }
