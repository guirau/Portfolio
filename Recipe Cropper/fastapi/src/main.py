"""Main FastAPI application."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from oma_recipecropper.src.api.router import api_router
from oma_recipecropper.src.config.settings import (ALLOWED_ORIGINS,
                                                   API_BASE_PATH)

app = FastAPI(
    title="Recipe Cropper API",
    description="API to crop recipe images",
    version="2.0",
    docs_url=f"{API_BASE_PATH}/docs",
    redoc_url=f"{API_BASE_PATH}/redoc",
    openapi_url=f"{API_BASE_PATH}/openapi.json",
)

app.include_router(api_router, prefix=API_BASE_PATH)


@app.get("/", include_in_schema=False)
def root():
    """
    Redirect the root URL ("/") to the API base path.
    This route will not be included in the OpenAPI schema.
    """
    return RedirectResponse(url=f"{API_BASE_PATH}")


# Redirect /api/v1 to /api/v2
@app.middleware("http")
async def redirect_v1(request: Request, call_next):
    """Middleware to redirect /api/v1 to /api/v2."""
    if request.url.path.startswith("/api/v1"):
        new_path = request.url.path.replace("/api/v1", "/api/v2")
        new_url = str(request.url.replace(path=new_path))
        return RedirectResponse(url=new_url, status_code=308)
    return await call_next(request)


# CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
