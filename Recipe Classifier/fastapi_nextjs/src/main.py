"""Main FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from oma_recipeclassifier.src.api.router import api_router
from oma_recipeclassifier.src.config.settings import ALLOWED_ORIGINS, API_BASE_PATH

app = FastAPI(
    title="Recipe Classifier",
    description="API to classify step recipe images",
    version="1.0.0",
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


# Add CORS to allow frontend access from specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
