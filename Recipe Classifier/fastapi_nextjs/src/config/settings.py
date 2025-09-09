"""General configuration."""

import os

# FastAPI settings
API_BASE_PATH = os.getenv("API_BASE_PATH", "/api/v1")  # Set in Helm chart
ENVIRONMENTS = ["dev", "staging", "live"]
ALLOWED_ORIGINS = [
    "http://localhost:3000",  # Local dev
    *[f"https://oma-ui.{env}.company.com" for env in ENVIRONMENTS],
    *[f"https://post-prod-demo.{env}.company.com" for env in ENVIRONMENTS],
]

# AWS settings
AWS_DEFAULT_REGION = "<AWS_DEFAULT_REGION>"
AWS_BUCKET_NAME = "<AWS_BUCKET_NAME>"
