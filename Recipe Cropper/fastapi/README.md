# Recipe Cropper

![Release: v2](https://img.shields.io/badge/release-v2-blue) [![Documentation: REST API](https://img.shields.io/badge/Docs-REST_API-228B22)](https://recipecropper.company.com/api/v2/docs) [![Documentation: Confluence](https://img.shields.io/badge/Docs-Confluence-228B22)](###)

> Note: All paths and deployment settings are based on the original folder name `oma_recipecropper`. Only using `fastapi` to better describe the project contents.

## Overview

**Recipe Cropper** is a ML Image Segmentation API that automates background removal for recipe images. Users can process an image by either uploading it directly or providing an image URL via the API. The model then removes the background and returns the processed image.

This API is designed primarily for two use cases:

- **Integration into existing post-production workflows** for cropping step images
- **Standalone service for marketing teams** to crop main recipe images

The API supports two segmentation models:

- **SegFormer** (NVIDIA's mit-b0 model): Used for segmenting step images
- **RMBG-2.0** (BRIA AI's non-commercial background removal model): Used for cropping main images

Processed images are uploaded to AWS S3 (`s3://<AWS_BUCKET_NAME>/recipecropper/` in the \<AWS_ACCOUNT_NAME\> account) and served to users as S3 object URLs or Cloudinary URLs.

## Purpose

This project automates the background removal process for step and main recipe images through an API service. It eliminates the neeed for manual cropping, streamlining both post-production and marketing workflows.

Key objectives:

- Automate background removal for step and main recipe images
- Provide an API with multiple endpoints to support different use cases
- Deliver high-quality and reliable results consistenly

## How to Run

The application is hosted at:

```
https://recipecropper.company.com/api/v2/
```

### Endpoints

The API provides separate endpoints for **step** and **main** images:

- `/{image_type}/predict` (POST) - Processes an image from a URL in the request body and returns the S3 URL of the result
- `/{image_type}/predict_upload` (POST) - Processes an uploaded image file and returns the S3 URL of the result
- `/{image_type}/crop` (GET) - Processes an image from a URL provided in a query parameter and redirects to a Cloudinary URL with the result
- `/{image_type}/health` (GET) - Checks the health status of the specific model

Where `{image_type}` should be replaced with either `step` or `main`.

Example:

```
https://recipecropper.company.com/api/v2/step/crop?<image_url>
```
 
## Developers

### Requirements

- VPN connection

### Deployment Environments

-  **Dev**. Deploy with: `make -C oma_recipecropper dev`
-  **Staging**: `https://recipecropper.staging.company.com`
-  **Live**: `https://recipecropper.company.com`

### Secrets and Environment Variables

**Secrets** stored in Vault:

- `aws_access_key_id`: AWS access key
- `aws_secret_access_key`: AWS secret key
- `slack_token`: Token for Slack notifications

**Environment variables** defined in `src/config/settings.py`:

- `AWS_DEFAULT_REGION`: Default is `<AWS_DEFAULT_REGION>`.
- `AWS_BUCKET_NAME`: Default is `<AWS_BUCKET_NAME>`

### Model Management

Models are handled automatically by the `ModelManager` class, which:

1. Checks for local model files
2. Downloads models from S3 if they're not present locally
3. Maintains separate models for step and main images

Latest model files are stored in S3 under:
```
s3://<AWS_BUCKET_NAME>/recipecropper/models/
├── segformer/       # Model for step images
└── briaai/          # Model for main images
```

### Model Configuration

The model configuration is defined in `src/config/model_config.py`. Local model paths within `model_config.py` must reference models stored under `/src/models` and align with the structure in S3 (under `/recipecropper/models`), as the `ModelManager` will download files from AWS to local storage preserving the directory structure.

### How to Test

Run: `make -C oma_qr_generator test`

### How to Deploy

-  **Dev**

Run: `make -C oma_recipecropper dev`

App will be deployed to: `https://recipecropper.dev.company.com`

-  **Staging**

Push changes to branch: `staging/oma_recipecropper`

App will be deployed to: `https://recipecropper.staging.company.com`

-  **Live**

Push changes to branch: `live/oma_recipecropper`

App will be deployed to: `https://recipecropper.company.com`

### How to Read Logs

#### Option 1: Using the Console

Logs are captured by Kubernetes, so make sure you’re logged into the correct AWS environment and Kubernetes context.

- Log into AWS: `aws sso login --profile <aws_profile>`
- Switch to the correct Kubernetes environment: `kubectl config use-context <k8s_context>`

Run the following commands for logs:

- Dev: `make -C oma_recipecropper logs-dev`
- Staging: `make -C oma_recipecropper logs-staging`
- Live: `make -C oma_recipecropper logs-live`

#### Option 2: Using Grafana

- [View logs in Grafana Staging](https://staging.company.grafana.net/)

- [View logs in Grafana Live](https://company.grafana.net/)
