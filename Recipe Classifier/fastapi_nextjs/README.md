# Recipe Classifier

![Release: v1](https://img.shields.io/badge/release-v1-blue) [![Documentation: REST API](https://img.shields.io/badge/Docs-REST_API-228B22)](https://recipeclassifier.company.com/api/v1/docs) [![Documentation: Confluence](https://img.shields.io/badge/Docs-Confluence-228B22)](###)

> Note: All paths and deployment settings are based on the original folder name `oma_recipeclassifier`. Only using `fastapi_nextjs` to better describe the project contents.

## Overview

**Recipe Classifier** is an ML Image Classification API that classifies step recipe images into a set of predefined categories. Users can classify an image by either uploading it directly or providing an image URL via the API. The model then classifies the image and returns the top 3 predictions with confidence scores.

This API is designed to fit into existing post-production workflows, automating step image processing without requiring any changes to the current setup.

The API uses a fine-tuned CLIP model:

- **[CLIP](https://github.com/openai/CLIP)** (OpenAI's Contrastive Language-Image Pre-Training Model): Using the `ViT-B/32` variant, fine-tuned to classify step images among a set of keywords.

## Purpose

This projects automates the classification of step images, a task currently handled manually by the post-production team before editing in Photoshop. By integrating this API, the classification process will be automated and run in batches, streamlining the workflow.

Key objectives:

- Automate the classification of step images
- Provide an API with multiple endpoints to support different use cases
- Deliver accurate predictons

## How to Run

The application is hosted at:

```
https://recipeclassifier.company.com/api/v1/
```

### Endpoints

The API provides the following endpoints

- `/predict` (POST) - Processes an uploaded image file and returns the top 3 classification results with confidence scores
- `/classify` (GET) - Processes an image from a URL provided in a query parameter and returns the top 3 classification results with confidence scores
- `/health` (GET) - Checks the health status of the model

Example:

```
https://recipeclassifier.company.com/api/v1/step/classify?<image_url>
```
 
## Developers

### Requirements

- VPN connection

### Deployment Environments

-  **Dev**. Deploy with: `make -C oma_recipeclassifier dev`
-  **Staging**: `https://recipeclassifier.staging.company.com`
-  **Live**: `https://recipeclassifier.company.com`

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

Latest model files are stored in S3 under:
```
s3://<AWS_BUCKET_NAME>/recipeclassifier/models/
```

### Model Configuration

The model configuration is defined in `src/config/model_config.py`. The configuration includes:

- `MODEL_NAME`: CLIP model variant (`ViT-B/32`)
- `MODEL_PATH`: Path to fine-tuned model weights
- `CLASSES`: List of classification categories
- `MODEL_BUCKET_PATH`: S3 path for model files.

 Local model paths within `model_config.py` must reference models stored under `/src/models` and align with the structure in S3 (under `/recipeclassifier/models`), as the `ModelManager` will download files from AWS to local storage preserving the directory structure.

### How to Test

Run: `make -C oma_recipeclassifier test`

### How to Deploy

-  **Dev**

Run: `make -C oma_recipeclassifier dev`

App will be deployed to: `https://recipeclassifier.dev.company.com`

-  **Staging**

Push changes to branch: `staging/oma_recipeclassifier`

App will be deployed to: `https://recipeclassifier.staging.company.com`

-  **Live**

Push changes to branch: `live/oma_recipeclassifier`

App will be deployed to: `https://recipeclassifier.company.com`

### How to Read Logs

#### Option 1: Using the Console

Logs are captured by Kubernetes, so make sure you’re logged into the correct AWS environment and Kubernetes context.

- Log into AWS: `aws sso login --profile <aws_profile>`
- Switch to the correct Kubernetes environment: `kubectl config use-context <k8s_context>`

Run the following commands for logs:

- Dev: `make -C oma_recipeclassifier logs-dev`
- Staging: `make -C oma_recipeclassifier logs-staging`
- Live: `make -C oma_recipeclassifier logs-live`

#### Option 2: Using Grafana

- [View logs in Grafana Staging](https://staging.company.grafana.net/)

- [View logs in Grafana Live](https://company.grafana.net/)
