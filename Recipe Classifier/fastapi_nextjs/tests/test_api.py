"""
Integration tests for oma_recipeclassifier.src.api.endpoints.predict

These tests validate the complete prediction pipeline using a real instance of ModelService,
including model predictions, keyword embedding, and S3 uploads.

Make sure to set up the environment variables for AWS credentials before running the tests.
Make sure the S3 bucket '<AWS_BUCKET_NAME>' is accessible.
"""

import os
import tempfile

import pyexiv2
import pytest
import requests
from loguru import logger
from PIL import Image

from oma_recipeclassifier.src.schemas.prediction import OutputType


def test_predict_endpoint_predictions(client, sample_image_data):
    """Test the /predict endpoint with predictions output."""
    response = client.post(
        "/api/v1/predict",
        files={"file": ("sample_image.jpg", sample_image_data, "image/jpg")},
        params={"output": OutputType.PREDICTIONS.value},
    )

    assert response.status_code == 200
    predictions = response.json()

    # Verify the response structure
    assert isinstance(predictions, list)
    assert len(predictions) > 0
    assert all("keyword" in item for item in predictions)
    assert all("confidence" in item for item in predictions)

    # Print top 3 predictions for debugging
    for pred in predictions:
        logger.info(
            f"Prediction: {pred['keyword']} - Confidence: {pred['confidence']:.2f}%"
        )

    # Verify confidence values make sense
    assert 0 <= predictions[0]["confidence"] <= 100

    # Assert top prediction has highest confidence
    assert all(
        predictions[0]["confidence"] >= pred["confidence"] for pred in predictions[1:]
    )

    # Check if top prediction is the expected keyword
    assert predictions[0]["keyword"] == "pan", "Top prediction keyword should be 'pan'"


def test_predict_endpoint_file(client, sample_image_data):
    """Test the /predict endpoint with file output and metadata verification."""
    response = client.post(
        "/api/v1/predict",
        files={"file": ("sample_image.jpg", sample_image_data, "image/jpg")},
        params={"output": OutputType.FILE.value},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpg"
    assert "content-disposition" in response.headers
    assert "attachment; filename=" in response.headers["content-disposition"]

    # Save the returned image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name

    try:
        expected_keyword = "pan"

        # Very image is valid
        img = Image.open(tmp_path)
        assert img.format in ["JPEG", "PNG"]

        # Verify metadata contains expected keyword
        with pyexiv2.Image(tmp_path) as img_metadata:
            iptc_data = img_metadata.read_iptc()

            # Verify keyword exists
            assert "Iptc.Application2.Keywords" in iptc_data
            keywords = iptc_data["Iptc.Application2.Keywords"]
            assert len(keywords) > 0

            logger.info(f"Expected keyword: {expected_keyword}")
            logger.info(f"Embedded keywords: {keywords}")

            # Check expected keyword is present
            assert any(
                expected_keyword.lower() in keyword.lower() for keyword in keywords
            )
    finally:
        # Clean up tmp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_predict_endpoint_s3_url(client, sample_image_data):
    """Test the /predict endpoint with S3 URL output."""
    response = client.post(
        "/api/v1/predict",
        files={"file": ("sample_image.jpg", sample_image_data, "image/jpg")},
        params={"output": OutputType.S3_URL.value},
    )

    assert response.status_code == 200
    result = response.json()

    # Verify output is a valid S3 URL
    assert "s3_url" in result
    assert result["s3_url"].startswith("https://")
    assert ".s3." in result["s3_url"]
    assert "recipeclassifier/results/" in result["s3_url"]

    # Verify the URL is accessible
    try:
        s3_response = requests.get(result["s3_url"], timeout=10)
        assert s3_response.status_code == 200
    except requests.RequestException as ex:
        pytest.skip(f"Could not access S3 URL: {ex}")

    # No check for specific keywords in S3 URL response
    # Assuming the keyword embedding is done correctly if test_predict_endpoint_file passes


def test_classify_endpoint(client, sample_image_url):
    """Test classify endpoint with image URL."""
    response = client.get(f"/api/v1/classify?{sample_image_url}")

    assert response.status_code == 200
    predictions = response.json()

    # Verify the response structure
    assert isinstance(predictions, list)
    assert len(predictions) > 0
    assert all("keyword" in item for item in predictions)
    assert all("confidence" in item for item in predictions)

    # Print top 3 predictions for debugging
    for pred in predictions:
        logger.info(
            f"Prediction: {pred['keyword']} - Confidence: {pred['confidence']:.2f}%"
        )

    # Verify confidence values make sense
    assert 0 <= predictions[0]["confidence"] <= 100

    # Assert top prediction has highest confidence
    assert all(
        predictions[0]["confidence"] >= pred["confidence"] for pred in predictions[1:]
    )

    # Check if top prediction is the expected keyword
    assert (
        predictions[0]["keyword"] == "medium"
    ), "Top prediction keyword should be 'medium'"


def test_predict_endpoint_invalid_file_type(client):
    """Test the /predict endpoint with an invalid file type."""
    response = client.post(
        "/api/v1/predict",
        files={"file": ("sample_text.txt", b"This is not an image", "text/plain")},
        params={"output": OutputType.PREDICTIONS.value},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "File must be an image"}
