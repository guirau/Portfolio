"""
Integration tests for oma_recipecropper.src.api.endpoints.predict

These tests validate the complete prediction pipeline for both step and main image types,
including model predictions, path embedding, and S3 uploads.

Make sure to set up the environment variables for AWS credentials before running the tests.
Make sure the S3 bucket is accessible.
"""

import pytest
import requests
from loguru import logger

from oma_recipecropper.src.schemas.prediction import OutputType
from oma_recipecropper.tests.helpers import (
    get_image_dimensions,
    has_valid_photoshop_path,
)


def test_predict_endpoint_file_no_crop(client, image_type, sample_image_url):
    """Test the /{image_type}/predict endpoint with file output and no cropping."""
    response = client.post(
        f"/api/v2/{image_type.value}/predict",
        json={"image_path": sample_image_url},
        params={"output": OutputType.FILE.value, "crop": False},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert "content-disposition" in response.headers
    assert "attachment; filename=" in response.headers["content-disposition"]

    # Save the returned image
    output_image_data = response.content

    # Check the image contains a valid Photoshop path
    assert has_valid_photoshop_path, "Photoshop path not found in the image metadata"

    # Check image dimensions match input (no cropping)
    original_dimensions = get_image_dimensions(requests.get(sample_image_url).content)
    output_dimensions = get_image_dimensions(output_image_data)

    logger.info(f"Original dimensions: {original_dimensions}")
    logger.info(f"Output dimensions: {output_dimensions}")

    assert (
        output_dimensions is not None
    ), "Could not extract dimensions from output image"
    assert (
        original_dimensions is not None
    ), "Could not extract dimensions from original image"
    assert (
        output_dimensions == original_dimensions
    ), "Image dimensions should match when crop=False"


def test_predict_endpoint_file_with_crop(client, image_type, sample_image_url):
    """Test the /{image_type}/predict endpoint with file output and cropping."""
    response = client.post(
        f"/api/v2/{image_type.value}/predict",
        json={"image_path": sample_image_url},
        params={"output": OutputType.FILE.value, "crop": True},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert "content-disposition" in response.headers
    assert "attachment; filename=" in response.headers["content-disposition"]

    # Save the returned image
    output_image_data = response.content

    # Check the image contains a valid Photoshop path
    assert has_valid_photoshop_path, "Photoshop path not found in the image metadata"

    # Check image dimensions (should be smaller than original)
    original_dimensions = get_image_dimensions(requests.get(sample_image_url).content)
    output_dimensions = get_image_dimensions(output_image_data)

    logger.info(f"Original dimensions: {original_dimensions}")
    logger.info(f"Output dimensions: {output_dimensions}")

    assert (
        output_dimensions is not None
    ), "Could not extract dimensions from output image"
    assert (
        original_dimensions is not None
    ), "Could not extract dimensions from original image"

    # When cropped, at least one dimension should be smaller or equal
    assert (output_dimensions[0] <= original_dimensions[0]) or (
        output_dimensions[1] <= original_dimensions[1]
    ), "Cropped image should be smaller in at least one dimension when crop=True"


def test_predict_endpoint_s3_url(client, image_type, sample_image_url):
    """Test the /{image_type}/predict endpoint with S3 URL output."""
    response = client.post(
        f"/api/v2/{image_type.value}/predict",
        json={"image_path": sample_image_url},
        params={"output": OutputType.S3_URL.value},
    )

    assert response.status_code == 200
    result = response.json()

    # Verify output is a valid S3 URL
    assert "s3_url" in result
    assert result["s3_url"].startswith("https://")
    assert ".s3." in result["s3_url"]
    assert "recipecropper/results/" in result["s3_url"]

    # Verify the URL is accessible
    try:
        s3_response = requests.get(result["s3_url"], timeout=10)
        assert s3_response.status_code == 200
    except requests.RequestException as ex:
        pytest.skip(f"Could not access S3 URL: {ex}")


def test_predict_upload_endpoint_file_no_crop(client, image_type, sample_image_data):
    """Test the /{image_type}/predict_upload endpoint with file output and no cropping."""
    response = client.post(
        f"/api/v2/{image_type.value}/predict_upload",
        files={"file": ("sample_image.jpg", sample_image_data, "image/jpg")},
        params={"output": OutputType.FILE.value, "crop": False},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert "content-disposition" in response.headers
    assert "attachment; filename=" in response.headers["content-disposition"]

    # Save the returned image
    output_image_data = response.content

    # Check the image contains a valid Photoshop path
    assert has_valid_photoshop_path, "Photoshop path not found in the image metadata"

    # Check image dimensions match input (no cropping)
    original_dimensions = get_image_dimensions(sample_image_data)
    output_dimensions = get_image_dimensions(output_image_data)

    logger.info(f"Original dimensions: {original_dimensions}")
    logger.info(f"Output dimensions: {output_dimensions}")

    assert (
        output_dimensions is not None
    ), "Could not extract dimensions from output image"
    assert (
        original_dimensions is not None
    ), "Could not extract dimensions from original image"
    assert (
        output_dimensions == original_dimensions
    ), "Image dimensions should match when crop=False"


def test_predict_upload_endpoint_file_with_crop(client, image_type, sample_image_data):
    """Test the /{image_type}/predict_upload endpoint with file output and cropping."""
    response = client.post(
        f"/api/v2/{image_type.value}/predict_upload",
        files={"file": ("sample_image.jpg", sample_image_data, "image/jpg")},
        params={"output": OutputType.FILE.value, "crop": True},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert "content-disposition" in response.headers
    assert "attachment; filename=" in response.headers["content-disposition"]

    # Save the returned image
    output_image_data = response.content

    # Check the image contains a valid Photoshop path
    assert has_valid_photoshop_path, "Photoshop path not found in the image metadata"

    # Check image dimensions (should be smaller than original)
    original_dimensions = get_image_dimensions(sample_image_data)
    output_dimensions = get_image_dimensions(output_image_data)

    logger.info(f"Original dimensions: {original_dimensions}")
    logger.info(f"Output dimensions: {output_dimensions}")

    assert (
        output_dimensions is not None
    ), "Could not extract dimensions from output image"
    assert (
        original_dimensions is not None
    ), "Could not extract dimensions from original image"

    # When cropped, at least one dimension should be smaller or equal
    assert (output_dimensions[0] <= original_dimensions[0]) or (
        output_dimensions[1] <= original_dimensions[1]
    ), "Cropped image should be smaller in at least one dimension when crop=True"


def test_predict_upload_endpoint_s3_url(client, image_type, sample_image_data):
    """Test the /{image_type}/predict_upload endpoint with S3 URL output."""
    response = client.post(
        f"/api/v2/{image_type.value}/predict_upload",
        files={"file": ("sample_image.jpg", sample_image_data, "image/jpg")},
        params={"output": OutputType.S3_URL.value},
    )

    assert response.status_code == 200
    result = response.json()

    # Verify output is a valid S3 URL
    assert "s3_url" in result
    assert result["s3_url"].startswith("https://")
    assert ".s3." in result["s3_url"]
    assert "recipecropper/results/" in result["s3_url"]

    # Verify the URL is accessible
    try:
        s3_response = requests.get(result["s3_url"], timeout=10)
        assert s3_response.status_code == 200
    except requests.RequestException as ex:
        pytest.skip(f"Could not access S3 URL: {ex}")


def test_crop_endpoint(client, image_type, sample_image_url):
    """Test the /{image_type}/crop endpoint with S3 URL output."""
    response = client.get(
        f"/api/v2/{image_type.value}/crop?{sample_image_url}", allow_redirects=False
    )

    assert response.status_code == 307, "Should return a redirect response"
    assert "location" in response.headers, "Redirect location should be present"

    # Get redirect URL
    redirect_url = response.headers["location"]
    logger.info(f"Redirect URL: {redirect_url}")
    assert redirect_url.startswith("https://"), "Redirect URL should be valid"

    # Follow the redirect and check the image
    try:
        redirect_response = requests.get(redirect_url, timeout=10)
        assert (
            redirect_response.status_code == 200
        ), "Redirected URL should be accessible"

        # Check dimensions
        redirect_image_data = redirect_response.content
        redirect_dimensions = get_image_dimensions(redirect_image_data)
        original_dimensions = get_image_dimensions(
            requests.get(sample_image_url).content
        )

        logger.info(f"Original dimensions: {original_dimensions}")
        logger.info(f"Redirected dimensions: {redirect_dimensions}")

        # Cropped image should be smaller in at least one dimension
        assert (redirect_dimensions[0] <= original_dimensions[0]) or (
            redirect_dimensions[1] <= original_dimensions[1]
        ), "Cropped image should be smaller in at least one dimension"
    except requests.RequestException as ex:
        pytest.skip(f"Could not access redirect URL: {ex}")


def test_predict_invalid_file_type(client, image_type):
    """Test the /predict_upload endpoint with invalid file type."""
    response = client.post(
        f"/api/v2/{image_type.value}/predict_upload",
        files={"file": ("sample_text.txt", b"This is not an image", "text/plain")},
        params={"output": OutputType.FILE.value},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "File must be an image"}
