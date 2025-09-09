"""Fixture functions for testing oma_recipeclassifier"""

import os

import pytest
from fastapi.testclient import TestClient

from oma_recipeclassifier.src.main import app


@pytest.fixture(scope="session")
def sample_image_path():
    """Path to a sample image file."""
    return os.path.join(os.path.dirname(__file__), "data", "sample_image.jpg")


@pytest.fixture(scope="session")
def sample_image_data(sample_image_path):
    """Sample image data as bytes."""
    with open(sample_image_path, "rb") as f:
        return f.read()


@pytest.fixture(scope="session")
def sample_image_url():
    """Sample image URL for testing."""
    return "https://img.company.com/step-39164f20.jpg"


@pytest.fixture()
def client():
    """FastAPI test client."""
    with TestClient(app) as test_client:
        yield test_client
