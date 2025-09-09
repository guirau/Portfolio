"""Pydantic schemas for prediction."""

from pydantic import BaseModel


class Prediction(BaseModel):
    """Response schema for classification prediction."""

    keyword: str
    confidence: float
