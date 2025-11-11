"""
Pydantic schemas for request/response validation
"""

from app.schemas.requests import PredictionRequest
from app.schemas.responses import PredictionResponse

__all__ = ["PredictionRequest", "PredictionResponse"]