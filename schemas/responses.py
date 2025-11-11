from pydantic import BaseModel, Field
from typing import Dict, Optional

class DiagnosisInfo(BaseModel):
    class_: int = Field(..., alias="class")
    name: str
    full_name: str
    description: str
    confidence: float

class RiskInfo(BaseModel):
    level: int
    name: str
    description: str
    recommendation: str
    color: str
    urgency: str
    confidence: float

class SexInfo(BaseModel):
    code: float
    name: str

class LocalizationInfo(BaseModel):
    code: float
    name: str
    description: str

class DxTypeInfo(BaseModel):
    code: float
    name: str
    description: str

class MetadataInfo(BaseModel):
    age: float
    sex: SexInfo
    localization: LocalizationInfo
    dx_type: DxTypeInfo

class PredictionResponse(BaseModel):
    """
    Schema for prediction response with exact mappings from notebook
    """
    success: bool
    diagnosis: Optional[DiagnosisInfo] = None
    risk: Optional[RiskInfo] = None
    metadata: Optional[MetadataInfo] = None
    probabilities: Optional[Dict] = None
    error: Optional[str] = None

    class Config:
        allow_population_by_field_name = True