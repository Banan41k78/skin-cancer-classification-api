from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    age: float = Field(..., ge=0, le=120, description="Patient age")
    sex: float = Field(..., ge=0, le=2, description="Sex (0-male, 1-female, 2-unknown)")
    localization: float = Field(..., ge=0, le=14, description="Lesion localization")
    dx_type: float = Field(..., ge=0, le=3, description="Diagnosis type")

    class Config:
        schema_extra = {
            "example": {
                "age": 45,
                "sex": 1,
                "localization": 5,
                "dx_type": 0
            }
        }