from pydantic import BaseModel
from typing import Dict

class PatientRecord(BaseModel):
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    probability: float
    prediction: int
