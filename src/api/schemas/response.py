"""
Response schemas for API
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class PredictionResponse(BaseModel):
    """Single prediction response"""
    prediction: int = Field(..., description="Predicted churn (0=No, 1=Yes)")
    probability: float = Field(..., ge=0, le=1, description="Churn probability")
    churn_risk: str = Field(..., description="Risk level: Low/Medium/High")
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 0,
                "probability": 0.23,
                "churn_risk": "Low",
                "model_version": "RandomForest_v1.0.0",
                "timestamp": "2024-01-15T10:30:00"
            }
        }

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_count: int
    churn_count: int
    churn_rate: float
    model_version: str
    timestamp: str

class ExplanationResponse(BaseModel):
    """SHAP explanation response"""
    features: Dict[str, float] = Field(..., description="Feature values")
    shap_values: Dict[str, float] = Field(..., description="SHAP values")
    prediction: int
    probability: float
    model_version: str
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool
    model_version: Optional[str]
    preprocessor_loaded: bool
    timestamp: str

class MetricsResponse(BaseModel):
    """Metrics response"""
    model_version: str
    metrics: Dict[str, Any]
    last_updated: str
