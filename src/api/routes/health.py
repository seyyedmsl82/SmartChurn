"""
Health check routes
"""
from fastapi import APIRouter, Depends
from datetime import datetime

from src.api.schemas.response import HealthResponse
from src.api.dependencies import get_prediction_service
from src.api.services.prediction_service import PredictionService

router = APIRouter(prefix="/api/v1/health", tags=["Health"])

@router.get("", response_model=HealthResponse)
async def health_check(
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Health check endpoint
    """
    metadata = service.get_metadata()
    
    return HealthResponse(
        status="healthy",
        model_loaded=service.model is not None,
        model_version=metadata['model_version'],
        preprocessor_loaded=service.preprocessor is not None,
        timestamp=datetime.now().isoformat()
    )

@router.get("/readiness")
async def readiness_check():
    """
    Readiness probe for Kubernetes
    """
    return {"status": "ready"}

@router.get("/liveness")
async def liveness_check():
    """
    Liveness probe for Kubernetes
    """
    return {"status": "alive"}
