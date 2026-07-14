"""
Prediction routes
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List
from loguru import logger

from src.api.schemas.request import PredictionRequest, BatchPredictionRequest
from src.api.schemas.response import PredictionResponse, BatchPredictionResponse
from src.api.dependencies import get_prediction_service
from src.api.services.prediction_service import PredictionService

router = APIRouter(prefix="/api/v1/predict", tags=["Prediction"])

@router.post("/single", response_model=PredictionResponse)
async def predict_single(
    request: PredictionRequest,
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Make a single prediction for a customer
    """
    try:
        logger.info(f"Single prediction request for customer")
        
        result = service.predict_single(request.customer)
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Make predictions for multiple customers
    """
    try:
        logger.info(f"Batch prediction request for {len(request.customers)} customers")
        
        results = service.predict_batch(request.customers)
        
        # Calculate stats
        churn_count = sum(1 for r in results if r['prediction'] == 1)
        total_count = len(results)
        
        # Convert to response objects
        predictions = [PredictionResponse(**r) for r in results]
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_count=total_count,
            churn_count=churn_count,
            churn_rate=churn_count / total_count if total_count > 0 else 0,
            model_version=results[0]['model_version'] if results else 'unknown',
            timestamp=results[0]['timestamp'] if results else ''
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
