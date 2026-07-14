"""
Metrics routes
"""
from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime
import json
from pathlib import Path

from src.api.schemas.response import MetricsResponse
from src.api.dependencies import get_prediction_service
from src.api.services.prediction_service import PredictionService

router = APIRouter(prefix="/api/v1/metrics", tags=["Metrics"])

@router.get("", response_model=MetricsResponse)
async def get_metrics(
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Get model performance metrics
    """
    try:
        # Load evaluation report
        report_path = Path("reports/evaluation_report.json")
        
        if report_path.exists():
            with open(report_path, 'r') as f:
                report = json.load(f)
            
            metrics = report.get('metrics', {})
        else:
            metrics = {"message": "Evaluation report not found"}
        
        return MetricsResponse(
            model_version=service.model_info.get('version', 'unknown'),
            metrics=metrics,
            last_updated=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
