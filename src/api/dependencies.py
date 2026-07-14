"""
API dependencies
"""
from functools import lru_cache
from src.api.services.prediction_service import PredictionService
from src.api.config import settings

@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    """
    Get prediction service instance (cached)
    """
    return PredictionService(
        model_path=settings.MODEL_PATH,
        preprocessor_path=settings.PREPROCESSOR_PATH,
        features_path=settings.FEATURES_PATH
    )
