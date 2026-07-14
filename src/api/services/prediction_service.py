"""
Prediction service - core business logic
"""
import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from loguru import logger

from src.data.preprocessor import DataPreprocessor
from src.features.engineering import AdvancedFeatureEngineer
from src.models.registry import ModelRegistry
from src.api.schemas.request import CustomerFeatures, PredictionRequest
from src.api.schemas.response import PredictionResponse

class PredictionService:
    """Service for making predictions"""
    
    def __init__(self, 
                 model_path: str = "models/registry",
                 preprocessor_path: str = "models/preprocessor.pkl",
                 features_path: str = "models/selected_features.json"):
        """
        Initialize prediction service
        """
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.features_path = Path(features_path)
        
        # Load components
        self.model = None
        self.model_info = None
        self.preprocessor = None
        self.selected_features = None
        self.engineer = AdvancedFeatureEngineer()
        
        self._load_components()
    
    def _load_components(self):
        """Load all required components"""
        try:
            # Load model from registry
            registry = ModelRegistry(self.model_path)
            self.model, self.model_info = registry.load_model(version='latest')
            
            # Load preprocessor
            self.preprocessor = DataPreprocessor.load(self.preprocessor_path)
            
            # Load selected features
            if self.features_path.exists():
                with open(self.features_path, 'r') as f:
                    self.selected_features = json.load(f)
            
            logger.info(f"Prediction service initialized with model {self.model_info.get('version', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to load components: {e}")
            raise
    
    def predict_single(self, customer: CustomerFeatures) -> Dict[str, Any]:
        """
        Make prediction for a single customer
        """
        print(customer)
        # Convert to DataFrame
        df = pd.DataFrame([customer.dict()])
        
        # Preprocess
        X_preprocessed = self.preprocessor.transform(df)
        
        # Feature engineering
        try:
            # Get feature names from preprocessor if available
            feature_names = self.preprocessor.get_feature_names()
            X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=feature_names)
        except:
            X_preprocessed_df = pd.DataFrame(X_preprocessed)
        
        X_engineered = self.engineer.transform(X_preprocessed_df)
        
        # Select features if needed
        if self.selected_features:
            available_features = [f for f in self.selected_features if f in X_engineered.columns]
            if available_features:
                X_final = X_engineered[available_features]
            else:
                X_final = X_engineered
        else:
            X_final = X_engineered
        
        # Make prediction
        prediction = int(self.model.predict(X_final)[0])
        print(prediction)
        
        try:
            probability = float(self.model.predict_proba(X_final)[0, 1])
        except:
            probability = 0.5
        
        # Determine risk level
        if probability < 0.5:
            risk = "Low"
        elif probability < 0.65:
            risk = "Medium"
        else:
            risk = "High"
        
        return {
            'prediction': prediction,
            'probability': probability,
            'churn_risk': risk,
            'model_version': self.model_info.get('version', 'unknown'),
            'timestamp': datetime.now().isoformat()
        }
    
    def predict_batch(self, customers: List[CustomerFeatures]) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple customers
        """
        results = []
        for customer in customers:
            result = self.predict_single(customer)
            results.append(result)
        
        return results
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get service metadata"""
        return {
            'model_version': self.model_info.get('version', 'unknown'),
            'model_name': self.model_info.get('model_name', 'unknown'),
            'features_count': len(self.selected_features) if self.selected_features else 'N/A',
            'preprocessor_loaded': self.preprocessor is not None,
            'timestamp': datetime.now().isoformat()
        }
