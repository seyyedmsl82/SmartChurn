"""
Data preprocessing pipeline for customer churn prediction
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import List, Optional, Tuple
from loguru import logger
import pickle
from pathlib import Path


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom feature engineering transformer
    """
    def __init__(self):
        self.feature_names_out_ = None
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering"""
        X = X.copy()
        
        # 1. Create tenure groups
        X['tenure_group'] = pd.cut(
            X['tenure'],
            bins=[0, 6, 12, 24, 48, 72],
            labels=['0-6', '6-12', '12-24', '24-48', '48-72']
        )
        
        # 2. Create average monthly spend (if TotalCharges and tenure exist)
        if 'TotalCharges' in X.columns and 'tenure' in X.columns:
            # Handle potential zero tenure
            X['avg_monthly_spend'] = np.where(
                X['tenure'] > 0,
                X['TotalCharges'] / X['tenure'],
                X['MonthlyCharges']
            )
        
        # 3. Create service usage intensity
        service_cols = [
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        # Count number of services (excluding 'No' and 'No internet service')
        if all(col in X.columns for col in service_cols):
            X['service_count'] = X[service_cols].apply(
                lambda row: sum(
                    1 for val in row 
                    if val not in ['No', 'No internet service']
                ),
                axis=1
            )
        
        # 4. Create senior citizen indicator (keep as is, but ensure it's numeric)
        if 'SeniorCitizen' in X.columns:
            X['SeniorCitizen'] = X['SeniorCitizen'].astype(int)
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names"""
        if self.feature_names_out_ is None:
            return ['tenure_group', 'avg_monthly_spend', 'service_count']
        return self.feature_names_out_

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Handle outliers using IQR method
    """
    
    def __init__(self, multiplier: float = 1.5):
        self.multiplier = multiplier
        self.upper_bounds = {}
        self.lower_bounds = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        """Calculate bounds for each numeric column"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            
            self.lower_bounds[col] = Q1 - self.multiplier * IQR
            self.upper_bounds[col] = Q3 + self.multiplier * IQR
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Cap outliers"""
        X = X.copy()
        
        for col, (lower, upper) in zip(
            self.lower_bounds.keys(),
            zip(self.lower_bounds.values(), self.upper_bounds.values())
        ):
            if col in X.columns:
                X[col] = X[col].clip(lower, upper)
        
        return X

class DataPreprocessor:
    """
    Main preprocessing pipeline for customer churn data
    """
    
    def __init__(
        self,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        target_col: str = 'Churn',
        handle_outliers: bool = True
    ):
        """
        Initialize the preprocessor
        
        Args:
            numeric_features: List of numeric column names
            categorical_features: List of categorical column names  
            target_col: Name of target column
            handle_outliers: Whether to handle outliers
        """
        self.numeric_features = numeric_features or [
            'tenure', 'MonthlyCharges', 'TotalCharges'
        ]
        self.categorical_features = categorical_features or [
            'gender', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod'
        ]
        self.target_col = target_col
        self.handle_outliers = handle_outliers
        self.pipeline = None
        self.label_encoders = {}
        
        self._build_pipeline()
    
    def _build_pipeline(self) -> None:
        """Build the full preprocessing pipeline"""
        
        # Numeric transformer
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('outlier_handler', OutlierHandler(multiplier=1.5)) if self.handle_outliers else ('passthrough', 'passthrough'),
            ('scaler', StandardScaler())
        ])
        
        # Categorical transformer
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'
        )
        
        # Full pipeline with feature engineering
        self.pipeline = Pipeline([
            ('feature_engineer', FeatureEngineer()),
            ('preprocessor', self.preprocessor)
        ])
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """
        Fit and transform the data
        
        Args:
            X: Feature DataFrame
            y: Target Series (optional)
            
        Returns:
            Transformed feature matrix
        """
        # Separate features and target if needed
        if self.target_col in X.columns:
            X = X.drop(columns=[self.target_col])
        
        # Encode target if provided
        if y is not None:
            self._encode_target(y)
        
        logger.info(f"Fitting and transforming data with {len(X)} rows")
        transformed = self.pipeline.fit_transform(X)
        logger.info(f"Transformed data shape: {transformed.shape}")
        
        return transformed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted pipeline
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Transformed feature matrix
        """
        if self.target_col in X.columns:
            X = X.drop(columns=[self.target_col])
        
        logger.info(f"Transforming data with {len(X)} rows")
        return self.pipeline.transform(X)
    
    def _encode_target(self, y: pd.Series) -> None:
        """Encode target variable"""
        le = LabelEncoder()
        le.fit(y)
        self.label_encoders[self.target_col] = le
        logger.info(f"Target encoded with classes: {le.classes_}")
    
    def get_feature_names(self) -> List[str]:
        """Get names of transformed features"""
        if hasattr(self.pipeline, 'get_feature_names_out'):
            return self.pipeline.get_feature_names_out()
        return []
    
    def save(self, path: str) -> None:
        """
        Save the preprocessor to disk
        
        Args:
            path: Path to save the preprocessor
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
        logger.info(f"Preprocessor saved to {path}")
    
    @staticmethod
    def load(path: str) -> 'DataPreprocessor':
        """
        Load a saved preprocessor
        
        Args:
            path: Path to the saved preprocessor
            
        Returns:
            Loaded DataPreprocessor instance
        """
        with open(path, 'rb') as f:
            preprocessor = pickle.load(f)
        
        logger.info(f"Preprocessor loaded from {path}")
        return preprocessor
