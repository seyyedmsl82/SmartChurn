"""
Data preprocessing pipeline for customer churn prediction
"""
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


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
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        X = X.copy()

        # Convert TotalCharges to numeric, handling ' ' or empty strings
        if 'TotalCharges' in X.columns:
            X['TotalCharges'] = pd.to_numeric(
                X['TotalCharges'].astype(str).str.strip().replace('', np.nan),
                errors='coerce'
            )
        
        # Ensure numeric columns are actually numeric
        numeric_cols = ['tenure', 'MonthlyCharges', 'SeniorCitizen']
        for col in numeric_cols:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # 1. Create tenure groups
        if 'tenure' in X.columns:
            try:
                X['tenure_group'] = pd.cut(
                    X['tenure'],
                    bins=[0, 6, 12, 24, 48, 72],
                    labels=['0-6', '6-12', '12-24', '24-48', '48-72']
                )
            except Exception as e:
                logger.warning(f"Could not create tenure_group: {e}")
                # Create a default tenure_group
                X['tenure_group'] = '0-6'
        
        # 2. Create average monthly spend (if TotalCharges and tenure exist)
        if 'TotalCharges' in X.columns and 'tenure' in X.columns and 'MonthlyCharges' in X.columns:
            # Handle potential zero tenure
            X['avg_monthly_spend'] = np.where(
                (X['tenure'] > 0) & (X['TotalCharges'].notna()),
                X['TotalCharges'] / X['tenure'],
                X['MonthlyCharges']
            )
            # Handle cases where MonthlyCharges might be NaN
            X['avg_monthly_spend'] = X['avg_monthly_spend'].fillna(X['MonthlyCharges'])
        
        # 3. Create service usage intensity
        service_cols = [
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        # Count number of services (excluding 'No' and 'No internet service')
        existing_service_cols = [col for col in service_cols if col in X.columns]
        if existing_service_cols:
            X['service_count'] = X[existing_service_cols].apply(
                lambda row: sum(
                    1 for val in row 
                    if val not in ['No', 'No internet service']
                ),
                axis=1
            )
        else:
            X['service_count'] = 0
        
        # 4. Create senior citizen indicator (keep as is, but ensure it's numeric)
        if 'SeniorCitizen' in X.columns:
            X['SeniorCitizen'] = pd.to_numeric(X['SeniorCitizen'], errors='coerce').fillna(0).astype(int)
        
        # 5. Fill any remaining NaN values in numeric columns
        numeric_engineered_cols = ['avg_monthly_spend', 'service_count', 'SeniorCitizen', 'tenure']
        for col in numeric_engineered_cols:
            if col in X.columns:
                X[col] = X[col].fillna(0)
        
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
        self.numeric_cols = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Calculate bounds for each numeric column"""
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Get numeric columns (excluding engineered categorical features)
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove columns that shouldn't be outlier-capped
        exclude_cols = ['service_count', 'SeniorCitizen']
        self.numeric_cols = [col for col in self.numeric_cols if col not in exclude_cols]
        
        for col in self.numeric_cols:
            # Skip if column has all NaN or no variance
            if X[col].isna().all() or X[col].std() == 0:
                self.lower_bounds[col] = X[col].min()
                self.upper_bounds[col] = X[col].max()
                continue
                
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            
            self.lower_bounds[col] = Q1 - self.multiplier * IQR
            self.upper_bounds[col] = Q3 + self.multiplier * IQR
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Cap outliers"""
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        X = X.copy()
        
        for col in self.numeric_cols:
            if col in X.columns:
                # Only cap if bounds are valid
                if col in self.lower_bounds and col in self.upper_bounds:
                    X[col] = X[col].clip(self.lower_bounds[col], self.upper_bounds[col])
        
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
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "avg_monthly_spend",
            "service_count",
        ]
        self.categorical_features = categorical_features or [
            'gender', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod', 'tenure_group',
        ]
        self.target_col = target_col
        self.handle_outliers = handle_outliers
        self.pipeline = None
        self.label_encoders = {}
        self.feature_names_ = []
        
        self._build_pipeline()
    
    def _build_pipeline(self) -> None:
        """Build the full preprocessing pipeline"""
        
        # Build numeric transformer
        numeric_steps = [
            ('imputer', SimpleImputer(strategy='median'))
        ]
        
        if self.handle_outliers:
            numeric_steps.append(('outlier_handler', OutlierHandler(multiplier=1.5)))
        
        numeric_steps.append(('scaler', StandardScaler()))
        
        numeric_transformer = Pipeline(numeric_steps)
        
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
        
        # Full pipeline: Feature engineering FIRST, then preprocessing
        self.pipeline = Pipeline([
            ('feature_engineer', FeatureEngineer()),
            ('preprocessor', self.preprocessor)
        ])
    
    def fit_transform(self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None
    ) -> np.ndarray:
        """
        Fit and transform the data
        
        Args:
            X: Feature DataFrame
            y: Target Series (optional)
            
        Returns:
            Transformed feature matrix
        """
        # Make a copy to avoid modifying original
        X = X.copy()
        
        # Separate features and target if needed
        if self.target_col in X.columns:
            X = X.drop(columns=[self.target_col])
        
        # Encode target if provided
        if y is not None:
            self._encode_target(y)
        
        logger.info(f"Fitting and transforming data with {len(X)} rows")
        
        # Fit and transform
        transformed = self.pipeline.fit_transform(X)
        
        logger.info(f"Transformed data shape: {transformed.shape}")
        
        # Store feature names for later use
        self._extract_feature_names()
        
        return transformed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted pipeline
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Transformed feature matrix
        """
        X = X.copy()
        
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
    
    def _extract_feature_names(self) -> None:
        """Extract feature names from the fitted pipeline"""
        try:
            # Try to get feature names from the preprocessor
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                # Get numeric feature names
                num_features = self.numeric_features.copy()
                
                # Get categorical feature names
                if hasattr(self.preprocessor.named_transformers_['cat'], 'get_feature_names_out'):
                    cat_features = self.preprocessor.named_transformers_['cat'].get_feature_names_out(
                        self.categorical_features
                    ).tolist()
                else:
                    cat_features = []
                
                # Combine all feature names
                self.feature_names_ = num_features + cat_features
                
        except Exception as e:
            logger.warning(f"Could not extract feature names: {e}")
            self.feature_names_ = []
    
    def get_feature_names(self) -> List[str]:
        """Get names of transformed features"""
        if self.feature_names_:
            return self.feature_names_
        
        # Fallback: generate generic names
        if hasattr(self.pipeline, 'get_feature_names_out'):
            try:
                return self.pipeline.get_feature_names_out().tolist()
            except:
                pass
        
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


def handle_missing_values(X: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Handle missing values in features
    
    Args:
        X: Feature DataFrame
        strategy: Imputation strategy ('mean', 'median', 'most_frequent', 'constant')
        
    Returns:
        DataFrame with missing values handled
    """
    # Check for missing values
    missing_cols = X.columns[X.isnull().any()].tolist()
    
    if not missing_cols:
        logger.info("   No missing values found")
        return X
    
    logger.info(f"   Found missing values in {len(missing_cols)} columns: {missing_cols[:5]}...")
    
    # Separate numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    
    X_clean = X.copy()
    
    # Impute numeric columns
    if len(numeric_cols) > 0:
        numeric_missing = [col for col in numeric_cols if col in missing_cols]
        if numeric_missing:
            logger.info(f"   Imputing {len(numeric_missing)} numeric columns with {strategy}")
            imputer = SimpleImputer(strategy=strategy)
            X_clean[numeric_cols] = imputer.fit_transform(X_clean[numeric_cols])
    
    # Impute categorical columns
    if len(categorical_cols) > 0:
        categorical_missing = [col for col in categorical_cols if col in missing_cols]
        if categorical_missing:
            logger.info(f"   Imputing {len(categorical_missing)} categorical columns with 'most_frequent'")
            imputer = SimpleImputer(strategy='most_frequent')
            X_clean[categorical_cols] = imputer.fit_transform(X_clean[categorical_cols])
    
    return X_clean
