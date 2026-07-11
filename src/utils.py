import json
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger
from sklearn.preprocessing import LabelEncoder
from src.data.preprocessor import handle_missing_values
from src.features.selection import FeatureSelector, map_selected_features


def encode_target(y: pd.Series) -> tuple:
    """
    Encode target variable to binary values
    
    Args:
        y: Target Series with string values
        
    Returns:
        Tuple of (encoded_y, label_encoder)
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    logger.info(f"   Target encoded: {le.classes_} -> {list(range(len(le.classes_)))}")
    return y_encoded, le

def load_selected_features(
    X_engineered: pd.DataFrame, 
    selected_features_path: Path = Path('models/selected_features.json')
) -> tuple:
    """
    Load previously selected features from file
    
    Args:
        X_engineered: Engineered features
        selected_features_path: Path to the JSON file containing selected features
        
    Returns:
        Tuple of (selected_features, X_selected)
    """
    selected_features_path = Path(selected_features_path)
    
    if not selected_features_path.exists():
        logger.warning("   No selected features file found")
        return None, X_engineered
    
    logger.info("   Loading existing selected features...")
    with open(selected_features_path, 'r') as f:
        selected_features = json.load(f)
    logger.info(f"   Loaded {len(selected_features)} selected features")
    
    # Filter X_engineered to only selected features
    available_features = [f for f in selected_features if f in X_engineered.columns]
    
    if not available_features:
        logger.warning("   No selected features found in data, using all features")
        return None, X_engineered
    
    if len(available_features) < len(selected_features):
        logger.warning(f"   Only {len(available_features)} of {len(selected_features)} features available")
    
    X_selected = X_engineered[available_features]
    logger.info(f"   Using {len(available_features)} available selected features")
    
    return available_features, X_selected

def run_feature_selection(X_engineered: pd.DataFrame, y: np.ndarray, 
                          n_features: int = 30, 
                          method: str = 'combined') -> tuple:
    """
    Run feature selection and return selected features
    
    Args:
        X_engineered: Engineered features
        y: Target variable
        n_features: Number of features to select
        method: Selection method
        
    Returns:
        Tuple of (selected_features, X_selected)
    """
    logger.info(f"   Running feature selection (n_features={n_features}, method='{method}')...")
    
    # Handle missing values before selection
    X_clean = handle_missing_values(X_engineered)
    
    # Encode categorical for selection
    X_encoded = pd.get_dummies(X_clean, drop_first=True)
    X_encoded = X_encoded.fillna(0)
    
    logger.info(f"   Encoded features: {X_encoded.shape[1]}")
    
    # Run feature selection
    selector = FeatureSelector(method=method, n_features=n_features)
    selected_encoded = selector.select_features(X_encoded, y)
    
    # Map back to original feature names
    selected_features = map_selected_features(selected_encoded, X_engineered.columns.tolist())
    
    # Remove duplicates while preserving order
    selected_features = list(dict.fromkeys(selected_features))
    
    logger.info(f"   Mapped to {len(selected_features)} original features")
    
    # Save selected features
    with open('models/selected_features.json', 'w') as f:
        json.dump(selected_features, f)
    logger.info(f"   Selected {len(selected_features)} features")
    logger.info(f"   Selected features: {selected_features[:5]}...")
    
    # Filter to selected features
    available_features = [f for f in selected_features if f in X_engineered.columns]
    X_selected = X_engineered[available_features]
    
    return selected_features, X_selected
