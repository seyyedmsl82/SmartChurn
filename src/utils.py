import json
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder


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
