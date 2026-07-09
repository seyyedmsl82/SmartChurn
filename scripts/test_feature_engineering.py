"""
Manual test for feature engineering
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.features.engineering import AdvancedFeatureEngineer
from src.features.selection import FeatureSelector
from loguru import logger

def test_feature_engineering():
    """Test feature engineering on real data"""
    
    logger.info("=" * 60)
    logger.info("Testing Feature Engineering")
    logger.info("=" * 60)
    
    # 1. Load data
    logger.info("\n[1/3] Loading data...")
    loader = DataLoader()
    
    try:
        df = loader.load_from_csv('data/processed/telco_churn_processed.csv')
    except:
        # Try raw
        df = loader.load_from_csv('data/raw/telco_churn.csv')
    
    logger.info(f"   Loaded {len(df)} rows")
    
    # 2. Preprocess basic features
    logger.info("\n[2/3] Engineering features...")
    
    # Separate target
    y = df['Churn']
    X = df.drop(columns=['Churn'])
    
    # Apply feature engineering
    engineer = AdvancedFeatureEngineer()
    X_engineered = engineer.transform(X)
    
    logger.info(f"   Original features: {X.shape[1]}")
    logger.info(f"   Engineered features: {X_engineered.shape[1]}")
    logger.info(f"   New features added: {X_engineered.shape[1] - X.shape[1]}")
    
    # Show new features
    new_features = set(X_engineered.columns) - set(X.columns)
    logger.info(f"   New features: {list(new_features)[:10]}...")
    
    # 3. Feature selection
    logger.info("\n[3/3] Selecting features...")
    
    # For selection, handle categorical variables
    # Convert categorical to numeric for selection
    X_encoded = pd.get_dummies(X_engineered, drop_first=True)
    X_encoded = X_encoded.fillna(0)
    
    selector = FeatureSelector(method='combined', n_features=30)
    selected = selector.select_features(X_encoded, y)
    selector.plot_feature_importance(X_encoded, y, top_n=20)
    
    logger.info(f"   Selected {len(selected)} features")
    
    # Show selected features
    logger.info(f"   Selected features: {selected[:10]}...")
    
    # 4. Report
    logger.info("\n" + "=" * 60)
    logger.info("Feature Engineering Summary:")
    logger.info(f"  Original features: {X.shape[1]}")
    logger.info(f"  After engineering: {X_engineered.shape[1]}")
    logger.info(f"  After selection: {len(selected)}")
    logger.info("=" * 60)
    
    return X_engineered, selected

if __name__ == "__main__":
    X_engineered, selected = test_feature_engineering()
    
    # Save selected features for later use
    import json
    with open('models/selected_features.json', 'w') as f:
        json.dump(selected, f)
    
    logger.success("\nFeature engineering tests completed!")
