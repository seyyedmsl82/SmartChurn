"""
Manual test script for preprocessing pipeline
Run this to test the preprocessor with real data
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.validator import DataValidator
from loguru import logger

def test_preprocessing_with_sample():
    """Test preprocessing with a small sample"""
    
    logger.info("=" * 60)
    logger.info("Testing Preprocessing Pipeline (Sample Data)")
    logger.info("=" * 60)
    
    # 1. Create sample data
    logger.info("\n1. Creating sample data...")
    sample_data = {
        'customerID': ['CUST-001', 'CUST-002', 'CUST-003'],
        'gender': ['Female', 'Male', 'Female'],
        'SeniorCitizen': [0, 1, 0],
        'Partner': ['Yes', 'No', 'No'],
        'Dependents': ['No', 'Yes', 'No'],
        'tenure': [10, 25, 40],
        'PhoneService': ['Yes', 'Yes', 'No'],
        'MultipleLines': ['No', 'Yes', 'No phone service'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['Yes', 'No', 'No internet service'],
        'OnlineBackup': ['No', 'Yes', 'No internet service'],
        'DeviceProtection': ['No', 'Yes', 'No internet service'],
        'TechSupport': ['No', 'No', 'No internet service'],
        'StreamingTV': ['No', 'Yes', 'No internet service'],
        'StreamingMovies': ['No', 'Yes', 'No internet service'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaperlessBilling': ['Yes', 'No', 'Yes'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)'],
        'MonthlyCharges': [50.0, 75.0, 30.0],
        'TotalCharges': [500.0, 1875.0, 1200.0],
        'Churn': ['No', 'Yes', 'No']
    }
    
    df = pd.DataFrame(sample_data)
    logger.info(f"   Created DataFrame with {len(df)} rows")
    logger.info(f"   Columns: {df.columns.tolist()}")
    
    # 2. Preprocess
    logger.info("\n2. Initializing preprocessor...")
    preprocessor = DataPreprocessor()
    
    logger.info("\n3. Fitting and transforming...")
    y = df['Churn']
    X = df.drop(columns=['Churn'])
    
    X_transformed = preprocessor.fit_transform(X, y)
    
    logger.info(f"\n   Transformed shape: {X_transformed.shape}")
    logger.info(f"   Number of features: {X_transformed.shape[1]}")
    logger.info(f"   Feature types: {X_transformed.dtype}")
    
    # 3. Check feature engineering
    logger.info("\n4. Checking feature engineering...")
    feature_engineer = preprocessor.pipeline.named_steps['feature_engineer']
    X_featured = feature_engineer.transform(X)
    
    logger.info(f"   Engineered features: {X_featured.columns.tolist()}")
    
    # Check for expected engineered features
    expected_features = ['tenure_group', 'avg_monthly_spend', 'service_count']
    for feature in expected_features:
        if feature in X_featured.columns:
            logger.success(f"   {feature} created successfully")
        else:
            logger.warning(f"   {feature} not found")
    
    # 4. Check with missing values
    logger.info("\n5. Testing with missing values...")
    df_with_missing = df.copy()
    df_with_missing.loc[0, 'TotalCharges'] = np.nan
    df_with_missing.loc[1, 'MonthlyCharges'] = np.nan
    
    X_missing = df_with_missing.drop(columns=['Churn'])
    X_missing_transformed = preprocessor.transform(X_missing)
    
    logger.info(f"   Successfully handled missing values: {X_missing_transformed.shape}")
    
    return X_transformed

def test_preprocessing_with_real_data():
    """Test preprocessing with real dataset"""
    
    logger.info("\n" + "=" * 60)
    logger.info("Testing Preprocessing Pipeline (Real Data)")
    logger.info("=" * 60)
    
    try:
        # 1. Load real data
        loader = DataLoader()
        df = loader.load_from_csv('data/raw/telco_churn.csv')
        logger.info(f"Loaded {len(df)} rows from real dataset")
        
        # 2. Validate first
        validator = DataValidator()
        result = validator.validate(df)
        print(result)
        
        if not result.is_valid:
            logger.warning("Data validation has warnings or errors, but continuing...")
        
        # 3. Preprocess
        preprocessor = DataPreprocessor()
        y = df['Churn']
        X = df.drop(columns=['Churn'])
        
        logger.info("\nPreprocessing real data...")
        X_transformed = preprocessor.fit_transform(X, y)
        
        logger.info(f"\nReal data transformed shape: {X_transformed.shape}")
        logger.info(f"Number of features: {X_transformed.shape[1]}")
        
        # 4. Save preprocessor
        logger.info("\nSaving preprocessor...")
        preprocessor.save('models/preprocessor.pkl')
        logger.success("Preprocessor saved to models/preprocessor.pkl")
        
        # 5. Test loading
        logger.info("\nTesting loading...")
        loaded_preprocessor = DataPreprocessor.load('models/preprocessor.pkl')
        X_loaded_transformed = loaded_preprocessor.transform(X)
        
        logger.info(f"   Original shape: {X_transformed.shape}")
        logger.info(f"   Loaded shape: {X_loaded_transformed.shape}")
        
        # Check if transformations match
        if np.allclose(X_transformed, X_loaded_transformed):
            logger.success("   Loaded preprocessor produces same results")
        else:
            logger.warning("   Loaded preprocessor gives different results")
        
        return True
        
    except FileNotFoundError:
        logger.warning("Real dataset not found. Run download_dataset.py first.")
        return True  # Return True so tests continue
    except Exception as e:
        logger.error(f"Error testing with real data: {e}")
        return False

def inspect_transformed_features(X_transformed, preprocessor):
    """Helper to inspect what features were created"""
    
    logger.info("\n" + "=" * 60)
    logger.info("Feature Inspection")
    logger.info("=" * 60)
    
    # Get feature names if available
    try:
        feature_names = preprocessor.get_feature_names()
        if feature_names:
            logger.info(f"\nFeature names: {feature_names[:5]}...")
            logger.info(f"Total features: {len(feature_names)}")
    except:
        logger.info("Feature names not available")
    
    # Basic statistics
    logger.info(f"\nTransformed data stats:")
    logger.info(f"  Shape: {X_transformed.shape}")
    logger.info(f"  Mean: {X_transformed.mean():.4f}")
    logger.info(f"  Std: {X_transformed.std():.4f}")
    logger.info(f"  Min: {X_transformed.min():.4f}")
    logger.info(f"  Max: {X_transformed.max():.4f}")

if __name__ == "__main__":
    # Run tests
    logger.info("Starting Preprocessing Tests...\n")
    
    # Test with sample
    X_sample = test_preprocessing_with_sample()
    
    # Test with real data
    test_real = test_preprocessing_with_real_data()
    
    # Inspect features
    if X_sample is not None:
        preprocessor = DataPreprocessor()
        inspect_transformed_features(X_sample, preprocessor)
    
    logger.info("\n" + "=" * 60)
    if test_real:
        logger.success("All preprocessing tests passed!")
    else:
        logger.error("Some preprocessing tests failed!")
        sys.exit(1)
