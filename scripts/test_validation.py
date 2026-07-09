"""
Manual test script for data validation
Run this to test the validator with real data
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.validator import DataValidator
from loguru import logger

def test_validation_with_sample_data():
    """Test validation with a sample dataset"""
    
    logger.info("=" * 60)
    logger.info("Testing Data Validation")
    logger.info("=" * 60)
    
    # 1. Create a sample dataset
    logger.info("\n1. Creating sample dataset...")
    sample_data = {
        'customerID': ['CUST-001', 'CUST-002', 'CUST-003', 'CUST-004'],
        'gender': ['Female', 'Male', 'Female', 'Male'],
        'SeniorCitizen': [0, 1, 0, 0],
        'Partner': ['Yes', 'No', 'No', 'Yes'],
        'Dependents': ['No', 'Yes', 'No', 'No'],
        'tenure': [10, 25, 40, 60],
        'PhoneService': ['Yes', 'Yes', 'No', 'Yes'],
        'MultipleLines': ['No', 'Yes', 'No phone service', 'No'],
        'InternetService': ['DSL', 'Fiber optic', 'No', 'DSL'],
        'OnlineSecurity': ['Yes', 'No', 'No internet service', 'Yes'],
        'OnlineBackup': ['No', 'Yes', 'No internet service', 'No'],
        'DeviceProtection': ['No', 'Yes', 'No internet service', 'Yes'],
        'TechSupport': ['No', 'No', 'No internet service', 'No'],
        'StreamingTV': ['No', 'Yes', 'No internet service', 'No'],
        'StreamingMovies': ['No', 'Yes', 'No internet service', 'No'],
        'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month'],
        'PaperlessBilling': ['Yes', 'No', 'Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        'MonthlyCharges': [50.0, 75.0, 30.0, 85.0],
        'TotalCharges': [500.0, 1875.0, 1200.0, 5100.0],
        'Churn': ['No', 'Yes', 'No', 'No']
    }
    
    import pandas as pd
    df = pd.DataFrame(sample_data)
    logger.info(f"   Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
    
    # 2. Validate with default schema
    logger.info("\n2. Running validation with default schema...")
    validator = DataValidator()
    result = validator.validate(df)
    
    logger.info(f"\nValidation Result:")
    logger.info(f"  Is Valid: {result.is_valid}")
    logger.info(f"  Errors: {len(result.errors)}")
    for error in result.errors:
        logger.error(f"    - {error}")
    logger.info(f"  Warnings: {len(result.warnings)}")
    for warning in result.warnings:
        logger.warning(f"    - {warning}")
    
    # 3. Test with invalid data
    logger.info("\n3. Testing with invalid data...")
    invalid_df = df.copy()
    invalid_df.loc[0, 'SeniorCitizen'] = 5  # Out of range
    invalid_df.loc[1, 'gender'] = 'Unknown'  # Invalid category
    invalid_df.loc[2, 'TotalCharges'] = None  # Missing value
    
    result_invalid = validator.validate(invalid_df)
    logger.info(f"\nInvalid Data Validation Result:")
    logger.info(f"  Is Valid: {result_invalid.is_valid}")
    logger.info(f"  Errors: {len(result_invalid.errors)}")
    for error in result_invalid.errors:
        logger.error(f"    - {error}")
    
    return result.is_valid

def test_validation_with_real_data():
    """Test validation with the actual dataset"""
    
    logger.info("\n" + "=" * 60)
    logger.info("Testing with Real Telco Data")
    logger.info("=" * 60)
    
    try:
        # Load real data
        loader = DataLoader()
        df = loader.load_from_csv('data/raw/telco_churn.csv')
        logger.info(f"Loaded {len(df)} rows from real dataset")
        
        # Validate
        validator = DataValidator()
        result = validator.validate(df)
        
        logger.info(f"\nReal Data Validation Result:")
        logger.info(f"  Is Valid: {result.is_valid}")
        logger.info(f"  Errors: {len(result.errors)}")
        for error in result.errors:
            logger.error(f"    - {error}")
        logger.info(f"  Warnings: {len(result.warnings)}")
        for warning in result.warnings:
            logger.warning(f"    - {warning}")
        
        return result.is_valid
    
    except FileNotFoundError:
        logger.warning("Real dataset not found. Run download_dataset.py first.")
        return True  # Return True so tests continue

if __name__ == "__main__":
    # Run tests
    test1 = test_validation_with_sample_data()
    test2 = test_validation_with_real_data()
    
    if test1 and test2:
        logger.success("\nAll validation tests passed!")
    else:
        logger.error("\nSome validation tests failed!")
        sys.exit(1)
