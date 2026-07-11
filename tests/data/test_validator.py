import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd

from src.data.validator import DataValidator


class TestDataValidator:
    """Test suite for DataValidator"""
    
    def test_validator_initialization(self):
        """Test DataValidator initialization"""
        validator = DataValidator()
        assert validator.schema is not None
        assert 'required_columns' in validator.schema
    
    def test_validate_valid_data(self):
        """Test validation on valid data"""
        # Create valid data
        df = pd.DataFrame({
            'customerID': ['1', '2', '3'],
            'gender': ['Female', 'Male', 'Female'],
            'SeniorCitizen': [0, 1, 0],
            'Partner': ['Yes', 'No', 'No'],
            'Dependents': ['No', 'Yes', 'No'],
            'tenure': [10, 20, 30],
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
            'TotalCharges': [500.0, 1500.0, 900.0],
            'Churn': ['No', 'Yes', 'No']
        })
        
        validator = DataValidator()
        result = validator.validate(df)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_missing_columns(self):
        """Test validation catches missing columns"""
        # Create data with missing column
        df = pd.DataFrame({
            'customerID': ['1', '2'],
            'gender': ['Female', 'Male']
        })
        
        validator = DataValidator()
        result = validator.validate(df)
        
        assert not result.is_valid
        assert any('Missing required columns' in error for error in result.errors)
    
    def test_validate_out_of_range(self):
        """Test validation catches out-of-range values"""
        df = pd.DataFrame({
            'customerID': ['1', '2'],
            'gender': ['Female', 'Male'],
            'SeniorCitizen': [0, 2],  # Invalid: > 1
            'Partner': ['Yes', 'No'],
            'Dependents': ['No', 'Yes'],
            'tenure': [10, 20],
            'PhoneService': ['Yes', 'Yes'],
            'MultipleLines': ['No', 'No'],
            'InternetService': ['DSL', 'Fiber optic'],
            'OnlineSecurity': ['Yes', 'No'],
            'OnlineBackup': ['No', 'Yes'],
            'DeviceProtection': ['No', 'Yes'],
            'TechSupport': ['No', 'No'],
            'StreamingTV': ['No', 'Yes'],
            'StreamingMovies': ['No', 'Yes'],
            'Contract': ['Month-to-month', 'One year'],
            'PaperlessBilling': ['Yes', 'No'],
            'PaymentMethod': ['Electronic check', 'Mailed check'],
            'MonthlyCharges': [50.0, 75.0],
            'TotalCharges': [500.0, 1500.0],
            'Churn': ['No', 'Yes']
        })
        
        validator = DataValidator()
        result = validator.validate(df)
        
        assert not result.is_valid
        assert any('SeniorCitizen' in error for error in result.errors)
