import pytest
import pandas as pd
import numpy as np
from src.data.preprocessor import DataPreprocessor

class TestDataPreprocessor:
    """Test suite for DataPreprocessor"""
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization"""
        preprocessor = DataPreprocessor()
        assert preprocessor.numeric_features == ['tenure', 'MonthlyCharges', 'TotalCharges']
        assert preprocessor.pipeline is not None
    
    def test_preprocessor_fit_transform(self):
        """Test fit_transform functionality"""
        # Create sample data
        X = pd.DataFrame({
            'tenure': [10, 20, 30],
            'MonthlyCharges': [50.0, 75.0, 30.0],
            'TotalCharges': [500.0, 1500.0, 900.0],
            'gender': ['Female', 'Male', 'Female'],
            'Partner': ['Yes', 'No', 'No'],
            'Dependents': ['No', 'Yes', 'No'],
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
            'Churn': ['No', 'Yes', 'No']
        })
        
        preprocessor = DataPreprocessor()
        y = X['Churn']
        X_transformed = preprocessor.fit_transform(X, y)
        
        # Check output shape
        assert X_transformed.shape[0] == 3
        assert X_transformed.shape[1] > 0  # Should have features
        
        # Check feature engineering
        assert hasattr(preprocessor.pipeline.named_steps['feature_engineer'], 'transform')
    
    def test_preprocessor_save_load(self, tmp_path):
        """Test saving and loading preprocessor"""
        # Create and fit preprocessor
        preprocessor = DataPreprocessor()
        X = pd.DataFrame({
            'tenure': [10, 20, 30],
            'MonthlyCharges': [50.0, 75.0, 30.0],
            'TotalCharges': [500.0, 1500.0, 900.0],
            'gender': ['Female', 'Male', 'Female'],
            'Partner': ['Yes', 'No', 'No'],
            'Dependents': ['No', 'Yes', 'No'],
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
            'Churn': ['No', 'Yes', 'No']
        })
        y = X['Churn']
        preprocessor.fit_transform(X, y)
        
        # Save
        save_path = tmp_path / 'preprocessor.pkl'
        preprocessor.save(str(save_path))
        
        # Load
        loaded_preprocessor = DataPreprocessor.load(str(save_path))
        
        assert loaded_preprocessor is not None
        assert loaded_preprocessor.pipeline is not None
        assert loaded_preprocessor.label_encoders is not None
