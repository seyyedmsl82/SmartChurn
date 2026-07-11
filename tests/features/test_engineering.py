"""
Tests for feature engineering
"""
import numpy as np
import pandas as pd
import pytest

from src.features.engineering import AdvancedFeatureEngineer


class TestAdvancedFeatureEngineer:
    """Test suite for AdvancedFeatureEngineer"""
    
    def test_engineer_initialization(self):
        """Test initialization"""
        engineer = AdvancedFeatureEngineer()
        assert engineer.tenure_bins == [0, 6, 12, 24, 48, 72]
        assert engineer.create_interaction_features is True
    
    def test_tenure_features(self):
        """Test tenure-related features"""
        # Create sample data
        X = pd.DataFrame({
            'tenure': [3, 15, 36, 60],
            'SeniorCitizen': [0, 1, 0, 1]
        })
        
        engineer = AdvancedFeatureEngineer()
        X_transformed = engineer._create_tenure_features(X)
        
        assert 'tenure_group' in X_transformed.columns
        assert 'tenure_risk_score' in X_transformed.columns
        assert 'tenure_log' in X_transformed.columns
        assert 'senior_tenure' in X_transformed.columns
        
        # Check tenure groups
        assert X_transformed['tenure_group'].iloc[0] == '0-6'
        assert X_transformed['tenure_group'].iloc[1] == '12-24'
    
    def test_monetary_features(self):
        """Test monetary features"""
        X = pd.DataFrame({
            'MonthlyCharges': [50, 100, 150],
            'TotalCharges': [500, 1500, np.nan],
            'tenure': [10, 15, 20]
        })
        
        engineer = AdvancedFeatureEngineer()
        X_transformed = engineer._create_monetary_features(X)
        
        assert 'monthly_charges_log' in X_transformed.columns
        assert 'total_charges_log' in X_transformed.columns
        assert 'avg_monthly_spend' in X_transformed.columns
        
        # Check missing value handling
        assert X_transformed['TotalCharges_clean'].iloc[2] == 0
    
    def test_full_transform(self):
        """Test full transform"""
        # Create comprehensive sample
        X = pd.DataFrame({
            'tenure': [3, 15, 36],
            'MonthlyCharges': [50, 100, 75],
            'TotalCharges': [150, 1500, 2700],
            'SeniorCitizen': [0, 1, 0],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'PaymentMethod': ['Electronic check', 'Credit card (automatic)', 'Mailed check'],
            'Dependents': ['No', 'Yes', 'No'],
            'Partner': ['Yes', 'No', 'No'],
            'PaperlessBilling': ['Yes', 'No', 'Yes'],
            'PhoneService': ['Yes', 'Yes', 'No'],
            'MultipleLines': ['No', 'Yes', 'No phone service'],
            'InternetService': ['DSL', 'Fiber optic', 'No'],
            'OnlineSecurity': ['Yes', 'No', 'No internet service'],
            'OnlineBackup': ['No', 'Yes', 'No internet service'],
            'DeviceProtection': ['No', 'Yes', 'No internet service'],
            'TechSupport': ['No', 'No', 'No internet service'],
            'StreamingTV': ['No', 'Yes', 'No internet service'],
            'StreamingMovies': ['No', 'Yes', 'No internet service']
        })
        
        engineer = AdvancedFeatureEngineer()
        X_transformed = engineer.transform(X)
        
        # Check that we have more features
        assert X_transformed.shape[1] > X.shape[1]
        
        # Check engineered features
        assert 'service_count' in X_transformed.columns
        assert 'risk_score' in X_transformed.columns
        assert 'has_family' in X_transformed.columns
        
        # Check risk score calculation
        assert X_transformed['risk_score'].iloc[0] > 0
