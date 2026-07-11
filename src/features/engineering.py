"""
Advanced feature engineering for customer churn prediction
"""
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings('ignore')


class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Advanced feature engineering with business logic
    """
    
    def __init__(self, 
        tenure_bins: Optional[List[int]] = None,
        create_interaction_features: bool = True,
        create_aggregate_features: bool = True
    ):
        """
        Initialize feature engineer
        
        Args:
            tenure_bins: Custom bins for tenure grouping
            create_interaction_features: Whether to create feature interactions
            create_aggregate_features: Whether to create aggregate statistics
        """
        self.tenure_bins = tenure_bins or [0, 6, 12, 24, 48, 72]
        self.create_interaction_features = create_interaction_features
        self.create_aggregate_features = create_aggregate_features
        self.feature_names_ = []
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering transformations"""
        X = X.copy()
        logger.info(f"Starting feature engineering on {len(X)} rows")
        
        # 1. Tenure-based features
        X = self._create_tenure_features(X)
        
        # 2. Monetary features
        X = self._create_monetary_features(X)
        
        # 3. Service usage features
        X = self._create_service_features(X)
        
        # 4. Interaction features
        if self.create_interaction_features:
            X = self._create_interaction_features(X)
        
        # 5. Aggregate features (customer-level)
        if self.create_aggregate_features:
            X = self._create_aggregate_features(X)
        
        # 6. Risk indicators
        X = self._create_risk_indicators(X)
        
        # Store feature names
        self.feature_names_ = X.columns.tolist()
        
        logger.info(f"Feature engineering complete: {len(X.columns)} features")
        return X
    
    def _create_tenure_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create features based on customer tenure"""
        if 'tenure' in X.columns:
            # Tenure groups
            X['tenure_group'] = pd.cut(
                X['tenure'],
                bins=self.tenure_bins,
                labels=[f'{self.tenure_bins[i]}-{self.tenure_bins[i+1]}' 
                        for i in range(len(self.tenure_bins)-1)]
            )
            
            # Tenure risk score (new customers are higher risk)
            X['tenure_risk_score'] = np.where(
                X['tenure'] < 6, 3,  # Very high risk
                np.where(X['tenure'] < 12, 2,  # High risk
                np.where(X['tenure'] < 24, 1,  # Medium risk
                np.where(X['tenure'] < 48, 0.5, 0)))  # Low risk
            )
            
            # Logarithmic tenure for non-linear effects
            X['tenure_log'] = np.log1p(X['tenure'])
            
            # Senior citizen tenure interaction
            if 'SeniorCitizen' in X.columns:
                X['senior_tenure'] = X['SeniorCitizen'] * X['tenure']
        
        return X
    
    def _create_monetary_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create monetary-related features"""
        # Monthly charges features
        if 'MonthlyCharges' in X.columns:
            # Log transform for skewed distribution
            X['monthly_charges_log'] = np.log1p(X['MonthlyCharges'])
            
            # Monthly charges squared (capture non-linear effects)
            X['monthly_charges_sq'] = X['MonthlyCharges'] ** 2
        
        # Total charges features
        if 'TotalCharges' in X.columns:
            # Clean TotalCharges (handle missing)
            X['TotalCharges_clean'] = X['TotalCharges'].fillna(0)
            
            # Log transform
            X['total_charges_log'] = np.log1p(X['TotalCharges_clean'])
            
            # Average monthly spend (if tenure exists)
            if 'tenure' in X.columns:
                X['avg_monthly_spend'] = np.where(
                    X['tenure'] > 0,
                    X['TotalCharges_clean'] / X['tenure'],
                    X['MonthlyCharges']
                )
                
                # Spend to tenure ratio
                X['spend_tenure_ratio'] = np.where(
                    X['tenure'] > 0,
                    X['TotalCharges_clean'] / X['tenure'],
                    X['MonthlyCharges']
                )
        
        # Charge ratio (monthly to total)
        if 'MonthlyCharges' in X.columns and 'TotalCharges' in X.columns:
            X['charge_ratio'] = np.where(
                X['TotalCharges_clean'] > 0,
                X['MonthlyCharges'] / X['TotalCharges_clean'],
                0
            )
        
        return X
    
    def _create_service_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create service-related features"""
        # Define service columns
        service_cols = [
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        existing_services = [col for col in service_cols if col in X.columns]
        
        if existing_services:
            # Count of services (excluding 'No' and 'No internet service')
            X['service_count'] = X[existing_services].apply(
                lambda row: sum(1 for val in row if val not in ['No', 'No internet service']),
                axis=1
            )
            
            # Count of additional services (beyond basic)
            X['additional_services'] = X['service_count'] - X['service_count'].min()
            
            # Service diversity (number of different service types)
            internet_services = ['InternetService', 'OnlineSecurity', 'OnlineBackup', 
                               'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
            internet_existing = [col for col in internet_services if col in X.columns]
            
            if internet_existing:
                X['internet_service_count'] = X[internet_existing].apply(
                    lambda row: sum(1 for val in row if val not in ['No', 'No internet service']),
                    axis=1
                )
            
            # Has multiple services (potential for upselling)
            X['has_multiple_services'] = (X['service_count'] > 1).astype(int)
            
            # Is fully bundled (has all services)
            max_services = len(existing_services)
            X['is_fully_bundled'] = (X['service_count'] == max_services).astype(int)
        
        return X
    
    def _create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create feature interactions"""
        # Contract and tenure interaction
        if 'Contract' in X.columns and 'tenure' in X.columns:
            # Tenure relative to contract (how much longer they have)
            contract_durations = {
                'Month-to-month': 1,
                'One year': 12,
                'Two year': 24
            }
            if X['Contract'].dtype == 'object':
                X['contract_duration_months'] = X['Contract'].map(contract_durations).fillna(1)
                X['tenure_contract_ratio'] = X['tenure'] / X['contract_duration_months']
        
        # Payment method and tenure
        if 'PaymentMethod' in X.columns and 'tenure' in X.columns:
            # Auto-payment indicator
            X['has_autopay'] = X['PaymentMethod'].str.contains('automatic|auto', case=False).astype(int)
            X['autopay_tenure'] = X['has_autopay'] * X['tenure']
        
        # Senior citizen with services
        if 'SeniorCitizen' in X.columns:
            if 'service_count' in X.columns:
                X['senior_service_count'] = X['SeniorCitizen'] * X['service_count']
            if 'MonthlyCharges' in X.columns:
                X['senior_monthly_charges'] = X['SeniorCitizen'] * X['MonthlyCharges']
        
        return X
    
    def _create_aggregate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create aggregate features at customer level"""
        # Customer has dependents or partner
        if 'Dependents' in X.columns and 'Partner' in X.columns:
            X['has_family'] = ((X['Dependents'] == 'Yes') | (X['Partner'] == 'Yes')).astype(int)
        
        # Customer with paperless billing (higher risk for some segments)
        if 'PaperlessBilling' in X.columns:
            X['is_paperless'] = (X['PaperlessBilling'] == 'Yes').astype(int)
        
        return X
    
    def _create_risk_indicators(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create risk indicators based on business rules"""
        # High churn risk factors
        risk_factors = []
        
        # Short tenure (< 6 months)
        if 'tenure' in X.columns:
            risk_factors.append(('short_tenure', X['tenure'] < 6))
        
        # High monthly charges
        if 'MonthlyCharges' in X.columns:
            risk_factors.append(('high_monthly_charges', X['MonthlyCharges'] > X['MonthlyCharges'].quantile(0.75)))
        
        # Month-to-month contract
        if 'Contract' in X.columns:
            risk_factors.append(('month_to_month', X['Contract'] == 'Month-to-month'))
        
        # No online security (higher risk)
        if 'OnlineSecurity' in X.columns:
            risk_factors.append(('no_security', X['OnlineSecurity'] == 'No'))
        
        # Electronic check (higher risk)
        if 'PaymentMethod' in X.columns:
            risk_factors.append(('electronic_check', X['PaymentMethod'] == 'Electronic check'))
        
        # Combine risk factors
        if risk_factors:
            risk_count = pd.DataFrame({
                name: indicator.astype(int) for name, indicator in risk_factors
            }).sum(axis=1)
            
            X['risk_score'] = risk_count
            
            # Risk level categories
            X['risk_level'] = pd.cut(
                risk_count,
                bins=[-1, 1, 3, 5],
                labels=['Low', 'Medium', 'High']
            )
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names after transformation"""
        if hasattr(self, 'feature_names_') and self.feature_names_:
            return self.feature_names_
        return []
