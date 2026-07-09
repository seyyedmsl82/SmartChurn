# Feature Engineering Usage Guide

## Quick Start

### 1. Basic Usage

```python
from src.features.engineering import AdvancedFeatureEngineer

# Initialize engineer
engineer = AdvancedFeatureEngineer()

# Transform your data
X_engineered = engineer.transform(X)

# X_engineered now contains all original + engineered features
```

### 2. With Feature Selection

```python
from src.features.engineering import AdvancedFeatureEngineer
from src.features.selection import FeatureSelector

# Engineer features
engineer = AdvancedFeatureEngineer()
X_engineered = engineer.transform(X)

# Select best features
selector = FeatureSelector(method='combined', n_features=30)
selected_features = selector.select_features(X_engineered, y)

# Keep only selected features
X_selected = X_engineered[selected_features]
```

### 3. Full Pipeline

```python
from sklearn.pipeline import Pipeline
from src.features.engineering import AdvancedFeatureEngineer
from src.features.selection import FeatureSelector
from sklearn.ensemble import RandomForestClassifier

# Create complete pipeline
pipeline = Pipeline([
    ('feature_engineering', AdvancedFeatureEngineer()),
    ('feature_selection', FeatureSelector(method='combined', n_features=30)),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)
```

## Configuration Options

### AdvancedFeatureEngineer Parameters

```python
engineer = AdvancedFeatureEngineer(
    tenure_bins=[0, 6, 12, 24, 48, 72],  # Custom tenure groups
    create_interaction_features=True,      # Enable/disable interactions
    create_aggregate_features=True         # Enable/disable aggregates
)
```

### FeatureSelector Parameters

```python
selector = FeatureSelector(
    n_features=30,          # Number of features to select
    method='combined',      # 'filter', 'wrapper', 'embedded', 'combined'
    random_state=42         # For reproducibility
)
```

## Validation and Testing

### Unit Tests

```python
import pytest
import pandas as pd
import numpy as np
from src.features.engineering import AdvancedFeatureEngineer

def test_feature_engineering_basic():
    # Create sample data
    X = pd.DataFrame({
        'tenure': [10, 20, 30],
        'MonthlyCharges': [50, 100, 75],
        'TotalCharges': [500, 2000, 2250],
        'SeniorCitizen': [0, 1, 0],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaymentMethod': ['Electronic check', 'Credit card (automatic)', 'Mailed check']
    })
    
    engineer = AdvancedFeatureEngineer()
    X_transformed = engineer.transform(X)
    
    # Check new features created
    assert 'tenure_group' in X_transformed.columns
    assert 'avg_monthly_spend' in X_transformed.columns
    assert 'risk_score' in X_transformed.columns
    assert X_transformed.shape[1] > X.shape[1]
```

### Integration Tests

```python
def test_full_feature_pipeline():
    # Load real data
    from src.data.loader import DataLoader
    loader = DataLoader()
    df = loader.load_from_csv('data/raw/telco_churn.csv')
    
    # Preprocess
    from src.data.preprocessor import DataPreprocessor
    preprocessor = DataPreprocessor()
    y = df['Churn']
    X = df.drop(columns=['Churn'])
    X_preprocessed = preprocessor.fit_transform(X, y)
    
    # Engineer features
    engineer = AdvancedFeatureEngineer()
    X_engineered = engineer.transform(X_preprocessed)
    
    # Select features
    selector = FeatureSelector(n_features=30, method='combined')
    selected = selector.select_features(X_engineered, y)
    
    # Validate
    assert len(selected) <= 30
    assert len(X_engineered) == len(X)
    assert X_engineered.shape[1] > X_preprocessed.shape[1]
```
