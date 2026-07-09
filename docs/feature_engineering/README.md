# Feature Engineering for Customer Churn Prediction

## Overview

The feature engineering module transforms raw customer data into predictive features for churn prediction. It creates **20+ new features** from the original dataset through domain-specific transformations, business logic, and statistical techniques.

## Why Feature Engineering Matters

In churn prediction, feature engineering is critical because:
- **Domain knowledge** transforms raw data into business insights
- **Interaction features** capture relationships between variables
- **Risk indicators** quantify churn propensity
- **Aggregate features** summarize customer behavior patterns

## Quick Start

```python
from src.features.engineering import AdvancedFeatureEngineer
from src.features.selection import FeatureSelector

# Initialize engineer
engineer = AdvancedFeatureEngineer()

# Transform your data
X_engineered = engineer.transform(X)

# Select best features
selector = FeatureSelector(method='combined', n_features=30)
selected_features = selector.select_features(X_engineered, y)
```

## Feature Categories

| Category | Description | Number of Features |
|----------|-------------|-------------------|
| **Tenure Features** | Customer loyalty and lifecycle indicators | 4 |
| **Monetary Features** | Financial behavior and spending patterns | 6 |
| **Service Features** | Product adoption and usage intensity | 5 |
| **Interaction Features** | Cross-variable relationships | 4 |
| **Risk Indicators** | Churn risk scoring and alerts | 3 |
| **Aggregate Features** | Customer-level summaries | 2 |

## Module Components

### 1. AdvancedFeatureEngineer
The main transformer that creates all engineered features.

### 2. FeatureSelector
Multi-method feature selection (filter, wrapper, embedded, combined).

### 3. Feature Analysis Tools
Correlation analysis, importance ranking, and visualization.

## Documentation Navigation

- **[Feature Descriptions](feature_descriptions.md)** - Detailed description of each engineered feature
- **[Feature Selection](feature_selection.md)** - Methodology for selecting optimal features
- **[Usage Guide](usage_guide.md)** - Step-by-step usage instructions
- **[Examples](examples.md)** - Code examples and use cases
