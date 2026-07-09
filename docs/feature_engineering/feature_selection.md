# Feature Selection Methodology

## Overview

Feature selection is critical for:
- **Reducing overfitting** and improving model generalization
- **Decreasing training time** and inference latency
- **Improving model interpretability**
- **Reducing data collection costs** in production

## Selection Methods

Our `FeatureSelector` implements four complementary methods:

### 1. Filter Method

**Approach**: Statistical tests on individual features.

**Used Tests**:
- **ANOVA F-test**: Measures linear relationship
- **Mutual Information**: Measures non-linear relationship

**Advantages**:
- Fastest method
- Independent of model
- Good for high-dimensional data

**Limitations**:
- Ignores feature interactions
- May select redundant features

**Use Case**: Initial screening, high-dimensional datasets (>1000 features)

### 2. Wrapper Method

**Approach**: Recursive Feature Elimination (RFE) with cross-validation.

**Algorithm**:
1. Train model on all features
2. Rank features by importance
3. Remove least important features
4. Repeat until target number reached

**Advantages**:
- Considers feature interactions
- Usually selects optimal subset

**Limitations**:
- Computationally expensive
- May overfit to specific model

**Use Case**: Final selection, small to medium datasets

### 3. Embedded Method

**Approach**: Feature importance from tree-based models.

**Used Model**: Random Forest with balanced class weights.

**Selection Criteria**:
- `SelectFromModel` with median threshold
- Features with importance above median retained

**Advantages**:
- Combines speed with accuracy
- Considers feature importance intrinsically

**Limitations**:
- Model-dependent
- May not capture linear relationships well

**Use Case**: Most practical scenarios, balanced approach

### 4. Combined Method (Default)

**Approach**: Consensus across all three methods.

**Algorithm**:
1. Run all three selection methods
2. Count how many methods select each feature
3. Retain features selected by ≥2 methods
4. If needed, limit to target number

**Advantages**:
- Most robust approach
- Reduces method-specific bias
- Often yields best results

**Limitations**:
- Most computationally expensive
- May be conservative

**Use Case**: Production models, when time allows

## Recommended Usage

### For Initial Exploration
```python
selector = FeatureSelector(method='filter', n_features=50)
selected = selector.select_features(X, y)
```

### For Final Model
```python
selector = FeatureSelector(method='combined', n_features=30)
selected = selector.select_features(X, y)
```

### For Performance Optimization
```python
selector = FeatureSelector(method='embedded', n_features=20)
selected = selector.select_features(X, y)
```

## Best Practices

### 1. Feature Selection Pipeline

```python
# Recommended pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selector', FeatureSelector(method='combined', n_features=30)),
    ('classifier', RandomForestClassifier())
])
```

### 2. Cross-Validation Integration

Always perform feature selection inside cross-validation:
```python
# Correct: Feature selection inside CV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('feature_selector', FeatureSelector(method='combined', n_features=30)),
    ('classifier', RandomForestClassifier())
])

scores = cross_val_score(pipeline, X, y, cv=5)
```

### 3. Correlation Analysis

Use correlation analysis to remove redundant features:
```python
selector = FeatureSelector()
high_corr = selector.analyze_correlations(X, threshold=0.9)
# Remove one feature from each highly correlated pair
```

### 4. Business Validation

Always validate selected features with domain experts:
- Do features make business sense?
- Are there regulatory concerns?
- Can features be collected in production?
