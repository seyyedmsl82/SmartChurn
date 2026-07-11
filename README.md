# SmartChurn - Customer Churn Prediction Pipeline

Production-ready ML pipeline for predicting customer churn using the IBM Telco dataset.

## Project Overview

SmartChurn is an end-to-end machine learning system that:

- **Ingests** customer data from multiple sources (CSV, URL, local cache)
- **Validates** data quality with schema enforcement
- **Engineers** domain-specific features (tenure groups, avg spend, service intensity)
- **Selects** optimal features using filter/wrapper/embedded methods
- **Trains** multiple models with imbalance handling
- **Tracks** experiments with MLflow
- **Versions** models for production deployment

### Key Results

| Model | F1 Score | ROC-AUC | Best Threshold |
|-------|----------|---------|----------------|
| RandomForest | **0.7838** | 0.8405 | 0.45 |
| LightGBM | 0.7671 | 0.8381 | 0.46 |
| LogisticRegression | 0.7503 | 0.8468 | 0.43 |
| XGBoost | 0.7493 | 0.8370 | 0.44 |

## Quick Start

### Installation

```bash
# Clone repository
git clone git@github.com:seyyedmsl82/SmartChurn.git
cd SmartChurn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements/requirements.txt

# Download the dataset
python scripts/download_dataset.py

# Run the full training pipeline
python scripts/train_pipeline.py

# Test everything
make test
```

### Quick Commands with Make

```bash
make install       # Install all dependencies
make data          # Download dataset
make preprocess    # Run preprocessing only
make train         # Train models
make test          # Run all tests
make lint          # Run linters
make clean         # Clean temporary files
make all           # Full pipeline (data → train → test)
```

## Convenience Scripts

| Script | What it does | Usage |
|--------|--------------|-------|
| `scripts/download_dataset.py` | Downloads IBM Telco dataset from GitHub | `python scripts/download_dataset.py` |
| `scripts/train_pipeline.py` | Complete training pipeline with feature selection | `python scripts/train_pipeline.py [--no-feature-selection] [--n-features 30]` |
| `scripts/test_pipeline.py` | Quick test of the entire pipeline | `python scripts/test_pipeline.py` |

### Training Script Options

```bash
# Skip feature selection (use all features)
python scripts/train_pipeline.py --no-feature-selection

# Use existing selected features (from previous run)
python scripts/train_pipeline.py --use-existing-selection

# Specify number of features to select
python scripts/train_pipeline.py --n-features 25

# Choose selection method
python scripts/train_pipeline.py --selection-method embedded

# Change imputation strategy
python scripts/train_pipeline.py --impute-strategy median

# Use custom config
python scripts/train_pipeline.py --config config/model_configs/default.yaml
```

## What's Been Built

### 1. Data Ingestion (`src/data/loader.py`)
Load data from multiple sources with local caching.

```python
from src.data.loader import DataLoader

loader = DataLoader()

# Load from URL
df = loader.load_from_url("https://raw.githubusercontent.com/...")

# Load from CSV
df = loader.load_from_csv("data/raw/file.csv")

# Get dataset info
info = loader.get_dataset_info(df)
```

### 2. Data Validation (`src/data/validator.py`)
Comprehensive data quality checks.

**Validations performed:**
- Required columns existence
- Data type matching
- Missing value thresholds
- Categorical value validation
- Numeric range validation
- Unique ID validation

### 3. Preprocessing Pipeline (`src/data/preprocessor.py`)
Feature engineering with sklearn Pipeline integration.

**Features created:**
- Tenure groups (0-6, 6-12, 12-24, 24-48, 48-72 months)
- Average monthly spend (TotalCharges / tenure)
- Service usage intensity (count of subscribed services)
- Senior citizen indicator

### 4. Feature Selection (`src/features/selection.py`)
Multiple methods for optimal feature selection.

### 5. Model Training (`src/models/train.py`)
Training with imbalance handling and MLflow tracking.

### 6. Model Registry (`src/models/registry.py`)
Versioned model storage.

## Results & Performance

### Current Best Model: RandomForest

| Metric | Value |
|--------|-------|
| **F1 Score** | **0.7838** |
| ROC-AUC | 0.8405 |
| Accuracy | ~84% |
| Optimal Threshold | 0.45 |


## Deployment

### Export Model for Production

```python
from src.models.registry import ModelRegistry
import pickle

registry = ModelRegistry()

# Get latest model
model = registry.load_model()

# Or specific version
model = registry.load_model(version='v1.0.0')

# Export for API service
with open('models/production_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

### FastAPI Integration (Coming Soon)

```python
from fastapi import FastAPI
from src.data.preprocessor import DataPreprocessor
from src.models.registry import ModelRegistry

app = FastAPI()
model = ModelRegistry().load_model()
preprocessor = DataPreprocessor.load('models/preprocessor.pkl')

@app.post("/predict")
def predict(features: dict):
    X = pd.DataFrame([features])
    X_processed = preprocessor.transform(X)
    prediction = model.predict(X_processed)
    return {"churn_prediction": bool(prediction[0])}
```

**Maintainer:** [S. R. Moslemi](https://github.com/seyyedmsl82)

**Last Updated:** July 2026
