# SmartChurn - Customer Churn Prediction Pipeline

Production-ready ML pipeline for predicting customer churn using the IBM Telco dataset.

## Quick Start

```bash
# Clone and setup
git clone git@github.com:seyyedmsl82/SmartChurn.git

cd SmartChurn
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements/requirements.txt

# Download the dataset
python scripts/download_dataset.py

# Test the full pipeline
python scripts/...
```

## Convenience Scripts

| Script | What it does |
|--------|--------------|
| `scripts/download_dataset.py` | Downloads IBM Telco dataset from GitHub |
| `scripts/test_validation.py` | Tests validator with sample + real data |
| `scripts/test_preprocessing.py` | Tests preprocessing with sample + real data |

Run any script directly:
```bash
python scripts/test_validation.py
python scripts/test_preprocessing.py
```

## What's Been Built

### 1. Data Ingestion (`src/data/loader.py`)
Load data from CSV or URL with local caching.

```python
from src.data.loader import DataLoader
loader = DataLoader()
df = loader.load_from_url("https://...")
df = loader.load_from_csv("data/raw/file.csv")
```

### 2. Data Validation (`src/data/validator.py`)
Validates schema, types, ranges, missing values, categorical values.

```python
from src.data.validator import DataValidator
validator = DataValidator()
result = validator.validate(df)
# result.is_valid, result.errors, result.warnings
```

### 3. Preprocessing (`src/data/preprocessor.py`)
Feature engineering + sklearn Pipeline with save/load.

```python
from src.data.preprocessor import DataPreprocessor
preprocessor = DataPreprocessor()
X_transformed = preprocessor.fit_transform(X, y)
preprocessor.save("models/preprocessor.pkl")
```

## Testing

```bash
# Unit tests
pytest tests/ -v

# Manual tests (using scripts above)
python scripts/test_validation.py
python scripts/test_preprocessing.py
python scripts/test_full_pipeline.py

# Coverage
pytest tests/ --cov=src.data --cov-report=term-missing
```

## Next Steps

- Model training with Optuna
- MLflow tracking
- FastAPI deployment
- Monitoring with Evidently

---

**Maintainer:** [S. R. Moslemi](https://github.com/seyyedmsl82)
