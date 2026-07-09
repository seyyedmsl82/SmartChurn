# SmartChurn - Customer Churn Prediction Pipeline

## Data Pipeline

### Data Ingestion
The data ingestion module (`src/data/loader.py`) handles loading data from various sources:
- **CSV files**: Load local CSV files
- **URLs**: Download datasets from URLs (e.g., IBM Telco Churn dataset)
- **Local cache**: Automatically saves downloaded data for reproducibility

### Dataset
The project uses the IBM Telco Customer Churn dataset, which contains 7,043 customer records with 20 features including customer tenure, service subscriptions, and billing information.

### How to Use
1. Download the dataset:
```bash
   python scripts/download_dataset.py
```

2. Load data in your code:
```python
    from src.data.loader import DataLoader
    loader = DataLoader()
    df = loader.load_from_csv('data/raw/telco_churn.csv')
```
