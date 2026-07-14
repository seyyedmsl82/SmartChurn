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
- **Serves** predictions via production-grade REST API
- **Monitors** data drift and model performance

### Key Results

| Model | F1 Score | ROC-AUC | Best Threshold |
|-------|----------|---------|----------------|
| RandomForest | **0.7838** | 0.8405 | 0.45 |
| LightGBM | 0.7671 | 0.8381 | 0.46 |
| LogisticRegression | 0.7503 | 0.8468 | 0.43 |
| XGBoost | 0.7493 | 0.8370 | 0.44 |

## Quick Start

### Docker Deployment (Recommended)

The fastest way to get started is with Docker Compose:

```bash
# Clone repository
git clone git@github.com:seyyedmsl82/SmartChurn.git
cd SmartChurn

# Create required directories
mkdir -p models data logs reports mlflow models/registry

# Build and start all services
docker-compose up -d

# Check if services are running
docker-compose ps

# View logs
docker-compose logs -f api
```

This starts:
- **API Service** on `http://localhost:8000`
- **MLflow Tracking** on `http://localhost:5001`
- **Automatic health checks** and restart policies

### Local Development Setup

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

# Preprocess the dataset and save the preprocessor
python scripts/preprocess_data.py

# Run the full training pipeline
python scripts/train_pipeline.py

# Run evaluation pipeline
python scripts/evaluate_model.py

# Start the API service
python src/api/app.py
```

## Docker Setup

### Dockerfile Features

The multi-stage Dockerfile provides:

- **Optimized Build**: Uses multi-stage builds for smaller production images
- **Security**: Runs as non-root user (appuser:1000)
- **Health Checks**: Automatic monitoring of API health
- **Production Ready**: Configured with environment variables for different deployments

```dockerfile
# Multi-stage build for production
FROM python:3.10-slim AS builder  # Build stage with compilers
FROM python:3.10-slim              # Production stage without build tools
```

### Docker Compose Services

| Service | Container Name | Port | Description |
|---------|---------------|------|-------------|
| API | smartchurn-api | 8000 | FastAPI application with ML predictions |
| MLflow | smartchurn-mlflow | 5001 | Experiment tracking and model registry |

### Docker Commands

```bash
# Build and start all services
docker-compose up -d

# Rebuild without cache
docker-compose build --no-cache

# View logs for specific service
docker-compose logs -f api
docker-compose logs -f mlflow

# Stop all services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v

# Execute commands inside container
docker-compose exec api bash
docker-compose exec api python -c "import src; print('OK')"

# Restart a specific service
docker-compose restart api

# Check health status
docker inspect smartchurn-api --format='{{.State.Health.Status}}'
```

### Persistent Data Volumes

The docker-compose mounts the following volumes for data persistence:

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./models` | `/app/models` | Trained models and registry |
| `./data` | `/app/data` | Datasets and processed data |
| `./logs` | `/app/logs` | Application logs |
| `./reports` | `/app/reports` | Evaluation reports |
| `./mlflow` | `/mlflow` | MLflow experiment data |

## API Service

### Interactive Documentation

Access automatic Swagger/OpenAPI docs at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/predict/single` | Single customer prediction |
| POST | `/api/v1/predict/batch` | Batch predictions (max 1000) |
| GET | `/api/v1/health` | Service health check |
| GET | `/api/v1/metrics` | Model performance metrics |
| GET | `/docs` | Interactive API documentation |
| GET | `/redoc` | ReDoc API documentation |

**Example Request:**

```bash
curl -X POST http://localhost:8000/api/v1/predict/single \
  -H "Content-Type: application/json" \
  -d '{
    "customer": {
      "gender": "Female",
      "SeniorCitizen": 0,
      "Partner": "Yes",
      "Dependents": "No",
      "tenure": 24,
      "Contract": "Month-to-month",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 75.0,
      "TotalCharges": 1800.0,
      "PhoneService": "Yes",
      "MultipleLines": "No",
      "InternetService": "DSL",
      "OnlineSecurity": "No",
      "OnlineBackup": "No",
      "DeviceProtection": "No",
      "TechSupport": "No",
      "StreamingTV": "No",
      "StreamingMovies": "No"
    }
  }'
```

**Example Response:**

```json
{
  "prediction": 0,
  "probability": 0.23,
  "churn_risk": "Low",
  "model_version": "RandomForest_v1.0.0",
  "timestamp": "2024-01-15T10:30:00"
}
```

## Convenience Scripts

| Script | What it does | Usage |
|--------|--------------|-------|
| `scripts/download_dataset.py` | Downloads IBM Telco dataset from GitHub | `python scripts/download_dataset.py` |
| `scripts/train_pipeline.py` | Complete training pipeline with feature selection | `python scripts/train_pipeline.py [--no-feature-selection] [--n-features 30]` |
| `scripts/evaluate_model.py` | Quick evaluation test of models | `python scripts/evaluate_model.py` |
| `scripts/test_api.sh` | Test API endpoints | `./scripts/test_api.sh` |
| `scripts/deploy.sh` | Deploy with Docker | `./scripts/deploy.sh` |

## What's Been Built

### Data Layer (`src/data/`)
- **Loader** - CSV, URL, and local caching support
- **Validator** - Schema validation, data quality checks
- **Preprocessor** - Pipeline with imputation, scaling, encoding

### Feature Layer (`src/features/`)
- **Engineering** - Tenure groups, avg spend, service intensity, risk indicators
- **Selection** - Filter, wrapper, and embedded methods

### Model Layer (`src/models/`)
- **Train** - Multiple models with imbalance handling
- **Registry** - Versioned model storage with metadata

### Evaluation Layer (`src/evaluation/`)
- **Metrics** - Business metrics with cost analysis

### API Layer (`src/api/`)
- **REST API** - FastAPI with prediction, batch, and explanation endpoints
- **Validation** - Pydantic schemas for request/response
- **Services** - Prediction service with model lifecycle management
- **Middleware** - Rate limiting, logging, CORS
- **Documentation** - Automatic Swagger/OpenAPI docs

## Configuration

### API Configuration (`.env`)

```env
APP_ENV=development
DEBUG=True
HOST=0.0.0.0
PORT=8000
WORKERS=1
MODEL_PATH=models/registry
PREPROCESSOR_PATH=models/preprocessor.pkl
FEATURES_PATH=models/selected_features.json
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=60
```

### Model Configuration

```yaml
# config/model_configs/default.yaml
models:
  logistic_regression:
    enabled: true
    C: 1.0
  random_forest:
    enabled: true
    n_estimators: 100
    max_depth: 10
  xgboost:
    enabled: true
    n_estimators: 100
    learning_rate: 0.1
hyperparameter_tuning:
  enabled: true
  n_trials: 30
  model_type: xgboost
```

## MLflow Tracking

MLflow is pre-configured in the docker-compose for experiment tracking:

- **Tracking UI**: `http://localhost:5001`
- **Backend Store**: Persistent volume at `./mlflow`
- **Auto-logging**: Configured in training scripts

### MLflow Commands

```bash
# Access MLflow UI
open http://localhost:5001

# List experiments (from inside container)
docker-compose exec api mlflow experiments list

# Download model artifacts
docker-compose exec api mlflow artifacts download --run-id <run_id>
```

## Monitoring & Health Checks

### API Health Check

```bash
# Check API health
curl http://localhost:8000/api/v1/health

# Check Docker health status
docker inspect smartchurn-api --format='{{.State.Health.Status}}'

# View health check logs
docker inspect smartchurn-api --format='{{.State.Health.Log}}'
```

### Service Status

```bash
# Check all services
docker-compose ps

# View resource usage
docker stats

# Check logs for errors
docker-compose logs --tail=50 api | grep ERROR
```

## Development Workflow

### With Docker (Recommended for Production)

```bash
# Build and run with volumes for development
docker-compose up --build

# Test changes by rebuilding
docker-compose build --no-cache api
docker-compose up -d api
```

### Local Development

```bash
# Activate virtual environment
source venv/bin/activate

# Install in development mode
pip install -e .

# Run tests
pytest tests/

# Run linter
flake8 src/
black src/  # Auto-format

# Start API with hot-reload
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

## Troubleshooting

### Docker Build Issues

**WSL Timeout (Windows):**
```bash
# Reset WSL
wsl --shutdown
# Restart Docker Desktop
```

**Permission Issues:**
```bash
# Ensure directories have correct permissions
mkdir -p models data logs reports mlflow
chmod 755 models data logs reports mlflow
```

**Port Conflicts:**
```bash
# Check if ports are in use
netstat -ano | findstr :8000
netstat -ano | findstr :5001

# Change ports in docker-compose.yml if needed
```

### Common Issues

**ModuleNotFoundError:**
```bash
# Rebuild with clear cache
docker-compose build --no-cache api
# Check installed packages
docker-compose exec api pip list
```

**MLflow Connection Failed:**
```bash
# Check MLflow is running
docker-compose ps mlflow
# Test connection
docker-compose exec api curl http://mlflow:5000
```

**Volume Mount Issues:**
```bash
# Verify volumes are mounted
docker inspect smartchurn-api --format='{{.Mounts}}'
# Check directory permissions on host
ls -la models/ data/ logs/
```

## Contributing

1. Create a feature branch from `develop`
2. Implement your changes with tests
3. Run `make lint` and `make test`
4. Create a pull request to `develop`

---

**Maintainer:** [S. R. Moslemi](https://github.com/seyyedmsl82)  
**Last Updated:** July 2026

---

## Detailed Documentation

For more detailed information about specific components, see the documentation in the `docs/` directory:

- [Data Ingestion](docs/data_ingestion.md) - Data loading and caching
- [Feature Engineering](docs/feature_engineering/) - Feature creation and selection
- [Model Training](docs/evaluation.md) - Evaluation pipeline and Evaluation Features
- [API Service](docs/api_service.md) - API endpoints and usage
