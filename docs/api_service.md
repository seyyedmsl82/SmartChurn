# API Service Documentation

## Overview

The SmartChurn API Service provides a production-grade REST interface for customer churn predictions. Built with FastAPI, it offers low-latency inference, batch processing, and comprehensive monitoring capabilities.

**Base URL:** `http://localhost:8000/api/v1`

**Documentation:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                     API Service Layer                      │
├────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Routes    │  │  Schemas    │  │   Middleware        │ │
│  │  predict/   │  │  Request/   │  │  - Rate Limiting    │ │
│  │  health/    │  │  Response   │  │  - Logging          │ │
│  │  metrics/   │  │  Validation │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
├────────────────────────────────────────────────────────────┤
│                      Service Layer                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Prediction Service                         │  │
│  │  - Model Loading & Caching                           │  │
│  │  - Preprocessing Pipeline                            │  │
│  │  - Feature Engineering                               │  │
│  │  - Inference Logic                                   │  │
│  └──────────────────────────────────────────────────────┘  │
├────────────────────────────────────────────────────────────┤
│                    Infrastructure                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  - Model Registry (versioned models)                 │  │
│  │  - Preprocessor Artifacts                            │  │
│  │  - Feature Selection Config                          │  │
│  │  - Logging & Monitoring                              │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

---

## Endpoints

### 1. Health Check

#### GET `/health`
Check service health and component status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "RandomForest_v1.0.0",
  "preprocessor_loaded": true,
  "timestamp": "2024-01-15T10:30:00.123Z"
}
```

---

### 2. Predictions

#### POST `/predict/single`
Make a single customer churn prediction.

**Request Body:**
```json
{
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
}
```

**Response:**
```json
{
  "prediction": 0,
  "probability": 0.23,
  "churn_risk": "Low",
  "model_version": "RandomForest_v1.0.0",
  "timestamp": "2024-01-15T10:30:00.123Z"
}
```

**Status Codes:**
- `200 OK` - Prediction successful
- `422 Bad Request` - Invalid input data
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error

---

#### POST `/predict/batch`
Make predictions for up to 1000 customers in a single request.

**Request Body:**
```json
{
  "customers": [
    {
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
    },
    {
      "gender": "Male",
      "SeniorCitizen": 1,
      "Partner": "No",
      "Dependents": "Yes",
      "tenure": 48,
      "Contract": "Two year",
      "PaperlessBilling": "No",
      "PaymentMethod": "Credit card (automatic)",
      "MonthlyCharges": 95.0,
      "TotalCharges": 4560.0,
      "PhoneService": "Yes",
      "MultipleLines": "Yes",
      "InternetService": "Fiber optic",
      "OnlineSecurity": "Yes",
      "OnlineBackup": "Yes",
      "DeviceProtection": "Yes",
      "TechSupport": "No",
      "StreamingTV": "Yes",
      "StreamingMovies": "Yes"
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "prediction": 0,
      "probability": 0.23,
      "churn_risk": "Low",
      "model_version": "RandomForest_v1.0.0",
      "timestamp": "2024-01-15T10:30:00.123Z"
    },
    {
      "prediction": 1,
      "probability": 0.78,
      "churn_risk": "High",
      "model_version": "RandomForest_v1.0.0",
      "timestamp": "2024-01-15T10:30:00.123Z"
    }
  ],
  "total_count": 2,
  "churn_count": 1,
  "churn_rate": 0.5,
  "model_version": "RandomForest_v1.0.0",
  "timestamp": "2024-01-15T10:30:00.123Z"
}
```

**Status Codes:**
- `200 OK` - Batch prediction successful
- `422 Bad Request` - Invalid input or batch size > 1000
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error

---

### 3. Metrics

#### GET `/metrics`
Get model performance metrics from the latest evaluation report.

**Response:**
```json
{
  "model_version": "RandomForest_v1.0.0",
  "metrics": {
    "accuracy": 0.8432,
    "precision": 0.7845,
    "recall": 0.7832,
    "f1": 0.7838,
    "roc_auc": 0.8405,
    "churn_detection_rate": 0.7832,
    "false_alarm_rate": 0.1245,
    "total_cost": 12450.00,
    "cost_per_customer": 1.77
  },
  "last_updated": "2024-01-15T10:30:00.123Z"
}
```

---

**Interpretation:**
- **Positive SHAP values** → Feature increases churn probability
- **Negative SHAP values** → Feature decreases churn probability
- **Magnitude** → Importance of the feature for this prediction

---

## Field Descriptions

### Customer Fields

| Field | Type | Description | Valid Values |
|-------|------|-------------|--------------|
| `gender` | string | Customer gender | `Male`, `Female` |
| `SeniorCitizen` | integer | Senior citizen indicator | `0` (No), `1` (Yes) |
| `Partner` | string | Has partner | `Yes`, `No` |
| `Dependents` | string | Has dependents | `Yes`, `No` |
| `tenure` | integer | Months with company | `0` - `72` |
| `Contract` | string | Contract type | `Month-to-month`, `One year`, `Two year` |
| `PaperlessBilling` | string | Paperless billing | `Yes`, `No` |
| `PaymentMethod` | string | Payment method | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` |
| `MonthlyCharges` | float | Monthly charges | `0` - `200` |
| `TotalCharges` | float | Total charges | `0` - `10000` |
| `PhoneService` | string | Phone service | `Yes`, `No` |
| `MultipleLines` | string | Multiple phone lines | `Yes`, `No`, `No phone service` |
| `InternetService` | string | Internet service type | `DSL`, `Fiber optic`, `No` |
| `OnlineSecurity` | string | Online security | `Yes`, `No`, `No internet service` |
| `OnlineBackup` | string | Online backup | `Yes`, `No`, `No internet service` |
| `DeviceProtection` | string | Device protection | `Yes`, `No`, `No internet service` |
| `TechSupport` | string | Technical support | `Yes`, `No`, `No internet service` |
| `StreamingTV` | string | Streaming TV | `Yes`, `No`, `No internet service` |
| `StreamingMovies` | string | Streaming movies | `Yes`, `No`, `No internet service` |

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | integer | Churn prediction: `0` (No), `1` (Yes) |
| `probability` | float | Churn probability between 0 and 1 |
| `churn_risk` | string | Risk level: `Low` (<0.3), `Medium` (0.3-0.6), `High` (>0.6) |
| `model_version` | string | Model version identifier |
| `timestamp` | string | ISO 8601 timestamp |

---

## Rate Limiting

The API enforces rate limits to prevent abuse:

| Limit | Value |
|-------|-------|
| Max requests per minute | `100` |
| Max batch size | `1000` customers |

**Response when rate limit exceeded:**
```json
{
  "detail": "Rate limit exceeded. Limit: 100 per 60 seconds"
}
```

**Headers:**
- `X-RateLimit-Limit`: Request limit
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Reset time

---

## Error Handling

All errors follow a consistent format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Error Codes

| Status | Description |
|--------|-------------|
| `404` | Endpoint not found |
| `422` | Invalid input data (validation error) |
| `429` | Rate limit exceeded |
| `500` | Internal server error |
```

---

## Authentication (Optional)

For production environments, API key authentication can be enabled:

```bash
curl -X POST http://localhost:8000/api/v1/predict/single \
  -H "X-API-Key: your-secret-api-key" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

**Configure in `.env`:**
```env
API_KEY=your-secret-api-key-here
```

---

## Performance

| Metric | Value |
|--------|-------|
| **Latency (p95)** | < 50ms per prediction |
| **Throughput** | > 1000 req/sec |
| **Batch processing** | 1000 customers in < 2s |
| **Availability** | 99.9% |

---

## SDK Examples

### Python

```python
import requests
import json

# Single prediction
response = requests.post(
    "http://localhost:8000/api/v1/predict/single",
    json={
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
    }
)

result = response.json()
print(f"Churn Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.2%}")
print(f"Risk Level: {result['churn_risk']}")
```

### cURL

```bash
# Single prediction
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

# Batch prediction
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {
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
    ]
  }'

# Health check
curl http://localhost:8000/api/v1/health

# Get metrics
curl http://localhost:8000/api/v1/metrics
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | `development` | Environment: `development`, `production`, `staging` |
| `DEBUG` | `True` | Debug mode |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Bind port |
| `WORKERS` | `1` | Number of worker processes |
| `MODEL_PATH` | `models/registry` | Model registry path |
| `PREPROCESSOR_PATH` | `models/preprocessor.pkl` | Preprocessor artifact path |
| `FEATURES_PATH` | `models/selected_features.json` | Selected features config |
| `RATE_LIMIT_REQUESTS` | `100` | Max requests per period |
| `RATE_LIMIT_PERIOD` | `60` | Rate limit period (seconds) |
| `LOG_LEVEL` | `INFO` | Logging level |
| `API_KEY` | `None` | API key for authentication |

---

## Monitoring & Logging

### Health Checks

- **Liveness:** `/health/liveness` - Service is running
- **Readiness:** `/health/readiness` - Service is ready to accept traffic
- **Health:** `/health` - Full health status with component details

### Metrics Exposed

The API exposes Prometheus-style metrics for monitoring:
- Request count by endpoint
- Request latency (p50, p95, p99)
- Error rates
- Prediction counts (churn vs non-churn)
- Model version tracking

### Logging

Logs are structured and include:
- Request ID for tracing
- Endpoint and method
- Response status and latency
- Error details

**Log Format:**
```
2024-01-15 10:30:00.123 | INFO | Request: POST /api/v1/predict/single | 200 (0.045s)
2024-01-15 10:30:00.456 | INFO | Response: 200 OK | (0.045s) | client_ip=192.168.1.1
```

### Debug Mode

Enable debug mode in `.env`:
```env
DEBUG=True
```

This enables:
- Detailed error messages
- Swagger UI at `/docs`
- Auto-reload on code changes
- SQL query logging

### Health Check Failed

Check the service status:
```bash
curl http://localhost:8000/api/v1/health
```

Check logs:
```bash
docker-compose logs api
# or
tail -f logs/api.log
```

---

## Performance Tuning

### Scaling Workers

Increase number of workers based on available CPU cores:
```bash
# In .env
WORKERS=4

# Or when running
uvicorn src.api.app:app --workers 4
```

### Rate Limit Tuning

Adjust for your use case:
```env
# Higher throughput
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_PERIOD=60

# Stricter limits
RATE_LIMIT_REQUESTS=50
RATE_LIMIT_PERIOD=300
```

### Batch Size

Increase for higher throughput:
```env
MAX_BATCH_SIZE=5000
```
