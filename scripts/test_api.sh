#!/bin/bash
# Test the API endpoints

echo "Testing API endpoints..."

# Health check
echo -e "\n1. Health Check:"
curl -s http://localhost:8000/api/v1/health | jq '.'

# Get metrics
echo -e "\n2. Metrics:"
curl -s http://localhost:8000/api/v1/metrics | jq '.'

# Prediction
echo -e "\n3. Prediction:"
curl -s -X POST http://localhost:8000/api/v1/predict/single \
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
  }' | jq '.'

echo -e "\nAPI test complete!"
