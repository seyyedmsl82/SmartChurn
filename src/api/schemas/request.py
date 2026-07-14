"""
Request schemas for API
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator

class CustomerFeatures(BaseModel):
    """Customer features for prediction"""
    
    # Customer demographics
    gender: str = Field(..., description="Customer gender (Male/Female)")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Senior citizen indicator (0/1)")
    Partner: str = Field(..., description="Has partner (Yes/No)")
    Dependents: str = Field(..., description="Has dependents (Yes/No)")
    
    # Account information
    tenure: int = Field(..., ge=0, le=72, description="Customer tenure in months")
    Contract: str = Field(..., description="Contract type")
    PaperlessBilling: str = Field(..., description="Paperless billing (Yes/No)")
    PaymentMethod: str = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges")
    TotalCharges: float = Field(..., ge=0, description="Total charges")
    
    # Services
    PhoneService: str = Field(..., description="Phone service (Yes/No)")
    MultipleLines: str = Field(..., description="Multiple lines")
    InternetService: str = Field(..., description="Internet service type")
    OnlineSecurity: str = Field(..., description="Online security")
    OnlineBackup: str = Field(..., description="Online backup")
    DeviceProtection: str = Field(..., description="Device protection")
    TechSupport: str = Field(..., description="Tech support")
    StreamingTV: str = Field(..., description="Streaming TV")
    StreamingMovies: str = Field(..., description="Streaming movies")
    
    @validator('gender')
    def validate_gender(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError('gender must be Male or Female')
        return v
    
    @validator('Contract')
    def validate_contract(cls, v):
        valid = ['Month-to-month', 'One year', 'Two year']
        if v not in valid:
            raise ValueError(f'Contract must be one of: {valid}')
        return v
    
    @validator('PaymentMethod')
    def validate_payment(cls, v):
        valid = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
        if v not in valid:
            raise ValueError(f'PaymentMethod must be one of: {valid}')
        return v

    @validator("MultipleLines")
    def validate_multiple_lines(cls, v):
        valid = ["Yes", "No", "No phone service"]
        if v not in valid:
            raise ValueError(f"Must be one of {valid}")
        return v

    @validator(
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    )
    def validate_service(cls, v):
        valid = ['Yes', 'No', 'No internet service']
        if v not in valid:
            raise ValueError(f'Value must be one of: {valid}')
        return v

class PredictionRequest(BaseModel):
    """Single prediction request"""
    customer: CustomerFeatures
    
    class Config:
        schema_extra = {
            "example": {
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
        }

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    customers: List[CustomerFeatures] = Field(..., max_items=1000)
    
    @validator('customers')
    def validate_batch_size(cls, v):
        if len(v) > 1000:
            raise ValueError('Batch size cannot exceed 1000')
        return v
