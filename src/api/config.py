"""
API configuration
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """API settings"""
    
    # API Settings
    APP_NAME: str = "SmartChurn API"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = Field(default="development", env="APP_ENV")
    DEBUG: bool = Field(default=True, env="DEBUG")
    
    # Server Settings
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    WORKERS: int = Field(default=1, env="WORKERS")
    
    # Model Settings
    MODEL_PATH: str = Field(default="models/registry", env="MODEL_PATH")
    PREPROCESSOR_PATH: str = Field(default="models/preprocessor.pkl", env="PREPROCESSOR_PATH")
    FEATURES_PATH: str = Field(default="models/selected_features.json", env="FEATURES_PATH")
    
    # API Limits
    MAX_BATCH_SIZE: int = Field(default=1000, env="MAX_BATCH_SIZE")
    RATE_LIMIT_REQUESTS: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    RATE_LIMIT_PERIOD: int = Field(default=60, env="RATE_LIMIT_PERIOD")  # seconds
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: str = Field(default="logs/api.log", env="LOG_FILE")
    
    # Security
    API_KEY: Optional[str] = Field(default=None, env="API_KEY")  # Optional auth
    ALLOWED_ORIGINS: list = Field(default=["*"], env="ALLOWED_ORIGINS")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
