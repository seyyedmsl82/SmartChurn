"""
Main FastAPI application
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from loguru import logger

from src.api.config import settings
from src.api.dependencies import get_prediction_service
from src.api.middleware.logging import LoggingMiddleware
from src.api.middleware.rate_limit import RateLimitMiddleware
from src.api.routes import health, metrics, predict

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-ready customer churn prediction API",
    docs_url="/docs" if settings.APP_ENV == "development" else None,
    redoc_url="/redoc" if settings.APP_ENV == "development" else None,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure in production
)

# Add custom middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)

# Include routers
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(metrics.router)

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.APP_ENV}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    # Load prediction service
    try:
        service = get_prediction_service()
        metadata = service.get_metadata()
        logger.info(f"Loaded model: {metadata['model_version']}")
        logger.info(f"Features count: {metadata['features_count']}")
    except Exception as e:
        logger.error(f"Failed to load prediction service: {e}")
        # Don't raise - app will still start but predict endpoints will fail

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("Shutting down API...")

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint"""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs" if settings.APP_ENV == "development" else None
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        reload=settings.DEBUG
    )
