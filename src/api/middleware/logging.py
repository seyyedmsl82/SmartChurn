"""
Logging middleware
"""
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger
import time

class LoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests and responses"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")
        logger.debug(f"Headers: {dict(request.headers)}")
        
        # Process request
        try:
            response = await call_next(request)
            
            # Log response
            duration = time.time() - start_time
            logger.info(f"Response: {response.status_code} ({duration:.3f}s)")
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Request failed: {e} ({duration:.3f}s)")
            raise
