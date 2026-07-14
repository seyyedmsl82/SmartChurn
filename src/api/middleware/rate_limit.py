"""
Rate limiting middleware
"""
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List

from src.api.config import settings

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter"""
    
    def __init__(self, app):
        super().__init__(app)
        self.requests: Dict[str, List[datetime]] = defaultdict(list)
        self.limit = settings.RATE_LIMIT_REQUESTS
        self.period = settings.RATE_LIMIT_PERIOD
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path.startswith("/health"):
            return await call_next(request)
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Clean old requests
        now = datetime.now()
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if req_time > now - timedelta(seconds=self.period)
        ]
        
        # Check limit
        if len(self.requests[client_ip]) >= self.limit:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Limit: {self.limit} per {self.period} seconds"
            )
        
        # Add request
        self.requests[client_ip].append(now)
        
        return await call_next(request)
