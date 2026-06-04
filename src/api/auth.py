"""
Simple API key authentication.

For a portfolio project, API key auth is sufficient.
Production systems would use OAuth2 or JWT.
"""

import os
import secrets
import hmac
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """Validate the API key from request header.
    
    Use hmac.compare_digest to prevent timing attacks -
    a constant-time comparison so attackers can't guess
    the key character by character based on response
    time.
    """

    expected = os.getenv("API_KEY")

    if not expected:
        raise HTTPException(500, "API_KEY not configured on server")
    
    if not hmac.compare_digest(api_key, expected):
        raise HTTPException(401, "Invalid API Key")
    
    return api_key

