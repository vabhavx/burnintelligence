"""
API key authentication for the Intelligence API.

Behaviour:
- If env var INTEL_API_KEY is unset (or empty), auth is disabled — every
  request passes. This preserves existing client compatibility.
- If INTEL_API_KEY is set, every protected route requires the header
  `Authorization: Bearer <INTEL_API_KEY>`. Anything else → HTTP 401.
"""

import os

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

_api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


def _expected_key() -> str | None:
    key = os.getenv("INTEL_API_KEY", "").strip()
    return key or None


async def require_api_key(authorization: str | None = Security(_api_key_header)) -> None:
    expected = _expected_key()
    if expected is None:
        return  # Auth disabled.

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token or token != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
