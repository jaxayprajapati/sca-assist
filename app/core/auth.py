"""Lightweight JWT auth utilities for FastAPI endpoints."""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config.settings import settings
from app.utils.logging import logger


_security = HTTPBearer(auto_error=False)

_CREDENTIALS_EXCEPTION = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def _build_payload(
    subject: str,
    scopes: Optional[List[str]] = None,
    expires_minutes: Optional[int] = None,
) -> Dict[str, Any]:
    """Build JWT payload with subject, scopes, and expiration."""
    expires_in = expires_minutes if expires_minutes is not None else settings.JWT_EXPIRES_MINUTES
    now = datetime.now(timezone.utc)
    expire_at = now + timedelta(minutes=expires_in)
    return {
        "sub": subject,
        "scopes": scopes or [],
        "exp": expire_at,
        "iat": now,
    }


def create_access_token(
    subject: str,
    scopes: Optional[List[str]] = None,
    expires_minutes: Optional[int] = None,
) -> str:
    """Create a signed JWT access token."""
    payload = _build_payload(subject, scopes, expires_minutes)
    token = jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    logger.debug(f"Access token created for subject: {subject}")
    return token


def decode_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        logger.debug(f"Token validated for subject: {payload.get('sub')}")
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Token validation failed: expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        logger.warning(f"Token validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_security),
) -> Dict[str, Any]:
    """FastAPI dependency to extract and validate current user from JWT."""
    if credentials is None:
        logger.warning("Authorization header missing")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_token(credentials.credentials)


def require_scope(required_scope: str):
    """
    Factory for a FastAPI dependency that enforces a specific scope.

    Usage:
        @router.get("/admin", dependencies=[Depends(require_scope("admin"))])
    """
    def dependency(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        scopes: List[str] = user.get("scopes", []) if isinstance(user, dict) else []
        if required_scope and required_scope not in scopes:
            logger.warning(f"Scope '{required_scope}' required but user has {scopes}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Scope '{required_scope}' required",
            )
        return user

    return dependency
