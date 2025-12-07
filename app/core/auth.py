"""Lightweight JWT auth utilities for FastAPI endpoints."""

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config.settings import settings


_security = HTTPBearer(auto_error=False)


def _build_payload(subject: str, scopes: Optional[List[str]] = None, expires_minutes: Optional[int] = None) -> Dict[str, Any]:
    expires_in = expires_minutes if expires_minutes is not None else settings.JWT_EXPIRES_MINUTES
    expire_at = datetime.utcnow() + timedelta(minutes=expires_in)
    return {
        "sub": subject,
        "scopes": scopes or [],
        "exp": expire_at,
        "iat": datetime.utcnow(),
    }


def create_access_token(subject: str, scopes: Optional[List[str]] = None, expires_minutes: Optional[int] = None) -> str:
    payload = _build_payload(subject, scopes, expires_minutes)
    token = jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    return token


def decode_token(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(_security)) -> Dict[str, Any]:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header missing")

    token = credentials.credentials
    return decode_token(token)


def require_scope(required_scope: str):
    def dependency(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        scopes = user.get("scopes", []) if isinstance(user, dict) else []
        if required_scope and required_scope not in scopes:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient scope")
        return user

    return dependency
