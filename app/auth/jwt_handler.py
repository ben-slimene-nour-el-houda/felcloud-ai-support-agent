"""JWT creation and validation for service-to-service authentication."""
import logging
from datetime import datetime, timedelta, timezone

import jwt
from jwt import PyJWTError

from app.config import settings

logger = logging.getLogger(__name__)


def create_access_token(role: str, subject: str) -> str:
    """
    Create a signed JWT containing the caller's verified role.

    Args:
        role: The verified role to embed in the token (e.g. "zeroclaw_agent").
        subject: Identifier of the token owner (e.g. "zeroclaw").
    """
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)

    payload = {
        "sub": subject,
        "role": role,
        "iat": now,
        "exp": expire,
    }

    token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    logger.info(f"Issued JWT for subject='{subject}' role='{role}' expiring at {expire.isoformat()}")
    return token


def decode_access_token(token: str) -> dict:
    """
    Decode and validate a JWT. Raises PyJWTError on invalid/expired tokens.
    """
    payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    return payload
