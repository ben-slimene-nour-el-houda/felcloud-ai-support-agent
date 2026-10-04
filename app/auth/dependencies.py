"""FastAPI dependencies for extracting a verified role from a JWT bearer token."""
import logging

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt import PyJWTError

from app.auth.jwt_handler import decode_access_token

logger = logging.getLogger(__name__)

# tokenUrl is documentation-only here (points at our token issuance endpoint)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


async def get_current_role(token: str = Depends(oauth2_scheme)) -> str:
    """
    Decode the bearer token and return the verified role.
    Use this as a dependency on any route/tool that needs a trusted role,
    instead of trusting a caller-supplied role parameter.
    """
    try:
        payload = decode_access_token(token)
    except PyJWTError as e:
        logger.warning(f"JWT validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    role = payload.get("role")
    if not role:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token missing role claim")

    return role
