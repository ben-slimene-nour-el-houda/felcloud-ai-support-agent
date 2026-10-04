"""Token issuance endpoint for service-to-service authentication (ZeroClaw)."""
import logging

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel

from app.auth.jwt_handler import create_access_token
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int


@router.post("/token", response_model=TokenResponse)
async def issue_token(x_api_key: str = Header(..., alias="X-API-Key")):
    """
    Exchange ZeroClaw's existing API key for a short-lived JWT.
    """
    if not settings.ZEROCLAW_API_KEY:
        logger.error("ZEROCLAW_API_KEY is not configured; cannot issue tokens.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Token issuance not configured")

    if x_api_key != settings.ZEROCLAW_API_KEY:
        logger.warning("Token request rejected: invalid API key.")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    token = create_access_token(role="zeroclaw_agent", subject="zeroclaw")
    return TokenResponse(access_token=token, expires_in_minutes=settings.JWT_EXPIRE_MINUTES)


from app.auth.dependencies import get_current_role
from fastapi import Depends


@router.get("/whoami")
async def whoami(role: str = Depends(get_current_role)):
    """Debug endpoint: returns the verified role extracted from the bearer token."""
    return {"role": role}
