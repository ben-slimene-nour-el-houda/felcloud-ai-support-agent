import hmac
import hashlib
import logging
import json
import redis.asyncio as redis
from typing import Optional

from fastapi import APIRouter, Header, Request, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from app.config import settings
from app.graph.workflow import app as langgraph_app
from app.graph.state import Channel, Language

logger = logging.getLogger(__name__)
REPLAY_WINDOW_SECONDS = 300

router = APIRouter()


def get_redis_client() -> redis.Redis:
    """Create a fresh Redis client bound to the current event loop."""
    return redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        decode_responses=True,
    )


class ZeroClawPayloadBody(BaseModel):
    message: str
    user_id: str
    channel: str
    language: str

class ZeroClawEvent(BaseModel):
    event_type: str
    session_id: str
    payload: ZeroClawPayloadBody

async def verify_signature(request: Request, x_zeroclaw_signature: Optional[str] = Header(None)):
    """
    Dependency to verify the HMAC-SHA256 signature from ZeroClaw,
    and reject replayed (already-seen) signed payloads.
    """
    if not x_zeroclaw_signature:
        logger.warning("Missing X-ZeroClaw-Signature header in webhook request.")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing signature")

    body = await request.body()
    secret = settings.ZEROCLAW_WEBHOOK_SECRET.encode("utf-8")

    expected_signature = hmac.new(secret, body, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(expected_signature, x_zeroclaw_signature):
        logger.warning("Invalid ZeroClaw signature received.")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid signature")

    # --- Anti-replay check ---
    client = get_redis_client()
    try:
        nonce_key = f"webhook:seen_sig:{expected_signature}"
        is_new = await client.set(nonce_key, "1", nx=True, ex=REPLAY_WINDOW_SECONDS)
        if not is_new:
            logger.warning("Replayed webhook payload detected (signature already seen).")
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Duplicate/replayed request")
    finally:
        await client.aclose()

@router.post(settings.ZEROCLAW_WEBHOOK_PATH, dependencies=[Depends(verify_signature)])
async def zeroclaw_webhook(request: Request):
    # 1. Parse and Validate Payload
    try:
        body_bytes = await request.body()
        body_json = json.loads(body_bytes)
        event = ZeroClawEvent(**body_json)
    except json.JSONDecodeError:
        logger.error("Failed to parse webhook JSON body.")
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"status": "error", "message": "Malformed JSON"})
    except ValidationError as e:
        logger.error(f"Webhook payload validation error: {e.errors()}")
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"status": "error", "message": "Invalid payload structure"})

    logger.info(f"Received valid ZeroClaw event: event_type={event.event_type}, session_id={event.session_id}")

    if len(event.payload.message) > 4000:
        logger.warning(f"Payload too large for session {event.session_id}")
        return JSONResponse(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, content={"status": "error", "message": "Message exceeds 4000 characters limit"})

    # 2. Map the validated payload into GraphState schema
    try:
        channel_enum = Channel(event.payload.channel.lower())
    except ValueError:
        logger.warning(f"Invalid channel received: {event.payload.channel}, defaulting to WEBHOOK")
        channel_enum = Channel.WEBHOOK

    try:
        language_enum = Language(event.payload.language.lower())
    except ValueError:
        logger.warning(f"Invalid language received: {event.payload.language}, defaulting to EN")
        language_enum = Language.EN

    state_input = {
        "conversation_id": event.session_id,
        "session_id": event.session_id,
        "channel": channel_enum.value,
        "user_message": event.payload.message,
        "language": language_enum.value,
        "chat_history": [],
        # HMAC signature already verified this request came from ZeroClaw
        # (see verify_signature dependency) — role is fixed accordingly.
        "caller_role": "zeroclaw_agent"
    }

    # 3. Invoke the compiled LangGraph workflow
    try:
        logger.info(f"Invoking LangGraph workflow for session: {event.session_id}")
        result = langgraph_app.invoke(state_input)
        logger.info(f"LangGraph invocation completed successfully for session: {event.session_id}")
        final_response = result.get("final_response")
        return {"status": "success", "reply": final_response}
    except Exception as e:
        logger.exception(f"LangGraph execution failed mid-invocation for session: {event.session_id}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "message": "An internal error occurred during graph processing"}
        )
