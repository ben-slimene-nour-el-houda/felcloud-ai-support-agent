import logging
import json
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from app.graph.workflow import app as langgraph_app
from app.graph.state import Channel, Language
from app.memory.redis_store import load_chat_history, save_chat_history
from app.channels.email_processor import normalize_email_body

logger = logging.getLogger(__name__)

router = APIRouter()

class EmailPayload(BaseModel):
    session_id: str
    message: str
    channel: str = "email"
    language: str = "en"
    user_metadata: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, Any]]] = None

@router.post("/webhook/email")
async def email_webhook(payload: EmailPayload):
    """
    Day 8 Email Webhook endpoint for LangGraph integration.
    """
    logger.info(f"Received email event for session: {payload.session_id}")
    
    if len(payload.message) > 4000:
        logger.warning(f"Payload too large for session {payload.session_id}")
        return JSONResponse(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, content={"status": "error", "message": "Message exceeds 4000 characters limit"})

    # Map the validated payload into GraphState schema
    try:
        channel_enum = Channel(payload.channel.lower())
    except ValueError:
        logger.warning(f"Invalid channel received: {payload.channel}, defaulting to EMAIL")
        channel_enum = Channel.EMAIL

    try:
        language_enum = Language(payload.language.lower())
    except ValueError:
        logger.warning(f"Invalid language received: {payload.language}, defaulting to EN")
        language_enum = Language.EN
        
    # Hydrate chat history from Redis
    # Use Redis as the source of truth, fallback to payload history if provided
    redis_history = load_chat_history(payload.session_id)
    chat_history = redis_history if redis_history else (payload.history or [])

    # Clean up the email body (strip quotes, signatures)
    cleaned_message = normalize_email_body(payload.message)

    state_input = {
        "conversation_id": payload.session_id,
        "session_id": payload.session_id,
        "channel": channel_enum.value,
        "user_message": cleaned_message,
        "language": language_enum.value,
        "chat_history": chat_history
    }

    # Invoke the compiled LangGraph workflow
    try:
        logger.info(f"Invoking LangGraph workflow for session: {payload.session_id}")
        
        # Invoke LangGraph
        result = langgraph_app.invoke(state_input)
        
        logger.info(f"LangGraph invocation completed successfully for session: {payload.session_id}")
        
        final_response = result.get("final_response")
        formatted_response = result.get("formatted_response")
        
        # Append this interaction to the chat history to persist
        # (Assuming the graph doesn't append it itself, or if it does, we save the graph's output)
        updated_history = result.get("chat_history", chat_history)
        
        # Manually append the new user/assistant messages if the graph doesn't mutate chat_history
        # This depends on how generation_node is written, but usually we just persist what it outputs.
        # Let's just save whatever came out of the graph (or append if empty)
        # Assuming generation_node might not append to it, we'll append here to be safe:
        if not result.get("chat_history") or result.get("chat_history") == chat_history:
            updated_history = chat_history + [
                {"role": "user", "content": payload.message},
                {"role": "assistant", "content": final_response}
            ]
            
        save_chat_history(payload.session_id, updated_history)
        
        # Return structured response (using the formatted one if available)
        reply = formatted_response if formatted_response else {"response": final_response}
        
        return {"status": "success", "reply": reply}
        
    except Exception as e:
        logger.exception(f"LangGraph execution failed mid-invocation for session: {payload.session_id}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "message": "An internal error occurred during graph processing"}
        )
