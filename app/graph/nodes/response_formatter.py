import logging
from typing import Dict, Any
from app.graph.state import GraphState, Channel

logger = logging.getLogger(__name__)

def format_for_rocketchat(response: str) -> Dict[str, Any]:
    """Format response for Rocket.Chat (supports Markdown)"""
    return {
        "text": response,
        "parseUrls": True,
        "alias": "Felcloud Support AI"
    }

def format_for_email(response: str, subject: str = "Re: Felcloud Support") -> Dict[str, Any]:
    """Format response for Email"""
    return {
        "subject": subject,
        "body": response,
        "is_html": False
    }

def format_for_webhook(response: str) -> Dict[str, Any]:
    """Format response for generic Webhook (JSON)"""
    return {
        "response": response,
        "status": "success"
    }

def response_formatter(state: GraphState) -> Dict[str, Any]:
    """
    Day 8: Channel-aware response formatting.
    Transforms the final string response into the appropriate format for the destination channel.
    """
    logger.info(f"--- NODE: response_formatter (channel: {state.channel}) ---")
    
    if not state.final_response:
        return {"formatted_response": None}
        
    formatted = None
    if state.channel == Channel.ROCKETCHAT:
        formatted = format_for_rocketchat(state.final_response)
    elif state.channel == Channel.EMAIL:
        # In a real scenario, we might extract the original subject from state to prefix with "Re:"
        formatted = format_for_email(state.final_response)
    elif state.channel == Channel.WEBHOOK:
        formatted = format_for_webhook(state.final_response)
    else:
        # Default fallback
        formatted = {"response": state.final_response}
        
    return {"formatted_response": formatted}
