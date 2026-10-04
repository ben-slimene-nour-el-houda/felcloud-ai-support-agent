from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class Channel(str, Enum):
    ROCKETCHAT = "rocketchat"
    EMAIL = "email"
    MQTT = "mqtt"
    WEBHOOK = "webhook"

class Language(str, Enum):
    EN = "en"
    FR = "fr"
    DARIJA = "darija"

class GraphState(BaseModel):
    """
    Shared state for the Felcloud AI Support Chatbot LangGraph workflow.
    This schema defines the contract across all nodes.
    """
    # Populated Day 1
    conversation_id: str
    session_id: str
    channel: Channel
    user_message: str
    chat_history: List[Dict[str, Any]] = Field(default_factory=list)
    language: Language

    # Populated Day 4
    intent: Optional[str] = None

    # Auth — populated at webhook entry from JWT-verified role (J11/J12)
    caller_role: Optional[str] = None

    # Populated Day 3
    retrieved_context: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None

    # Populated Day 5
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None

    # Populated Day 6
    validation_status: Optional[str] = None
    retry_count: int = Field(default=0, ge=0)

    # Populated Day 7
    escalation_flag: bool = False
    ticket_id: Optional[str] = None
    
    # Final Output
    final_response: Optional[str] = None
    formatted_response: Optional[Any] = None
