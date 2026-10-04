import logging
import uuid
import requests
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from app.config import settings

logger = logging.getLogger(__name__)

# Simple in-memory set to ensure idempotency for mock implementation
_created_tickets = set()

class CreateTicketSchema(BaseModel):
    user_id: str = Field(description="The ID of the user requesting support")
    issue_description: str = Field(description="Detailed description of the issue to escalate")
    priority: str = Field(default="medium", description="Priority level: low, medium, high, critical")
    idempotency_key: str = Field(description="Unique key to prevent duplicate tickets for the same issue")

@tool("create_ticket", args_schema=CreateTicketSchema)
def create_ticket(user_id: str, issue_description: str, priority: str = "medium", idempotency_key: str = "") -> dict:
    """Creates a support ticket and escalates it to human agents. Uses an idempotency key to prevent duplicates."""
    logger.info(f"Executing create_ticket for user: {user_id}, priority: {priority}")
    
    if not idempotency_key:
        idempotency_key = str(uuid.uuid4())
        
    if idempotency_key in _created_tickets:
        logger.info(f"Ticket creation skipped: duplicate idempotency_key {idempotency_key}")
        return {
            "status": "skipped",
            "message": "Ticket already exists for this issue.",
            "ticket_id": f"TKT-{idempotency_key[:8].upper()}"
        }
        
    if settings.TICKETING_API_URL:
        try:
            headers = {"Authorization": f"Bearer {settings.TICKETING_API_KEY}"} if settings.TICKETING_API_KEY else {}
            payload = {
                "user_id": user_id,
                "issue_description": issue_description,
                "priority": priority,
                "idempotency_key": idempotency_key
            }
            response = requests.post(f"{settings.TICKETING_API_URL}/api/v1/tickets", json=payload, headers=headers, timeout=5)
            response.raise_for_status()
            _created_tickets.add(idempotency_key)
            data = response.json()
            return {
                "status": "success",
                "ticket_id": data.get("ticket_id", f"TKT-{idempotency_key[:8].upper()}"),
                "message": data.get("message", "Successfully created ticket via API.")
            }
        except Exception as e:
            logger.error(f"Failed to create ticket via API: {e}")
            return {"status": "error", "message": f"Failed to create ticket: {str(e)}"}

    # Fallback to mock implementation
    _created_tickets.add(idempotency_key)
    
    ticket_id = f"TKT-{idempotency_key[:8].upper()}"
    
    return {
        "status": "success",
        "ticket_id": ticket_id,
        "message": f"Successfully created ticket {ticket_id} with {priority} priority."
    }

ticketing_tools = [create_ticket]
