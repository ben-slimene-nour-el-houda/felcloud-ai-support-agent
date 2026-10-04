import logging
import uuid
from app.graph.state import GraphState
from app.tools.ticketing import create_ticket

logger = logging.getLogger(__name__)

def escalation_node(state: GraphState) -> GraphState:
    """
    Day 7: Escalation Node
    Escalates to a human agent and generates a ticket idempotently.
    """
    logger.info("--- NODE: escalation_node ---")
    
    # We use a unique idempotency key based on conversation ID or session if available
    # For now, just generate a uuid or use user_message as a proxy for the issue
    id_key = str(uuid.uuid5(uuid.NAMESPACE_OID, state.user_message))
    
    ticket_result = create_ticket.invoke({
        "user_id": "user_unknown", # In a real scenario, this comes from state/metadata
        "issue_description": state.user_message,
        "priority": "high",
        "idempotency_key": id_key
    })
    
    if ticket_result.get("status") == "success":
        state.final_response = (
            f"Je ne peux pas résoudre ce problème directement. "
            f"J'ai créé le ticket d'assistance {ticket_result['ticket_id']} "
            f"et un agent humain vous contactera prochainement."
        )
    else:
        state.final_response = (
            f"J'ai déjà signalé ce problème à notre équipe de support. "
            f"(Ticket: {ticket_result.get('ticket_id')})"
        )
        
    return state
