import logging
from langgraph.graph import END
from app.graph.state import GraphState

logger = logging.getLogger(__name__)

def route_intent(state: GraphState) -> str:
    """
    Day 4/5: Router function.
    Reads state.intent and routes to the appropriate node.
    """
    logger.info(f"--- ROUTER: route_intent ({state.intent}) ---")
    
    if state.intent == "escalation":
        return "escalation_node"
    elif state.intent == "infra_alert":
        return "infra_alert_node"
    elif state.intent == "general_chat":
        return "generation_node"
    else:
        # Default to retrieval for support
        return "retrieval_node"

def route_validation(state: GraphState) -> str:
    """
    Day 6: Route based on validation status.
    Routes to retry_node if validation failed and under max retries.
    Otherwise, routes to END.
    """
    MAX_RETRIES = 2
    logger.info(f"--- ROUTER: route_validation (status: {state.validation_status}, retries: {state.retry_count}) ---")
    
    if state.validation_status == "failed" and state.retry_count < MAX_RETRIES:
        return "retry_node"
    
    return "response_formatter"
