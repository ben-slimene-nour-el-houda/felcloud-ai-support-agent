import logging
from app.graph.state import GraphState

logger = logging.getLogger(__name__)

def retry_node(state: GraphState) -> GraphState:
    """
    Day 6: Retry Node
    Handles retries if validation fails.
    """
    logger.info("--- NODE: retry_node ---")
    
    state.retry_count += 1
    logger.info(f"Retry count is now: {state.retry_count}")
    
    return state
