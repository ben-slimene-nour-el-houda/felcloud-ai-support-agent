import logging
from app.graph.state import GraphState

logger = logging.getLogger(__name__)

def validation_node(state: GraphState) -> GraphState:
    """
    Day 6: Validation Node
    Validates the generated response or tool output.
    """
    logger.info("--- NODE: validation_node ---")
    
    response = state.final_response
    is_valid = True
    
    if not response or len(response.strip()) < 10:
        is_valid = False
    else:
        lower_resp = response.lower()
        error_phrases = [
            "désolé, je n'ai pas pu",
            "i don't know",
            "i cannot answer",
            "as an ai",
            "i am an ai language model"
        ]
        if any(phrase in lower_resp for phrase in error_phrases):
            is_valid = False
            
    state.validation_status = "passed" if is_valid else "failed"
    logger.info(f"Validation status set to: {state.validation_status}")
    
    return state
