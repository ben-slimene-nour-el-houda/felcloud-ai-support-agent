from app.graph.state import GraphState
from app.rag.retrieval.search import retrieve_context
import logging

logger = logging.getLogger(__name__)

def retrieval_node(state: GraphState) -> GraphState:
    """
    Day 3: Retrieval Node
    Retrieves context from Qdrant based on the user message.
    Applies hybrid search, RRF fusion, and cross-encoder reranking.
    """
    print("--- NODE: retrieval_node ---")
    
    query = state.user_message
    language_val = state.language.value if state.language else None
    
    try:
        context, sources = retrieve_context(
            query=query,
            language=language_val,
            limit=10,
            top_n=3
        )
        
        # We mutate state to include retrieved values
        state.retrieved_context = context
        state.sources = sources
        
        logger.info(f"Successfully retrieved {len(sources)} sources.")
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        state.retrieved_context = "Erreur lors de la récupération du contexte."
        state.sources = []
        
    return state
