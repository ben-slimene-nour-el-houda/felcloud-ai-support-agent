import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.graph.state import GraphState
from app.config import settings

logger = logging.getLogger(__name__)

LITELLM_BASE_URL = settings.LLM_BASE_URL
LITELLM_API_KEY = settings.LLM_API_KEY
MODEL_NAME = settings.GENERATION_MODEL

def generation_node(state: GraphState) -> GraphState:
    """
    Day 5/6: Generation Node
    Generates a response using Llama 3.2:3B, attributing sources.
    """
    logger.info("--- NODE: generation_node ---")
    
    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=LITELLM_BASE_URL,
        api_key=LITELLM_API_KEY,
        temperature=0.3
    )
    
    system_prompt = "You are a helpful and professional AI support agent for Felcloud. "
    
    if state.intent == "general_chat":
        system_prompt += "Respond politely to the user's conversational message."
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=state.user_message)]
    else:
        system_prompt += (
            "Use the provided context to answer the user's question. "
            "If the answer is not in the context, say that you don't know based on the available documentation."
        )
        context = state.retrieved_context or "No context available."
        prompt = f"Context:\n{context}\n\nUser Question:\n{state.user_message}"
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
        
    try:
        response = llm.invoke(messages)
        final_text = response.content
        
        # Source attribution (J4)
        if state.intent == "support" and state.sources and len(state.sources) > 0:
            sources_text = "\n\n**Sources utilisées :**\n"
            for idx, source in enumerate(state.sources, 1):
                doc_id = source.get("metadata", {}).get("doc_id", "Inconnu")
                title = source.get("metadata", {}).get("title", f"Document {doc_id}")
                sources_text += f"{idx}. {title}\n"
            final_text += sources_text
            
        state.final_response = final_text
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        state.final_response = "Désolé, je n'ai pas pu générer une réponse pour le moment."
        
    return state
