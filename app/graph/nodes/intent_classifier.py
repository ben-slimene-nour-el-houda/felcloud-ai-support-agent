import logging
from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.graph.state import GraphState, Channel
from app.config import settings

logger = logging.getLogger(__name__)

LITELLM_BASE_URL = settings.LLM_BASE_URL
LITELLM_API_KEY = settings.LLM_API_KEY
MODEL_NAME = settings.STRUCTURED_OUTPUT_MODEL

class IntentClassification(BaseModel):
    intent: Literal["support", "escalation", "general_chat"] = Field(
        description="The intent of the user message. 'support' for technical questions needing documentation. 'escalation' for complaints or explicit requests to talk to a human. 'general_chat' for greetings, thanks, or casual conversation."
    )

def intent_classifier(state: GraphState) -> GraphState:
    """
    Day 4: Intent Classifier Node
    Classifies the user message intent using an LLM.
    """
    logger.info("--- NODE: intent_classifier ---")
    
    # Day 11: Prompt Injection Protection
    lower_msg = state.user_message.lower()
    injection_keywords = ["ignore previous instructions", "system prompt", "forget all instructions", "you are now"]
    if any(kw in lower_msg for kw in injection_keywords):
        logger.warning(f"Potential prompt injection detected: {state.user_message}")
        state.intent = "escalation"
        state.final_response = "Request blocked due to security policies."
        return state

    if state.channel == Channel.MQTT:
        state.intent = "infra_alert"
        logger.info("MQTT channel detected, bypassing LLM classification -> infra_alert")
        return state

    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=LITELLM_BASE_URL,
        api_key=LITELLM_API_KEY,
        temperature=0.0
    )
    
    structured_llm = llm.with_structured_output(IntentClassification)
    
    system_prompt = """You are an intent classification system for the Felcloud AI support agent.
Analyze the user's message and determine the most appropriate intent:
- 'support': The user is asking a technical question, troubleshooting issue, or requesting information about Felcloud products/services that requires knowledge base lookup.
- 'escalation': The user is explicitly asking for a human agent, complaining about service, or expressing extreme frustration.
- 'general_chat': The user is just saying hello, thank you, or making casual conversation that does not require documentation lookup.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state.user_message)
    ]
    
    try:
        result = structured_llm.invoke(messages)
        state.intent = result.intent
        logger.info(f"Intent classified as: {state.intent}")
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        # Default to support on failure
        state.intent = "support"
        
    return state
