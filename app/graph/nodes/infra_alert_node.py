import logging
import json
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.graph.state import GraphState
from app.config import settings

logger = logging.getLogger(__name__)

LITELLM_BASE_URL = settings.LLM_BASE_URL
LITELLM_API_KEY = settings.LLM_API_KEY
MODEL_NAME = settings.STRUCTURED_OUTPUT_MODEL

class StructuredAlertSummary(BaseModel):
    severity: str = Field(description="Severity of the alert: 'critical', 'warning', or 'info'")
    component: str = Field(description="The system component affected (e.g., 'database', 'network', 'api')")
    summary: str = Field(description="A concise 1-2 sentence summary of the infrastructure alert.")
    action_required: bool = Field(description="Whether immediate human action is required.")

def infra_alert_node(state: GraphState) -> GraphState:
    """
    Day 9: Infrastructure Alert Node
    Processes incoming MQTT infrastructure events and generates a structured summary.
    """
    logger.info("--- NODE: infra_alert_node ---")
    
    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=LITELLM_BASE_URL,
        api_key=LITELLM_API_KEY,
        temperature=0.0
    )
    
    structured_llm = llm.with_structured_output(StructuredAlertSummary)
    
    system_prompt = """You are an infrastructure alert analyzer for Felcloud.
Extract the key details from the raw alert payload and provide a structured summary.
"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state.user_message)
    ]
    
    try:
        result = structured_llm.invoke(messages)
        # Store the structured JSON string as the final response
        state.final_response = result.model_dump_json()
        state.validation_status = "success"
        logger.info(f"Generated structured alert: {state.final_response}")
    except Exception as e:
        logger.error(f"Failed to generate structured alert: {e}")
        state.final_response = json.dumps({
            "severity": "critical",
            "component": "unknown",
            "summary": "Failed to parse alert payload.",
            "action_required": True
        })
        state.validation_status = "success"
        
    return state
