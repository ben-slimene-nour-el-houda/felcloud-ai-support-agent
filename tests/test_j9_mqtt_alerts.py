import json
import pytest
from app.graph.state import GraphState, Channel
from app.graph.nodes.intent_classifier import intent_classifier
from app.graph.nodes.infra_alert_node import infra_alert_node
from app.graph.router import route_intent

def test_j9_mqtt_bypasses_llm():
    state = GraphState(
        conversation_id="mqtt_conv",
        session_id="mqtt_sess",
        channel=Channel.MQTT,
        user_message="CRITICAL: Node down",
        language="en"
    )
    
    # Should instantly set intent to 'infra_alert' without calling LLM
    updated_state = intent_classifier(state)
    assert updated_state.intent == "infra_alert"
    
def test_j9_mqtt_router():
    state = GraphState(
        conversation_id="mqtt_conv",
        session_id="mqtt_sess",
        channel=Channel.MQTT,
        user_message="CRITICAL: Node down",
        language="en",
        intent="infra_alert"
    )
    
    next_node = route_intent(state)
    assert next_node == "infra_alert_node"

# Note: We won't test infra_alert_node fully as it calls the LLM, 
# but we can mock it or just verify the node signature/imports work.
