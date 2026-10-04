from langgraph.graph import StateGraph, END
from app.graph.state import GraphState
from app.graph.nodes.intent_classifier import intent_classifier
from app.graph.nodes.retrieval_node import retrieval_node
from app.graph.nodes.generation_node import generation_node
from app.graph.nodes.validation_node import validation_node
from app.graph.nodes.retry_node import retry_node
from app.graph.nodes.escalation_node import escalation_node
from app.graph.nodes.infra_alert_node import infra_alert_node
from app.graph.nodes.response_formatter import response_formatter

from app.graph.router import route_intent, route_validation

def build_workflow():
    """
    Builds and compiles the LangGraph workflow for the AI support agent.
    """
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("intent_classifier", intent_classifier)
    workflow.add_node("retrieval_node", retrieval_node)
    workflow.add_node("generation_node", generation_node)
    workflow.add_node("validation_node", validation_node)
    workflow.add_node("retry_node", retry_node)
    workflow.add_node("escalation_node", escalation_node)
    workflow.add_node("infra_alert_node", infra_alert_node)
    workflow.add_node("response_formatter", response_formatter)
    
    # Entry Point
    workflow.set_entry_point("intent_classifier")
    
    # Routing from intent classifier
    workflow.add_conditional_edges(
        "intent_classifier",
        route_intent,
        {
            "retrieval_node": "retrieval_node",
            "escalation_node": "escalation_node",
            "generation_node": "generation_node",
            "infra_alert_node": "infra_alert_node"
        }
    )
    
    # Linear and loop edges
    workflow.add_edge("retrieval_node", "generation_node")
    workflow.add_edge("generation_node", "validation_node")
    workflow.add_edge("infra_alert_node", "validation_node")

    
    workflow.add_conditional_edges(
        "validation_node",
        route_validation,
        {
            "retry_node": "retry_node",
            "response_formatter": "response_formatter"
        }
    )
    
    workflow.add_edge("response_formatter", END)
    
    workflow.add_edge("retry_node", "generation_node")
    workflow.add_edge("escalation_node", END)
    
    return workflow.compile()

app = build_workflow()
