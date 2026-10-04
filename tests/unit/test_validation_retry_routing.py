import pytest
from app.graph.state import GraphState, Channel, Language
from app.graph.router import route_validation
from app.graph.nodes.validation_node import validation_node
from app.graph.nodes.retry_node import retry_node
from langgraph.graph import END

def get_base_state() -> GraphState:
    """Helper to generate a base valid state for testing."""
    return GraphState(
        conversation_id="123",
        session_id="123",
        channel=Channel.WEBHOOK,
        user_message="Hello",
        language=Language.EN,
        chat_history=[]
    )

def test_route_validation_retry_when_failed_and_under_limit():
    """route_validation returns 'retry_node' when validation_status='failed' and retry_count < MAX_RETRIES"""
    state = get_base_state()
    state.validation_status = "failed"
    state.retry_count = 0
    assert route_validation(state) == "retry_node"

def test_route_validation_end_when_failed_and_over_limit():
    """route_validation returns END when validation_status='failed' and retry_count >= MAX_RETRIES"""
    state = get_base_state()
    state.validation_status = "failed"
    # MAX_RETRIES is 2 in router.py
    state.retry_count = 2
    assert route_validation(state) == "response_formatter"
    
    state.retry_count = 3
    assert route_validation(state) == "response_formatter"

def test_route_validation_end_when_passed():
    """route_validation returns END when validation_status='passed'"""
    state = get_base_state()
    state.validation_status = "passed"
    state.retry_count = 0
    assert route_validation(state) == "response_formatter"
    
    state.retry_count = 1
    assert route_validation(state) == "response_formatter"

def test_validation_node_fails_empty_response():
    """validation_node sets validation_status='failed' on empty/None final_response"""
    state = get_base_state()
    state.final_response = None
    result = validation_node(state)
    assert result.validation_status == "failed"

    state.final_response = ""
    result = validation_node(state)
    assert result.validation_status == "failed"

    state.final_response = "   "
    result = validation_node(state)
    assert result.validation_status == "failed"

def test_validation_node_fails_short_response():
    """validation_node sets validation_status='failed' on suspiciously short response"""
    state = get_base_state()
    state.final_response = "short"
    result = validation_node(state)
    assert result.validation_status == "failed"

def test_validation_node_passes_valid_response():
    """validation_node sets validation_status='passed' on a normal, non-empty response"""
    state = get_base_state()
    state.final_response = "This is a sufficiently long and valid response from the model."
    result = validation_node(state)
    assert result.validation_status == "passed"
    
def test_validation_node_fails_error_phrases():
    """validation_node sets validation_status='failed' on AI fallback phrases"""
    state = get_base_state()
    state.final_response = "I am an AI language model and cannot assist with this."
    result = validation_node(state)
    assert result.validation_status == "failed"

def test_retry_node_increments_count():
    """retry_node increments retry_count by exactly 1 and preserves other state fields"""
    state = get_base_state()
    state.retry_count = 1
    state.final_response = "Something"
    
    result = retry_node(state)
    
    assert result.retry_count == 2
    assert result.final_response == "Something"
    assert result.session_id == "123"
