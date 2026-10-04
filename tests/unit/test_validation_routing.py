"""
J6 — Unit Test: Validation Routing
===================================
Goal: Verify that the route_validation router function correctly 
directs the flow based on validation status and retry count.
"""

import pytest
from app.graph.state import GraphState, Channel, Language
from app.graph.router import route_validation
from langgraph.graph import END

@pytest.fixture
def base_state():
    return GraphState(
        conversation_id="test-1",
        session_id="sess-1",
        channel=Channel.WEBHOOK,
        user_message="Test message",
        language=Language.EN
    )

def test_route_validation_failed_retry(base_state):
    """
    Builds a state with validation_status="failed", retry_count=0 
    → asserts graph routes to retry_node 
    """
    base_state.validation_status = "failed"
    base_state.retry_count = 0
    
    route = route_validation(base_state)
    assert route == "retry_node"

def test_route_validation_failed_max_retries(base_state):
    """
    Builds a state with validation_status="failed", retry_count=MAX_RETRIES 
    → asserts graph routes to END (no infinite loop)
    """
    base_state.validation_status = "failed"
    base_state.retry_count = 2  # MAX_RETRIES in router is 2
    
    route = route_validation(base_state)
    assert route == "response_formatter"

def test_route_validation_passed(base_state):
    """
    Builds a state with validation_status="passed" 
    → asserts graph routes to END
    """
    base_state.validation_status = "passed"
    base_state.retry_count = 0
    
    route = route_validation(base_state)
    assert route == "response_formatter"
