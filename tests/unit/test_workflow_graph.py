import pytest
from unittest.mock import patch
from app.graph.workflow import build_workflow
from app.graph.state import GraphState, Channel, Language

def get_base_state() -> GraphState:
    return GraphState(
        conversation_id="123",
        session_id="123",
        channel=Channel.WEBHOOK,
        user_message="Hello",
        language=Language.EN,
        chat_history=[]
    )

def test_build_workflow_compiles():
    """build_workflow() compiles without errors and returns an executable graph."""
    app = build_workflow()
    assert app is not None

@patch('app.graph.workflow.generation_node')
@patch('app.graph.workflow.intent_classifier')
def test_full_graph_success_no_retries(mock_intent, mock_generation):
    """
    Full graph run: valid state with validation_status ending "passed" 
    reaches END with unchanged retry_count.
    """
    # Mock intent to route directly to generation_node
    def mock_intent_effect(state):
        state.intent = "general_chat"
        return state
    mock_intent.side_effect = mock_intent_effect

    # Mock generation to yield a valid response
    def mock_generation_effect(state):
        state.final_response = "This is a sufficiently long and valid response."
        return state
    mock_generation.side_effect = mock_generation_effect

    app = build_workflow()
    state = get_base_state()
    
    result = app.invoke(state.model_dump())
    
    assert result["validation_status"] == "passed"
    assert result["retry_count"] == 0
    assert mock_generation.call_count == 1

@patch('app.graph.workflow.generation_node')
@patch('app.graph.workflow.intent_classifier')
def test_full_graph_retry_then_success(mock_intent, mock_generation):
    """
    Full graph run: state that fails validation once then passes on retry 
    ends at END with retry_count == 1, and generation_node was invoked twice.
    """
    def mock_intent_effect(state):
        state.intent = "general_chat"
        return state
    mock_intent.side_effect = mock_intent_effect

    call_count = [0]
    def mock_generation_effect(state):
        if call_count[0] == 0:
            state.final_response = "short"  # Causes validation failure
        else:
            state.final_response = "This is a sufficiently long and valid response." # Causes validation success
        call_count[0] += 1
        return state
    
    mock_generation.side_effect = mock_generation_effect

    app = build_workflow()
    state = get_base_state()
    
    result = app.invoke(state.model_dump())
    
    assert result["validation_status"] == "passed"
    assert result["retry_count"] == 1
    assert mock_generation.call_count == 2

@patch('app.graph.workflow.generation_node')
@patch('app.graph.workflow.intent_classifier')
def test_full_graph_max_retries_loop_break(mock_intent, mock_generation):
    """
    Full graph run: state that keeps failing validation stops looping at 
    MAX_RETRIES and reaches END, guarding against infinite loops.
    """
    def mock_intent_effect(state):
        state.intent = "general_chat"
        return state
    mock_intent.side_effect = mock_intent_effect

    # Always yield an invalid response
    def mock_generation_effect(state):
        state.final_response = "short"
        return state
    mock_generation.side_effect = mock_generation_effect

    app = build_workflow()
    state = get_base_state()
    
    result = app.invoke(state.model_dump())
    
    # MAX_RETRIES is 2. The generation_node is called:
    # 1st time (retry_count=0) -> validation fails -> retry_node increments to 1 -> routes to generation_node
    # 2nd time (retry_count=1) -> validation fails -> retry_node increments to 2 -> routes to generation_node
    # 3rd time (retry_count=2) -> validation fails -> route_validation sees retry_count=2 -> routes to END
    assert result["validation_status"] == "failed"
    assert result["retry_count"] == 2
    assert mock_generation.call_count == 3
