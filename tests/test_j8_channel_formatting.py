import pytest
from app.graph.state import GraphState, Channel
from app.graph.nodes.response_formatter import response_formatter

def test_j8_rocketchat_formatting():
    state = GraphState(
        conversation_id="c1",
        session_id="s1",
        channel=Channel.ROCKETCHAT,
        user_message="Hello",
        language="en",
        final_response="This is a test response."
    )
    
    result = response_formatter(state)
    formatted = result["formatted_response"]
    
    assert formatted["text"] == "This is a test response."
    assert formatted["parseUrls"] is True
    assert "alias" in formatted

def test_j8_email_formatting():
    state = GraphState(
        conversation_id="c2",
        session_id="s2",
        channel=Channel.EMAIL,
        user_message="Hello",
        language="en",
        final_response="This is a test response."
    )
    
    result = response_formatter(state)
    formatted = result["formatted_response"]
    
    assert formatted["body"] == "This is a test response."
    assert "subject" in formatted
    assert formatted["is_html"] is False

def test_j8_webhook_formatting():
    state = GraphState(
        conversation_id="c3",
        session_id="s3",
        channel=Channel.WEBHOOK,
        user_message="Hello",
        language="en",
        final_response="This is a test response."
    )
    
    result = response_formatter(state)
    formatted = result["formatted_response"]
    
    assert formatted["response"] == "This is a test response."
    assert formatted["status"] == "success"
