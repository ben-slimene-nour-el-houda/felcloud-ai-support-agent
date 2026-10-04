"""
Tests for J1: LangGraph state & workflow
  - Unit: GraphState Pydantic validation
  - Unit: router.py conditional edges
  - Integration: graph compiles
  - Integration: ZeroClaw → LangGraph handoff via server.py
"""
import sys
import json
import hmac
import hashlib
from types import ModuleType
from unittest.mock import patch, MagicMock

import pytest

# Sys modules mock and helpers are now in conftest.py

from app.graph.state import GraphState, Channel, Language
from app.graph.router import route_intent, route_validation


# ==========================================================================
# Unit: GraphState validates correctly with Pydantic
# ==========================================================================

class TestGraphStateValidation:
    """Reject malformed input, missing required fields, invalid enums."""

    def test_valid_state(self):
        state = GraphState(
            conversation_id="conv-123",
            session_id="sess-123",
            channel=Channel.WEBHOOK,
            user_message="Hello",
            language=Language.FR,
        )
        assert state.conversation_id == "conv-123"
        assert state.retry_count == 0
        assert state.escalation_flag is False
        assert state.final_response is None

    def test_missing_required_fields(self):
        with pytest.raises(ValueError):
            GraphState(conversation_id="conv-123")  # missing session_id, channel, etc.

    def test_invalid_channel_enum(self):
        with pytest.raises(ValueError):
            GraphState(
                conversation_id="conv-123",
                session_id="sess-123",
                channel="invalid_channel",
                user_message="Hello",
                language=Language.FR,
            )

    def test_invalid_language_enum(self):
        with pytest.raises(ValueError):
            GraphState(
                conversation_id="conv-123",
                session_id="sess-123",
                channel=Channel.WEBHOOK,
                user_message="Hello",
                language="klingon",
            )

    def test_defaults_are_set(self):
        state = GraphState(
            conversation_id="c", session_id="s",
            channel=Channel.EMAIL, user_message="hi", language=Language.EN,
        )
        assert state.chat_history == []
        assert state.intent is None
        assert state.retry_count == 0


# ==========================================================================
# Unit: router.py — conditional edges route to the correct next node
# ==========================================================================

class TestRouterConditionalEdges:

    # -- route_intent --
    def test_route_intent_escalation(self, make_state):
        state = make_state(intent="escalation")
        assert route_intent(state) == "escalation_node"

    def test_route_intent_general_chat(self, make_state):
        state = make_state(intent="general_chat")
        assert route_intent(state) == "generation_node"

    def test_route_intent_support_default(self, make_state):
        state = make_state(intent="support")
        assert route_intent(state) == "retrieval_node"

    def test_route_intent_unknown_falls_to_retrieval(self, make_state):
        state = make_state(intent="something_unknown")
        assert route_intent(state) == "retrieval_node"

    # -- route_validation --
    def test_route_validation_passed(self, make_state):
        """Validation passed → END."""
        state = make_state(validation_status="passed", retry_count=0)
        from langgraph.graph import END
        result = route_validation(state)
        assert result == END  # should go to END

    def test_route_validation_failed_under_max(self, make_state):
        """Validation failed with retries left → retry_node."""
        state = make_state(validation_status="failed", retry_count=1)
        assert route_validation(state) == "retry_node"

    def test_route_validation_failed_at_max(self, make_state):
        """Validation failed at max retries → END."""
        state = make_state(validation_status="failed", retry_count=2)
        from langgraph.graph import END
        result = route_validation(state)
        assert result == END


# ==========================================================================
# Integration: full graph compiles without errors
# ==========================================================================

def test_graph_compiles():
    """build_workflow() should return a compiled graph without raising."""
    from app.graph.workflow import build_workflow
    compiled = build_workflow()
    assert compiled is not None


# ==========================================================================
# Integration: ZeroClaw → LangGraph handoff via agent/server.py
# ==========================================================================

class TestZeroClawHandoff:
    """Mock the compiled LangGraph app and verify the webhook endpoint
    correctly initialises GraphState from an incoming ZeroClaw payload."""

    @patch("app.channels.webhooks.zeroclaw_webhook.langgraph_app")
    def test_valid_webhook_invokes_graph(self, mock_langgraph, webhook_path, sign_payload):
        from fastapi.testclient import TestClient
        from agent.server import app

        mock_langgraph.invoke.return_value = {"final_response": "Mocked reply"}

        payload = {
            "event_type": "message",
            "session_id": "sess-1",
            "payload": {
                "message": "I need help",
                "user_id": "user-42",
                "channel": "webhook",
                "language": "fr",
            },
        }
        body = json.dumps(payload).encode()
        from app.config import settings
        sig = sign_payload(body, settings.ZEROCLAW_WEBHOOK_SECRET)

        client = TestClient(app)
        resp = client.post(
            webhook_path,
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-ZeroClaw-Signature": sig,
            },
        )

        assert resp.status_code == 200
        mock_langgraph.invoke.assert_called_once()
        call_args = mock_langgraph.invoke.call_args[0][0]
        assert call_args["user_message"] == "I need help"
        assert call_args["session_id"] == "sess-1"

    @patch("app.channels.webhooks.zeroclaw_webhook.langgraph_app")
    def test_missing_signature_returns_401(self, mock_langgraph, webhook_path):
        from fastapi.testclient import TestClient
        from agent.server import app

        client = TestClient(app)
        resp = client.post(
            webhook_path,
            json={"event_type": "message", "session_id": "s", "payload": {
                "message": "hi", "user_id": "u", "channel": "webhook", "language": "fr"
            }},
        )
        assert resp.status_code == 401
