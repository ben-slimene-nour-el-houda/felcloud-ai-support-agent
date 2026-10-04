"""
Tests for J7: Ticketing & Escalation
  - Unit: Pydantic schema validation for create_ticket
  - Unit: create_ticket idempotency logic
  - Integration: escalation_node generating response with ticket ID
  - Security: verify TICKETING_API_KEY handling
"""
import uuid
from unittest.mock import patch, MagicMock

import pytest
from pydantic import ValidationError

from app.tools.ticketing import create_ticket, CreateTicketSchema, _created_tickets
from app.graph.state import GraphState, Channel, Language
from app.graph.nodes.escalation_node import escalation_node


# ==========================================================================
# Helpers
# ==========================================================================

@pytest.fixture(autouse=True)
def clear_ticket_cache():
    """Clear the in-memory idempotency cache before each test."""
    _created_tickets.clear()
    yield
    _created_tickets.clear()

def _make_state(message="I want to talk to a human", **kw):
    defaults = dict(
        conversation_id="c-123", session_id="s-123",
        channel=Channel.WEBHOOK, user_message=message, language=Language.FR,
    )
    defaults.update(kw)
    return GraphState(**defaults)


# ==========================================================================
# Unit: Pydantic schema for create_ticket
# ==========================================================================

class TestCreateTicketSchema:

    def test_valid_args(self):
        s = CreateTicketSchema(
            user_id="u1",
            issue_description="help me",
            priority="high",
            idempotency_key="key-1"
        )
        assert s.priority == "high"
        assert s.idempotency_key == "key-1"

    def test_missing_required_args(self):
        with pytest.raises(ValidationError):
            CreateTicketSchema(user_id="u1")  # missing issue_description and idempotency_key

    def test_default_priority(self):
        s = CreateTicketSchema(
            user_id="u1",
            issue_description="help me",
            idempotency_key="key-1"
        )
        assert s.priority == "medium"


# ==========================================================================
# Unit: create_ticket tool and idempotency
# ==========================================================================

class TestCreateTicketTool:

    @patch("app.tools.ticketing.settings")
    def test_fallback_creates_ticket_successfully(self, mock_settings):
        """Test fallback implementation when TICKETING_API_URL is None."""
        mock_settings.TICKETING_API_URL = None
        key = str(uuid.uuid4())
        
        result = create_ticket.invoke({
            "user_id": "u1",
            "issue_description": "App crash",
            "priority": "high",
            "idempotency_key": key
        })
        
        assert result["status"] == "success"
        assert "TKT-" in result["ticket_id"]
        assert key in _created_tickets

    @patch("app.tools.ticketing.settings")
    def test_idempotency_prevents_duplicates(self, mock_settings):
        """Test that submitting the same idempotency_key twice skips the second time."""
        mock_settings.TICKETING_API_URL = None
        key = "idem-duplicate-123"
        
        # First call
        res1 = create_ticket.invoke({
            "user_id": "u1",
            "issue_description": "Network issue",
            "idempotency_key": key
        })
        assert res1["status"] == "success"
        
        # Second call with same key
        res2 = create_ticket.invoke({
            "user_id": "u1",
            "issue_description": "Network issue again",
            "idempotency_key": key
        })
        assert res2["status"] == "skipped"
        assert res2["ticket_id"] == res1["ticket_id"]

    @patch("app.tools.ticketing.requests.post")
    @patch("app.tools.ticketing.settings")
    def test_api_success_with_credentials(self, mock_settings, mock_post):
        """Test API integration sends auth headers when key is configured."""
        mock_settings.TICKETING_API_URL = "http://ticket-api"
        mock_settings.TICKETING_API_KEY = "secret_token"
        
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ticket_id": "API-123", "message": "Done"}
        mock_post.return_value = mock_resp
        
        result = create_ticket.invoke({
            "user_id": "u1",
            "issue_description": "API test",
            "idempotency_key": "api-key-1"
        })
        
        assert result["status"] == "success"
        assert result["ticket_id"] == "API-123"
        
        # Verify headers included Bearer token
        mock_post.assert_called_once()
        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer secret_token"

    @patch("app.tools.ticketing.requests.post")
    @patch("app.tools.ticketing.settings")
    def test_api_failure_handled_gracefully(self, mock_settings, mock_post):
        mock_settings.TICKETING_API_URL = "http://ticket-api"
        mock_post.side_effect = Exception("API down")
        
        result = create_ticket.invoke({
            "user_id": "u1",
            "issue_description": "Fail test",
            "idempotency_key": "fail-key-1"
        })
        
        assert result["status"] == "error"
        assert "Failed to create ticket" in result["message"]


# ==========================================================================
# Integration: escalation_node
# ==========================================================================

class TestEscalationNode:

    @patch("app.graph.nodes.escalation_node.create_ticket")
    def test_escalation_node_success(self, mock_tool):
        """Node should populate state.final_response with the created ticket ID."""
        mock_tool.invoke.return_value = {"status": "success", "ticket_id": "TKT-TEST"}
        
        state = _make_state("I need a human")
        result = escalation_node(state)
        
        assert "TKT-TEST" in result.final_response
        assert "J'ai créé le ticket" in result.final_response

    @patch("app.graph.nodes.escalation_node.create_ticket")
    def test_escalation_node_duplicate(self, mock_tool):
        """Node should handle 'skipped' (duplicate) ticketing gracefully."""
        mock_tool.invoke.return_value = {"status": "skipped", "ticket_id": "TKT-TEST"}
        
        state = _make_state("I need a human")
        result = escalation_node(state)
        
        assert "TKT-TEST" in result.final_response
        assert "J'ai déjà signalé ce problème" in result.final_response

    def test_node_idempotency_key_generation(self):
        """Node should generate identical UUID5 keys for identical messages."""
        with patch("app.graph.nodes.escalation_node.create_ticket") as mock_tool:
            mock_tool.invoke.return_value = {"status": "success", "ticket_id": "TKT-IDEM"}
            
            # Send same message twice
            state1 = _make_state("Identical issue")
            escalation_node(state1)
            
            state2 = _make_state("Identical issue")
            escalation_node(state2)
            
            call1_args = mock_tool.invoke.call_args_list[0][0][0]
            call2_args = mock_tool.invoke.call_args_list[1][0][0]
            
            # They should have the same generated idempotency key
            assert call1_args["idempotency_key"] == call2_args["idempotency_key"]
