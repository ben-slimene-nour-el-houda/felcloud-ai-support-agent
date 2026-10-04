"""
Tests for J4: Intent classification & webhooks
  - Unit: intent_classifier.py — multilingual utterance→intent mapping
  - Unit: zeroclaw_webhook.py — HMAC signature validation
  - Security: replay attack / idempotency check
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
from app.graph.nodes.intent_classifier import (
    IntentClassification,
    intent_classifier,
)
from app.config import settings


# ==========================================================================
# Helpers
# ==========================================================================

# Helpers


def _make_webhook_payload(**overrides) -> dict:
    base = {
        "event_type": "message",
        "session_id": "sess-test",
        "payload": {
            "message": "Hello",
            "user_id": "usr-1",
            "channel": "webhook",
            "language": "fr",
        },
    }
    base.update(overrides)
    return base


# ==========================================================================
# Unit: IntentClassification Pydantic schema
# ==========================================================================

class TestIntentClassificationSchema:
    """The Pydantic schema only allows the three defined intent labels."""

    @pytest.mark.parametrize("label", ["support", "escalation", "general_chat"])
    def test_valid_intents(self, label):
        obj = IntentClassification(intent=label)
        assert obj.intent == label

    def test_invalid_intent_rejected(self):
        with pytest.raises(ValueError):
            IntentClassification(intent="billing")


# ==========================================================================
# Unit: intent_classifier node — multilingual fixture set
# ==========================================================================

# Each tuple: (utterance, language, expected_intent)
_INTENT_FIXTURES = [
    # --- English ---
    ("My server is not responding, can you help?", Language.EN, "support"),
    ("I want to talk to a real person NOW!", Language.EN, "escalation"),
    ("Thanks, that was helpful!", Language.EN, "general_chat"),
    ("How do I configure SSL on my VM?", Language.EN, "support"),
    # --- French ---
    ("Mon serveur ne répond plus, que faire ?", Language.FR, "support"),
    ("Je veux parler à un humain immédiatement", Language.FR, "escalation"),
    ("Bonjour, comment ça va ?", Language.FR, "general_chat"),
    ("Comment redémarrer mon instance ?", Language.FR, "support"),
    # --- Darija ---
    ("serveur te3i ma khdemch, 3awnouni", Language.DARIJA, "support"),
    ("bghit nhedder m3a wahd f support", Language.DARIJA, "escalation"),
    ("salam, labas ?", Language.DARIJA, "general_chat"),
]


class TestIntentClassifierNode:
    """Mock the LLM and verify the node correctly assigns state.intent."""

    @pytest.mark.parametrize("utterance,lang,expected", _INTENT_FIXTURES)
    @patch("app.graph.nodes.intent_classifier.ChatOpenAI")
    def test_intent_mapping(self, mock_chat_cls, utterance, lang, expected, make_state):
        # Wire up: ChatOpenAI().with_structured_output().invoke() → IntentClassification
        mock_llm_instance = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = IntentClassification(intent=expected)
        mock_llm_instance.with_structured_output.return_value = mock_structured
        mock_chat_cls.return_value = mock_llm_instance

        state = make_state(user_message=utterance, language=lang)
        result = intent_classifier(state)

        assert result.intent == expected
        mock_structured.invoke.assert_called_once()

    @patch("app.graph.nodes.intent_classifier.ChatOpenAI")
    def test_llm_failure_defaults_to_support(self, mock_chat_cls, make_state):
        """If the LLM call raises, the node should default to 'support'."""
        mock_llm_instance = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = RuntimeError("LLM down")
        mock_llm_instance.with_structured_output.return_value = mock_structured
        mock_chat_cls.return_value = mock_llm_instance

        state = make_state(user_message="anything")
        result = intent_classifier(state)

        assert result.intent == "support"


# ==========================================================================
# Unit: HMAC signature validation on the webhook endpoint
# ==========================================================================

class TestWebhookHMACValidation:
    """Verify verify_signature correctly accepts/rejects requests."""

    def _client(self):
        from fastapi.testclient import TestClient
        from agent.server import app
        return TestClient(app)

    @patch("app.channels.webhooks.zeroclaw_webhook.langgraph_app")
    def test_valid_signature_accepted(self, mock_lg, webhook_path, sign_payload):
        mock_lg.invoke.return_value = {"final_response": "OK"}
        payload = _make_webhook_payload()
        body = json.dumps(payload).encode()
        from app.config import settings
        sig = sign_payload(body, settings.ZEROCLAW_WEBHOOK_SECRET)

        resp = self._client().post(
            webhook_path, content=body,
            headers={"Content-Type": "application/json", "X-ZeroClaw-Signature": sig},
        )
        assert resp.status_code == 200

    @patch("app.channels.webhooks.zeroclaw_webhook.langgraph_app")
    def test_missing_signature_returns_401(self, mock_lg, webhook_path):
        payload = _make_webhook_payload()
        resp = self._client().post(webhook_path, json=payload)
        assert resp.status_code == 401

    @patch("app.channels.webhooks.zeroclaw_webhook.langgraph_app")
    def test_wrong_signature_returns_401(self, mock_lg, webhook_path):
        payload = _make_webhook_payload()
        body = json.dumps(payload).encode()
        resp = self._client().post(
            webhook_path, content=body,
            headers={
                "Content-Type": "application/json",
                "X-ZeroClaw-Signature": "deadbeef0000",
            },
        )
        assert resp.status_code == 401

    @patch("app.channels.webhooks.zeroclaw_webhook.langgraph_app")
    def test_tampered_payload_rejected(self, mock_lg, webhook_path, sign_payload):
        """Sign the original payload, then modify it — signature no longer matches."""
        original = _make_webhook_payload()
        body_original = json.dumps(original).encode()
        from app.config import settings
        sig = sign_payload(body_original, settings.ZEROCLAW_WEBHOOK_SECRET)

        # Tamper: change the message
        tampered = _make_webhook_payload()
        tampered["payload"]["message"] = "DROP TABLE users;"
        body_tampered = json.dumps(tampered).encode()

        resp = self._client().post(
            webhook_path, content=body_tampered,
            headers={"Content-Type": "application/json", "X-ZeroClaw-Signature": sig},
        )
        assert resp.status_code == 401

    @patch("app.channels.webhooks.zeroclaw_webhook.langgraph_app")
    def test_malformed_json_returns_400(self, mock_lg, webhook_path, sign_payload):
        """Non-JSON body should return 400 (after passing signature)."""
        body = b"this is not json"
        from app.config import settings
        sig = sign_payload(body, settings.ZEROCLAW_WEBHOOK_SECRET)
        resp = self._client().post(
            webhook_path, content=body,
            headers={"Content-Type": "application/json", "X-ZeroClaw-Signature": sig},
        )
        assert resp.status_code == 400

    @patch("app.channels.webhooks.zeroclaw_webhook.langgraph_app")
    def test_invalid_payload_schema_returns_400(self, mock_lg, webhook_path, sign_payload):
        """Valid JSON but missing required fields → 400."""
        body = json.dumps({"event_type": "message"}).encode()  # missing session_id, payload
        from app.config import settings
        sig = sign_payload(body, settings.ZEROCLAW_WEBHOOK_SECRET)
        resp = self._client().post(
            webhook_path, content=body,
            headers={"Content-Type": "application/json", "X-ZeroClaw-Signature": sig},
        )
        assert resp.status_code == 400


# ==========================================================================
# Security: replay attack
# ==========================================================================

class TestReplayAttack:
    """
    The webhook now has replay protection: a signature is remembered in
    Redis for REPLAY_WINDOW_SECONDS. Resending the same signed payload
    within that window is rejected with 409 Conflict.
    """

    @patch("app.channels.webhooks.zeroclaw_webhook.langgraph_app")
    def test_replay_same_payload_rejected(self, mock_lg, webhook_path, sign_payload):
        """Same payload + signature resent twice → second one rejected (replay guard)."""
        from fastapi.testclient import TestClient
        from agent.server import app

        mock_lg.invoke.return_value = {"final_response": "OK"}
        payload = _make_webhook_payload()
        body = json.dumps(payload).encode()
        from app.config import settings
        sig = sign_payload(body, settings.ZEROCLAW_WEBHOOK_SECRET)
        headers = {"Content-Type": "application/json", "X-ZeroClaw-Signature": sig}

        client = TestClient(app)
        resp1 = client.post(webhook_path, content=body, headers=headers)
        resp2 = client.post(webhook_path, content=body, headers=headers)

        # First request succeeds, replayed one is rejected.
        assert resp1.status_code == 200
        assert resp2.status_code == 409
