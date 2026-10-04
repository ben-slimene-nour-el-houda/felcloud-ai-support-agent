"""
Shared Pytest Fixtures and Helpers for Felcloud AI Support Agent tests.
"""
import sys
import hmac
import hashlib
from types import ModuleType
from unittest.mock import MagicMock

import pytest
import redis as redis_sync

# ==========================================================================
# 1. Global Module Mocks (Injected before any graph imports)
# ==========================================================================

_fake = ModuleType("langchain_openai")
_fake.ChatOpenAI = MagicMock()  # type: ignore[attr-defined]
_fake.OpenAIEmbeddings = MagicMock()  # type: ignore[attr-defined]
sys.modules.setdefault("langchain_openai", _fake)

from app.graph.state import GraphState, Channel, Language
from app.config import settings

# ==========================================================================
# 2. GraphState Factory
# ==========================================================================

@pytest.fixture
def make_state():
    """
    Returns a factory function to create a GraphState object with sensible defaults.
    """
    def _factory(**kwargs):
        defaults = dict(
            conversation_id="c-123",
            session_id="s-123",
            channel=Channel.WEBHOOK,
            user_message="Hello",
            language=Language.FR,
        )
        defaults.update(kwargs)
        return GraphState(**defaults)
    return _factory

# ==========================================================================
# 3. Webhook Security Helpers
# ==========================================================================

@pytest.fixture
def webhook_path():
    """Returns the dynamic webhook path from settings."""
    return settings.ZEROCLAW_WEBHOOK_PATH

@pytest.fixture
def sign_payload():
    """Returns a helper function to sign a payload with HMAC-SHA256."""
    def _sign(payload: bytes, secret: str) -> str:
        return hmac.new(
            secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
    return _sign

# ==========================================================================
# 4. Redis cleanup — avoid replay-protection false positives between tests
# ==========================================================================

@pytest.fixture(autouse=True)
def clear_webhook_replay_keys():
    """
    Runs before every test. Clears any 'webhook:seen_sig:*' keys left in Redis,
    so that tests reusing the same payload/signature don't get falsely
    rejected as replays by tests that ran earlier.
    """
    client = redis_sync.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        decode_responses=True,
    )
    try:
        for key in client.scan_iter("webhook:seen_sig:*"):
            client.delete(key)
    except Exception:
        pass  # Redis not reachable in some test contexts — don't block tests
    finally:
        client.close()
    yield
