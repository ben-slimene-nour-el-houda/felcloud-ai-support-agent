"""
Tests for J5: Generation Node
"""
import sys
from types import ModuleType
from unittest.mock import patch, MagicMock

import pytest

# Mock langchain_openai
_fake = ModuleType("langchain_openai")
_fake.ChatOpenAI = MagicMock()  # type: ignore[attr-defined]
sys.modules.setdefault("langchain_openai", _fake)

from app.graph.state import GraphState, Channel, Language
from app.graph.nodes.generation_node import generation_node

def _make_state(intent="support", user_message="Hello", **kw):
    defaults = dict(
        conversation_id="c-123", session_id="s-123",
        channel=Channel.WEBHOOK, user_message=user_message, language=Language.FR,
        intent=intent
    )
    defaults.update(kw)
    return GraphState(**defaults)

class TestGenerationNode:

    @patch("app.graph.nodes.generation_node.ChatOpenAI")
    def test_generation_node_general_chat(self, mock_chat):
        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "Hello there!"
        mock_llm.invoke.return_value = mock_resp
        mock_chat.return_value = mock_llm
        
        state = _make_state(intent="general_chat")
        result = generation_node(state)
        
        assert result.final_response == "Hello there!"
        # Check that context wasn't explicitly injected in prompt
        call_args = mock_llm.invoke.call_args[0][0]
        assert "Respond politely" in call_args[0].content

    @patch("app.graph.nodes.generation_node.ChatOpenAI")
    def test_generation_node_support_with_sources(self, mock_chat):
        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "Here is how you do it."
        mock_llm.invoke.return_value = mock_resp
        mock_chat.return_value = mock_llm
        
        state = _make_state(
            intent="support", 
            retrieved_context="Doc content",
            sources=[{"metadata": {"title": "The Manual", "doc_id": "M1"}}]
        )
        result = generation_node(state)
        
        # Should append sources
        assert "Here is how you do it." in result.final_response
        assert "Sources utilisées" in result.final_response
        assert "The Manual" in result.final_response
        
        # Check prompt contains context
        call_args = mock_llm.invoke.call_args[0][0]
        assert "Doc content" in call_args[1].content

    @patch("app.graph.nodes.generation_node.ChatOpenAI")
    def test_generation_node_llm_failure(self, mock_chat):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM Down")
        mock_chat.return_value = mock_llm
        
        state = _make_state(intent="support")
        result = generation_node(state)
        
        assert "Désolé" in result.final_response
