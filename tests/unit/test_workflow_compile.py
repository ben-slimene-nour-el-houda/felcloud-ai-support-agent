"""
J1 — Unit Test: LangGraph Workflow Compilation & Execution
==========================================================
Goal: Verify the LangGraph foundation works independently.

✅ GraphState is created correctly
✅ LangGraph workflow compiles without errors
✅ All nodes execute in the expected order
✅ final_response is produced at the end of the graph

All LLM and RAG calls are mocked so this test runs
with zero external dependencies.
"""

import pytest
from unittest.mock import patch, MagicMock


# ── helpers ──────────────────────────────────────────────────────────────────

def _fake_structured_llm(intent: str):
    """Return a mock structured-output chain whose .invoke() returns the given intent."""
    result = MagicMock()
    result.intent = intent

    chain = MagicMock()
    chain.invoke.return_value = result
    return chain


def _fake_llm_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = text
    return msg


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def general_chat_state():
    return {
        "conversation_id": "test-conv-001",
        "session_id": "test-sess-001",
        "channel": "webhook",
        "user_message": "Hello, how are you?",
        "language": "en",
        "chat_history": [],
    }


@pytest.fixture
def support_state():
    return {
        "conversation_id": "test-conv-002",
        "session_id": "test-sess-002",
        "channel": "rocketchat",
        "user_message": "How do I configure the Felcloud firewall?",
        "language": "fr",
        "chat_history": [],
    }


@pytest.fixture
def escalation_state():
    return {
        "conversation_id": "test-conv-003",
        "session_id": "test-sess-003",
        "channel": "webhook",
        "user_message": "I want to speak to a human agent NOW!",
        "language": "en",
        "chat_history": [],
    }


# ── Test 1: GraphState schema validation ─────────────────────────────────────

class TestGraphStateCreation:
    """Verify GraphState (Pydantic model) is created and validated correctly."""

    def test_valid_minimal_state(self, general_chat_state):
        from app.graph.state import GraphState, Channel, Language

        state = GraphState(**general_chat_state)
        assert state.conversation_id == "test-conv-001"
        assert state.session_id == "test-sess-001"
        assert state.channel == Channel.WEBHOOK
        assert state.user_message == "Hello, how are you?"
        assert state.language == Language.EN
        assert state.chat_history == []
        # Optional fields should default to None / False / 0
        assert state.intent is None
        assert state.retrieved_context is None
        assert state.sources is None
        assert state.final_response is None
        assert state.escalation_flag is False
        assert state.retry_count == 0

    def test_invalid_channel_raises_validation_error(self):
        from pydantic import ValidationError
        from app.graph.state import GraphState

        with pytest.raises(ValidationError):
            GraphState(
                conversation_id="x",
                session_id="x",
                channel="fax_machine",  # not in Channel enum
                user_message="hi",
                language="en",
            )

    def test_invalid_language_raises_validation_error(self):
        from pydantic import ValidationError
        from app.graph.state import GraphState

        with pytest.raises(ValidationError):
            GraphState(
                conversation_id="x",
                session_id="x",
                channel="webhook",
                user_message="hi",
                language="klingon",  # not in Language enum
            )

    def test_all_channel_values_are_valid(self):
        from app.graph.state import GraphState, Channel

        for ch in Channel:
            state = GraphState(
                conversation_id="x",
                session_id="x",
                channel=ch.value,
                user_message="test",
                language="en",
            )
            assert state.channel == ch

    def test_all_language_values_are_valid(self):
        from app.graph.state import GraphState, Language

        for lang in Language:
            state = GraphState(
                conversation_id="x",
                session_id="x",
                channel="webhook",
                user_message="test",
                language=lang.value,
            )
            assert state.language == lang


# ── Test 2: Workflow compilation ──────────────────────────────────────────────

class TestWorkflowCompilation:
    """Verify the LangGraph workflow builds and compiles without errors."""

    def test_build_workflow_returns_compiled_graph(self):
        from app.graph.workflow import build_workflow
        from langgraph.graph.state import CompiledStateGraph

        graph = build_workflow()
        assert graph is not None
        assert isinstance(graph, CompiledStateGraph), (
            f"Expected CompiledStateGraph, got {type(graph)}"
        )

    def test_workflow_module_app_is_compiled(self):
        from app.graph import workflow
        from langgraph.graph.state import CompiledStateGraph

        assert isinstance(workflow.app, CompiledStateGraph)


# ── Test 3: General-chat path ────────────────────────────────────────────────

class TestGeneralChatPath:
    """
    Expected path: intent_classifier → generation_node → validation_node → END
    """

    def test_general_chat_produces_final_response(self, general_chat_state):
        # Pre-import target modules so patch() can resolve them
        import app.graph.nodes.intent_classifier as ic_mod
        import app.graph.nodes.generation_node as gen_mod

        with (
            patch.object(ic_mod, "ChatOpenAI") as mock_intent_cls,
            patch.object(gen_mod, "ChatOpenAI") as mock_gen_cls,
        ):
            intent_instance = MagicMock()
            intent_instance.with_structured_output.return_value = _fake_structured_llm("general_chat")
            mock_intent_cls.return_value = intent_instance

            gen_instance = MagicMock()
            gen_instance.invoke.return_value = _fake_llm_response("Hello! How can I help?")
            mock_gen_cls.return_value = gen_instance

            from app.graph.workflow import build_workflow

            graph = build_workflow()
            result = graph.invoke(general_chat_state)

        assert result is not None
        assert result.get("intent") == "general_chat"
        assert result.get("final_response") is not None
        assert len(result["final_response"]) > 0

    def test_general_chat_node_order(self, general_chat_state):
        """Verify intent_classifier fires before generation_node."""
        import app.graph.nodes.intent_classifier as ic_mod
        import app.graph.nodes.generation_node as gen_mod

        execution_order = []

        with (
            patch.object(ic_mod, "ChatOpenAI") as mock_intent_cls,
            patch.object(gen_mod, "ChatOpenAI") as mock_gen_cls,
        ):
            # Intent chain tracking
            intent_result = MagicMock()
            intent_result.intent = "general_chat"
            intent_chain = MagicMock()

            def intent_invoke(messages):
                execution_order.append("intent_classifier")
                return intent_result

            intent_chain.invoke.side_effect = intent_invoke
            intent_instance = MagicMock()
            intent_instance.with_structured_output.return_value = intent_chain
            mock_intent_cls.return_value = intent_instance

            # Generation tracking
            def gen_invoke(messages):
                execution_order.append("generation_node")
                return _fake_llm_response("Hello there! How can I help you today?")

            gen_instance = MagicMock()
            gen_instance.invoke.side_effect = gen_invoke
            mock_gen_cls.return_value = gen_instance

            from app.graph.workflow import build_workflow

            graph = build_workflow()
            graph.invoke(general_chat_state)

        assert execution_order == ["intent_classifier", "generation_node"], (
            f"Unexpected node order: {execution_order}"
        )


# ── Test 4: Support path ──────────────────────────────────────────────────────

class TestSupportPath:
    """
    Expected path: intent_classifier → retrieval_node → generation_node → validation_node → END
    """

    def test_support_produces_final_response(self, support_state):
        import app.graph.nodes.intent_classifier as ic_mod
        import app.graph.nodes.generation_node as gen_mod
        import app.graph.nodes.retrieval_node as ret_mod

        with (
            patch.object(ic_mod, "ChatOpenAI") as mock_intent_cls,
            patch.object(gen_mod, "ChatOpenAI") as mock_gen_cls,
            patch.object(ret_mod, "retrieve_context") as mock_retrieve,
        ):
            intent_instance = MagicMock()
            intent_instance.with_structured_output.return_value = _fake_structured_llm("support")
            mock_intent_cls.return_value = intent_instance

            mock_retrieve.return_value = (
                "Felcloud firewall is configured via the admin panel.",
                [{"metadata": {"doc_id": "fw-001", "title": "Firewall Setup Guide"}}],
            )

            gen_instance = MagicMock()
            gen_instance.invoke.return_value = _fake_llm_response(
                "Configure the firewall in the admin panel."
            )
            mock_gen_cls.return_value = gen_instance

            from app.graph.workflow import build_workflow

            graph = build_workflow()
            result = graph.invoke(support_state)

        assert result.get("intent") == "support"
        assert result.get("retrieved_context") is not None
        assert result.get("sources") is not None and len(result["sources"]) == 1
        assert result.get("final_response") is not None

    def test_support_node_order(self, support_state):
        """retrieval_node must execute before generation_node."""
        import app.graph.nodes.intent_classifier as ic_mod
        import app.graph.nodes.generation_node as gen_mod
        import app.graph.nodes.retrieval_node as ret_mod

        execution_order = []

        with (
            patch.object(ic_mod, "ChatOpenAI") as mock_intent_cls,
            patch.object(gen_mod, "ChatOpenAI") as mock_gen_cls,
            patch.object(ret_mod, "retrieve_context") as mock_retrieve,
        ):
            intent_result = MagicMock()
            intent_result.intent = "support"
            intent_chain = MagicMock()

            def intent_invoke(messages):
                execution_order.append("intent_classifier")
                return intent_result

            intent_chain.invoke.side_effect = intent_invoke
            intent_instance = MagicMock()
            intent_instance.with_structured_output.return_value = intent_chain
            mock_intent_cls.return_value = intent_instance

            def retrieve_side(*args, **kwargs):
                execution_order.append("retrieval_node")
                return ("ctx", [])

            mock_retrieve.side_effect = retrieve_side

            def gen_invoke(messages):
                execution_order.append("generation_node")
                return _fake_llm_response("Here is the answer to your question.")

            gen_instance = MagicMock()
            gen_instance.invoke.side_effect = gen_invoke
            mock_gen_cls.return_value = gen_instance

            from app.graph.workflow import build_workflow

            graph = build_workflow()
            graph.invoke(support_state)

        assert execution_order == [
            "intent_classifier",
            "retrieval_node",
            "generation_node",
        ], f"Unexpected node order: {execution_order}"


# ── Test 5: Escalation path ───────────────────────────────────────────────────

class TestEscalationPath:
    """
    Expected path: intent_classifier → escalation_node → END
    """

    def test_escalation_routes_correctly(self, escalation_state):
        import app.graph.nodes.intent_classifier as ic_mod

        with patch.object(ic_mod, "ChatOpenAI") as mock_intent_cls:
            intent_instance = MagicMock()
            intent_instance.with_structured_output.return_value = _fake_structured_llm("escalation")
            mock_intent_cls.return_value = intent_instance

            from app.graph.workflow import build_workflow

            graph = build_workflow()
            result = graph.invoke(escalation_state)

        # Graph must complete without raising
        assert result is not None
        assert result.get("intent") == "escalation"
        # escalation_node is still a stub: final_response stays None — that's expected
        # The key assertion: graph didn't crash and intent is correct


# ── Test 6: Retrieval failure resilience ──────────────────────────────────────

class TestResilienceOnRetrievalFailure:
    """The graph must survive a Qdrant/retrieval crash and still return a response."""

    def test_graph_survives_retrieval_failure(self, support_state):
        import app.graph.nodes.intent_classifier as ic_mod
        import app.graph.nodes.generation_node as gen_mod
        import app.graph.nodes.retrieval_node as ret_mod

        with (
            patch.object(ic_mod, "ChatOpenAI") as mock_intent_cls,
            patch.object(gen_mod, "ChatOpenAI") as mock_gen_cls,
            patch.object(ret_mod, "retrieve_context") as mock_retrieve,
        ):
            intent_instance = MagicMock()
            intent_instance.with_structured_output.return_value = _fake_structured_llm("support")
            mock_intent_cls.return_value = intent_instance

            # Simulated Qdrant outage
            mock_retrieve.side_effect = ConnectionError("Qdrant unreachable")

            gen_instance = MagicMock()
            gen_instance.invoke.return_value = _fake_llm_response(
                "I don't have enough context to answer right now."
            )
            mock_gen_cls.return_value = gen_instance

            from app.graph.workflow import build_workflow

            graph = build_workflow()
            result = graph.invoke(support_state)

        assert result is not None
        # retrieval_node fallback sets these on error
        assert result.get("retrieved_context") == "Erreur lors de la récupération du contexte."
        assert result.get("sources") == []
        # Graph still produced a final response via generation_node
        assert result.get("final_response") is not None
