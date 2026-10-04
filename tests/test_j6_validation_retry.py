"""
Tests for J6: Validation & retry logic
  - Unit: validation_node.py — valid output passes, invalid/malformed flagged
  - Unit: retry_node.py — retry increments count, respects max
  - Integration: full validation→retry→exhaustion routing through the graph router
"""
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

# Sys modules mock is now in conftest.py

from langgraph.graph import END

from app.graph.state import GraphState, Channel, Language
from app.graph.nodes.validation_node import validation_node
from app.graph.nodes.retry_node import retry_node
from app.graph.router import route_validation


# Helpers


# ==========================================================================
# Unit: validation_node.py
# ==========================================================================

class TestValidationNode:
    """validation_node inspects state.final_response and sets validation_status."""

    # -- Valid outputs that should PASS --

    def test_good_response_passes(self, make_state):
        state = make_state(final_response="Votre serveur est en cours de redémarrage. Veuillez patienter 2 minutes.")
        result = validation_node(state)
        assert result.validation_status == "passed"

    def test_long_response_passes(self, make_state):
        state = make_state(final_response="A" * 200)
        result = validation_node(state)
        assert result.validation_status == "passed"

    # -- Invalid / malformed outputs that should FAIL --

    def test_none_response_fails(self, make_state):
        state = make_state(final_response=None)
        result = validation_node(state)
        assert result.validation_status == "failed"

    def test_empty_response_fails(self, make_state):
        state = make_state(final_response="")
        result = validation_node(state)
        assert result.validation_status == "failed"

    def test_whitespace_only_fails(self, make_state):
        state = make_state(final_response="    ")
        result = validation_node(state)
        assert result.validation_status == "failed"

    def test_too_short_response_fails(self, make_state):
        """Responses shorter than 10 characters are considered empty/garbage."""
        state = make_state(final_response="OK")
        result = validation_node(state)
        assert result.validation_status == "failed"

    # -- Error-phrase detection --

    @pytest.mark.parametrize("phrase", [
        "Désolé, je n'ai pas pu générer une réponse.",
        "I don't know the answer to that.",
        "I cannot answer this question.",
        "As an AI, I cannot do that.",
        "I am an AI language model and I cannot help.",
    ])
    def test_error_phrases_flagged(self, phrase, make_state):
        state = make_state(final_response=phrase)
        result = validation_node(state)
        assert result.validation_status == "failed"

    def test_valid_response_with_partial_match_passes(self, make_state):
        """A response that *contains* a keyword but is genuinely useful should pass."""
        # "as an ai" is a blocked phrase, but "as an aide" is not
        state = make_state(final_response="En tant qu'aide technique, voici la solution à suivre pour résoudre ce problème.")
        result = validation_node(state)
        assert result.validation_status == "passed"


# ==========================================================================
# Unit: retry_node.py
# ==========================================================================

class TestRetryNode:

    def test_increments_retry_count(self, make_state):
        state = make_state(retry_count=0)
        result = retry_node(state)
        assert result.retry_count == 1

    def test_second_retry(self, make_state):
        state = make_state(retry_count=1)
        result = retry_node(state)
        assert result.retry_count == 2

    def test_preserves_other_state(self, make_state):
        state = make_state(
            retry_count=0,
            final_response="old response",
            validation_status="failed",
        )
        result = retry_node(state)
        assert result.retry_count == 1
        assert result.final_response == "old response"
        assert result.validation_status == "failed"


# ==========================================================================
# Integration: route_validation — happy path, retry, and exhausted retries
# ==========================================================================

class TestValidationRetryRouting:
    """End-to-end routing logic: validation_node → route_validation → retry or END."""

    # -- Happy path: validation passes immediately --

    def test_valid_response_routes_to_end(self, make_state):
        state = make_state(final_response="Here is your answer in detail with sources.", retry_count=0)
        state = validation_node(state)
        assert state.validation_status == "passed"

        next_node = route_validation(state)
        assert next_node == END

    # -- Retry path: validation fails, retries remain --

    def test_failed_validation_retry_0_routes_to_retry(self, make_state):
        state = make_state(final_response="", retry_count=0)
        state = validation_node(state)
        assert state.validation_status == "failed"

        next_node = route_validation(state)
        assert next_node == "retry_node"

    def test_failed_validation_retry_1_routes_to_retry(self, make_state):
        state = make_state(final_response="I don't know", retry_count=1)
        state = validation_node(state)
        assert state.validation_status == "failed"

        next_node = route_validation(state)
        assert next_node == "retry_node"

    # -- Exhausted retries: fails at MAX_RETRIES → END --

    def test_exhausted_retries_routes_to_end(self, make_state):
        """MAX_RETRIES is 2. At retry_count=2, even a failed validation goes to END."""
        state = make_state(final_response="", retry_count=2)
        state = validation_node(state)
        assert state.validation_status == "failed"

        next_node = route_validation(state)
        assert next_node == END

    def test_exhausted_retries_at_3_routes_to_end(self, make_state):
        """retry_count beyond MAX goes to END."""
        state = make_state(final_response="I cannot answer", retry_count=5)
        state = validation_node(state)
        next_node = route_validation(state)
        assert next_node == END

    # -- Full loop simulation: fail → retry → fail → retry → fail → END --

    def test_full_retry_loop_simulation(self, make_state):
        """Simulate the complete validation/retry loop until exhaustion."""
        state = make_state(final_response="", retry_count=0)

        loop_count = 0
        while True:
            state = validation_node(state)
            next_node = route_validation(state)

            if next_node == END:
                break

            assert next_node == "retry_node"
            state = retry_node(state)
            loop_count += 1

            # Safety: avoid infinite loop in test
            assert loop_count <= 10, "Infinite retry loop detected!"

        # With MAX_RETRIES=2, we should have looped exactly 2 times
        assert loop_count == 2
        assert state.retry_count == 2
        assert state.validation_status == "failed"
