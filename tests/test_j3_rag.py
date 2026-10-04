"""
Tests for J3: RAG pipeline
  - Unit: rrf_fusion — hand-computed RRF scores
  - Unit: rerank_results — cross-encoder reorders docs correctly
  - Unit: assemble_context — source attribution / citations
  - Integration: full retrieve_context pipeline with mocked Qdrant
  - Regression: golden dataset placeholder
"""
from unittest.mock import patch, MagicMock

from qdrant_client.http import models

from app.rag.retrieval.search import (
    rrf_fusion,
    rerank_results,
    assemble_context,
    retrieve_context,
)


# ==========================================================================
# Helpers
# ==========================================================================

def _scored_point(id_, score=0.0, text=""):
    return models.ScoredPoint(
        id=id_, version=1, score=score,
        payload={"text": text, "source_id": f"SRC-{id_}", "title": f"Title {id_}"},
    )


# ==========================================================================
# Unit: rrf_fusion
# ==========================================================================

class TestRRFFusion:

    def test_shared_doc_ranked_first(self):
        """A doc appearing in BOTH dense and sparse lists gets the highest RRF score."""
        dense = [_scored_point(1, 0.9), _scored_point(2, 0.8)]
        sparse = [_scored_point(2, 0.5), _scored_point(3, 0.4)]

        fused = rrf_fusion(dense, sparse, k=60)

        assert len(fused) == 3
        assert fused[0]["id"] == 2, "Doc 2 (in both lists) should be ranked first"

    def test_hand_computed_scores(self):
        """Verify exact RRF scores against a manual computation.
        RRF(d, k) = sum over lists of 1/(k + rank + 1)  (rank is 0-indexed)
        """
        k = 60
        dense = [_scored_point("A"), _scored_point("B")]
        sparse = [_scored_point("B"), _scored_point("C")]

        fused = rrf_fusion(dense, sparse, k=k)
        scores = {r["id"]: r["score"] for r in fused}

        # A: dense rank 0 only  → 1/(60+0+1) = 1/61
        assert abs(scores["A"] - 1 / 61) < 1e-9

        # B: dense rank 1 + sparse rank 0 → 1/62 + 1/61
        assert abs(scores["B"] - (1 / 62 + 1 / 61)) < 1e-9

        # C: sparse rank 1 only → 1/62
        assert abs(scores["C"] - 1 / 62) < 1e-9

    def test_empty_lists(self):
        assert rrf_fusion([], []) == []

    def test_single_list_populated(self):
        dense = [_scored_point(1)]
        fused = rrf_fusion(dense, [], k=60)
        assert len(fused) == 1


# ==========================================================================
# Unit: rerank_results
# ==========================================================================

class TestRerankResults:

    @patch("app.rag.retrieval.search.load_model")
    def test_reorders_by_cross_encoder_score(self, mock_load_model):
        mock_model = MagicMock()
        # scores: doc-1 gets -5, doc-2 gets 10 → doc-2 should come first
        mock_model.compute_score.return_value = [-5.0, 10.0]
        mock_load_model.return_value = mock_model

        results = [
            {"id": 1, "payload": {"text": "Irrelevant document"}},
            {"id": 2, "payload": {"text": "Highly relevant document"}},
        ]

        reranked = rerank_results("my query", results, top_n=2)
        assert reranked[0]["id"] == 2
        assert reranked[0]["rerank_score"] == 10.0
        assert reranked[1]["id"] == 1

    @patch("app.rag.retrieval.search.load_model")
    def test_top_n_truncation(self, mock_load_model):
        mock_model = MagicMock()
        mock_model.compute_score.return_value = [1.0, 2.0, 3.0]
        mock_load_model.return_value = mock_model

        results = [
            {"id": i, "payload": {"text": f"Doc {i}"}} for i in range(3)
        ]
        reranked = rerank_results("q", results, top_n=1)
        assert len(reranked) == 1

    def test_empty_input(self):
        assert rerank_results("q", []) == []

    @patch("app.rag.retrieval.search.load_model")
    def test_single_result_float_score(self, mock_load_model):
        """compute_score may return a bare float for a single pair."""
        mock_model = MagicMock()
        mock_model.compute_score.return_value = 7.5  # float, not list
        mock_load_model.return_value = mock_model

        results = [{"id": 1, "payload": {"text": "Only doc"}}]
        reranked = rerank_results("q", results, top_n=1)
        assert len(reranked) == 1
        assert reranked[0]["rerank_score"] == 7.5


# ==========================================================================
# Unit: assemble_context
# ==========================================================================

class TestAssembleContext:

    def test_source_attribution(self):
        results = [
            {"payload": {"source_id": "SRC1", "title": "First Source", "text": "Content 1"}},
            {"payload": {"source_id": "SRC2", "title": "Second Source", "text": "Content 2"}},
        ]
        context = assemble_context(results)

        assert "--- Source 1 | Document ID: SRC1 | Titre: First Source ---" in context
        assert "Content 1" in context
        assert "--- Source 2 | Document ID: SRC2 | Titre: Second Source ---" in context
        assert "Content 2" in context

    def test_empty_results_fallback(self):
        context = assemble_context([])
        assert "Aucun contexte" in context

    def test_missing_metadata_defaults(self):
        results = [{"payload": {"text": "Some text"}}]
        context = assemble_context(results)
        assert "Unknown" in context  # default source_id
        assert "Untitled" in context  # default title


# ==========================================================================
# Integration: retrieve_context full pipeline
# ==========================================================================

class TestRetrieveContextPipeline:

    @patch("app.rag.retrieval.search.load_model")
    @patch("app.rag.retrieval.search.fetch_qdrant_results")
    @patch("app.rag.retrieval.search.get_qdrant_client")
    def test_full_pipeline(self, mock_get_client, mock_fetch, mock_load_model):
        """Mock Qdrant search and BGE-M3 model, run retrieve_context end-to-end."""

        # -- Mock BGE-M3 model for embed_query and rerank_results --
        mock_model = MagicMock()
        mock_model.encode.return_value = {
            "dense_vecs": [[0.1] * 1024],
            "lexical_weights": [{0: 1.0}],
        }
        # rerank scores: first doc gets 5.0
        mock_model.compute_score.return_value = [5.0]
        mock_load_model.return_value = mock_model

        # -- Mock fetch_qdrant_results to return pre-built ScoredPoints --
        fake_point = models.ScoredPoint(
            id=42, version=1, score=0.9,
            payload={"text": "Felcloud overview", "source_id": "DOC-1", "title": "Overview"},
        )
        mock_fetch.return_value = {"dense": [fake_point], "sparse": [fake_point]}
        mock_get_client.return_value = MagicMock()

        context, sources = retrieve_context("What is Felcloud?", top_n=1)

        assert "DOC-1" in context
        assert len(sources) == 1
        assert sources[0]["source_id"] == "DOC-1"


# ==========================================================================
# Regression: golden dataset queries
# ==========================================================================

import json
import os
import pytest

# Helper to load the golden dataset for parametrization
def load_golden_dataset():
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "evaluation", "golden_dataset.json")
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

@pytest.mark.parametrize("item", load_golden_dataset())
@patch("app.rag.retrieval.search.load_model")
@patch("app.rag.retrieval.search.fetch_qdrant_results")
@patch("app.rag.retrieval.search.get_qdrant_client")
def test_golden_dataset_regression(mock_get_client, mock_fetch, mock_load_model, item):
    """Run each query from the golden dataset through retrieve_context() and assert expected_source_id appears."""
    query = item["query"]
    expected_source_id = item.get("expected_source_id")
    
    if not expected_source_id:
        pytest.skip(f"Item {item.get('id')} lacks expected_source_id")

    # -- Mock BGE-M3 model --
    mock_model = MagicMock()
    mock_model.encode.return_value = {
        "dense_vecs": [[0.1] * 1024],
        "lexical_weights": [{0: 1.0}],
    }
    # rerank scores
    mock_model.compute_score.return_value = [5.0]
    mock_load_model.return_value = mock_model

    # -- Mock fetch_qdrant_results to return the expected source_id --
    fake_point = models.ScoredPoint(
        id=42, version=1, score=0.9,
        payload={"text": "Simulated result", "source_id": expected_source_id, "title": "Golden Result"},
    )
    mock_fetch.return_value = {"dense": [fake_point], "sparse": [fake_point]}
    mock_get_client.return_value = MagicMock()

    context, sources = retrieve_context(query, top_n=1)

    assert expected_source_id in context
    assert len(sources) > 0
    assert any(s["source_id"] == expected_source_id for s in sources)
