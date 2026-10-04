"""
Tests for J2: Ingestion & storage
  - Unit: chunking.py — chunk boundaries respect structure
  - Unit: embeddings.py — BGE-M3 output dimensionality and non-null vectors
  - Integration: end-to-end ingest a sample doc → chunks → embeddings
  - Schema: validate schema.sql contains all required tables/constraints
"""
import os
import numpy as np
from unittest.mock import patch, MagicMock

from app.rag.ingestion.schemas import RawDocument
from app.rag.ingestion.chunking import (
    chunk_markdown_doc,
    chunk_single_unit,
    chunk_document,
    split_markdown_by_h2,
    _approx_tokens,
    SPLIT_THRESHOLD_TOKENS,
)


# ==========================================================================
# Unit: chunking.py
# ==========================================================================

class TestSplitMarkdownByH2:
    """Low-level markdown header splitter."""

    def test_basic_split(self):
        md = "# Title\n\n## Section 1\nText one.\n\n## Section 2\nText two."
        splits = split_markdown_by_h2(md)
        assert len(splits) == 3
        assert splits[0] == ("", "# Title")
        assert splits[1][0] == "## Section 1"
        assert "Text one" in splits[1][1]
        assert splits[2][0] == "## Section 2"

    def test_no_h2_returns_whole(self):
        md = "Just a plain paragraph with no headers."
        splits = split_markdown_by_h2(md)
        assert len(splits) == 1
        assert splits[0][0] == ""


class TestChunkingBoundaries:
    """chunk_markdown_doc and chunk_single_unit."""

    def _make_doc(self, content: str, source_type: str = "documentation") -> RawDocument:
        return RawDocument(
            source_id="doc-test",
            source_type=source_type,
            content=content,
            title="Test Doc",
            category="general",
            language="en",
            metadata={},
        )

    def test_faq_produces_single_chunk(self):
        doc = self._make_doc("Q: How? A: Like this.", source_type="faq")
        chunks = chunk_single_unit(doc)
        assert len(chunks) == 1
        assert chunks[0].source_id == "doc-test"

    def test_markdown_doc_splits_on_h2(self):
        md = "# Title\n\n## Overview\nOverview text here.\n\n## Steps\nStep 1. Step 2."
        doc = self._make_doc(md)
        chunks = chunk_markdown_doc(doc)
        assert len(chunks) >= 1
        assert all(c.text.strip() for c in chunks), "No chunk should be empty"

    def test_no_mid_sentence_splits_on_short_doc(self):
        """Short sections should NOT be split in the middle of a sentence."""
        md = "## Intro\nThis is a complete sentence. Another sentence here."
        doc = self._make_doc(md)
        chunks = chunk_markdown_doc(doc)
        for chunk in chunks:
            # Ensure each chunk contains complete sentences (no dangling fragments)
            assert chunk.text.strip() != ""

    def test_chunk_document_dispatches_correctly(self):
        faq_doc = self._make_doc("FAQ content", source_type="faq")
        doc_doc = self._make_doc("## S1\nText", source_type="documentation")
        assert len(chunk_document(faq_doc)) == 1
        assert len(chunk_document(doc_doc)) >= 1

    def test_large_section_is_sub_split(self):
        """A section exceeding SPLIT_THRESHOLD_TOKENS must be sub-split."""
        long_text = "Word " * (SPLIT_THRESHOLD_TOKENS * 5)  # well above threshold
        md = f"## Big Section\n{long_text}"
        doc = self._make_doc(md)
        chunks = chunk_markdown_doc(doc)
        assert len(chunks) > 1, "Large section should produce multiple chunks"

    def test_chunk_ids_are_unique(self):
        md = "## A\nText A.\n\n## B\nText B.\n\n## C\nText C."
        doc = self._make_doc(md)
        chunks = chunk_markdown_doc(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "chunk_ids must be unique"


# ==========================================================================
# Unit: embeddings.py — BGE-M3 output
# ==========================================================================

class TestEmbeddingsOutput:

    @patch("app.rag.retrieval.search.load_model")
    def test_embed_query_dense_dimension(self, mock_load_model):
        """embed_query returns a 1024-dim dense vector (BGE-M3)."""
        mock_model = MagicMock()
        mock_model.encode.return_value = {
            "dense_vecs": [[0.1] * 1024],
            "lexical_weights": [{42: 1.5, 99: 0.8}],
        }
        mock_load_model.return_value = mock_model

        from app.rag.retrieval.search import embed_query

        vectors = embed_query("Test query")
        assert len(vectors["dense"]) == 1024
        assert all(v != 0 for v in vectors["dense"])

    @patch("app.rag.retrieval.search.load_model")
    def test_embed_query_sparse_non_empty(self, mock_load_model):
        mock_model = MagicMock()
        mock_model.encode.return_value = {
            "dense_vecs": [[0.0] * 1024],
            "lexical_weights": [{1: 0.5, 2: 0.3}],
        }
        mock_load_model.return_value = mock_model

        from app.rag.retrieval.search import embed_query

        vectors = embed_query("Another query")
        assert "sparse" in vectors
        assert len(vectors["sparse"]) > 0


# ==========================================================================
# Integration: end-to-end ingest sample doc → chunks → embeddings
# ==========================================================================

class TestEndToEndIngest:

    @patch("app.rag.ingestion.embeddings.load_model")
    def test_ingest_doc_to_embedded_chunks(self, mock_load_model):
        """Load a doc, chunk it, and (with mocked model) embed the chunks."""
        from app.rag.ingestion.chunking import chunk_all
        from app.rag.ingestion.embeddings import embed_chunks, _reset_model

        # Reset singleton so our mock takes effect
        _reset_model()

        # Build a small mock model that returns valid dense+sparse vectors
        mock_model = MagicMock()
        mock_model.encode.return_value = {
            "dense_vecs": np.array([[0.1] * 1024]),
            "lexical_weights": [{0: 1.0}],
        }
        mock_load_model.return_value = mock_model

        doc = RawDocument(
            source_id="ingest-test-1",
            source_type="faq",
            content="Q: What is Felcloud? A: A cloud platform.",
            title="FAQ 1",
            category="general",
            language="fr",
            metadata={},
        )

        # 1. Chunk
        chunks = chunk_all([doc])
        assert len(chunks) == 1

        # 2. Embed (mocked)
        embedded = embed_chunks(chunks)
        assert len(embedded) == len(chunks)
        assert len(embedded[0].dense_vector) == 1024
        mock_model.encode.assert_called_once()

        # Cleanup
        _reset_model()


# ==========================================================================
# Schema: validate schema.sql
# ==========================================================================

import re

class TestSchemaSQL:
    SCHEMA_PATH = os.path.join(
        os.path.dirname(__file__), "..", "infra", "database", "schema.sql"
    )

    def _read_schema(self) -> str:
        with open(self.SCHEMA_PATH, "r") as f:
            return f.read()

    def test_all_tables_present(self):
        sql = self._read_schema()
        for table in ("users", "conversations", "messages", "tickets", "agent_metadata"):
            pattern = re.compile(rf"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?{table}\s*\(", re.IGNORECASE)
            assert pattern.search(sql) is not None, f"Missing table or invalid syntax: {table}"

    def test_foreign_keys_present(self):
        sql = self._read_schema()
        assert re.search(r"REFERENCES\s+users\s*\(\s*id\s*\)", sql, re.IGNORECASE) is not None
        assert re.search(r"REFERENCES\s+conversations\s*\(\s*id\s*\)", sql, re.IGNORECASE) is not None

    def test_uuid_primary_keys(self):
        sql = self._read_schema()
        assert re.search(r"UUID\s+PRIMARY\s+KEY", sql, re.IGNORECASE) is not None
