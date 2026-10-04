"""
Tests for J3: Qdrant Client (Direct connection logic)
"""
import pytest
from unittest.mock import patch, MagicMock

from app.rag.retrieval.qdrant_client import get_qdrant_client, setup_collection, upsert_chunks
from app.rag.ingestion.embeddings import EmbeddedChunk


class TestQdrantClient:

    @patch("app.rag.retrieval.qdrant_client.QdrantClient")
    @patch("app.rag.retrieval.qdrant_client.settings")
    def test_get_qdrant_client(self, mock_settings, mock_client_cls):
        mock_settings.QDRANT_URL = "http://localhost:6333"
        mock_settings.QDRANT_API_KEY = "secret"
        
        get_qdrant_client()
        mock_client_cls.assert_called_once_with(url="http://localhost:6333", api_key="secret")

    def test_setup_collection_already_exists(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        
        setup_collection(mock_client, "test_collection")
        mock_client.create_collection.assert_not_called()

    def test_setup_collection_creates_new(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        
        setup_collection(mock_client, "test_collection")
        mock_client.create_collection.assert_called_once()
        assert mock_client.create_collection.call_args[1]["collection_name"] == "test_collection"
        assert "bge_dense" in mock_client.create_collection.call_args[1]["vectors_config"]
        assert "bge_sparse" in mock_client.create_collection.call_args[1]["sparse_vectors_config"]
        
        # Verify payload indexes were created
        assert mock_client.create_payload_index.call_count == 3

    def test_upsert_chunks_empty(self):
        mock_client = MagicMock()
        upsert_chunks(mock_client, "col", [])
        mock_client.upload_points.assert_not_called()

    def test_upsert_chunks_success(self):
        mock_client = MagicMock()
        chunk = EmbeddedChunk(
            chunk_id="c1", source_id="s1", source_type="faq", chunk_index=0,
            text="text", title="t", category="c", language="en", metadata={},
            dense_vector=[0.1] * 1024, sparse_vector={"123": 0.5}
        )
        
        upsert_chunks(mock_client, "col", [chunk])
        mock_client.upload_points.assert_called_once()
        
        points_arg = mock_client.upload_points.call_args[1]["points"]
        assert len(points_arg) == 1
        
        # Validate point structure mapping
        point = points_arg[0]
        assert point.payload["chunk_id"] == "c1"
        assert len(point.vector["bge_dense"]) == 1024
        # Since we use qdrant_client.http.models, sparse vectors are indices/values
        assert point.vector["bge_sparse"].indices == [123]
        assert point.vector["bge_sparse"].values == [0.5]
