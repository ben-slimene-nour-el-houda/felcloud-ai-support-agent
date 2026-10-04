import uuid
import logging
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.rag.ingestion.embeddings import EmbeddedChunk
from app.config import settings

logger = logging.getLogger(__name__)

def get_qdrant_client() -> QdrantClient:
    """
    Initializes and returns the Qdrant client based on environment variables.
    This makes it infra-agnostic (works locally or with Yosra's infra).
    """
    url = settings.QDRANT_URL
    api_key = settings.QDRANT_API_KEY
    
    logger.info(f"Connecting to Qdrant at {url}")
    return QdrantClient(url=url, api_key=api_key)

def setup_collection(client: QdrantClient, collection_name: str) -> None:
    """
    Creates the collection for Hybrid Search (Dense + Sparse) if it doesn't exist.
    """
    if client.collection_exists(collection_name):
        logger.info(f"Collection '{collection_name}' already exists.")
        return
        
    logger.info(f"Creating collection '{collection_name}' for hybrid search...")
    
    # We configure two vectors: one dense (BGE-M3 is 1024-dim) and one sparse
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "bge_dense": models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "bge_sparse": models.SparseVectorParams()
        }
    )
    
    # Create payload indexes to speed up metadata filtering
    client.create_payload_index(collection_name, field_name="source_type", field_schema=models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(collection_name, field_name="category", field_schema=models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(collection_name, field_name="language", field_schema=models.PayloadSchemaType.KEYWORD)
    
    logger.info(f"Collection '{collection_name}' created successfully.")

def upsert_chunks(client: QdrantClient, collection_name: str, chunks: List[EmbeddedChunk]) -> None:
    """
    Converts EmbeddedChunks to Qdrant points and upserts them.
    """
    if not chunks:
        logger.warning("No chunks provided for upsert.")
        return
        
    points = []
    for chunk in chunks:
        # Generate a deterministic UUID based on the string chunk_id
        point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, chunk.chunk_id))
        
        # Convert sparse vector dict back to Qdrant format (integer indices)
        sparse_indices = [int(k) for k in chunk.sparse_vector.keys()]
        sparse_values = list(chunk.sparse_vector.values())
        
        point = models.PointStruct(
            id=point_id,
            vector={
                "bge_dense": chunk.dense_vector,
                "bge_sparse": models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values
                )
            },
            payload={
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "source_type": chunk.source_type,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "title": chunk.title,
                "category": chunk.category,
                "language": chunk.language,
                "metadata": chunk.metadata
            }
        )
        points.append(point)
        
    logger.info(f"Upserting {len(points)} points into '{collection_name}'...")
    
    # Qdrant client handles batching internally if we pass a large list, 
    # but we can explicitly use batching logic if needed. The SDK's upload_points is optimized for this.
    client.upload_points(
        collection_name=collection_name,
        points=points,
        batch_size=100
    )
    
    logger.info(f"Successfully upserted {len(points)} points.")
