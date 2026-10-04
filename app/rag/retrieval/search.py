import logging
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.rag.ingestion.embeddings import load_model
from app.rag.retrieval.qdrant_client import get_qdrant_client
from app.config import settings

logger = logging.getLogger(__name__)

def embed_query(query: str) -> Dict[str, Any]:
    """
    Embeds a single user query into dense and sparse vectors using BGE-M3.
    """
    model = load_model()
    # encode() returns a dict with "dense_vecs", "lexical_weights", etc.
    # For a single string or list of 1 string, it returns batched arrays.
    output = model.encode([query], return_dense=True, return_sparse=True)
    
    dense_vec = output["dense_vecs"][0]
    sparse_vec = output["lexical_weights"][0]
    
    if hasattr(dense_vec, "tolist"):
        dense_vec = dense_vec.tolist()
        
    return {
        "dense": dense_vec,
        "sparse": {str(k): float(v) for k, v in sparse_vec.items()}
    }

def fetch_qdrant_results(
    client: QdrantClient, 
    collection_name: str, 
    query_vectors: Dict[str, Any], 
    filters: Optional[models.Filter] = None,
    limit: int = 10
) -> Dict[str, List[models.ScoredPoint]]:
    """
    Performs separate dense and sparse searches in Qdrant.
    """
    # 1. Dense Search
    dense_results = client.search(
        collection_name=collection_name,
        query_vector=models.NamedVector(
            name="bge_dense",
            vector=query_vectors["dense"]
        ),
        query_filter=filters,
        limit=limit,
        with_payload=True
    )
    
    # 2. Sparse Search
    sparse_indices = [int(k) for k in query_vectors["sparse"].keys()]
    sparse_values = list(query_vectors["sparse"].values())
    
    sparse_results = client.search(
        collection_name=collection_name,
        query_vector=models.NamedSparseVector(
            name="bge_sparse",
            vector=models.SparseVector(
                indices=sparse_indices,
                values=sparse_values
            )
        ),
        query_filter=filters,
        limit=limit,
        with_payload=True
    )
    
    return {
        "dense": dense_results,
        "sparse": sparse_results
    }

def rrf_fusion(dense_results: List[models.ScoredPoint], sparse_results: List[models.ScoredPoint], k: int = 60) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF) algorithm to combine dense and sparse results.
    """
    scores = {}
    points = {}
    
    # Process Dense
    for rank, point in enumerate(dense_results):
        point_id = point.id
        scores[point_id] = scores.get(point_id, 0.0) + 1.0 / (k + rank + 1)
        points[point_id] = point
        
    # Process Sparse
    for rank, point in enumerate(sparse_results):
        point_id = point.id
        scores[point_id] = scores.get(point_id, 0.0) + 1.0 / (k + rank + 1)
        points[point_id] = point
        
    # Sort by RRF score descending
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    fused_results = []
    for point_id, score in fused:
        result_dict = {
            "id": point_id,
            "score": score,  # RRF score
            "payload": points[point_id].payload
        }
        fused_results.append(result_dict)
        
    return fused_results

def rerank_results(query: str, results: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
    """
    Reranks the fused results using the BGE-M3 cross-encoder capabilities.
    """
    if not results:
        return []
        
    model = load_model()
    
    # Prepare pairs for cross-encoder: (query, document_text)
    pairs = [[query, res["payload"]["text"]] for res in results]
    
    # Compute scores (higher is better)
    rerank_scores = model.compute_score(pairs)
    
    # If it's a single float, convert to list to avoid iteration errors
    if isinstance(rerank_scores, float):
        rerank_scores = [rerank_scores]
        
    # Attach new scores and sort
    for idx, res in enumerate(results):
        res["rerank_score"] = float(rerank_scores[idx])
        
    reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_n]

def assemble_context(results: List[Dict[str, Any]]) -> str:
    """
    Assembles the final context string for the LLM with clear source attribution.
    """
    if not results:
        return "Aucun contexte pertinent trouvé dans la base de connaissances."
        
    context_parts = []
    for idx, res in enumerate(results):
        payload = res["payload"]
        source_id = payload.get("source_id", "Unknown")
        title = payload.get("title", "Untitled")
        text = payload.get("text", "")
        
        part = f"--- Source {idx+1} | Document ID: {source_id} | Titre: {title} ---\n{text}\n"
        context_parts.append(part)
        
    return "\n".join(context_parts)

def retrieve_context(
    query: str, 
    language: Optional[str] = None, 
    category: Optional[str] = None,
    limit: int = 10,
    top_n: int = 3
) -> tuple[str, List[Dict[str, Any]]]:
    """
    Main entry point for the RAG retrieval pipeline (J3).
    Executes hybrid search, filtering, RRF fusion, reranking, and context assembly.
    Returns the assembled context string and the list of source payloads.
    """
    logger.info(f"Retrieving context for query: '{query}'")
    
    client = get_qdrant_client()
    collection_name = settings.QDRANT_COLLECTION
    
    # Build Filters
    must_conditions = []
    if language:
        must_conditions.append(models.FieldCondition(key="language", match=models.MatchValue(value=language)))
    if category:
        must_conditions.append(models.FieldCondition(key="category", match=models.MatchValue(value=category)))
        
    qdrant_filter = models.Filter(must=must_conditions) if must_conditions else None
    
    # 1. Embed query
    query_vectors = embed_query(query)
    
    # 2. Fetch Hybrid Results
    raw_results = fetch_qdrant_results(
        client=client, 
        collection_name=collection_name, 
        query_vectors=query_vectors, 
        filters=qdrant_filter,
        limit=limit
    )
    
    # 3. RRF Fusion
    fused_results = rrf_fusion(raw_results["dense"], raw_results["sparse"])
    logger.info(f"Found {len(fused_results)} unique documents after RRF fusion.")
    
    # 4. Reranking
    reranked_results = rerank_results(query, fused_results, top_n=top_n)
    logger.info(f"Reranked and kept top {len(reranked_results)} documents.")
    
    # 5. Assemble Context
    context = assemble_context(reranked_results)
    sources = [res["payload"] for res in reranked_results]
    
    return context, sources
