"""
embeddings.py — BGE-M3 dense + sparse embedding step for the Felcloud RAG ingestion pipeline.

Converts List[Chunk] → List[EmbeddedChunk], each carrying a 1024-dim dense vector
and a sparse lexical-weight dict, ready for Qdrant upsert (formatting happens downstream).

Model: BAAI/bge-m3 via the FlagEmbedding library (BGEM3FlagModel).

Configuration (via app.config.settings):
  EMBEDDING_DEVICE        — "cpu", "cuda", or "auto" (default: "auto" → cuda if available)
  EMBEDDING_BATCH_SIZE    — integer batch size for encode() (default: 16)
  EMBEDDING_USE_COLBERT   — bool — whether to compute ColBERT vectors (default: False)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel

from app.rag.ingestion.chunking import Chunk
from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

_DEFAULT_BATCH_SIZE = 16
_BGE_M3_DENSE_DIM = 1024
_MODEL_NAME = "BAAI/bge-m3"


def _get_device() -> str:
    """Resolve the device to load the model on."""
    env = settings.EMBEDDING_DEVICE.lower()
    if env in ("cuda", "cpu"):
        return env
    # auto-detect
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _get_batch_size() -> int:
    """Read batch size from env or fall back to default."""
    try:
        return int(settings.EMBEDDING_BATCH_SIZE)
    except ValueError:
        logger.warning(
            "Invalid EMBEDDING_BATCH_SIZE config, falling back to %d", _DEFAULT_BATCH_SIZE
        )
        return _DEFAULT_BATCH_SIZE


def _use_colbert() -> bool:
    """Whether to compute ColBERT vectors (off by default — not needed for RRF fusion)."""
    return settings.EMBEDDING_USE_COLBERT


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class EmbeddedChunk(BaseModel):
    """A Chunk enriched with dense and sparse vector representations."""

    chunk_id: str
    source_id: str
    source_type: str
    chunk_index: int
    text: str
    title: str
    category: str
    language: str
    metadata: dict
    dense_vector: List[float]   # BGE-M3 dense output, 1024-dim
    sparse_vector: dict         # BGE-M3 lexical weights: {token_id: weight}


# ---------------------------------------------------------------------------
# Singleton model loader
# ---------------------------------------------------------------------------

_model_instance: Optional[Any] = None


def load_model():
    """
    Load BAAI/bge-m3 via FlagEmbedding's BGEM3FlagModel.

    The model is cached in a module-level singleton so it is loaded exactly once
    per process, even if ``embed_chunks`` is called multiple times.
    """
    global _model_instance  # noqa: PLW0603

    if _model_instance is not None:
        return _model_instance

    device = _get_device()
    logger.info("Loading BGE-M3 model '%s' on device='%s' ...", _MODEL_NAME, device)

    from FlagEmbedding import BGEM3FlagModel  # import here to avoid slow top-level import

    use_fp16 = device == "cuda"
    _model_instance = BGEM3FlagModel(
        _MODEL_NAME,
        use_fp16=use_fp16,
        device=device,
    )

    logger.info("BGE-M3 model loaded successfully.")
    return _model_instance


def _reset_model() -> None:
    """Reset the singleton — only used by tests."""
    global _model_instance  # noqa: PLW0603
    _model_instance = None


# ---------------------------------------------------------------------------
# Core embedding function
# ---------------------------------------------------------------------------

def embed_chunks(
    chunks: List[Chunk],
    batch_size: Optional[int] = None,
) -> List[EmbeddedChunk]:
    """
    Embed a list of Chunks using BGE-M3.

    Parameters
    ----------
    chunks : List[Chunk]
        The chunks produced by the chunking step.
    batch_size : int, optional
        Number of texts per encode() call.  Falls back to the
        ``EMBEDDING_BATCH_SIZE`` env var, then to 16.

    Returns
    -------
    List[EmbeddedChunk]
        One EmbeddedChunk per input Chunk (empty-text chunks are skipped
        with a warning).
    """
    if batch_size is None:
        batch_size = _get_batch_size()

    model = load_model()
    use_colbert = _use_colbert()

    # --- filter empty-text chunks (should never happen post-chunking, but be safe) ---
    valid_chunks: List[Chunk] = []
    for chunk in chunks:
        if not chunk.text or not chunk.text.strip():
            logger.warning(
                "Skipping chunk '%s' — empty text (should not happen post-chunking validation).",
                chunk.chunk_id,
            )
            continue
        valid_chunks.append(chunk)

    if not valid_chunks:
        logger.warning("embed_chunks called with 0 valid chunks — returning empty list.")
        return []

    # --- collect texts and batch-encode ---
    texts = [c.text for c in valid_chunks]
    total = len(texts)
    num_batches = math.ceil(total / batch_size)

    logger.info(
        "Embedding %d chunks in %d batches (batch_size=%d, colbert=%s) ...",
        total, num_batches, batch_size, use_colbert,
    )

    all_dense: List[np.ndarray] = []
    all_sparse: List[Dict[str, float]] = []

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]

        logger.info("  Batch %d/%d  (%d texts) ...", batch_idx + 1, num_batches, len(batch_texts))

        output = model.encode(
            batch_texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=use_colbert,
        )

        # output is a dict with keys "dense_vecs", "lexical_weights", optionally "colbert_vecs"
        batch_dense = output["dense_vecs"]          # np.ndarray (batch, 1024)
        batch_sparse = output["lexical_weights"]    # list of dicts {token_id: weight}

        for i in range(len(batch_texts)):
            all_dense.append(batch_dense[i])
            all_sparse.append(batch_sparse[i])

    # --- assemble EmbeddedChunks ---
    embedded: List[EmbeddedChunk] = []
    for idx, chunk in enumerate(valid_chunks):
        dense_vec = all_dense[idx]
        sparse_vec = all_sparse[idx]

        # Ensure dense vector is a plain Python list of floats
        if isinstance(dense_vec, np.ndarray):
            dense_list = dense_vec.tolist()
        else:
            dense_list = list(dense_vec)

        # Sparse vector: ensure keys are strings (token ids) and values are floats
        sparse_dict = {str(k): float(v) for k, v in sparse_vec.items()}

        embedded.append(
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                source_id=chunk.source_id,
                source_type=chunk.source_type,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                title=chunk.title,
                category=chunk.category,
                language=chunk.language,
                metadata=chunk.metadata,
                dense_vector=dense_list,
                sparse_vector=sparse_dict,
            )
        )

    logger.info("Embedding complete: %d EmbeddedChunks produced.", len(embedded))
    return embedded


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_embeddings(embedded: List[EmbeddedChunk]) -> dict:
    """
    Post-embedding validation pass.

    Checks
    ------
    - Every dense_vector has the expected dimensionality (1024 for BGE-M3).
    - No all-zero dense vectors.
    - No NaN values in dense vectors.
    - Sparse vectors are non-empty for non-trivial text.

    Returns
    -------
    dict
        A validation report with keys:
          total, passed, failures (list of {chunk_id, reason}),
          avg_sparse_density, dimension_expected.
    """
    failures: List[Dict[str, str]] = []
    sparse_lengths: List[int] = []

    for ec in embedded:
        cid = ec.chunk_id

        # --- dimension check ---
        dim = len(ec.dense_vector)
        if dim != _BGE_M3_DENSE_DIM:
            failures.append({
                "chunk_id": cid,
                "reason": f"dense_vector dim={dim}, expected {_BGE_M3_DENSE_DIM}",
            })

        # --- NaN check ---
        if any(math.isnan(v) for v in ec.dense_vector):
            failures.append({
                "chunk_id": cid,
                "reason": "dense_vector contains NaN values",
            })

        # --- all-zero check ---
        if all(v == 0.0 for v in ec.dense_vector):
            failures.append({
                "chunk_id": cid,
                "reason": "dense_vector is all zeros",
            })

        # --- sparse non-empty check (only if text has substance) ---
        text_len = len(ec.text.strip())
        if text_len > 10 and len(ec.sparse_vector) == 0:
            failures.append({
                "chunk_id": cid,
                "reason": "sparse_vector is empty for non-trivial text",
            })

        sparse_lengths.append(len(ec.sparse_vector))

    avg_sparse_density = (
        sum(sparse_lengths) / len(sparse_lengths) if sparse_lengths else 0.0
    )

    report = {
        "total": len(embedded),
        "passed": len(embedded) - len(failures),
        "failures": failures,
        "avg_sparse_density": round(avg_sparse_density, 2),
        "dimension_expected": _BGE_M3_DENSE_DIM,
    }

    # --- log the report ---
    logger.info("=== Embedding Validation Report ===")
    logger.info("Total chunks embedded : %d", report["total"])
    logger.info("Passed                : %d", report["passed"])
    logger.info("Failures              : %d", len(failures))
    logger.info("Avg sparse density    : %.2f tokens/chunk", avg_sparse_density)

    if failures:
        for f in failures:
            logger.error("  FAIL [%s]: %s", f["chunk_id"], f["reason"])
    else:
        logger.info("All embeddings passed validation.")

    return report


# ---------------------------------------------------------------------------
# __main__ — end-to-end smoke test: load → normalize → chunk → embed → validate
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    from app.rag.ingestion.main import run_ingestion_pipeline
    from app.rag.ingestion.chunking import chunk_all

    logger.info("=" * 60)
    logger.info("FULL PIPELINE SMOKE TEST: load → normalize → chunk → embed")
    logger.info("=" * 60)

    # Step 1 & 2: load + normalize
    raw_docs = run_ingestion_pipeline()
    logger.info("Loaded %d raw documents.", len(raw_docs))

    # Step 3: chunk
    chunks = chunk_all(raw_docs)
    logger.info("Produced %d chunks.", len(chunks))

    # Step 4: embed
    embedded = embed_chunks(chunks)
    logger.info("Produced %d embedded chunks.", len(embedded))

    # Step 5: validate
    report = validate_embeddings(embedded)

    logger.info("=" * 60)
    logger.info("SMOKE TEST COMPLETE")
    logger.info("=" * 60)

    if report["failures"]:
        logger.error("Validation found %d failures — see above.", len(report["failures"]))
        sys.exit(1)
    else:
        logger.info("All %d embeddings passed validation.", report["total"])
