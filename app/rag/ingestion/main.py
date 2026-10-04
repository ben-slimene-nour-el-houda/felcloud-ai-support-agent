import logging
from pathlib import Path
from typing import List

from app.rag.ingestion.schemas import RawDocument
from app.rag.ingestion.loaders import load_faqs, load_techdocs, load_troubleshooting, load_tickets
from app.rag.ingestion.normalize import (
    normalize_faq, normalize_techdoc, normalize_troubleshooting, normalize_ticket, validate_raw_documents
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_ingestion_pipeline(base_data_dir: str = "data/raw") -> List[RawDocument]:
    base_path = Path(base_data_dir)
    raw_documents = []
    
    logger.info("Loading FAQs...")
    faqs = load_faqs(str(base_path / "faq"))
    for faq in faqs:
        raw_documents.append(normalize_faq(faq))
        
    logger.info("Loading Technical Documentation...")
    docs = load_techdocs(str(base_path / "documentation"))
    for doc in docs:
        raw_documents.append(normalize_techdoc(doc))
        
    logger.info("Loading Troubleshooting Scenarios...")
    scenarios = load_troubleshooting(str(base_path / "troubleshooting"))
    for scenario in scenarios:
        raw_documents.append(normalize_troubleshooting(scenario))
        
    logger.info("Loading Support Tickets...")
    tickets = load_tickets(str(base_path / "tickets"))
    for ticket in tickets:
        raw_documents.append(normalize_ticket(ticket))
        
    logger.info(f"Loaded and normalized {len(raw_documents)} documents. Running validation...")
    
    report = validate_raw_documents(raw_documents)
    
    logger.info("Validation Report:")
    for key, value in report["counts_per_type"].items():
        logger.info(f"  {key}: {value}")
        
    if report["issues"]:
        logger.warning(f"Found {len(report['issues'])} issues:")
        for issue in report["issues"]:
            logger.warning(f"  - {issue}")
    else:
        logger.info("All documents valid.")
        
    return raw_documents

if __name__ == "__main__":
    from app.rag.ingestion.chunking import chunk_all
    from app.rag.ingestion.embeddings import embed_chunks
    from app.rag.retrieval.qdrant_client import get_qdrant_client, setup_collection, upsert_chunks
    from app.config import settings

    logger.info("Starting Full Ingestion Pipeline...")
    
    # 1. Load and Normalize
    docs = run_ingestion_pipeline()
    
    # 2. Chunking
    logger.info("Chunking documents...")
    chunks = chunk_all(docs)
    logger.info(f"Generated {len(chunks)} chunks.")
    
    # 3. Embedding (BGE-M3)
    logger.info("Embedding chunks...")
    embedded_chunks = embed_chunks(chunks)
    logger.info(f"Generated {len(embedded_chunks)} embedded chunks.")
    
    # 4. Qdrant Population
    collection_name = settings.QDRANT_COLLECTION
    
    try:
        qdrant_client = get_qdrant_client()
        setup_collection(qdrant_client, collection_name)
        upsert_chunks(qdrant_client, collection_name, embedded_chunks)
        logger.info("Ingestion pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Failed to populate Qdrant: {e}")
        logger.info("Note: Make sure Qdrant is running and accessible.")
