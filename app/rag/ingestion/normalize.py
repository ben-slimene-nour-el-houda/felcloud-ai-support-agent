import re
import logging
import unicodedata
from typing import List, Dict, Any, Union

from app.rag.ingestion.schemas import (
    FAQ, TechDoc, Troubleshooting, SupportTicket, RawDocument
)

logger = logging.getLogger(__name__)

VALID_CATEGORIES = {
    "Compute", "Storage", "Networking", "Billing & Account", "Console & Interface"
}
VALID_LANGUAGES = {"en", "fr", "darija"}

def clean_text(text: str) -> str:
    """Cleans text by normalizing unicode, and stripping excess whitespace."""
    if not text:
        return ""
    
    # Normalize unicode to avoid weird encoding artifacts if any
    text = unicodedata.normalize("NFKC", text)
    
    # Strip leading/trailing whitespaces
    text = text.strip()
    return text

def normalize_faq(faq: FAQ) -> RawDocument:
    content = f"Question: {faq.question}\nAnswer: {faq.answer}"
    return RawDocument(
        source_id=faq.source_id,
        source_type="faq",
        title=faq.question,
        content=clean_text(content),
        category=faq.category,
        language=faq.language,
        metadata={
            "tags": faq.tags
        }
    )

def normalize_techdoc(doc: TechDoc) -> RawDocument:
    # Intentionally NOT stripping internal newlines to preserve Markdown structure,
    # but clean_text strips leading/trailing
    content = clean_text(doc.content)
    return RawDocument(
        source_id=doc.source_id,
        source_type="documentation",
        title=doc.title,
        content=content,
        category=doc.category,
        language=doc.language,
        metadata={
            "service": doc.service,
            "description": doc.description,
            "prerequisites": doc.prerequisites,
            "steps": doc.steps
        }
    )

def normalize_troubleshooting(ts: Troubleshooting) -> RawDocument:
    symptoms_str = "\n".join(f"- {s}" for s in ts.symptoms)
    solution_str = "\n".join(f"- {s}" for s in ts.solution)
    content = (
        f"Problem: {ts.problem}\n\n"
        f"Symptoms:\n{symptoms_str}\n\n"
        f"Cause: {ts.cause}\n\n"
        f"Solution:\n{solution_str}"
    )
    return RawDocument(
        source_id=ts.source_id,
        source_type="troubleshooting",
        title=ts.title,
        content=clean_text(content),
        category=ts.category,
        language=ts.language,
        metadata={
            "service": ts.service
        }
    )

def normalize_ticket(ticket: SupportTicket) -> RawDocument:
    content = f"Customer issue: {ticket.customer_issue}\nResolution: {ticket.agent_resolution}"
    return RawDocument(
        source_id=ticket.source_id,
        source_type="support_ticket",
        title=ticket.title,
        content=clean_text(content),
        category=ticket.category,
        language=ticket.language,
        metadata={
            "ticket_id": ticket.ticket_id,
            "service": ticket.service,
            "severity": ticket.severity,
            "status": ticket.status
        }
    )

def validate_raw_documents(docs: List[RawDocument]) -> Dict[str, Any]:
    report = {
        "counts_per_type": {
            "faq": 0,
            "documentation": 0,
            "troubleshooting": 0,
            "support_ticket": 0
        },
        "issues": []
    }
    
    seen_ids = set()
    
    for doc in docs:
        report["counts_per_type"][doc.source_type] += 1
        
        # Check duplicate source_id
        if doc.source_id in seen_ids:
            report["issues"].append(f"Duplicate source_id found: {doc.source_id}")
        else:
            seen_ids.add(doc.source_id)
            
        # Check empty content
        if not doc.content:
            report["issues"].append(f"Empty content in document: {doc.source_id}")
            
        # Check category
        if doc.category not in VALID_CATEGORIES:
            report["issues"].append(f"Invalid category '{doc.category}' in document: {doc.source_id}")
            
        # Check language
        if doc.language not in VALID_LANGUAGES:
            report["issues"].append(f"Invalid language '{doc.language}' in document: {doc.source_id}")
            
    if report["issues"]:
        for issue in report["issues"]:
            logger.warning(f"Validation Issue: {issue}")
    else:
        logger.info("Validation passed with 0 issues.")
        
    return report
