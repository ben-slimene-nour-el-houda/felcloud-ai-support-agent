from typing import List, Literal, Dict, Any
from pydantic import BaseModel

class FAQ(BaseModel):
    source_id: str
    question: str
    answer: str
    category: str
    tags: List[str]
    language: str
    source_type: str

class TechDoc(BaseModel):
    source_id: str
    title: str
    service: str
    description: str
    prerequisites: List[str]
    steps: List[str]
    content: str
    category: str
    language: str
    source_type: str

class Troubleshooting(BaseModel):
    source_id: str
    title: str
    problem: str
    symptoms: List[str]
    cause: str
    solution: List[str]
    service: str
    category: str
    language: str
    source_type: str

class SupportTicket(BaseModel):
    source_id: str
    ticket_id: str
    title: str
    customer_issue: str
    agent_resolution: str
    service: str
    category: str
    severity: str
    status: str
    language: str
    source_type: str

class RawDocument(BaseModel):
    source_id: str
    source_type: Literal["faq", "documentation", "troubleshooting", "support_ticket"]
    title: str
    content: str
    category: str
    language: str
    metadata: Dict[str, Any]
