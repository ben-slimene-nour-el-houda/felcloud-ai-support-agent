"""
Tests for J2: Ingestion Data Loaders, Normalization, and Main pipeline
  - Unit: normalize.py
  - Unit: loaders.py
  - Unit: main.py pipeline wrapper
"""
import pytest
from unittest.mock import patch, mock_open, MagicMock

from app.rag.ingestion.schemas import FAQ, TechDoc, Troubleshooting, SupportTicket
from app.rag.ingestion.normalize import (
    clean_text, normalize_faq, normalize_techdoc, normalize_troubleshooting,
    normalize_ticket, validate_raw_documents
)
from app.rag.ingestion.loaders import (
    load_json_files, load_faqs, load_techdocs, load_troubleshooting, load_tickets
)
from app.rag.ingestion.main import run_ingestion_pipeline


# ==========================================================================
# Unit: normalize.py
# ==========================================================================

class TestNormalize:

    def test_clean_text(self):
        assert clean_text("  hello  ") == "hello"
        assert clean_text(None) == ""
        # test unicode normalization
        assert clean_text("café") == "café"

    def test_normalize_faq(self):
        faq = FAQ(
            source_id="faq-1", question="What?", answer="This.",
            category="Compute", language="en", tags=["test"], source_type="faq"
        )
        doc = normalize_faq(faq)
        assert doc.source_type == "faq"
        assert "Question: What?" in doc.content
        assert "Answer: This." in doc.content
        assert doc.metadata["tags"] == ["test"]

    def test_normalize_techdoc(self):
        td = TechDoc(
            source_id="td-1", title="Doc", content="Line 1\nLine 2",
            service="EC2", description="test", category="Compute", language="en",
            prerequisites=["p1"], steps=["s1"], source_type="documentation"
        )
        doc = normalize_techdoc(td)
        assert doc.source_type == "documentation"
        assert doc.content == "Line 1\nLine 2"
        assert doc.metadata["service"] == "EC2"

    def test_normalize_troubleshooting(self):
        ts = Troubleshooting(
            source_id="ts-1", title="TS", problem="Crash", symptoms=["S1", "S2"],
            cause="Bug", solution=["Fix 1"], service="EC2", category="Compute", language="en",
            source_type="troubleshooting"
        )
        doc = normalize_troubleshooting(ts)
        assert "Problem: Crash" in doc.content
        assert "- S1" in doc.content
        assert "Cause: Bug" in doc.content
        assert "- Fix 1" in doc.content

    def test_normalize_ticket(self):
        st = SupportTicket(
            source_id="t-1", ticket_id="TKT-1", title="Ticket",
            customer_issue="Broken", agent_resolution="Fixed",
            service="EC2", severity="high", status="closed", category="Compute", language="en",
            source_type="support_ticket"
        )
        doc = normalize_ticket(st)
        assert "Customer issue: Broken" in doc.content
        assert "Resolution: Fixed" in doc.content
        assert doc.metadata["ticket_id"] == "TKT-1"

    def test_validate_raw_documents(self):
        docs = [
            normalize_faq(FAQ(source_id="f1", question="q", answer="a", category="Compute", language="en", tags=[], source_type="faq")),
            normalize_faq(FAQ(source_id="f1", question="q2", answer="a2", category="Compute", language="en", tags=[], source_type="faq")), # duplicate
            normalize_faq(FAQ(source_id="f2", question="q", answer="a", category="Invalid", language="en", tags=[], source_type="faq")), # bad cat
            normalize_techdoc(TechDoc(source_id="f3", title="t", content="", category="Compute", language="en", service="s", description="d", prerequisites=[], steps=[], source_type="documentation")), # empty content
            normalize_faq(FAQ(source_id="f4", question="q", answer="a", category="Compute", language="xx", tags=[], source_type="faq")), # bad lang
        ]
        
        report = validate_raw_documents(docs)
        assert report["counts_per_type"]["faq"] == 4
        assert report["counts_per_type"]["documentation"] == 1
        
        issues = " ".join(report["issues"])
        assert "Duplicate source_id" in issues
        assert "Invalid category" in issues
        assert "Empty content" in issues
        assert "Invalid language" in issues


# ==========================================================================
# Unit: loaders.py
# ==========================================================================

class TestLoaders:

    @patch("app.rag.ingestion.loaders.Path")
    def test_load_json_files_directory_not_found(self, mock_path_cls):
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_path_cls.return_value = mock_path
        
        items, errors = load_json_files("bad_dir", FAQ)
        assert len(items) == 0
        assert len(errors) == 0

    @patch("app.rag.ingestion.loaders.Path")
    def test_load_json_files_success(self, mock_path_cls):
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_file = MagicMock()
        mock_path.glob.return_value = [mock_file]
        mock_path_cls.return_value = mock_path
        
        json_data = '{"source_id": "f1", "question": "q", "answer": "a", "category": "Compute", "language": "en", "tags": [], "source_type": "faq"}'
        
        with patch("app.rag.ingestion.loaders.open", mock_open(read_data=json_data)):
            items, errors = load_json_files("test_dir", FAQ)
            
        assert len(errors) == 0
        assert len(items) == 1
        assert items[0].source_id == "f1"

    @patch("app.rag.ingestion.loaders.Path")
    def test_load_json_files_invalid_json(self, mock_path_cls):
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.glob.return_value = [MagicMock()]
        mock_path_cls.return_value = mock_path
        
        with patch("app.rag.ingestion.loaders.open", mock_open(read_data='{bad json}')):
            items, errors = load_json_files("test_dir", FAQ)
            
        assert len(items) == 0
        assert len(errors) == 1
        assert "Invalid JSON" in errors[0]


# ==========================================================================
# Unit: main.py pipeline wrapper
# ==========================================================================

class TestMainPipeline:

    @patch("app.rag.ingestion.main.load_tickets")
    @patch("app.rag.ingestion.main.load_troubleshooting")
    @patch("app.rag.ingestion.main.load_techdocs")
    @patch("app.rag.ingestion.main.load_faqs")
    def test_run_ingestion_pipeline(self, mock_faqs, mock_docs, mock_ts, mock_tickets):
        mock_faqs.return_value = [FAQ(source_id="f1", question="q", answer="a", category="Compute", language="en", tags=[], source_type="faq")]
        mock_docs.return_value = [TechDoc(source_id="d1", title="t", content="c", category="Compute", language="en", service="s", description="d", prerequisites=[], steps=[], source_type="documentation")]
        mock_ts.return_value = [Troubleshooting(source_id="ts1", title="t", problem="p", symptoms=["s"], cause="c", solution=["s"], service="s", category="Compute", language="en", source_type="troubleshooting")]
        mock_tickets.return_value = [SupportTicket(source_id="t1", ticket_id="t", title="t", customer_issue="i", agent_resolution="r", service="s", severity="low", status="closed", category="Compute", language="en", source_type="support_ticket")]
        
        docs = run_ingestion_pipeline("dummy_dir")
        assert len(docs) == 4
        assert {d.source_type for d in docs} == {"faq", "documentation", "troubleshooting", "support_ticket"}
