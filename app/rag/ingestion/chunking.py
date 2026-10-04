"""
chunking.py — Source-type-aware chunking step for the Felcloud RAG ingestion pipeline.

Converts List[RawDocument] → List[Chunk], ready for BGE-M3 embedding.

Strategy:
  - faq / troubleshooting / support_ticket : single chunk = whole content (atomic units)
  - documentation : structural split on ## Markdown headers, with sub-split for long
    sections and merge for near-empty sections.
"""

from __future__ import annotations

import re
import logging
from typing import List, Tuple

from pydantic import BaseModel

from app.rag.ingestion.schemas import RawDocument

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

# Token approximation: 1 token ≈ 4 characters (conservative, works for en/fr/darija)
CHARS_PER_TOKEN: int = 4

# Thresholds (in tokens)
TARGET_MIN_TOKENS: int = 150
TARGET_MAX_TOKENS: int = 400
SPLIT_THRESHOLD_TOKENS: int = 500   # sub-split a section above this
MERGE_THRESHOLD_TOKENS: int = 40    # merge a section below this into the next one

# Warn thresholds
WARN_TOO_LARGE_TOKENS: int = 500
WARN_TOO_SMALL_TOKENS: int = 20


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    chunk_id: str           # f"{source_id}-chunk-{index}"
    source_id: str
    source_type: str
    chunk_index: int        # 0-indexed position within parent doc
    text: str               # actual content to embed
    title: str              # parent doc title, repeated for context
    category: str
    language: str
    metadata: dict          # passed through unchanged from parent RawDocument


# ---------------------------------------------------------------------------
# Token / character helpers
# ---------------------------------------------------------------------------

def _approx_tokens(text: str) -> int:
    """Approximate token count via character count (1 token ≈ 4 chars)."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def _chars(tokens: int) -> int:
    """Convert token count to approximate character count."""
    return tokens * CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Markdown header splitting — isolated helper (easy to unit-test)
# ---------------------------------------------------------------------------

# Matches a ## level-2 header line (the structural boundary used in our docs)
_H2_PATTERN = re.compile(r"^(##\s+.+)$", re.MULTILINE)


def split_markdown_by_h2(content: str) -> List[Tuple[str, str]]:
    """
    Split *content* at every ``## ...`` boundary.

    Returns a list of (header, body) tuples where:
      - ``header`` is the raw ``## Header text`` string (empty string for any
        leading content before the first header)
      - ``body``   is the section text that follows that header (stripped)

    Leading content before the first ## header is returned with an empty header
    string; callers may discard it if it's pure whitespace.

    Example
    -------
    >>> pairs = split_markdown_by_h2("# Title\\n\\n## Overview\\nfoo\\n## Steps\\nbar")
    >>> pairs[0]  # pre-header content
    ('', '# Title')
    >>> pairs[1]
    ('## Overview', 'foo')
    >>> pairs[2]
    ('## Steps', 'bar')
    """
    splits: List[Tuple[str, str]] = []

    # Find all header positions
    matches = list(_H2_PATTERN.finditer(content))

    if not matches:
        # No ## headers found — treat the whole thing as one body with no header
        return [("", content.strip())]

    # Content before the first ## header
    pre = content[: matches[0].start()].strip()
    if pre:
        splits.append(("", pre))

    for i, m in enumerate(matches):
        header = m.group(1).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[body_start:body_end].strip()
        splits.append((header, body))

    return splits


# ---------------------------------------------------------------------------
# Recursive character-level sub-splitter
# ---------------------------------------------------------------------------

def _recursive_split(text: str, max_tokens: int, overlap_tokens: int = 20) -> List[str]:
    """
    Split *text* into chunks of at most *max_tokens* tokens with a small overlap.
    Uses a greedy paragraph → sentence → character cascade.
    """
    max_chars = _chars(max_tokens)
    overlap_chars = _chars(overlap_tokens)

    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    # Try splitting on double newlines first (paragraphs), then single newline, then brute-force
    for separator in ("\n\n", "\n", ". ", " ", ""):
        if separator:
            parts = text.split(separator)
        else:
            # brute-force character split
            parts = [text[i: i + max_chars] for i in range(0, len(text), max_chars - overlap_chars)]

        current: List[str] = []
        current_len = 0

        for part in parts:
            part_len = len(part) + len(separator)
            if current_len + part_len > max_chars and current:
                chunks.append(separator.join(current).strip())
                # Keep last bit for overlap
                overlap_text = separator.join(current[-2:]) if len(current) >= 2 else separator.join(current)
                overlap_text = overlap_text[-overlap_chars:] if len(overlap_text) > overlap_chars else overlap_text
                current = [overlap_text, part] if overlap_text else [part]
                current_len = len(overlap_text) + part_len
            else:
                current.append(part)
                current_len += part_len

        if current:
            chunks.append(separator.join(current).strip())

        # If we managed to split, return; otherwise try a finer separator
        if len(chunks) > 1:
            return [c for c in chunks if c.strip()]

    return [text]  # fallback: return unsplit (already tried everything)


# ---------------------------------------------------------------------------
# Public chunking functions
# ---------------------------------------------------------------------------

def chunk_single_unit(doc: RawDocument) -> List[Chunk]:
    """
    For faq / troubleshooting / support_ticket: the whole content is one chunk.

    These document types are already coherent, self-contained units.
    Always returns exactly one Chunk.
    """
    text = doc.content.strip()
    chunk = Chunk(
        chunk_id=f"{doc.source_id}-chunk-0",
        source_id=doc.source_id,
        source_type=doc.source_type,
        chunk_index=0,
        text=text,
        title=doc.title,
        category=doc.category,
        language=doc.language,
        metadata=doc.metadata,
    )
    _warn_chunk(chunk)
    return [chunk]


def chunk_markdown_doc(doc: RawDocument) -> List[Chunk]:
    """
    For documentation: structural split on ## Markdown headers.

    Rules applied in order:
      1. Split content at each ## boundary via ``split_markdown_by_h2()``.
      2. Merge sections whose token count is below MERGE_THRESHOLD_TOKENS into
         the *following* section (so small headers like "## Prerequisites" with
         one bullet point don't become isolated noise chunks).
      3. Sub-split any remaining section that exceeds SPLIT_THRESHOLD_TOKENS
         using a recursive character splitter.
      4. Prefix every final sub-chunk with the doc title + header for standalone
         readability.
    """
    raw_sections = split_markdown_by_h2(doc.content)

    # ------------------------------------------------------------------
    # Step 1: build initial section list (header, body) — discard empty bodies
    # ------------------------------------------------------------------
    sections: List[Tuple[str, str]] = []
    for header, body in raw_sections:
        if body.strip():
            sections.append((header, body.strip()))
        elif header:
            # Keep headerless-but-non-empty sections that have *only* a header line
            # (body might be empty if the section was purely the header text)
            sections.append((header, ""))

    if not sections:
        # Degenerate: no meaningful content at all — fall back to single chunk
        return chunk_single_unit(doc)

    # ------------------------------------------------------------------
    # Step 2: merge short sections into the following one
    # ------------------------------------------------------------------
    merged: List[Tuple[str, str]] = []
    pending_header: str = ""
    pending_body: str = ""

    for header, body in sections:
        combined_text = f"{header}\n\n{body}".strip() if header else body.strip()
        token_count = _approx_tokens(combined_text)

        if pending_header or pending_body:
            # We have accumulated content from a previous short section
            if token_count < MERGE_THRESHOLD_TOKENS:
                # Both current and previous are short — keep accumulating
                sep = "\n\n"
                if pending_header and header:
                    pending_body = f"{pending_body}{sep}{header}\n\n{body}".strip()
                elif header:
                    pending_body = f"{pending_body}{sep}{header}\n\n{body}".strip()
                else:
                    pending_body = f"{pending_body}{sep}{body}".strip()
                # pending_header stays as the *first* short section's header
            else:
                # Current section is big enough — flush pending first, then start fresh
                merged.append((pending_header, pending_body))
                pending_header = header
                pending_body = body
        else:
            if token_count < MERGE_THRESHOLD_TOKENS:
                # Short section — accumulate rather than emit immediately
                pending_header = header
                pending_body = body
            else:
                merged.append((header, body))

    # Flush any remaining pending content
    if pending_header or pending_body:
        if merged:
            # Merge leftover into the last emitted section
            last_header, last_body = merged[-1]
            sep = "\n\n"
            extra = f"{pending_header}\n\n{pending_body}".strip() if pending_header else pending_body.strip()
            merged[-1] = (last_header, f"{last_body}{sep}{extra}".strip())
        else:
            merged.append((pending_header, pending_body))

    # ------------------------------------------------------------------
    # Step 3 & 4: sub-split long sections, build Chunks with context prefix
    # ------------------------------------------------------------------
    chunks: List[Chunk] = []
    chunk_index = 0

    for header, body in merged:
        # Build the fully-prefixed section text for embedding context
        if header:
            section_text = f"{doc.title} — {header}\n\n{body}".strip()
        else:
            # Pre-header content (e.g. a Markdown # Title line)
            section_text = f"{doc.title}\n\n{body}".strip()

        token_count = _approx_tokens(section_text)

        if token_count > SPLIT_THRESHOLD_TOKENS:
            # Sub-split the body portion (preserve title+header prefix on each sub-chunk)
            prefix = f"{doc.title} — {header}\n\n" if header else f"{doc.title}\n\n"
            sub_parts = _recursive_split(body, max_tokens=TARGET_MAX_TOKENS)

            for part in sub_parts:
                if not part.strip():
                    continue
                sub_text = f"{prefix}{part}".strip()
                chunk = Chunk(
                    chunk_id=f"{doc.source_id}-chunk-{chunk_index}",
                    source_id=doc.source_id,
                    source_type=doc.source_type,
                    chunk_index=chunk_index,
                    text=sub_text,
                    title=doc.title,
                    category=doc.category,
                    language=doc.language,
                    metadata=doc.metadata,
                )
                _warn_chunk(chunk)
                chunks.append(chunk)
                chunk_index += 1
        else:
            if not section_text.strip():
                continue
            chunk = Chunk(
                chunk_id=f"{doc.source_id}-chunk-{chunk_index}",
                source_id=doc.source_id,
                source_type=doc.source_type,
                chunk_index=chunk_index,
                text=section_text,
                title=doc.title,
                category=doc.category,
                language=doc.language,
                metadata=doc.metadata,
            )
            _warn_chunk(chunk)
            chunks.append(chunk)
            chunk_index += 1

    if not chunks:
        # Last-resort fallback
        logger.warning(
            "chunk_markdown_doc produced 0 chunks for doc '%s' — falling back to single chunk.",
            doc.source_id,
        )
        return chunk_single_unit(doc)

    return chunks


def chunk_document(doc: RawDocument) -> List[Chunk]:
    """Dispatch chunking strategy based on source_type."""
    if doc.source_type in ("faq", "troubleshooting", "support_ticket"):
        return chunk_single_unit(doc)
    elif doc.source_type == "documentation":
        return chunk_markdown_doc(doc)
    else:
        logger.warning(
            "Unknown source_type '%s' for doc '%s' — defaulting to single-chunk.",
            doc.source_type,
            doc.source_id,
        )
        return chunk_single_unit(doc)


def chunk_all(docs: List[RawDocument]) -> List[Chunk]:
    """
    Top-level entry point.

    Runs ``chunk_document`` over the full list, concatenates results, runs the
    validation pass, and returns the full List[Chunk].
    """
    all_chunks: List[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))

    _validate_chunks(docs, all_chunks)
    return all_chunks


# ---------------------------------------------------------------------------
# Validation pass
# ---------------------------------------------------------------------------

def _warn_chunk(chunk: Chunk) -> None:
    """Emit warnings for degenerate individual chunks."""
    tokens = _approx_tokens(chunk.text)
    if tokens > WARN_TOO_LARGE_TOKENS:
        logger.warning(
            "LARGE CHUNK detected [chunk_id=%s, ~%d tokens] — consider tightening split threshold.",
            chunk.chunk_id,
            tokens,
        )
    elif tokens < WARN_TOO_SMALL_TOKENS:
        logger.warning(
            "SMALL CHUNK detected [chunk_id=%s, ~%d tokens] — consider raising merge threshold.",
            chunk.chunk_id,
            tokens,
        )


def _validate_chunks(docs: List[RawDocument], chunks: List[Chunk]) -> None:
    """
    Post-chunking validation pass.

    Checks:
      - Total chunk count >= len(docs) (only documentation can produce > 1 chunk)
      - No duplicate chunk_ids
      - No empty/whitespace-only chunk text
      - Reports per-type counts and token distribution statistics
    """
    issues: List[str] = []

    # --- sanity: count ---
    if len(chunks) < len(docs):
        issues.append(
            f"SANITY FAIL: chunk count ({len(chunks)}) < doc count ({len(docs)}). "
            "Every document must produce at least one chunk."
        )

    # --- duplicate chunk_ids ---
    seen_ids: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen_ids:
            issues.append(f"DUPLICATE chunk_id: {chunk.chunk_id}")
        seen_ids.add(chunk.chunk_id)

    # --- empty chunk text ---
    empty = [c.chunk_id for c in chunks if not c.text.strip()]
    if empty:
        issues.append(f"EMPTY chunks: {empty}")

    # --- log issues ---
    for issue in issues:
        logger.error("Chunking validation: %s", issue)

    # --- distribution stats ---
    by_type: dict[str, List[int]] = {}
    for chunk in chunks:
        tokens = _approx_tokens(chunk.text)
        by_type.setdefault(chunk.source_type, []).append(tokens)

    logger.info("=== Chunking Report ===")
    logger.info("Total docs: %d | Total chunks: %d", len(docs), len(chunks))
    for stype, token_list in sorted(by_type.items()):
        avg = sum(token_list) / len(token_list)
        logger.info(
            "  %-20s count=%-4d  avg_tokens=%-6.1f  min=%-4d  max=%d",
            stype,
            len(token_list),
            avg,
            min(token_list),
            max(token_list),
        )

    if not issues:
        logger.info("Validation passed with 0 issues.")
