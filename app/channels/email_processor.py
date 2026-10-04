import re
from typing import Dict, Any, Tuple

# Simple regex patterns to strip common quoted reply noise
QUOTE_PATTERNS = [
    re.compile(r"(?m)^>.*$"),                           # Standard > quotes
    re.compile(r"(?i)On\s+.*wrote:"),                   # "On [date] [name] wrote:"
    re.compile(r"(?i)Le\s+.*a\s+écrit\s*:"),            # French variant
    re.compile(r"(?i)---+\s*Original Message\s*---+")   # Block quotes separator
]

# Simple regex for signatures
SIGNATURE_PATTERN = re.compile(r"(?m)^--\s*[\r\n]+.*", re.DOTALL)

def normalize_email_body(body: str, is_html: bool = False) -> str:
    """
    Cleans up the email body by stripping quotes and signatures.
    If the email is HTML, it should ideally be converted to plain text first,
    but we assume the input here is plain text (or roughly converted).
    """
    cleaned = body
    
    # Strip signature if standard "-- " pattern is used
    cleaned = SIGNATURE_PATTERN.sub("", cleaned)
    
    # Strip quoted lines
    for pattern in QUOTE_PATTERNS:
        cleaned = pattern.sub("", cleaned)
        
    # Remove multiple blank lines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    return cleaned.strip()

def process_inbound_email(payload: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Parses an inbound email webhook payload into variables ready for GraphState.
    Returns:
        user_message: The cleaned email body
        thread_id: The identifier for mapping to session
        sender: The sender's email address
    """
    subject = payload.get("subject", "")
    body = payload.get("body", "")
    is_html = payload.get("is_html", False)
    
    # In a real setup, thread_id comes from Message-ID or In-Reply-To header
    # Yosra is building this mapping, we just extract it from the agreed payload schema
    thread_id = payload.get("thread_id", payload.get("message_id", "unknown_thread"))
    sender = payload.get("sender", "unknown_sender")
    
    user_message = normalize_email_body(body, is_html)
    
    # Optionally prefix the subject if it's not a reply already
    if subject and not subject.lower().startswith("re:"):
        # We just want to extract context, subject can be part of the prompt
        user_message = f"Subject: {subject}\n\n{user_message}"
        
    return user_message, thread_id, sender
