import pytest
from app.channels.email_processor import normalize_email_body, process_inbound_email

def test_j8_normalize_email_body():
    raw_body = """Hi there,

I need help with my server.

Thanks,
Bob
-- 
Bob Builder
Director
"""
    cleaned = normalize_email_body(raw_body)
    assert "Bob Builder" not in cleaned
    assert "I need help with my server." in cleaned
    
    raw_body_with_quotes = """Yes, that works for me.

On Thu, Oct 26, 2023 at 10:00 AM Felcloud Support <support@felcloud.com> wrote:
> Hello Bob,
> Did you try turning it off and on again?
> 
> Regards,
> Support Team
"""
    cleaned2 = normalize_email_body(raw_body_with_quotes)
    assert "Yes, that works for me." in cleaned2
    assert "Did you try turning it off" not in cleaned2

def test_j8_process_inbound_email():
    payload = {
        "subject": "Server Down",
        "body": "My server is not responding.\n\nOn Jan 1, 2023 Support wrote:\n> Reboot it.",
        "sender": "bob@example.com",
        "thread_id": "thread_123"
    }
    
    user_message, thread_id, sender = process_inbound_email(payload)
    
    assert "Subject: Server Down" in user_message
    assert "My server is not responding." in user_message
    assert "Reboot it" not in user_message
    assert thread_id == "thread_123"
    assert sender == "bob@example.com"
