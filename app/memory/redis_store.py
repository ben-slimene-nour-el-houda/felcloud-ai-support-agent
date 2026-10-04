import json
import redis
from app.config import settings

redis_client = redis.Redis(
    host=settings.REDIS_HOST or "localhost",
    port=settings.REDIS_PORT,
    password=settings.REDIS_PASSWORD,
    decode_responses=True,
)

def save_chat_history(session_id: str, chat_history: list):
    """Save the chat history list for a specific session to Redis."""
    key = f"session:{session_id}:history"
    redis_client.set(key, json.dumps(chat_history), ex=3600)

def load_chat_history(session_id: str) -> list:
    """Load the chat history list for a specific session from Redis."""
    key = f"session:{session_id}:history"
    data = redis_client.get(key)
    return json.loads(data) if data else []
