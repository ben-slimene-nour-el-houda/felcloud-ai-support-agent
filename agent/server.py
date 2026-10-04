import logging

import uvicorn
from fastapi import FastAPI

from app.config import settings
from app.channels.webhooks.zeroclaw_webhook import router as zeroclaw_router
from app.auth.router import router as auth_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Felcloud AI Support Agent API")

# Register channel routers
app.include_router(zeroclaw_router)
app.include_router(auth_router)
# Future (J8/J9): app.include_router(email_router)
#                 app.include_router(mqtt_router)  # if MQTT needs an HTTP bridge


@app.get("/health")
async def health_check():
    """Basic liveness check for deployment/monitoring."""
    return {"status": "ok"}


if __name__ == "__main__":
    host = settings.ZEROCLAW_WEBHOOK_HOST
    port = settings.ZEROCLAW_WEBHOOK_PORT
    logger.info(f"Starting Felcloud AI Support Agent API on {host}:{port}...")
    uvicorn.run("agent.server:app", host=host, port=port)