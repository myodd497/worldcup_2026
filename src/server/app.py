"""
FastAPI webhook server.
Entry point: uvicorn src.server.app:app
"""
import logging

from fastapi import FastAPI, Request, Response

from src.data.startup_etl import run_full_etl_once
from src.server.whatsapp_handler import handle_incoming

app = FastAPI(title="World Cup 2026 AI Insights", version="1.0.0")
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def _run_bootstrap_etl() -> None:
    """Runs full ETL once at app process startup (not per chat/request)."""
    try:
        result = run_full_etl_once(trigger="fastapi_startup")
        logger.info("Startup ETL status: %s", result)
    except Exception:
        # Keep API process alive; existing data may still serve traffic.
        logger.exception("Startup ETL failed.")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/webhook")
async def webhook(request: Request) -> Response:
    """Twilio calls this endpoint for every incoming WhatsApp message."""
    return await handle_incoming(request)
