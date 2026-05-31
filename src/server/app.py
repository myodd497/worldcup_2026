"""
FastAPI webhook server.
Entry point: uvicorn src.server.app:app
"""
from fastapi import FastAPI, Request, Response
from src.server.whatsapp_handler import handle_incoming

app = FastAPI(title="World Cup 2026 AI Insights", version="1.0.0")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/webhook")
async def webhook(request: Request) -> Response:
    """Twilio calls this endpoint for every incoming WhatsApp message."""
    return await handle_incoming(request)
