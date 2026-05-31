"""
Validates Twilio webhook signatures, parses incoming WhatsApp messages,
dispatches to the orchestrator agent, and returns TwiML responses.
"""
import os
import logging

from fastapi import Request, Response
from twilio.request_validator import RequestValidator
from twilio.twiml.messaging_response import MessagingResponse

from src.agents.orchestrator import run_orchestrator


logger = logging.getLogger(__name__)


async def handle_incoming(request: Request) -> Response:
    # Validate Twilio signature to prevent spoofed requests
    validator = RequestValidator(os.environ["TWILIO_AUTH_TOKEN"])
    form_data = await request.form()
    url = str(request.url)
    signature = request.headers.get("X-Twilio-Signature", "")

    if not validator.validate(url, dict(form_data), signature):
        return Response(content="Forbidden", status_code=403)

    from_number: str = form_data.get("From", "")
    body: str = form_data.get("Body", "").strip()

    try:
        reply_text = await run_orchestrator(user_message=body, user_id=from_number)
    except Exception:
        logger.exception("Unhandled error while processing incoming WhatsApp message")
        twiml = MessagingResponse()
        twiml.message(
            "I had a temporary issue processing your request. "
            "Please try again in a few moments."
        )
        return Response(content=str(twiml), media_type="application/xml", status_code=200)

    twiml = MessagingResponse()
    twiml.message(reply_text)
    return Response(content=str(twiml), media_type="application/xml")
