"""Streamlit chat UI for World Cup 2026 assistant.

Run:
    poetry run streamlit run src/server/streamlit_app.py
"""
from __future__ import annotations

import asyncio
import json
import os

import streamlit as st

from src.agents.orchestrator import run_orchestrator


def _hydrate_env_from_streamlit_secrets() -> None:
    """Copies Streamlit secrets into process env for downstream libraries.

    Also converts [gcp_service_account] TOML section into a JSON env var used by
    src.tools.bigquery_tools._client.
    """
    for key, value in st.secrets.items():
        if isinstance(value, (str, int, float, bool)):
            os.environ[key] = str(value)

    gcp_info = st.secrets.get("gcp_service_account")
    if gcp_info:
        os.environ["GOOGLE_SERVICE_ACCOUNT_INFO"] = json.dumps(dict(gcp_info))


_hydrate_env_from_streamlit_secrets()


st.set_page_config(page_title="World Cup 2026 Chat", page_icon="⚽", layout="centered")
st.title("World Cup 2026 Chat")
st.caption("Send a message like WhatsApp and get the same orchestrated reply.")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_id" not in st.session_state:
    st.session_state.user_id = "web_user"


with st.sidebar:
    st.subheader("Session")
    st.session_state.user_id = st.text_input("User ID", value=st.session_state.user_id)
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def _run_orchestrator_sync(user_message: str, user_id: str, conversation_history: list[dict[str, str]] | None = None) -> str:
    """Executes async orchestrator from Streamlit sync context."""
    try:
        return asyncio.run(
            run_orchestrator(
                user_message=user_message,
                user_id=user_id,
                conversation_history=conversation_history,
            )
        )
    except RuntimeError:
        # Fallback in case an event loop is already running in this process.
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                run_orchestrator(
                    user_message=user_message,
                    user_id=user_id,
                    conversation_history=conversation_history,
                )
            )
        finally:
            loop.close()


prompt = st.chat_input("Ask about matches, predictions, standings, sentiment...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Send only prior turns as context; current prompt is passed separately.
            history = st.session_state.messages[:-1]
            reply = _run_orchestrator_sync(prompt, st.session_state.user_id, history)
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
