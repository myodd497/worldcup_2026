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
from src.data.startup_etl import run_full_etl_once
from src.tools.bigquery_tools import run_query


def _hydrate_env_from_streamlit_secrets() -> None:
    """Copies Streamlit secrets into process env for downstream libraries.

    Also converts [gcp_service_account] TOML section into a JSON env var used by
    src.tools.bigquery_tools._client.
    """
    try:
        secrets = st.secrets
    except Exception:
        # No secrets file found — skip hydration gracefully.
        return

    for key, value in secrets.items():
        if isinstance(value, (str, int, float, bool)):
            os.environ[key] = str(value)

    gcp_info = secrets.get("gcp_service_account")
    if gcp_info:
        os.environ["GOOGLE_SERVICE_ACCOUNT_INFO"] = json.dumps(dict(gcp_info))


_hydrate_env_from_streamlit_secrets()


@st.cache_resource(show_spinner=False)
def _run_bootstrap_etl_once() -> dict[str, object]:
    """Runs full ETL once per Streamlit server process startup."""
    return run_full_etl_once(trigger="streamlit_startup")


def _startup_etl_enabled() -> bool:
    value = os.getenv("RUN_FULL_ETL_ON_STARTUP", "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


@st.cache_data(ttl=60)
def _get_last_etl_status() -> dict[str, str] | None:
    """Returns latest ETL run metadata from BigQuery status table."""
    project = os.environ.get("BIGQUERY_PROJECT_ID")
    dataset = os.environ.get("BIGQUERY_DATASET_ID")
    if not project or not dataset:
        return None

    sql = f"""
    SELECT
      trigger,
      status,
      CAST(started_at AS STRING) AS started_at,
      CAST(finished_at AS STRING) AS finished_at,
      CAST(duration_s AS STRING) AS duration_s,
      error_message
    FROM `{project}.{dataset}.etl_run_status`
    ORDER BY started_at DESC
    LIMIT 1
    """
    try:
        df = run_query(sql)
    except Exception:
        return None
    if df.empty:
        return None
    row = df.iloc[0]
    return {
        "trigger": str(row.get("trigger", "")),
        "status": str(row.get("status", "")),
        "started_at": str(row.get("started_at", "")),
        "finished_at": str(row.get("finished_at", "")),
        "duration_s": str(row.get("duration_s", "")),
        "error_message": str(row.get("error_message", "")),
    }


_etl_bootstrap_error: str | None = None
if _startup_etl_enabled():
    with st.spinner("Initializing data pipeline for app startup..."):
        try:
            _run_bootstrap_etl_once()
        except Exception as exc:
            _etl_bootstrap_error = str(exc)


st.set_page_config(page_title="World Cup 2026 Chat", page_icon="⚽", layout="centered")
st.title("World Cup 2026 Chat")
st.caption("Send a message like WhatsApp and get the same orchestrated reply.")

if _etl_bootstrap_error:
    st.warning(
        "Startup ETL did not run successfully. The app will continue with existing data. "
        f"Details: {_etl_bootstrap_error}"
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_id" not in st.session_state:
    st.session_state.user_id = "web_user"


with st.sidebar:
    st.subheader("Session")
    st.session_state.user_id = st.text_input("User ID", value=st.session_state.user_id)

    st.divider()
    st.subheader("ETL Status")
    last_etl = _get_last_etl_status()
    if last_etl is None:
        st.caption("No ETL status found yet.")
    else:
        st.caption(f"Last run at: {last_etl['started_at']}")
        st.caption(f"Status: {last_etl['status']}")
        st.caption(f"Trigger: {last_etl['trigger']}")
        if last_etl["duration_s"] and last_etl["duration_s"] != "nan":
            st.caption(f"Duration: {last_etl['duration_s']}s")
        if last_etl["status"].startswith("FAILED") and last_etl["error_message"]:
            st.caption(f"Error: {last_etl['error_message'][:160]}")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def _run_orchestrator_sync(user_message: str, user_id: str, conversation_history: list[dict[str, str]] | None = None) -> str:
    """Executes async orchestrator from Streamlit sync context."""
    async def _invoke() -> str:
        try:
            return await run_orchestrator(
                user_message=user_message,
                user_id=user_id,
                conversation_history=conversation_history,
            )
        except TypeError as exc:
            if "unexpected keyword argument 'conversation_history'" not in str(exc):
                raise
            # Backward compatibility for deployments with older orchestrator signature.
            return await run_orchestrator(
                user_message=user_message,
                user_id=user_id,
            )

    try:
        return asyncio.run(_invoke())
    except RuntimeError:
        # Fallback in case an event loop is already running in this process.
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_invoke())
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
