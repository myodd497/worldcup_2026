"""Streamlit chat UI for World Cup 2026 assistant.

Run:
    streamlit run src/server/streamlit_app.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

# Ensure the repo root is on sys.path so `src.agents.orchestrator` etc. resolve
# regardless of working directory (critical for Streamlit Cloud).
_sys_path_hint = os.path.join(os.path.dirname(__file__), "..", "..")
if _sys_path_hint not in sys.path:
    sys.path.insert(0, os.path.abspath(_sys_path_hint))

import streamlit as st

from src.agents.orchestrator import run_orchestrator
from src.data.startup_etl import run_full_etl_once
from src.server.worldcup_style import (
    get_world_cup_css,
    world_cup_header_html,
    CROWD_ROAR_HTML,
    confidence_stars_html,
    inject_flag_emojis,
    inject_player_images,
    get_next_match_html,
    get_standings_html,
    get_standings_groups,
)
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


st.set_page_config(page_title="🏆 World Cup 2026 Chat", page_icon="🏆", layout="centered")

# ── Inject custom World Cup CSS (trophy background, glass chat bubbles, gold accents, animations) ──
st.markdown(get_world_cup_css(), unsafe_allow_html=True)

# ── Crowd roar audio on assistant reply (Web Audio API, no external file) ──
st.markdown(CROWD_ROAR_HTML, unsafe_allow_html=True)

# ── Title area: trophy logo + title, all centered (above cards) ──
st.markdown(
    '<div style="text-align:center;margin-top:4px;margin-bottom:12px;">'
    '<div style="display:flex;align-items:center;justify-content:center;gap:8px;">'
    f'{world_cup_header_html()}'
    '<h1 style="margin:0;padding:0;display:inline;">World Cup 2026</h1>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Three cards side-by-side: Match | Standings | Top Scorers ──
_col1, _col2, _col3 = st.columns([1, 1, 1], gap="small")
with _col1:
    st.markdown(get_next_match_html(), unsafe_allow_html=True)
with _col2:
    # Default group selector
    groups = get_standings_groups()
    if "selected_group" not in st.session_state:
        st.session_state.selected_group = groups[0] if groups else "A"
    if st.session_state.selected_group not in groups:
        st.session_state.selected_group = groups[0] if groups else "A"

    st.markdown(get_standings_html(st.session_state.selected_group), unsafe_allow_html=True)

    # Dropdown group picker — uses on_change callback, no st.rerun()
    if groups:
        def _on_group_change():
            st.session_state.selected_group = st.session_state.group_dropdown_key  # type: ignore[attr-defined]
        st.selectbox(
            "Group",
            options=groups,
            index=groups.index(st.session_state.selected_group),
            key="group_dropdown_key",
            label_visibility="collapsed",
            on_change=_on_group_change,
        )
with _col3:
    from src.server.worldcup_style import get_top_scorers_html, get_top_scorer_metrics

    # Default metric
    if "selected_metric" not in st.session_state:
        st.session_state.selected_metric = "goals"
    metrics = get_top_scorer_metrics()
    if st.session_state.selected_metric not in metrics:
        st.session_state.selected_metric = "goals"

    st.markdown(
        get_top_scorers_html(st.session_state.selected_metric),
        unsafe_allow_html=True,
    )

    # Dropdown metric picker — uses on_change callback, no st.rerun()
    if metrics:
        current_idx = metrics.index(st.session_state.selected_metric)
        def _on_metric_change():
            st.session_state.selected_metric = st.session_state.metric_dropdown_key  # type: ignore[attr-defined]
        st.selectbox(
            "Metric",
            options=metrics,
            index=current_idx,
            key="metric_dropdown_key",
            label_visibility="collapsed",
            on_change=_on_metric_change,
        )

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
    st.markdown("### 🏆 World Cup 2026")
    st.markdown("*AI-Powered Football Insights*")
    st.divider()
    st.subheader("👤 Session")
    st.session_state.user_id = st.text_input("User ID", value=st.session_state.user_id)

    st.divider()
    st.subheader("📡 ETL Status")
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
    role = msg["role"]
    content = msg["content"]
    # Inject flag emojis for national teams
    content = inject_flag_emojis(content)
    # Embed player images for known players
    content = inject_player_images(content)
    with st.chat_message(role, avatar="assistant" if role == "assistant" else "👤"):
        st.markdown(content, unsafe_allow_html=True)


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
    # Inject flag emojis into user message before storing
    display_prompt = inject_flag_emojis(prompt)
    display_prompt = inject_player_images(display_prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(display_prompt, unsafe_allow_html=True)

    with st.chat_message("assistant", avatar="assistant"):
        with st.spinner("🏆 Thinking..."):
            # Send only prior turns as context; current prompt is passed separately.
            history = st.session_state.messages[:-1]
            reply = _run_orchestrator_sync(prompt, st.session_state.user_id, history)
        # Inject flag emojis, player images, and confidence stars into assistant reply
        styled_reply = inject_flag_emojis(reply)
        styled_reply = inject_player_images(styled_reply)
        st.markdown(styled_reply, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": reply})
