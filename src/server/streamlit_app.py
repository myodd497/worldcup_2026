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

# ── Title area: trophy logo + title, right at the top ──
st.markdown(
    '<div style="text-align:center;margin-top:-1.5rem;margin-bottom:0;">'
    '<div style="display:flex;align-items:center;justify-content:center;gap:8px;">'
    f'{world_cup_header_html()}'
    '<h1 style="margin:0;padding:0;display:inline;">World Cup 2026</h1>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Two tabs: Dashboard (cards) | Chat Assistant ──
_tab_dashboard, _tab_chat = st.tabs(["📊 Dashboard", "💬 Chat Assistant"])

# ═════════════════════════════════════════════
# TAB 1: Dashboard — cards + charts
# ═════════════════════════════════════════════
with _tab_dashboard:
    # ── Row 1: Three data cards ──
    _col1, _col2, _col3 = st.columns([1, 1, 1], gap="small")
    with _col1:
        st.markdown(get_next_match_html(), unsafe_allow_html=True)
    with _col2:
        groups = get_standings_groups()
        if "selected_group" not in st.session_state:
            st.session_state.selected_group = groups[0] if groups else "A"
        if st.session_state.selected_group not in groups:
            st.session_state.selected_group = groups[0] if groups else "A"
        st.markdown(get_standings_html(st.session_state.selected_group), unsafe_allow_html=True)
        if groups:
            def _on_group_change():
                st.session_state.selected_group = st.session_state.group_dropdown_dk  # type: ignore[attr-defined]
            st.selectbox("Group", options=groups,
                         index=groups.index(st.session_state.selected_group),
                         key="group_dropdown_dk", label_visibility="collapsed",
                         on_change=_on_group_change)
    with _col3:
        from src.server.worldcup_style import get_top_scorers_html, get_top_scorer_metrics
        if "selected_metric" not in st.session_state:
            st.session_state.selected_metric = "goals"
        metrics = get_top_scorer_metrics()
        if st.session_state.selected_metric not in metrics:
            st.session_state.selected_metric = "goals"
        st.markdown(get_top_scorers_html(st.session_state.selected_metric), unsafe_allow_html=True)
        if metrics:
            def _on_metric_change():
                st.session_state.selected_metric = st.session_state.metric_dropdown_dk  # type: ignore[attr-defined]
            st.selectbox("Metric", options=metrics,
                         index=metrics.index(st.session_state.selected_metric),
                         key="metric_dropdown_dk", label_visibility="collapsed",
                         on_change=_on_metric_change)

    # ── Row 2: Top Scorers Bar Chart │ Attack vs Defense Scatter ──
    from src.server.dashboard_charts import (
        top_scorers_bar_chart, team_attack_defense_scatter,
        group_standings_chart, player_comparison_radar, get_available_players,
    )
    _c1, _c2 = st.columns([1, 1], gap="small")
    with _c1:
        fig1 = top_scorers_bar_chart()
        if fig1:
            st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("Chart data not available yet.")
    with _c2:
        fig2 = team_attack_defense_scatter()
        if fig2:
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("Chart data not available yet.")

    # ── Row 3: Group Standings │ Player Comparison Radar ──
    _c3, _c4 = st.columns([1, 1], gap="small")
    with _c3:
        fig3 = group_standings_chart(st.session_state.get("selected_group", "A"))
        if fig3:
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("Chart data not available yet.")
    with _c4:
        # Player comparison
        players = get_available_players()
        if len(players) >= 2:
            if "radar_p1" not in st.session_state:
                st.session_state.radar_p1 = players[0]
            if "radar_p2" not in st.session_state:
                st.session_state.radar_p2 = players[1] if len(players) > 1 else players[0]
            _p1, _p2 = st.columns(2)
            with _p1:
                if players:
                    def _on_p1_change():
                        st.session_state.radar_p1 = st.session_state.radar_p1_key  # type: ignore[attr-defined]
                    st.selectbox("Player 1", players,
                                 index=players.index(st.session_state.radar_p1) if st.session_state.radar_p1 in players else 0,
                                 key="radar_p1_key", label_visibility="collapsed",
                                 on_change=_on_p1_change)
            with _p2:
                if players:
                    def _on_p2_change():
                        st.session_state.radar_p2 = st.session_state.radar_p2_key  # type: ignore[attr-defined]
                    st.selectbox("Player 2", players,
                                 index=players.index(st.session_state.radar_p2) if st.session_state.radar_p2 in players else min(1, len(players)-1),
                                 key="radar_p2_key", label_visibility="collapsed",
                                 on_change=_on_p2_change)
            fig4 = player_comparison_radar(st.session_state.radar_p1, st.session_state.radar_p2)
            if fig4:
                st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})
            else:
                st.caption("Chart data not available yet.")
        else:
            st.caption("Not enough player data for comparison.")

    if _etl_bootstrap_error:
        st.warning(f"Startup ETL failed. Details: {_etl_bootstrap_error}")

# ═════════════════════════════════════════════
# TAB 2: Chat Assistant
# ═════════════════════════════════════════════
with _tab_chat:
    # Display chat history
    for msg in st.session_state.messages:
        role = msg["role"]
        content = inject_flag_emojis(msg["content"])
        content = inject_player_images(content)
        with st.chat_message(role, avatar="assistant" if role == "assistant" else "👤"):
            st.markdown(content, unsafe_allow_html=True)

    prompt = st.chat_input("Ask about matches, predictions, standings, sentiment...")
    if prompt:
        display_prompt = inject_player_images(inject_flag_emojis(prompt))
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(display_prompt, unsafe_allow_html=True)
        with st.chat_message("assistant", avatar="assistant"):
            with st.spinner("🏆 Thinking..."):
                history = st.session_state.messages[:-1]
                reply = _run_orchestrator_sync(prompt, st.session_state.user_id, history)
            styled_reply = inject_player_images(inject_flag_emojis(reply))
            st.markdown(styled_reply, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": reply})
