"""Datamodel catalog — single source of truth for agent-facing metadata.

Builds a structured catalog by introspecting the SCHEMA constants exported by
every dim/fact/mart module. No duplication, no drift: change a column in a
module's SCHEMA and the catalog updates automatically.

Public API:
  - get_catalog()                       → dict[table_name, TableSpec]
  - get_table(name)                     → TableSpec
  - list_tables(agent_visible=True)     → list[TableSpec]
  - format_catalog_for_llm(...)         → markdown summary suitable for prompts
  - format_table_detail_for_llm(name)   → full per-column markdown for prompts
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from google.cloud import bigquery

from src.data.datamodel import (
    dim_team, dim_competition, dim_venue, dim_date,
    fact_match, fact_match_team, fact_match_event, fact_standings_snapshot,
    mart_team_profile, mart_team_form, mart_head_to_head,
    mart_match_history, mart_match_upcoming, mart_tournament_state,
)


@dataclass(frozen=True)
class TableSpec:
    name: str
    layer: str                      # 'dim' | 'fact' | 'mart'
    grain: str                      # one-line natural-language grain description
    description: str                # what the table represents (agent-readable)
    schema: tuple[bigquery.SchemaField, ...]
    primary_keys: tuple[str, ...]
    agent_visible: bool             # may the agent query this table?
    preferred: bool = False         # surface FIRST in catalog summary
    usage_hint: str = ""            # extra guidance for the LLM (when to use this)
    example_questions: tuple[str, ...] = field(default_factory=tuple)


# ─────────────────────────────────────────────────────────────────────────────
# Catalog definition — one entry per table.
# `schema` is pulled live from each module so column definitions never drift.
# ─────────────────────────────────────────────────────────────────────────────

_CATALOG_DEFS: tuple[TableSpec, ...] = (
    # ── DIMS ────────────────────────────────────────────────────────────────
    TableSpec(
        name="dim_team", layer="dim",
        grain="One row per team.",
        description="Team master. Use to resolve team_name → team_id and to flag WC 2026 participants.",
        schema=tuple(dim_team.DIM_TEAM_SCHEMA),
        primary_keys=("team_id",),
        agent_visible=True,
        usage_hint="Always use this for team-name → team_id lookups. Prefer is_wc2026_participant filter for WC questions.",
        example_questions=("Is Morocco a WC 2026 participant?", "Find the team_id for Argentina."),
    ),
    TableSpec(
        name="dim_competition", layer="dim",
        grain="One row per competition.",
        description="Competition master with is_world_cup flag and seasons_observed array.",
        schema=tuple(dim_competition.DIM_COMPETITION_SCHEMA),
        primary_keys=("competition_id",),
        agent_visible=True,
        usage_hint="World Cup competition_id = 1. Use is_world_cup or competition_id=1 to filter.",
    ),
    TableSpec(
        name="dim_venue", layer="dim",
        grain="One row per venue.",
        description="Venue master with surrogate venue_key (id-based when API has it, else name+city).",
        schema=tuple(dim_venue.DIM_VENUE_SCHEMA),
        primary_keys=("venue_key",),
        agent_visible=True,
    ),
    TableSpec(
        name="dim_date", layer="dim",
        grain="One row per calendar date.",
        description="Calendar dimension spanning historical + 1 year ahead.",
        schema=tuple(dim_date.DIM_DATE_SCHEMA),
        primary_keys=("calendar_date",),
        agent_visible=True,
    ),

    # ── FACTS ───────────────────────────────────────────────────────────────
    TableSpec(
        name="fact_match", layer="fact",
        grain="One row per match_id.",
        description="Canonical match fact. Final scores, status, winner, venue. Use for ad-hoc match-level queries.",
        schema=tuple(fact_match.FACT_MATCH_SCHEMA),
        primary_keys=("match_id",),
        agent_visible=True,
        usage_hint=(
            "match_status is normalized to: SCHEDULED, LIVE, FINISHED, POSTPONED, CANCELLED, ABANDONED. "
            "Use is_completed for played-match filters. winner_team_id is NULL on draw or not-completed. "
            "Prefer marts (mart_match_history / mart_match_upcoming) when they cover the question."
        ),
    ),
    TableSpec(
        name="fact_match_team", layer="fact",
        grain="One row per (match_id, team_id). Two rows per match.",
        description="Team-side match fact. Outcome (W/D/L, points, goals_for/against) + wide-pivoted stats (possession_pct, shots_total_count, xg, etc.).",
        schema=tuple(fact_match_team.FACT_MATCH_TEAM_SCHEMA),
        primary_keys=("match_id", "team_id"),
        agent_visible=True,
        usage_hint=(
            "Use when you need per-team stats or W/D/L from a team's perspective. "
            "result is one of 'W','D','L' (NULL if not completed). "
            "Stats are NULL for international matches (API-Football limitation)."
        ),
    ),
    TableSpec(
        name="fact_match_event", layer="fact",
        grain="One row per (match_id, event_seq).",
        description="Match events: goals, cards, substitutions, VAR. Includes player and assist names.",
        schema=tuple(fact_match_event.FACT_MATCH_EVENT_SCHEMA),
        primary_keys=("match_id", "event_seq"),
        agent_visible=True,
        usage_hint="Filter is_goal/is_yellow_card/is_red_card/is_substitution for typed event queries.",
    ),
    TableSpec(
        name="fact_standings_snapshot", layer="fact",
        grain="One row per (competition_id, season_year, team_id, snapshot_date).",
        description="Standings snapshots over time. For 'current standings' prefer mart_tournament_state.",
        schema=tuple(fact_standings_snapshot.FACT_STANDINGS_SCHEMA),
        primary_keys=("competition_id", "season_year", "team_id", "snapshot_date"),
        agent_visible=True,
    ),

    # ── MARTS (preferred — agent should start here) ─────────────────────────
    TableSpec(
        name="mart_team_profile", layer="mart",
        grain="One row per team_id.",
        description="Lifetime team profile: matches_played, wins/draws/losses, win_pct, goals_for/against, clean_sheets.",
        schema=tuple(mart_team_profile.SCHEMA),
        primary_keys=("team_id",),
        agent_visible=True, preferred=True,
        usage_hint="Use for any 'overall', 'historically', 'all-time' team statistic question.",
        example_questions=(
            "What is Argentina's all-time win rate?",
            "Which WC2026 team has the most clean sheets?",
        ),
    ),
    TableSpec(
        name="mart_team_form", layer="mart",
        grain="One row per team_id.",
        description="Rolling form over last 10 completed matches: WDL string, points, gf/ga, clean sheets.",
        schema=tuple(mart_team_form.SCHEMA),
        primary_keys=("team_id",),
        agent_visible=True, preferred=True,
        usage_hint="Use for 'recent form', 'last N matches', 'on a winning streak' questions. recent_form_string is newest-first (e.g. 'WWDLW').",
        example_questions=(
            "What is Portugal's recent form?",
            "Who has the best form among WC2026 teams?",
        ),
    ),
    TableSpec(
        name="mart_head_to_head", layer="mart",
        grain="One row per unordered team pair (team_lo_id < team_hi_id).",
        description="Pairwise head-to-head aggregates: matches_played, team_lo_wins, team_hi_wins, draws, first/last meeting.",
        schema=tuple(mart_head_to_head.SCHEMA),
        primary_keys=("team_lo_id", "team_hi_id"),
        agent_visible=True, preferred=True,
        usage_hint=(
            "IMPORTANT: pair key is sorted. To query the pair (A, B) use "
            "team_lo_id = LEAST(A_id, B_id) AND team_hi_id = GREATEST(A_id, B_id). "
            "team_lo_wins is wins by the team with the lower id, NOT by the home team."
        ),
        example_questions=(
            "How many times have Argentina and Brazil played?",
            "What is the head-to-head between Portugal and Morocco?",
        ),
    ),
    TableSpec(
        name="mart_match_history", layer="mart",
        grain="One row per completed match_id.",
        description="Completed matches with wide-pivoted home_* / away_* stat columns. Single-row-per-match view of facts.",
        schema=tuple(mart_match_history.SCHEMA),
        primary_keys=("match_id",),
        agent_visible=True, preferred=True,
        usage_hint=(
            "Use for any past-match question with stats. "
            "match_result_label is human-readable ('Home win', 'Draw', 'Home win on penalties', ...). "
            "Stat columns are prefixed home_/away_ (e.g. home_possession_pct, away_shots_total_count)."
        ),
        example_questions=(
            "Show Argentina's last 5 matches with stats.",
            "What was the possession in Portugal vs Spain?",
        ),
    ),
    TableSpec(
        name="mart_match_upcoming", layer="mart",
        grain="One row per upcoming/live/postponed match_id (match_date >= today-1).",
        description="Upcoming matches pre-enriched with last-5 form strings and head-to-head record.",
        schema=tuple(mart_match_upcoming.SCHEMA),
        primary_keys=("match_id",),
        agent_visible=True, preferred=True,
        usage_hint="Use for 'next match', 'upcoming fixtures', 'preview <A> vs <B>'. days_until_kickoff is signed (positive = future).",
        example_questions=(
            "What is Portugal's next match?",
            "List the next 5 World Cup matches.",
        ),
    ),
    TableSpec(
        name="mart_tournament_state", layer="mart",
        grain="One row per (competition_id, season_year, team_id).",
        description="Current tournament state per team: latest standings + next/last match.",
        schema=tuple(mart_tournament_state.SCHEMA),
        primary_keys=("competition_id", "season_year", "team_id"),
        agent_visible=True, preferred=True,
        usage_hint="Use for 'standings', 'group stage', 'where does X stand'. For WC2026 filter competition_id=1 AND season_year=2026.",
        example_questions=(
            "What are the current WC2026 group standings?",
            "Show Group A of the 2026 World Cup.",
        ),
    ),
)

_CATALOG: dict[str, TableSpec] = {t.name: t for t in _CATALOG_DEFS}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def get_catalog() -> dict[str, TableSpec]:
    return dict(_CATALOG)


def get_table(name: str) -> TableSpec:
    if name not in _CATALOG:
        raise KeyError(f"Unknown table '{name}'. Known: {sorted(_CATALOG)}")
    return _CATALOG[name]


def list_tables(*, agent_visible: bool | None = True, layer: str | None = None) -> list[TableSpec]:
    out = list(_CATALOG.values())
    if agent_visible is not None:
        out = [t for t in out if t.agent_visible == agent_visible]
    if layer is not None:
        out = [t for t in out if t.layer == layer]
    return out


def fqn(table: str) -> str:
    """Fully-qualified `project.dataset.table` (backticked)."""
    p = os.environ["BIGQUERY_PROJECT_ID"]; d = os.environ["BIGQUERY_DATASET_ID"]
    return f"`{p}.{d}.{table}`"


# ─────────────────────────────────────────────────────────────────────────────
# LLM-formatted views
# ─────────────────────────────────────────────────────────────────────────────


def format_catalog_for_llm(*, agent_visible_only: bool = True) -> str:
    """Compact markdown summary for the system prompt.

    Groups by layer; marts shown first (preferred); each entry on one line.
    """
    tables = list_tables(agent_visible=True if agent_visible_only else None)
    by_layer: dict[str, list[TableSpec]] = {"mart": [], "fact": [], "dim": []}
    for t in tables:
        by_layer.setdefault(t.layer, []).append(t)

    sections: list[str] = []

    if by_layer.get("mart"):
        sections.append("## Marts (preferred — start here for agent questions)")
        for t in by_layer["mart"]:
            sections.append(f"- `{t.name}` — {t.description} **Grain:** {t.grain}")

    if by_layer.get("dim"):
        sections.append("\n## Dimensions (use for name → id resolution)")
        for t in by_layer["dim"]:
            sections.append(f"- `{t.name}` — {t.description}")

    if by_layer.get("fact"):
        sections.append("\n## Facts (use only when no mart can answer)")
        for t in by_layer["fact"]:
            sections.append(f"- `{t.name}` — {t.description} **Grain:** {t.grain}")

    return "\n".join(sections)


def format_table_detail_for_llm(name: str) -> str:
    """Full per-table reference: grain + usage hint + column list with types and descriptions."""
    t = get_table(name)
    lines: list[str] = [
        f"# `{t.name}` ({t.layer})",
        f"**Grain:** {t.grain}",
        f"**Description:** {t.description}",
    ]
    if t.primary_keys:
        lines.append(f"**Primary key:** {', '.join(t.primary_keys)}")
    if t.usage_hint:
        lines.append(f"**Usage hint:** {t.usage_hint}")
    if t.example_questions:
        lines.append("**Example questions:**")
        for q in t.example_questions:
            lines.append(f"  - {q}")
    lines.append("")
    lines.append("**Columns:**")
    for col in t.schema:
        nullable = "" if col.mode == "REQUIRED" else " NULL"
        desc = f" — {col.description}" if col.description else ""
        lines.append(f"  - `{col.name}` {col.field_type}{nullable}{desc}")
    return "\n".join(lines)


def column_names(name: str) -> list[str]:
    return [c.name for c in get_table(name).schema]


def agent_visible_table_names() -> set[str]:
    return {t.name for t in list_tables(agent_visible=True)}
