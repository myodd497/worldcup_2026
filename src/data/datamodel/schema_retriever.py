"""Schema retriever — returns only the 3-5 tables most relevant to a question.

Replaces dumping the entire catalog into every system prompt.

Strategy:
  1. Embedding-based semantic match (text-embedding-3-small) over a rich
     per-table document (description + usage_hint + example questions + column names).
  2. Keyword + topic-boost score as a tiebreaker and as a full fallback when
     embeddings are unavailable (no network / no API key).

Public API:
  - search_schema(question, top_k=5) → list[TableSpec]
  - search_schema_tool(question, top_k=5) → markdown string for LLM
"""
from __future__ import annotations

import re
from collections import Counter

from src.agents.embeddings import cosine, embed, embed_one
from src.data.datamodel.catalog import (
    TableSpec,
    format_table_detail_for_llm,
    list_tables,
)


_STOPWORDS = {
    "what", "which", "who", "when", "where", "how", "why", "the", "a", "an",
    "and", "or", "of", "to", "for", "in", "on", "with", "by", "from", "is",
    "are", "was", "were", "has", "have", "had", "do", "does", "did", "be",
    "been", "this", "that", "these", "those", "it", "its", "their", "show",
    "tell", "give", "list", "all", "any", "me", "us", "you", "team", "teams",
    "player", "players", "game", "games", "match", "matches", "between",
    "last", "next", "most", "best", "top", "more", "less", "than", "vs",
}

# Topic → preferred tables (boosts). Keeps retrieval intent-aware without an LLM.
_TOPIC_BOOSTS: dict[str, tuple[str, ...]] = {
    # player stats
    "goal": ("fact_player_match_stat", "fact_match_event", "mart_team_profile"),
    "assist": ("fact_player_match_stat",),
    "contribution": ("fact_player_match_stat",),
    "scorer": ("fact_player_match_stat", "dim_player"),
    "scored": ("fact_player_match_stat", "fact_match_team"),
    "yellow": ("fact_player_match_stat", "fact_match_event"),
    "red": ("fact_player_match_stat", "fact_match_event"),
    "card": ("fact_player_match_stat",),
    "discipline": ("fact_player_match_stat",),
    "minute": ("fact_player_match_stat",),
    "minutes": ("fact_player_match_stat",),
    "played": ("fact_player_match_stat", "fact_match"),
    "rating": ("fact_player_match_stat",),
    "passes": ("fact_player_match_stat",),
    "pass": ("fact_player_match_stat",),
    "save": ("fact_player_match_stat",),
    "saves": ("fact_player_match_stat",),
    "tackle": ("fact_player_match_stat",),
    "interception": ("fact_player_match_stat",),
    "goalkeeper": ("dim_player", "fact_player_match_stat"),
    "captain": ("fact_player_match_stat", "dim_player"),

    # team stats
    "form": ("mart_team_form", "fact_match_team"),
    "streak": ("mart_team_form",),
    "recent": ("mart_team_form", "fact_match_team", "mart_match_history"),
    "possession": ("fact_match_team",),
    "shot": ("fact_match_team",),
    "shots": ("fact_match_team",),
    "target": ("fact_match_team",),
    "conceded": ("fact_match_team", "mart_team_profile"),
    "defense": ("mart_team_profile", "fact_match_team"),
    "defence": ("mart_team_profile", "fact_match_team"),
    "attack": ("mart_team_profile", "fact_match_team"),
    "xg": ("fact_match_team",),
    "clean": ("mart_team_profile", "fact_match_team"),
    "win": ("mart_team_profile", "mart_team_form", "fact_match_team"),
    "loss": ("mart_team_profile", "mart_team_form", "fact_match_team"),
    "draw": ("mart_team_profile", "mart_team_form", "fact_match_team"),

    # fixtures / schedule
    "fixture": ("mart_match_upcoming", "mart_match_history", "fact_match"),
    "schedule": ("mart_match_upcoming",),
    "upcoming": ("mart_match_upcoming",),
    "next": ("mart_match_upcoming",),
    "previous": ("mart_match_history",),
    "history": ("mart_match_history", "fact_match"),
    "h2h": ("mart_head_to_head",),
    "head": ("mart_head_to_head",),
    "venue": ("dim_venue", "fact_match"),
    "stadium": ("dim_venue", "fact_match"),
    "referee": ("fact_match",),
    "kickoff": ("fact_match", "mart_match_upcoming"),
    "result": ("mart_match_history", "fact_match", "fact_match_team"),

    # tournament
    "group": ("mart_tournament_state", "fact_standings_snapshot"),
    "standings": ("fact_standings_snapshot", "mart_tournament_state"),
    "table": ("fact_standings_snapshot", "mart_tournament_state"),
    "qualify": ("dim_team", "fact_standings_snapshot"),
    "participant": ("dim_team",),
    "participants": ("dim_team",),
    "wc2026": ("dim_team", "mart_tournament_state"),
    "world cup": ("dim_competition", "dim_team"),

    # players / teams entities
    "name": ("dim_player", "dim_team"),
    "country": ("dim_team",),
    "competition": ("dim_competition",),
    "season": ("dim_competition", "fact_match"),
    "event": ("fact_match_event",),
    "events": ("fact_match_event",),
}


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z][a-z0-9\-]+", (text or "").lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


def _column_terms(spec: TableSpec) -> set[str]:
    out: set[str] = set()
    for col in spec.schema:
        for part in re.split(r"[_\s]+", col.name.lower()):
            if len(part) > 2 and part not in _STOPWORDS:
                out.add(part)
    return out


def _score_table(spec: TableSpec, q_tokens: list[str], q_text: str) -> float:
    score = 0.0
    col_terms = _column_terms(spec)
    # description / hint / example word overlap
    bag = " ".join([
        spec.description or "",
        spec.usage_hint or "",
        spec.grain or "",
        " ".join(spec.example_questions),
    ]).lower()
    bag_terms = set(_tokens(bag))

    counts = Counter(q_tokens)
    for tok, n in counts.items():
        if tok in col_terms:
            score += 2.0 * n
        if tok in bag_terms:
            score += 1.2 * n
        if tok == spec.name.replace("_", "").lower():
            score += 5.0
        if tok in spec.name.lower():
            score += 0.8

    # Topic boosts (multi-word topics check the raw question)
    for topic, tables in _TOPIC_BOOSTS.items():
        if topic in q_text.lower() and spec.name in tables:
            score += 2.5

    # Layer preference: marts > facts > dims (start with aggregated answer if possible)
    if spec.layer == "mart":
        score += 0.6
    elif spec.layer == "fact":
        score += 0.3

    return score


def search_schema(question: str, top_k: int = 5) -> list[TableSpec]:
    """Return the top-k tables most relevant to the question.

    Hybrid scorer: semantic (embeddings) + keyword/topic boosts. Keyword score
    serves as both the tiebreaker and the full fallback when embeddings are
    unavailable.
    """
    q_tokens = _tokens(question)
    q_text = (question or "").lower()
    tables = list_tables(agent_visible=True)

    # Keyword-only scores.
    kw_scores = {spec.name: _score_table(spec, q_tokens, q_text) for spec in tables}

    # Semantic scores (no-op if embeddings unavailable).
    q_vec = embed_one((question or "").strip())
    semantic: dict[str, float] = {}
    if q_vec is not None:
        table_docs = [_table_doc(t) for t in tables]
        table_vecs = embed(table_docs)
        for spec, vec in zip(tables, table_vecs):
            semantic[spec.name] = cosine(q_vec, vec) if vec is not None else 0.0

    def _final(spec: TableSpec) -> float:
        # Semantic dominates when available; keyword is the tiebreaker.
        s = semantic.get(spec.name, 0.0)
        k = kw_scores.get(spec.name, 0.0)
        # Normalise keyword score to roughly [0,1] then blend.
        k_norm = min(1.0, k / 10.0)
        if semantic:
            return 0.75 * s + 0.25 * k_norm
        return k_norm

    scored = sorted(((spec, _final(spec)) for spec in tables), key=lambda kv: kv[1], reverse=True)
    selected = [s for s, sc in scored if sc > 0][:top_k]
    # Always include dim_team if the user likely names a country.
    if not any(s.name == "dim_team" for s in selected):
        for spec in tables:
            if spec.name == "dim_team":
                selected.append(spec)
                break
    return selected[:top_k + 1]


def _table_doc(spec: TableSpec) -> str:
    """Rich per-table document used for semantic matching."""
    cols = ", ".join(c.name for c in spec.schema)
    examples = " | ".join(spec.example_questions) if spec.example_questions else ""
    return (
        f"Table {spec.name} ({spec.layer}). Grain: {spec.grain}. "
        f"{spec.description} {spec.usage_hint} "
        f"Columns: {cols}. Example questions: {examples}"
    )


def search_schema_tool(question: str, top_k: int = 5) -> str:
    """Markdown bundle of the most relevant tables — drop into the prompt."""
    tables = search_schema(question, top_k=top_k)
    if not tables:
        return "_(no relevant tables found)_"
    parts = [f"## Relevant tables (top {len(tables)})"]
    for spec in tables:
        parts.append("")
        parts.append(format_table_detail_for_llm(spec.name))
    return "\n".join(parts)
