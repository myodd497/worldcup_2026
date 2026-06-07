"""Deterministic entity resolution for teams, players, and competitions.

The BigQuery agent should NEVER write `LOWER(team_name) LIKE '%portugal%'`.
It should call this resolver, get back a `team_id` (or a disambiguation list),
and then write `WHERE team_id = <id>`. This eliminates a huge class of
"nonsense reply" failures caused by name ambiguity (e.g. "Korea" matches both
Korea Republic and Korea DPR; "Mexico" can match women's teams in mixed data).

Public API:
  - resolve_team(name)         → ResolvedEntity
  - resolve_player(name)       → ResolvedEntity
  - resolve_team_tool(name)    → JSON string (LLM-callable)
  - resolve_player_tool(name)  → JSON string (LLM-callable)
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Literal

from src.data.datamodel.catalog import fqn
from src.tools.bigquery_tools import run_query

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedEntity:
    kind: Literal["team", "player"]
    query: str
    matched: bool
    id: int | None
    name: str | None
    confidence: float
    alternatives: tuple[dict, ...]
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["alternatives"] = list(d["alternatives"])
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


# ─────────────────────────────────────────────────────────────────────────────
# Team resolver
# ─────────────────────────────────────────────────────────────────────────────

# Common nicknames / colloquialisms → canonical team_name fragments to match.
_TEAM_ALIASES: dict[str, str] = {
    "usa": "united states",
    "us": "united states",
    "states": "united states",
    "korea": "korea republic",
    "south korea": "korea republic",
    "north korea": "korea dpr",
    "ivory coast": "cote d ivoire",
    "england": "england",
    "holland": "netherlands",
    "the netherlands": "netherlands",
    "czech": "czech republic",
    "czechia": "czech republic",
    "uae": "united arab emirates",
    "drc": "congo dr",
    "dr congo": "congo dr",
    "iran": "iran",
    "russia": "russia",
}


@lru_cache(maxsize=1)
def _load_teams() -> tuple[dict, ...]:
    """Load all teams from dim_team once per process."""
    sql = f"SELECT team_id, team_name, is_wc2026_participant FROM {fqn('dim_team')}"
    try:
        df = run_query(sql)
    except Exception as exc:
        logger.warning("entity_resolver: failed to load dim_team: %s", exc)
        return tuple()
    rows: list[dict] = []
    for r in df.to_dict(orient="records"):
        rows.append({
            "team_id": int(r["team_id"]),
            "team_name": str(r.get("team_name") or ""),
            "is_wc2026_participant": bool(r.get("is_wc2026_participant") or False),
            "_norm_name": _norm(r.get("team_name") or ""),
        })
    return tuple(rows)


def resolve_team(name: str) -> ResolvedEntity:
    """Resolve a team name → team_id with confidence and alternatives."""
    query = (name or "").strip()
    if not query:
        return ResolvedEntity("team", query, False, None, None, 0.0, tuple(),
                              "Empty query.")

    q = _norm(query)
    q = _TEAM_ALIASES.get(q, q)

    teams = _load_teams()
    if not teams:
        return ResolvedEntity("team", query, False, None, None, 0.0, tuple(),
                              "dim_team is empty or unreachable.")

    # Score each team: exact > prefix > substring > fuzzy. Boost WC2026 participants.
    scored: list[tuple[float, dict]] = []
    for t in teams:
        nname = t["_norm_name"]
        score = 0.0
        if not nname:
            continue
        if q == nname:
            score = 1.0
        elif nname.startswith(q):
            score = 0.92
        elif q in nname.split():
            score = 0.88
        elif q in nname:
            score = 0.80
        else:
            sim = _similarity(q, nname)
            if sim >= 0.75:
                score = sim * 0.85
        if score > 0:
            if t["is_wc2026_participant"]:
                score = min(1.0, score + 0.03)
            scored.append((score, t))

    if not scored:
        return ResolvedEntity("team", query, False, None, None, 0.0, tuple(),
                              f"No team matched '{query}'.")

    scored.sort(key=lambda kv: kv[0], reverse=True)
    best_score, best = scored[0]
    alts = tuple(
        {"team_id": t["team_id"], "team_name": t["team_name"], "score": round(s, 3)}
        for s, t in scored[1:4]
    )

    # Ambiguity check: if the second-best is within 0.05 of the best, flag it.
    note = ""
    if len(scored) > 1 and (best_score - scored[1][0]) < 0.05 and best_score < 0.99:
        note = "Ambiguous: top matches are very close. Consider asking the user to disambiguate."

    return ResolvedEntity(
        kind="team",
        query=query,
        matched=True,
        id=best["team_id"],
        name=best["team_name"],
        confidence=round(min(best_score, 1.0), 3),
        alternatives=alts,
        note=note,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Player resolver
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_players() -> tuple[dict, ...]:
    """Load all players from dim_player once per process."""
    sql = f"""
        SELECT player_id, player_name, primary_team_id, primary_team_name, is_wc2026_participant
        FROM {fqn('dim_player')}
    """
    try:
        df = run_query(sql)
    except Exception as exc:
        logger.warning("entity_resolver: failed to load dim_player: %s", exc)
        return tuple()
    rows: list[dict] = []
    for r in df.to_dict(orient="records"):
        rows.append({
            "player_id": int(r["player_id"]),
            "player_name": str(r.get("player_name") or ""),
            "team_id": int(r["primary_team_id"]) if r.get("primary_team_id") is not None else None,
            "team_name": str(r.get("primary_team_name") or ""),
            "is_wc2026_participant": bool(r.get("is_wc2026_participant") or False),
            "_norm_name": _norm(r.get("player_name") or ""),
        })
    return tuple(rows)


def resolve_player(name: str) -> ResolvedEntity:
    """Resolve a player name → player_id with confidence and alternatives."""
    query = (name or "").strip()
    if not query:
        return ResolvedEntity("player", query, False, None, None, 0.0, tuple(),
                              "Empty query.")

    q = _norm(query)
    players = _load_players()
    if not players:
        return ResolvedEntity("player", query, False, None, None, 0.0, tuple(),
                              "dim_player is empty or unreachable.")

    scored: list[tuple[float, dict]] = []
    for p in players:
        nname = p["_norm_name"]
        if not nname:
            continue
        score = 0.0
        if q == nname:
            score = 1.0
        elif nname.endswith(" " + q) or nname.startswith(q + " "):
            score = 0.93
        elif q in nname.split():
            score = 0.86
        elif q in nname:
            score = 0.78
        else:
            sim = _similarity(q, nname)
            if sim >= 0.80:
                score = sim * 0.85
        if score > 0:
            if p["is_wc2026_participant"]:
                score = min(1.0, score + 0.05)
            scored.append((score, p))

    if not scored:
        return ResolvedEntity("player", query, False, None, None, 0.0, tuple(),
                              f"No player matched '{query}'.")

    scored.sort(key=lambda kv: kv[0], reverse=True)
    best_score, best = scored[0]
    alts = tuple(
        {
            "player_id": p["player_id"],
            "player_name": p["player_name"],
            "team_name": p["team_name"],
            "score": round(s, 3),
        }
        for s, p in scored[1:5]
    )

    note = ""
    if len(scored) > 1 and (best_score - scored[1][0]) < 0.05 and best_score < 0.99:
        note = "Ambiguous: multiple players match. Consider showing alternatives or asking for the team."

    return ResolvedEntity(
        kind="player",
        query=query,
        matched=True,
        id=best["player_id"],
        name=best["player_name"],
        confidence=round(min(best_score, 1.0), 3),
        alternatives=alts,
        note=note,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM-callable tool wrappers
# ─────────────────────────────────────────────────────────────────────────────

def resolve_team_tool(name: str) -> str:
    """JSON string for LLM consumption."""
    return json.dumps(resolve_team(name).to_dict(), default=str)


def resolve_player_tool(name: str) -> str:
    return json.dumps(resolve_player(name).to_dict(), default=str)


def clear_cache() -> None:
    """For tests / hot-reload."""
    _load_teams.cache_clear()
    _load_players.cache_clear()
