"""Conversation memory — rolling LLM summary + structured entity store.

Replaces the naive "last 6 turns" window. Two layers:

1. **Rolling summary** (LLM-compressed). When history > N turns, the older
   turns are summarised into a single paragraph. New turns are appended raw.
2. **Entity store** (deterministic). Tracks last-mentioned teams, players,
   competitions, season, and the last topic. This lets follow-up questions
   like "and their last 5?" resolve "their" without LLM guesswork.

Public API:
  - ConversationMemory.update(user_msg, assistant_msg, entities)
  - ConversationMemory.context_block() → str (drop into agent prompts)
  - ConversationMemory.entity_hint() → dict (passed to planner)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI

from src.agents.llm_config import create_chat_model


_RAW_TURNS_KEPT = 4               # always show the last 4 raw turns
_SUMMARISE_AFTER = 8              # compress older turns once history exceeds this


_summary_llm: ChatOpenAI | None = None


def _llm() -> ChatOpenAI:
    global _summary_llm
    if _summary_llm is None:
        _summary_llm = create_chat_model("simple", temperature=0)
    return _summary_llm


# ─────────────────────────────────────────────────────────────────────────────
# Entity store
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EntityStore:
    teams: list[dict] = field(default_factory=list)   # [{id, name}]
    players: list[dict] = field(default_factory=list)
    season_year: int | None = None
    competition_id: int | None = None
    last_topic: str = ""

    def merge(self, other: "EntityStore") -> None:
        # Prepend new entities so the most recent are first.
        for t in other.teams:
            if t and t not in self.teams:
                self.teams.insert(0, t)
        for p in other.players:
            if p and p not in self.players:
                self.players.insert(0, p)
        if other.season_year is not None:
            self.season_year = other.season_year
        if other.competition_id is not None:
            self.competition_id = other.competition_id
        if other.last_topic:
            self.last_topic = other.last_topic
        # Cap to avoid unbounded growth.
        self.teams = self.teams[:6]
        self.players = self.players[:6]

    def to_dict(self) -> dict:
        return {
            "teams": self.teams,
            "players": self.players,
            "season_year": self.season_year,
            "competition_id": self.competition_id,
            "last_topic": self.last_topic,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Conversation memory
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConversationMemory:
    rolling_summary: str = ""
    raw_turns: list[dict[str, str]] = field(default_factory=list)
    entities: EntityStore = field(default_factory=EntityStore)

    # ── lifecycle ─────────────────────────────────────────────────────────

    @classmethod
    def from_messages(cls, messages: list[dict[str, str]] | None) -> "ConversationMemory":
        """Construct from a list of turns coming from the chat surface."""
        mem = cls()
        if not messages:
            return mem
        mem.raw_turns = list(messages)
        # Heuristic seed entities from prior turns (cheap; LLM resolver runs later).
        mem.entities = _extract_entities_from_text(
            " ".join(m.get("content", "") for m in messages)
        )
        return mem

    def append_user(self, message: str) -> None:
        self.raw_turns.append({"role": "user", "content": message})

    def append_assistant(self, message: str) -> None:
        self.raw_turns.append({"role": "assistant", "content": message})
        self._maybe_summarise()

    def update_entities(self, **kwargs: Any) -> None:
        new = EntityStore(
            teams=list(kwargs.get("teams") or []),
            players=list(kwargs.get("players") or []),
            season_year=kwargs.get("season_year"),
            competition_id=kwargs.get("competition_id"),
            last_topic=str(kwargs.get("last_topic") or ""),
        )
        self.entities.merge(new)

    # ── exposure ──────────────────────────────────────────────────────────

    def context_block(self) -> str:
        """A compact block to drop into any agent prompt."""
        parts: list[str] = []
        if self.rolling_summary:
            parts.append("**Earlier-conversation summary:** " + self.rolling_summary)
        if self.raw_turns:
            recent = self.raw_turns[-_RAW_TURNS_KEPT:]
            lines = [
                f"{m.get('role','user')}: {str(m.get('content',''))[:400]}"
                for m in recent if (m.get("content") or "").strip()
            ]
            if lines:
                parts.append("**Recent turns:**\n" + "\n".join(lines))
        ent = self.entities.to_dict()
        if any(ent.values()):
            parts.append("**Known entities:** " + json.dumps(ent, default=str))
        return "\n\n".join(parts) if parts else "None"

    def entity_hint(self) -> dict:
        return self.entities.to_dict()

    # ── internals ─────────────────────────────────────────────────────────

    def _maybe_summarise(self) -> None:
        if len(self.raw_turns) <= _SUMMARISE_AFTER:
            return
        older = self.raw_turns[:-_RAW_TURNS_KEPT]
        if not older:
            return
        try:
            new_summary = _summarise_turns(older, prior=self.rolling_summary)
        except Exception:
            return  # never break the chat path because of summarisation
        self.rolling_summary = new_summary
        self.raw_turns = self.raw_turns[-_RAW_TURNS_KEPT:]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _summarise_turns(turns: list[dict[str, str]], *, prior: str) -> str:
    transcript = "\n".join(
        f"{t.get('role','user')}: {str(t.get('content','')).strip()[:600]}"
        for t in turns if (t.get("content") or "").strip()
    )
    prompt = (
        "Summarise the following football-assistant conversation in 4-6 short bullets. "
        "Capture: teams/players discussed, time window, key facts the user already learned, "
        "and any unresolved follow-ups. Be concrete with names and numbers.\n\n"
        f"Prior summary (carry forward if still relevant):\n{prior or 'None'}\n\n"
        f"New transcript:\n{transcript}"
    )
    try:
        return _llm().invoke(prompt).content.strip()
    except Exception:
        return prior or ""


_YEAR_RE = re.compile(r"\b(20\d{2})\b")
# Lightweight heuristic — proper resolution is done by entity_resolver tool.
_CAPITALISED_RE = re.compile(r"\b([A-Z][a-zA-Z]{2,}(?:\s[A-Z][a-zA-Z]+)?)\b")


def _extract_entities_from_text(text: str) -> EntityStore:
    season = None
    m = _YEAR_RE.search(text or "")
    if m:
        season = int(m.group(1))
    # We intentionally do NOT pre-resolve names here — the planner/agent calls
    # entity_resolver per turn. This keeps memory I/O-free.
    return EntityStore(season_year=season)
