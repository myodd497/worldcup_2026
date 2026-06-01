"""
Orchestrator Agent — receives the user's WhatsApp message, classifies intent,
routes to the appropriate specialist agent via LangGraph, computes confidence,
and uses a dedicated result composer to format the final response.
"""
from __future__ import annotations

import json
import operator
from typing import Annotated, Any, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.agents.workflow_logger import get_tracker


# ── State schema ────────────────────────────────────────────────────────────

class OrchestratorState(TypedDict):
    user_id: str
    user_message: str
    intent: str
    selected_agent: str
    selected_agents: list[str]
    agent_outputs: dict[str, dict[str, Any]]
    agent_payload: dict[str, Any]
    confidence_score: float
    confidence_label: str
    confidence_reason: str
    final_reply: str
    messages: Annotated[list, operator.add]


# ── LLM ─────────────────────────────────────────────────────────────────────

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_INTENTS = ["news", "sentiment", "match_facts", "prediction", "chat"]
_AGENTS = ["news", "sentiment", "match_facts", "prediction", "chat"]

_INTENT_PROMPT = """\
You are a routing manager for a football assistant.

Task:
- Classify the user message into exactly one of: {intents}

Routing policy:
- `match_facts`: schedules, fixtures, results, standings, historical matches, lineups, venues, referees.
- `prediction`: user asks for forecast, probability, odds, who will win.
- `sentiment`: user asks fan sentiment, public opinion, social reaction.
- `news`: user asks for latest updates, headlines, rumours, transfers, injuries from media/web.
- `chat`: generic conversation, ambiguous football talk, greetings, or any request that does not clearly fit above.

Conversation context (earlier turns, if any):
{context}

User message: "{message}"

Reply with only one label and nothing else.
""".strip()


_CHAT_PROMPT = """\
You are a helpful football assistant.
Keep the reply conversational and concise.
If the user asks for data-heavy details (fixtures, results, standings, predictions), gently ask a short follow-up question.

Conversation context (earlier turns, if any):
{context}

User message: "{message}"
""".strip()

_AGENT_PLANNER_PROMPT = """\
You are a football assistant orchestrator planner.

Task:
- Select one or more specialist agents to answer the user request.
- Return only valid JSON: {"agents": ["..."]}

Available agents:
- news: latest media updates.
- sentiment: fan/social sentiment.
- match_facts: fixtures, results, standings, venues, referees, historical facts.
- prediction: win/draw/loss probabilities.
- chat: generic conversation.

Selection rules:
- Choose ALL agents that are useful for the question (not only one).
- If the user asks for prediction/forecast, include both prediction and match_facts.
- If the user asks for factual match data, include match_facts.
- Use chat only for clearly generic conversation.
- Keep between 1 and 3 agents.

Conversation context (earlier turns, if any):
{context}

User message: "{message}"
""".strip()


def _format_recent_history(messages: list[dict[str, str]], max_messages: int = 8, max_chars: int = 400) -> str:
    """Formats a bounded recent history window for prompts."""
    if not messages:
        return "None"

    recent = messages[-max_messages:]
    lines: list[str] = []
    for msg in recent:
        role = str(msg.get("role", "user")).strip().lower() or "user"
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        lines.append(f"{role}: {content[:max_chars]}")

    return "\n".join(lines) if lines else "None"


def _build_contextual_user_message(state: OrchestratorState) -> str:
    """Adds recent conversation context to help specialist agents resolve references."""
    context = _format_recent_history(state.get("messages", []), max_messages=6, max_chars=300)
    if context == "None":
        return state["user_message"]

    return (
        "Conversation context (earlier turns):\n"
        f"{context}\n\n"
        "Current user message:\n"
        f"{state['user_message']}\n\n"
        "Use the context to resolve references like 'that team' or 'the previous match'."
    )


# ── Nodes ────────────────────────────────────────────────────────────────────

def classify_intent(state: OrchestratorState) -> OrchestratorState:
    tracker = get_tracker()
    prompt = _INTENT_PROMPT.format(
        intents=", ".join(_INTENTS),
        context=_format_recent_history(state.get("messages", [])),
        message=state["user_message"],
    )
    intent = _llm.invoke(prompt).content.strip().lower()
    if intent not in _INTENTS:
        intent = "chat"

    result = {**state, "intent": intent}
    tracker.log_step(
        "classify",
        status="executed",
        input_data={"user_message": state["user_message"][:100]},
        output_data={"intent": intent},
    )
    return result


def route_request(state: OrchestratorState) -> OrchestratorState:
    """Planner node: selects one or more specialist agents."""
    tracker = get_tracker()

    fallback_mapping = {
        "news": ["news", "match_facts"],
        "sentiment": ["sentiment", "news"],
        "match_facts": ["match_facts"],
        "prediction": ["prediction", "match_facts"],
        "chat": ["chat"],
    }

    selected_agents = fallback_mapping.get(state["intent"], ["chat"])
    try:
        raw = _llm.invoke(
            _AGENT_PLANNER_PROMPT.format(
                context=_format_recent_history(state.get("messages", [])),
                message=state["user_message"],
            )
        ).content.strip()
        parsed = json.loads(raw)
        candidates = parsed.get("agents", []) if isinstance(parsed, dict) else []
        cleaned = [a for a in candidates if isinstance(a, str) and a in _AGENTS]
        if cleaned:
            selected_agents = cleaned[:3]
    except Exception:
        pass

    # Deduplicate while preserving order.
    deduped: list[str] = []
    for agent in selected_agents:
        if agent not in deduped:
            deduped.append(agent)

    selected_agents = deduped or ["chat"]
    result = {
        **state,
        "selected_agents": selected_agents,
        "selected_agent": "multi" if len(selected_agents) > 1 else selected_agents[0],
    }
    tracker.log_step(
        "router",
        status="executed",
        input_data={"intent": state["intent"]},
        output_data={"selected_agents": selected_agents},
    )
    return result


def _run_news(state: OrchestratorState) -> dict[str, Any]:
    from src.agents.news_agent import run_structured as run_news
    return run_news(_build_contextual_user_message(state))


def _run_sentiment(state: OrchestratorState) -> dict[str, Any]:
    from src.agents.sentiment_agent import run_structured as run_sentiment
    return run_sentiment(_build_contextual_user_message(state))


def _run_match_facts(state: OrchestratorState) -> dict[str, Any]:
    from src.agents.match_facts_agent import run_structured as run_facts
    return run_facts(_build_contextual_user_message(state))


def _run_prediction(state: OrchestratorState) -> dict[str, Any]:
    from src.agents.prediction_agent import run_structured as run_prediction
    return run_prediction(_build_contextual_user_message(state))


def _run_chat(state: OrchestratorState) -> dict[str, Any]:
    answer = _llm.invoke(
        _CHAT_PROMPT.format(
            message=state["user_message"],
            context=_format_recent_history(state.get("messages", [])),
        )
    ).content.strip()
    return {
        "answer": answer,
        "confidence_score": 0.7,
        "confidence_reason": "Conversational response from LLM manager path.",
    }


def execute_agents_node(state: OrchestratorState) -> OrchestratorState:
    tracker = get_tracker()
    runners = {
        "news": _run_news,
        "sentiment": _run_sentiment,
        "match_facts": _run_match_facts,
        "prediction": _run_prediction,
        "chat": _run_chat,
    }

    outputs: dict[str, dict[str, Any]] = {}
    for agent in state.get("selected_agents", ["chat"]):
        runner = runners.get(agent)
        if runner is None:
            continue
        try:
            payload = runner(state)
        except Exception as exc:
            payload = {
                "answer": f"{agent} agent failed: {exc}",
                "confidence_score": 0.2,
                "confidence_reason": f"{agent} execution failed.",
                "metadata": {"data_source": "error"},
            }
        outputs[agent] = payload

    if not outputs:
        outputs["chat"] = _run_chat(state)

    result = {**state, "agent_outputs": outputs}
    tracker.log_step(
        "agent_execution",
        status="executed",
        input_data={"selected_agents": state.get("selected_agents", [])},
        output_data={
            "executed_agents": list(outputs.keys()),
        },
    )
    return result


def _data_source_rank(payload: dict[str, Any]) -> int:
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    source = str(metadata.get("data_source", "")).lower()
    if source == "bigquery":
        return 3
    if "bigquery" in source:
        return 2
    if source == "api":
        return 1
    return 0


def _pick_primary_payload(agent_outputs: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    items = list(agent_outputs.items())
    ranked = sorted(
        items,
        key=lambda kv: (
            _data_source_rank(kv[1]),
            float(kv[1].get("confidence_score", 0.0) or 0.0),
        ),
        reverse=True,
    )
    if not ranked:
        return "chat", {"answer": "I could not generate an answer.", "confidence_score": 0.2}
    return ranked[0]


def aggregate_outputs_node(state: OrchestratorState) -> OrchestratorState:
    tracker = get_tracker()
    outputs = state.get("agent_outputs", {})
    primary_agent, primary_payload = _pick_primary_payload(outputs)

    bullets: list[str] = []
    for agent, payload in outputs.items():
        answer = str(payload.get("answer", "")).strip()
        source = str((payload.get("metadata", {}) or {}).get("data_source", "unknown"))
        if not answer:
            continue
        bullets.append(f"[{agent} | source={source}]\n{answer}")

    synthesis_prompt = (
        "You are an orchestration synthesizer for a football assistant.\n"
        "Combine agent outputs into one coherent response.\n"
        "Critical rule: when there is BigQuery-backed output, treat it as source of truth for factual data.\n"
        "Use other sources only as complementary context.\n"
        "If sources conflict, follow BigQuery-backed facts and mention uncertainty briefly.\n"
        "Return concise markdown suitable for chat.\n\n"
        f"User question: {state['user_message']}\n"
        f"Primary source agent: {primary_agent}\n"
        f"Primary payload data source: {(primary_payload.get('metadata', {}) or {}).get('data_source', 'unknown')}\n\n"
        "Agent outputs:\n"
        + "\n\n".join(bullets)
    )

    try:
        merged_answer = _llm.invoke(synthesis_prompt).content.strip()
    except Exception:
        merged_answer = str(primary_payload.get("answer", "I could not generate an answer."))

    confidence_values = [
        float((p or {}).get("confidence_score", 0.0) or 0.0)
        for p in outputs.values()
    ]
    aggregate_score = max(confidence_values) if confidence_values else 0.4
    if len(confidence_values) > 1:
        aggregate_score = max(aggregate_score, sum(confidence_values) / len(confidence_values))

    data_sources = {
        agent: str(((payload or {}).get("metadata", {}) or {}).get("data_source", "unknown"))
        for agent, payload in outputs.items()
    }

    payload = {
        "answer": merged_answer,
        "confidence_score": max(0.0, min(1.0, aggregate_score)),
        "confidence_reason": (
            "Final answer synthesized from multiple agents with BigQuery-priority source selection."
        ),
        "metadata": {
            "has_match": True,
            "data_source": str((primary_payload.get("metadata", {}) or {}).get("data_source", "unknown")),
            "primary_agent": primary_agent,
            "agents_used": list(outputs.keys()),
            "agent_sources": data_sources,
        },
    }

    tracker.log_step(
        "aggregate",
        status="executed",
        input_data={"agents": list(outputs.keys())},
        output_data={
            "primary_agent": primary_agent,
            "primary_data_source": payload["metadata"]["data_source"],
        },
    )
    return {**state, "agent_payload": payload, "selected_agent": primary_agent}


def score_confidence(state: OrchestratorState) -> OrchestratorState:
    """Normalises confidence signal from specialist outputs into shared state."""
    tracker = get_tracker()
    payload = state.get("agent_payload", {})
    raw_score = payload.get("confidence_score", 0.6)
    score = max(0.0, min(1.0, float(raw_score)))

    if score >= 0.8:
        label = "high"
    elif score >= 0.55:
        label = "medium"
    else:
        label = "low"

    reason = payload.get("confidence_reason", "Confidence estimated from data coverage and source quality.")
    result = {
        **state,
        "confidence_score": score,
        "confidence_label": label,
        "confidence_reason": reason,
    }
    tracker.log_step(
        "confidence",
        status="executed",
        input_data={"raw_score": raw_score},
        output_data={
            "confidence_score": score,
            "confidence_label": label,
            "confidence_reason": reason[:50],
        },
    )
    return result


def compose_reply(state: OrchestratorState) -> OrchestratorState:
    """Final node: uses result composer to format a WhatsApp-friendly reply."""
    from src.agents.result_composer_agent import compose
    from src.agents.docs_agent import log_session
    
    tracker = get_tracker()

    final_reply = compose(
        user_message=state["user_message"],
        intent=state["intent"],
        selected_agent=state["selected_agent"],
        payload=state["agent_payload"],
        confidence_score=state["confidence_score"],
        confidence_label=state["confidence_label"],
        confidence_reason=state["confidence_reason"],
    )
    
    log_session(state["user_id"], state["user_message"], final_reply)
    
    tracker.log_step(
        "compose",
        status="executed",
        input_data={"confidence_label": state["confidence_label"]},
        output_data={"final_reply": final_reply[:50]},
    )
    
    return {**state, "final_reply": final_reply}


# ── Graph ────────────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    g = StateGraph(OrchestratorState)

    g.add_node("classify", classify_intent)
    g.add_node("router", route_request)
    g.add_node("execute_agents", execute_agents_node)
    g.add_node("aggregate", aggregate_outputs_node)
    g.add_node("confidence", score_confidence)
    g.add_node("compose", compose_reply)

    g.set_entry_point("classify")
    g.add_edge("classify", "router")
    g.add_edge("router", "execute_agents")
    g.add_edge("execute_agents", "aggregate")
    g.add_edge("aggregate", "confidence")
    g.add_edge("confidence", "compose")
    g.add_edge("compose", END)

    return g.compile()


_graph = _build_graph()


# ── Public API ───────────────────────────────────────────────────────────────

async def run_orchestrator(
    user_message: str,
    user_id: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    from src.agents.workflow_logger import reset_tracker
    
    # Reset tracker for this new message
    reset_tracker()
    
    initial_state: OrchestratorState = {
        "user_id": user_id,
        "user_message": user_message,
        "intent": "",
        "selected_agent": "",
        "selected_agents": [],
        "agent_outputs": {},
        "agent_payload": {},
        "confidence_score": 0.0,
        "confidence_label": "low",
        "confidence_reason": "",
        "final_reply": "",
        "messages": conversation_history or [],
    }
    result = _graph.invoke(initial_state)
    return result["final_reply"]
