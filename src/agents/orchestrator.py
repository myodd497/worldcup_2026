"""
Orchestrator Agent — receives the user's WhatsApp message, classifies intent,
routes to the appropriate specialist agent via LangGraph, computes confidence,
and uses a dedicated result composer to format the final response.
"""
from __future__ import annotations

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
    agent_payload: dict[str, Any]
    confidence_score: float
    confidence_label: str
    confidence_reason: str
    final_reply: str
    messages: Annotated[list, operator.add]


# ── LLM ─────────────────────────────────────────────────────────────────────

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_INTENTS = ["news", "sentiment", "match_facts", "prediction", "other"]

_INTENT_PROMPT = """\
Classify the following user message into exactly one of these intents:
{intents}

User message: "{message}"

Reply with only the intent label, nothing else.
""".strip()


# ── Nodes ────────────────────────────────────────────────────────────────────

def classify_intent(state: OrchestratorState) -> OrchestratorState:
    tracker = get_tracker()
    prompt = _INTENT_PROMPT.format(
        intents=", ".join(_INTENTS),
        message=state["user_message"],
    )
    intent = _llm.invoke(prompt).content.strip().lower()
    if intent not in _INTENTS:
        intent = "other"
    result = {**state, "intent": intent}
    tracker.log_step(
        "classify",
        status="executed",
        input_data={"user_message": state["user_message"][:100]},
        output_data={"intent": intent},
    )
    return result


def route_request(state: OrchestratorState) -> OrchestratorState:
    """Router node: maps intent to specialist agent for execution."""
    tracker = get_tracker()
    mapping = {
        "news": "news",
        "sentiment": "sentiment",
        "match_facts": "match_facts",
        "prediction": "prediction",
        "other": "other",
    }
    selected = mapping.get(state["intent"], "other")
    result = {**state, "selected_agent": selected}
    tracker.log_step(
        "router",
        status="executed",
        input_data={"intent": state["intent"]},
        output_data={"selected_agent": selected},
    )
    return result


def route_to_agent(state: OrchestratorState) -> str:
    return state["selected_agent"]


def news_node(state: OrchestratorState) -> OrchestratorState:
    from src.agents.news_agent import run_structured as run_news
    tracker = get_tracker()
    payload = run_news(state["user_message"])
    result = {**state, "agent_payload": payload}
    tracker.log_step(
        "news_agent",
        status="executed",
        input_data={"user_message": state["user_message"][:100]},
        output_data={
            "answer": payload.get("answer", "")[:50],
            "confidence_score": payload.get("confidence_score"),
        },
    )
    return result


def sentiment_node(state: OrchestratorState) -> OrchestratorState:
    from src.agents.sentiment_agent import run_structured as run_sentiment
    tracker = get_tracker()
    payload = run_sentiment(state["user_message"])
    result = {**state, "agent_payload": payload}
    tracker.log_step(
        "sentiment_agent",
        status="executed",
        input_data={"user_message": state["user_message"][:100]},
        output_data={
            "answer": payload.get("answer", "")[:50],
            "confidence_score": payload.get("confidence_score"),
        },
    )
    return result


def match_facts_node(state: OrchestratorState) -> OrchestratorState:
    from src.agents.match_facts_agent import run_structured as run_facts
    tracker = get_tracker()
    payload = run_facts(state["user_message"])
    result = {**state, "agent_payload": payload}
    tracker.log_step(
        "match_facts_agent",
        status="executed",
        input_data={"user_message": state["user_message"][:100]},
        output_data={
            "answer": payload.get("answer", "")[:50],
            "confidence_score": payload.get("confidence_score"),
        },
    )
    return result


def prediction_node(state: OrchestratorState) -> OrchestratorState:
    from src.agents.prediction_agent import run_structured as run_prediction
    tracker = get_tracker()
    payload = run_prediction(state["user_message"])
    result = {**state, "agent_payload": payload}
    tracker.log_step(
        "prediction_agent",
        status="executed",
        input_data={"user_message": state["user_message"][:100]},
        output_data={
            "answer": payload.get("answer", "")[:50],
            "confidence_score": payload.get("confidence_score"),
        },
    )
    return result


def other_node(state: OrchestratorState) -> OrchestratorState:
    tracker = get_tracker()
    result = {
        **state,
        "agent_payload": {
            "answer": "I can help with: news, sentiment, match facts, and predictions. Try asking about a specific game!",
            "confidence_score": 0.75,
            "confidence_reason": "General guidance response.",
        },
    }
    tracker.log_step(
        "other_agent",
        status="executed",
        input_data={"user_message": state["user_message"][:100]},
        output_data={
            "answer": result["agent_payload"]["answer"][:50],
            "confidence_score": result["agent_payload"]["confidence_score"],
        },
    )
    return result


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
    
    # Append workflow summary to the final reply (in debug/detail section)
    workflow_summary = tracker.get_summary()
    full_reply_with_workflow = f"{final_reply}\n\n{workflow_summary}"
    
    log_session(state["user_id"], state["user_message"], final_reply)
    
    tracker.log_step(
        "compose",
        status="executed",
        input_data={"confidence_label": state["confidence_label"]},
        output_data={"final_reply": final_reply[:50]},
    )
    
    return {**state, "final_reply": full_reply_with_workflow}


# ── Graph ────────────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    g = StateGraph(OrchestratorState)

    g.add_node("classify", classify_intent)
    g.add_node("router", route_request)
    g.add_node("news", news_node)
    g.add_node("sentiment", sentiment_node)
    g.add_node("match_facts", match_facts_node)
    g.add_node("prediction", prediction_node)
    g.add_node("other", other_node)
    g.add_node("confidence", score_confidence)
    g.add_node("compose", compose_reply)

    g.set_entry_point("classify")
    g.add_edge("classify", "router")
    g.add_conditional_edges(
        "router",
        route_to_agent,
        {
            "news": "news",
            "sentiment": "sentiment",
            "match_facts": "match_facts",
            "prediction": "prediction",
            "other": "other",
        },
    )
    for node in ["news", "sentiment", "match_facts", "prediction", "other"]:
        g.add_edge(node, "confidence")
    g.add_edge("confidence", "compose")
    g.add_edge("compose", END)

    return g.compile()


_graph = _build_graph()


# ── Public API ───────────────────────────────────────────────────────────────

async def run_orchestrator(user_message: str, user_id: str) -> str:
    from src.agents.workflow_logger import reset_tracker
    
    # Reset tracker for this new message
    reset_tracker()
    
    initial_state: OrchestratorState = {
        "user_id": user_id,
        "user_message": user_message,
        "intent": "",
        "selected_agent": "",
        "agent_payload": {},
        "confidence_score": 0.0,
        "confidence_label": "low",
        "confidence_reason": "",
        "final_reply": "",
        "messages": [],
    }
    result = _graph.invoke(initial_state)
    return result["final_reply"]
