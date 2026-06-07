"""
Orchestrator — single LangGraph pipeline for the football assistant.

Pipeline (5 nodes):
  plan → execute → verify → compose → END

- `plan`     : one gpt-4o call selects 1-2 specialists + topic + verifier flag
- `execute`  : runs the chosen specialists (sequentially); each returns {answer, confidence, metadata}
- `verify`   : when warranted (and there's bigquery output), gpt-4o critic scores groundedness
               and can request ONE repair from the bigquery agent with a structured hint
- `compose`  : gpt-4o-mini formats the final WhatsApp-friendly reply with confidence stars

Conversation context is supplied via `ConversationMemory` (rolling LLM summary
+ structured entity store). The naive "last N raw turns" window is gone.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.agents.conversation_memory import ConversationMemory
from src.agents.workflow_logger import get_tracker


# ── State ───────────────────────────────────────────────────────────────────

class OrchestratorState(TypedDict):
    user_id: str
    user_message: str
    conversation_context: str
    topic: str
    selected_agent: str
    selected_agents: list[str]
    needs_verifier: bool
    agent_outputs: dict[str, dict[str, Any]]
    agent_payload: dict[str, Any]
    verifier_verdict: dict[str, Any]
    confidence_score: float
    confidence_label: str
    confidence_reason: str
    final_reply: str
    messages: Annotated[list, operator.add]


_chat_llm: ChatOpenAI | None = None


def _chat() -> ChatOpenAI:
    """Cheap chat LLM for the small-talk specialist (chitchat only)."""
    global _chat_llm
    if _chat_llm is None:
        _chat_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    return _chat_llm


_CHAT_PROMPT = """\
You are a helpful football assistant.
Keep the reply conversational and concise.
If the user asks for data-heavy details (fixtures, results, standings, predictions), gently ask a short follow-up question.

Conversation context:
{context}

User message: "{message}"
""".strip()


# ── Node: plan ──────────────────────────────────────────────────────────────

def plan_node(state: OrchestratorState) -> OrchestratorState:
    tracker = get_tracker()
    from src.agents.planner_agent import plan_response

    plan = plan_response(
        user_message=state["user_message"],
        conversation_context=state.get("conversation_context") or "None",
    )

    agents = plan.get("agents") or ["chat"]
    primary = plan.get("primary_agent") or agents[0]
    needs_verifier = bool(plan.get("needs_verifier", "bigquery" in agents))

    result: OrchestratorState = {
        **state,
        "selected_agents": agents,
        "selected_agent": primary,
        "topic": plan.get("topic") or "",
        "needs_verifier": needs_verifier,
    }
    tracker.log_step(
        "plan", status="executed",
        input_data={"user_message": state["user_message"]},
        output_data={
            "agents": agents, "primary_agent": primary,
            "topic": plan.get("topic"), "needs_verifier": needs_verifier,
            "reason": plan.get("reason"),
        },
    )
    return result


# ── Node: execute ───────────────────────────────────────────────────────────

def _run_bigquery(state: OrchestratorState) -> dict[str, Any]:
    from src.agents.bigquery_agent import run_structured as run_bq
    return run_bq(state["user_message"], conversation_context=state.get("conversation_context"))


def _run_prediction(state: OrchestratorState) -> dict[str, Any]:
    from src.agents.prediction_agent import run_structured as run_pred
    return run_pred(state["user_message"])


def _run_news(state: OrchestratorState) -> dict[str, Any]:
    from src.agents.news_agent import run_structured as run_news
    return run_news(state["user_message"])


def _run_sentiment(state: OrchestratorState) -> dict[str, Any]:
    from src.agents.sentiment_agent import run_structured as run_sent
    return run_sent(state["user_message"])


def _run_rules(state: OrchestratorState) -> dict[str, Any]:
    from src.agents.rules_agent import run_structured as run_rules
    return run_rules(state["user_message"])


def _run_chat(state: OrchestratorState) -> dict[str, Any]:
    answer = _chat().invoke(
        _CHAT_PROMPT.format(
            context=state.get("conversation_context") or "None",
            message=state["user_message"],
        )
    ).content.strip()
    return {
        "answer": answer,
        "confidence_score": 0.7,
        "confidence_reason": "Conversational reply.",
        "metadata": {"data_source": "chat"},
    }


_RUNNERS = {
    "bigquery": _run_bigquery,
    "prediction": _run_prediction,
    "news": _run_news,
    "sentiment": _run_sentiment,
    "rules": _run_rules,
    "chat": _run_chat,
}


def execute_node(state: OrchestratorState) -> OrchestratorState:
    tracker = get_tracker()
    outputs: dict[str, dict[str, Any]] = {}
    for agent in state.get("selected_agents", ["chat"]):
        runner = _RUNNERS.get(agent)
        if runner is None:
            continue
        try:
            outputs[agent] = runner(state)
        except Exception as exc:
            outputs[agent] = {
                "answer": f"{agent} agent failed: {exc}",
                "confidence_score": 0.2,
                "confidence_reason": f"{agent} execution failed.",
                "metadata": {"data_source": "error", "error": str(exc)},
            }
    if not outputs:
        outputs["chat"] = _run_chat(state)

    tracker.log_step(
        "execute", status="executed",
        input_data={"agents": state.get("selected_agents", [])},
        output_data={
            agent: {
                "confidence_score": p.get("confidence_score"),
                "data_source": (p.get("metadata") or {}).get("data_source"),
            }
            for agent, p in outputs.items()
        },
    )
    return {**state, "agent_outputs": outputs}


# ── Node: verify ────────────────────────────────────────────────────────────

def _pick_primary(outputs: dict[str, dict[str, Any]], hinted_primary: str) -> tuple[str, dict[str, Any]]:
    # Prefer bigquery, then the planner's hint, then highest confidence.
    if "bigquery" in outputs:
        return "bigquery", outputs["bigquery"]
    if hinted_primary in outputs:
        return hinted_primary, outputs[hinted_primary]
    best = max(outputs.items(), key=lambda kv: float(kv[1].get("confidence_score", 0) or 0))
    return best[0], best[1]


def verify_node(state: OrchestratorState) -> OrchestratorState:
    tracker = get_tracker()
    outputs = state.get("agent_outputs", {})
    primary_name, primary_payload = _pick_primary(outputs, state.get("selected_agent", ""))

    verdict_dict: dict[str, Any] = {}

    # Only verify BigQuery answers — that's where nonsense slips through.
    if state.get("needs_verifier") and primary_name == "bigquery":
        from src.agents.verifier_agent import verify

        meta = primary_payload.get("metadata") or {}
        verdict = verify(
            question=state["user_message"],
            answer=str(primary_payload.get("answer", "")),
            sql_executed=list(meta.get("sql_executed") or []),
            row_samples=list(meta.get("row_samples") or []),
        )
        verdict_dict = verdict.to_dict()

        if verdict.needs_repair and verdict.repair_hint:
            # ONE repair attempt — feed the hint back to the BigQuery agent.
            from src.agents.bigquery_agent import run_structured as run_bq
            repair_q = (
                f"{state['user_message']}\n\n"
                f"[Reviewer feedback to fix the previous attempt — address ALL items]\n"
                f"- Issues: {'; '.join(verdict.issues) or '(none)'}\n"
                f"- Hint: {verdict.repair_hint}"
            )
            try:
                repaired = run_bq(repair_q, conversation_context=state.get("conversation_context"))
                if repaired and repaired.get("answer"):
                    outputs["bigquery"] = repaired
                    primary_payload = repaired
                    verdict_dict["repair_attempted"] = True
            except Exception as exc:
                verdict_dict["repair_error"] = str(exc)

        # Verifier may downgrade the confidence.
        if not verdict.grounded or not verdict.answers_question:
            primary_payload = {
                **primary_payload,
                "confidence_score": min(
                    float(primary_payload.get("confidence_score", 0.5) or 0.5),
                    verdict.confidence,
                ),
                "confidence_reason": (
                    f"Reviewer flagged issues: {'; '.join(verdict.issues) or 'low groundedness'}."
                ),
            }

    score = float(primary_payload.get("confidence_score", 0.6) or 0.6)
    score = max(0.0, min(1.0, score))
    label = "high" if score >= 0.8 else ("medium" if score >= 0.55 else "low")

    tracker.log_step(
        "verify", status="executed",
        input_data={"needs_verifier": state.get("needs_verifier"), "primary": primary_name},
        output_data={"verdict": verdict_dict, "confidence": score, "label": label},
    )
    return {
        **state,
        "agent_payload": primary_payload,
        "selected_agent": primary_name,
        "verifier_verdict": verdict_dict,
        "confidence_score": score,
        "confidence_label": label,
        "confidence_reason": str(primary_payload.get("confidence_reason", "")),
    }


# ── Node: compose ───────────────────────────────────────────────────────────

def compose_node(state: OrchestratorState) -> OrchestratorState:
    from src.agents.docs_agent import log_session
    from src.agents.result_composer_agent import compose

    tracker = get_tracker()
    final_reply = compose(
        user_message=state["user_message"],
        intent=state.get("topic") or state.get("selected_agent", ""),
        selected_agent=state["selected_agent"],
        payload=state["agent_payload"],
        confidence_score=state["confidence_score"],
        confidence_label=state["confidence_label"],
        confidence_reason=state["confidence_reason"],
    )
    try:
        log_session(state["user_id"], state["user_message"], final_reply)
    except Exception:
        pass

    tracker.log_step(
        "compose", status="executed",
        input_data={
            "confidence_label": state["confidence_label"],
            "confidence_score": state["confidence_score"],
            "selected_agent": state["selected_agent"],
        },
        output_data={"final_reply": final_reply},
    )
    return {**state, "final_reply": final_reply}


# ── Graph ───────────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    g = StateGraph(OrchestratorState)
    g.add_node("plan", plan_node)
    g.add_node("execute", execute_node)
    g.add_node("verify", verify_node)
    g.add_node("compose", compose_node)
    g.set_entry_point("plan")
    g.add_edge("plan", "execute")
    g.add_edge("execute", "verify")
    g.add_edge("verify", "compose")
    g.add_edge("compose", END)
    return g


_graph = _build_graph().compile()


async def run_orchestrator(
    user_message: str,
    user_id: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    memory = ConversationMemory.from_messages(conversation_history)
    memory.append_user(user_message)

    state: OrchestratorState = {
        "user_id": user_id,
        "user_message": user_message,
        "conversation_context": memory.context_block(),
        "topic": "",
        "selected_agent": "",
        "selected_agents": [],
        "needs_verifier": False,
        "agent_outputs": {},
        "agent_payload": {},
        "verifier_verdict": {},
        "confidence_score": 0.0,
        "confidence_label": "low",
        "confidence_reason": "",
        "final_reply": "",
        "messages": [],
    }
    out = await _graph.ainvoke(state)
    return out["final_reply"]
