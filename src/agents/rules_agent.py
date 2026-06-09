"""
Rules Agent — answers questions about the FIFA World Cup 2026 regulations.

Loads the official FIFA World Cup 26™ Regulations (extracted from PDF) as
a system-prompt context and uses an LLM to retrieve, interpret, and explain
specific rules, articles, and competition procedures.

The regulations cover:
  I.   General Provisions (Articles 1-6)
  II.  Disciplinary Matters and Procedures (Articles 7-10)
  III. Competition Format (Articles 11-14)
  IV.  Competition Preparation (Articles 15-18)
  V.   Stadiums and Training Sites (Articles 19-21)
  VI.  Players' and Officials' Lists (Articles 22-27)
  VII. Kit and Team Equipment (Articles 28-31)
  VIII. Match Organisation (Articles 32-35)
  IX.  Refereeing (Articles 36-37)
  X.   Financial Provisions (Articles 38-40)
  XI.  Medical (Articles 41-43)
  XII. Commercial Rights (Article 44)
  XIII. Awards (Article 45)
  XIV. Closing Provisions (Articles 46-52)
"""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_openai import ChatOpenAI

from src.agents.llm_config import create_chat_model

logger = logging.getLogger(__name__)

# ── Load regulations text ───────────────────────────────────────────────────

_REGULATIONS_PATH = Path(__file__).resolve().parents[2] / "Docs" / "FWC26_regulations_EN.txt"

_regulations_text: str | None = None


def _load_regulations() -> str:
    """Load the full regulations text, cached after first read."""
    global _regulations_text
    if _regulations_text is not None:
        return _regulations_text

    if not _REGULATIONS_PATH.exists():
        logger.warning("Regulations file not found at %s", _REGULATIONS_PATH)
        _regulations_text = ""
        return _regulations_text

    text = _REGULATIONS_PATH.read_text(encoding="utf-8")
    if not text.strip():
        _regulations_text = ""
        return _regulations_text

    _regulations_text = text
    return _regulations_text


# ── System prompt ───────────────────────────────────────────────────────────


def _system_prompt() -> str:
    regulations = _load_regulations()
    if not regulations:
        return (
            "You are a football regulations assistant. "
            "The official FIFA World Cup 2026 regulations are not available. "
            "Politely tell the user you cannot access the regulations document right now."
        )

    # Use the full regulations text; the document is ~128K chars (~32K tokens).
    return (
        "You are an expert on the FIFA World Cup 2026™ official regulations.\n"
        "You have access to the full Regulations document below.\n"
        "Answer user questions accurately by citing the relevant article and paragraph numbers.\n"
        "\n"
        "Rules:\n"
        "- Always cite the specific Article and paragraph (e.g., 'Article 6.2 states...').\n"
        "- If the regulations do not cover the question, say so clearly.\n"
        "- Provide the exact text from the regulations when possible.\n"
        "- Be concise but thorough — users want precise answers, not summaries.\n"
        "- Format with markdown for readability.\n"
        "\n"
        "=== FIFA WORLD CUP 26™ REGULATIONS ===\n\n"
        f"{regulations}"
    )


# ── LLM ─────────────────────────────────────────────────────────────────────

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    """Lazy-initialize the LLM client."""
    global _llm
    if _llm is None:
        _llm = create_chat_model("simple", temperature=0)
    return _llm


# ── Public API ──────────────────────────────────────────────────────────────


def run_structured(query: str) -> dict:
    """Run the rules agent and return a structured payload.

    Contract (matches all specialist agents):
        answer: str
        confidence_score: float
        confidence_reason: str
        metadata: dict
    """
    regulations = _load_regulations()

    if not regulations:
        return {
            "answer": (
                "I'm sorry, but I cannot access the official FIFA World Cup 2026 regulations "
                "document right now. Please ensure the regulations file is available at "
                "`Docs/FWC26_regulations_EN.txt`."
            ),
            "confidence_score": 0.1,
            "confidence_reason": "Regulations document not found or empty.",
            "metadata": {"data_source": "none", "regulations_loaded": False},
        }

    try:
        messages = [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": query},
        ]
        answer = _get_llm().invoke(messages).content.strip()

        # Confidence: moderate-high since the LLM has the full regulations
        # and the simple tier is reliable for retrieval tasks.
        return {
            "answer": answer,
            "confidence_score": 0.80,
            "confidence_reason": (
                "Answer grounded in the official FIFA World Cup 26 Regulations document. "
                "Confidence is high for factual rule retrieval."
            ),
            "metadata": {
                "data_source": "official_regulations_pdf",
                "regulations_loaded": True,
                "document": "FWC26_regulations_EN.pdf",
            },
        }
    except Exception as exc:
        logger.exception("rules_agent failed")
        return {
            "answer": f"I encountered an error while looking up the regulations: {exc}",
            "confidence_score": 0.2,
            "confidence_reason": f"Rules agent execution failed: {exc}",
            "metadata": {"data_source": "error", "regulations_loaded": bool(regulations)},
        }


def run(query: str) -> str:
    """Run the rules agent and return just the answer string."""
    return run_structured(query)["answer"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    import sys

    q = " ".join(sys.argv[1:]) or "How many teams participate in the 2026 World Cup and what is the format?"
    print(run_structured(q)["answer"])
