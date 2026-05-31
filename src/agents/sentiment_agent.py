"""
Sentiment Agent — analyses social media sentiment around a match or team.
Uses Twitter/X API v2 + VADER scoring (fast) with optional DistilBERT pass.
"""
from __future__ import annotations

from src.tools.twitter_sentiment import get_sentiment_summary


def run_structured(query: str) -> dict:
    summary = get_sentiment_summary(query)
    pos = summary["positive_pct"]
    neg = summary["negative_pct"]
    neu = summary["neutral_pct"]
    total = summary["tweet_count"]

    sentiment_label = "POSITIVE" if pos > 50 else ("NEGATIVE" if neg > 50 else "MIXED")

    answer = (
        f"*Social Media Sentiment — {query}*\n"
        f"Overall: {sentiment_label}\n"
        f"📊 {total} tweets analysed\n"
        f"✅ Positive: {pos:.0f}%\n"
        f"❌ Negative: {neg:.0f}%\n"
        f"➖ Neutral: {neu:.0f}%\n\n"
        f"Top topic: {summary.get('top_topic', 'N/A')}"
    )

    if total == 0:
        score = 0.2
        reason = "No social posts were retrieved."
    elif total < 20:
        score = 0.45
        reason = "Low sample size (<20 posts)."
    elif total < 60:
        score = 0.65
        reason = "Moderate sample size."
    else:
        score = 0.82
        reason = "Strong sample size and stable polarity distribution."

    return {
        "answer": answer,
        "confidence_score": score,
        "confidence_reason": reason,
        "metadata": {"tweet_count": total, "positive_pct": pos, "negative_pct": neg, "neutral_pct": neu},
    }


def run(query: str) -> str:
    return run_structured(query)["answer"]


if __name__ == "__main__":
    print(run("Portugal World Cup 2026"))
