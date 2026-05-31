"""
Twitter/X sentiment tool — fetches recent tweets and scores them with VADER.
"""
from __future__ import annotations

import os


def _is_social_sentiment_enabled() -> bool:
    return os.getenv("ENABLE_SOCIAL_SENTIMENT", "false").lower() == "true"


def _get_client():
    import tweepy

    return tweepy.Client(bearer_token=os.environ["TWITTER_BEARER_TOKEN"])


def get_sentiment_summary(query: str, max_results: int = 50) -> dict:
    if not _is_social_sentiment_enabled():
        return {
            "tweet_count": 0,
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "neutral_pct": 100.0,
            "top_topic": query,
            "sample_tweets": [],
        }

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    client = _get_client()
    response = client.search_recent_tweets(
        query=f"{query} -is:retweet lang:en",
        max_results=min(max_results, 100),
        tweet_fields=["text"],
    )

    tweets = [t.text for t in (response.data or [])]
    if not tweets:
        return {
            "tweet_count": 0,
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "neutral_pct": 100.0,
            "top_topic": "",
        }

    scores = [analyzer.polarity_scores(t) for t in tweets]
    positive = sum(1 for s in scores if s["compound"] >= 0.05)
    negative = sum(1 for s in scores if s["compound"] <= -0.05)
    neutral = len(scores) - positive - negative
    total = len(scores)

    return {
        "tweet_count": total,
        "positive_pct": positive / total * 100,
        "negative_pct": negative / total * 100,
        "neutral_pct": neutral / total * 100,
        "top_topic": query,
        "sample_tweets": tweets[:3],
    }


if __name__ == "__main__":
    print(get_sentiment_summary("Portugal World Cup"))
