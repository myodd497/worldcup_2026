"""
Smoke tests for all specialist agents.
Run: pytest tests/ -v
"""
import pytest
from unittest.mock import patch, MagicMock


# ── News Agent ───────────────────────────────────────────────────────────────

@patch("src.tools.news_search.search_news")
def test_news_agent(mock_search):
    mock_search.return_value = [
        {"title": "Portugal wins", "url": "http://example.com", "source": "ESPN", "published_at": "2026-06-11"}
    ]
    from src.agents.news_agent import run
    result = run("Portugal vs Morocco")
    assert "Portugal wins" in result


# ── Sentiment Agent ──────────────────────────────────────────────────────────

@patch("src.tools.twitter_sentiment.get_sentiment_summary")
def test_sentiment_agent(mock_sentiment):
    mock_sentiment.return_value = {
        "tweet_count": 100,
        "positive_pct": 60.0,
        "negative_pct": 20.0,
        "neutral_pct": 20.0,
        "top_topic": "Portugal",
    }
    from src.agents.sentiment_agent import run
    result = run("Portugal")
    assert "POSITIVE" in result
    assert "100" in result


# ── Match Facts Agent ────────────────────────────────────────────────────────

@patch("src.agents.match_facts_agent._llm_format_fixtures_answer")
@patch("src.agents.match_facts_agent.get_fixtures_cache_first")
@patch("src.tools.weather.get_venue_weather")
def test_match_facts_agent(mock_weather, mock_fixtures, mock_format):
    mock_fixtures.return_value = (
        [
            {
                "fixture_id": 1,
                "home_team": "Portugal",
                "away_team": "Morocco",
                "date": "2026-06-15T18:00:00",
                "venue": "MetLife Stadium",
                "venue_city": "New York",
                "referee": "Howard Webb",
                "home_goals": None,
                "away_goals": None,
                "status": "NS",
                "season": 2026,
            }
        ],
        "bigquery",
    )
    mock_format.return_value = "Portugal vs Morocco at MetLife Stadium"
    mock_weather.return_value = {"description": "Sunny", "temp_c": 24.0}
    from src.agents.match_facts_agent import run
    result = run("Portugal")
    assert "Portugal" in result
    assert "MetLife Stadium" in result


# ── Prediction Agent ─────────────────────────────────────────────────────────

@patch("src.models.predict.predict_match")
@patch("langchain_openai.ChatOpenAI.invoke")
def test_prediction_agent(mock_llm, mock_predict):
    mock_predict.return_value = {
        "home_team": "Portugal",
        "away_team": "Morocco",
        "home_win_pct": 55.0,
        "draw_pct": 22.0,
        "away_win_pct": 23.0,
        "model_version": "xgboost_v1",
    }
    mock_llm.return_value = MagicMock(content="Portugal have stronger form.")
    from src.agents.prediction_agent import run
    result = run("Portugal vs Morocco")
    assert "55%" in result or "55.0%" in result


# ── BigQuery Agent ───────────────────────────────────────────────────────────

@patch("src.agents.bigquery_agent.run_query")
@patch("src.agents.bigquery_agent._llm")
def test_bigquery_agent(mock_llm, mock_run_query):
    import pandas as pd

    mock_llm.invoke.return_value = MagicMock(
        content=(
            '{"sql": "SELECT fixture_id, home_team_name, away_team_name FROM `worldcup2026.fact_fixture` LIMIT 2", '
            '"tables_used": ["fact_fixture"], '
            '"explanation": "Use canonical fixture facts", '
            '"answer_style": "analytics"}'
        )
    )
    mock_run_query.side_effect = [
        pd.DataFrame(),
        pd.DataFrame(
            [
                {"fixture_id": 1, "home_team_name": "Portugal", "away_team_name": "Morocco"},
                {"fixture_id": 2, "home_team_name": "Brazil", "away_team_name": "Argentina"},
            ]
        ),
    ]

    from src.agents.bigquery_agent import run_structured

    result = run_structured("show me upcoming fixtures")
    assert result["metadata"]["data_source"] == "bigquery"
    assert "fact_fixture" in result["metadata"]["tables_used"]
    assert "BigQuery" in result["answer"]


# ── Docs Agent ───────────────────────────────────────────────────────────────

def test_docs_agent(tmp_path):
    import src.agents.docs_agent as docs_agent
    docs_agent._DOCS_DIR = tmp_path
    docs_agent.log_session("whatsapp:+351912345678", "Who will win?", "Portugal!")
    files = list(tmp_path.glob("*.md"))
    assert len(files) == 1
    assert "Portugal!" in files[0].read_text()
