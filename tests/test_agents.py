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

@patch.dict("os.environ", {"BIGQUERY_PROJECT_ID": "test-project", "BIGQUERY_DATASET_ID": "worldcup2026"})
@patch("src.agents.bigquery_agent.run_query")
@patch("src.agents.bigquery_agent._llm")
def test_bigquery_agent(mock_llm, mock_run_query):
    import pandas as pd

    # The new pipeline performs multiple LLM calls in order:
    # 1) entity extraction  2) table selection  3) SQL generation  4) final composition
    mock_llm.invoke.side_effect = [
        MagicMock(content='{"teams": [], "is_head_to_head": false, "needs_upcoming": true}'),
        MagicMock(content='{"tables": ["v_next_fixtures"], "reason": "upcoming fixtures"}'),
        MagicMock(content=(
            '{"queries": [{"name": "upcoming", "purpose": "next fixtures", '
            '"sql": "SELECT fixture_id, home_team_name, away_team_name '
            'FROM `test-project.worldcup2026.v_next_fixtures` LIMIT 2"}]}'
        )),
        MagicMock(content="Upcoming fixtures summary."),
    ]
    # column metadata + contract + sql execution
    mock_run_query.side_effect = [
        pd.DataFrame(),  # _load_column_metadata
        pd.DataFrame(),  # _load_contract (forces fallback to static catalog)
        pd.DataFrame(   # actual user query
            [
                {"fixture_id": 1, "home_team_name": "Portugal", "away_team_name": "Morocco"},
                {"fixture_id": 2, "home_team_name": "Brazil", "away_team_name": "Argentina"},
            ]
        ),
    ]

    from src.agents.bigquery_agent import run_structured

    result = run_structured("show me upcoming fixtures")
    assert result["metadata"]["data_source"] == "bigquery"
    assert "v_next_fixtures" in result["metadata"]["tables_used"]
    assert result["answer"]


@patch.dict("os.environ", {"BIGQUERY_PROJECT_ID": "test-project", "BIGQUERY_DATASET_ID": "worldcup2026"})
@patch("src.agents.bigquery_agent.run_query")
@patch("src.agents.bigquery_agent._llm")
def test_bigquery_agent_head_to_head_last_result_and_stats_uses_deterministic_plan(mock_llm, mock_run_query):
    import pandas as pd

    mock_llm.invoke.side_effect = [
        MagicMock(content='{"teams": ["Portugal", "Morocco"], "season": null, "is_head_to_head": true, "is_specific_match": true, "needs_recent_form": false, "needs_upcoming": false, "needs_match_stats": true, "needs_events": false}'),
        MagicMock(content='{"tables": ["fact_fixture", "fact_fixture_team_stat"], "reason": "last shared fixture and stats"}'),
        MagicMock(content="## Portugal vs Morocco Last Result\n- **Final Score**: Morocco 1 - 0 Portugal"),
    ]
    mock_run_query.side_effect = [
        pd.DataFrame(
            [
                {"team_id": 27, "team_name": "Portugal"},
                {"team_id": 31, "team_name": "Morocco"},
            ]
        ),
        pd.DataFrame(),
        pd.DataFrame(
            [
                {
                    "fixture_id": 1,
                    "fixture_date": "2022-12-10",
                    "fixture_datetime": "2022-12-10T15:00:00",
                    "competition_name": "World Cup",
                    "competition_round": "Quarter-finals",
                    "home_team_id": 31,
                    "home_team_name": "Morocco",
                    "away_team_id": 27,
                    "away_team_name": "Portugal",
                    "venue_name": "Al Thumama Stadium",
                    "venue_city": "Doha",
                    "referee": "F. Tello",
                    "status": "FT",
                    "home_goals": 1,
                    "away_goals": 0,
                }
            ]
        ),
        pd.DataFrame(
            [
                {"team_id": 31, "team_name": "Morocco", "shots_on_goal": "3", "ball_possession": "27%", "shots_off_goal": "6", "total_shots": "9", "corner_kicks": "3", "fouls": "15"},
                {"team_id": 27, "team_name": "Portugal", "shots_on_goal": "3", "ball_possession": "73%", "shots_off_goal": "6", "total_shots": "12", "corner_kicks": "9", "fouls": "9"},
            ]
        ),
    ]

    from src.agents.bigquery_agent import run_structured

    result = run_structured("What was Portugal vs Morocco last result and stats?")
    assert len(result["metadata"]["queries"]) == 2
    assert all(q["repair_note"] is None for q in result["metadata"]["queries"])
    assert "fact_fixture" in result["metadata"]["queries"][0]["sql"]
    assert "fact_fixture_team_stat" in result["metadata"]["queries"][1]["sql"]


# ── Planner Agent ────────────────────────────────────────────────────────────

@patch("src.agents.planner_agent._llm")
def test_planner_agent_selects_bigquery(mock_llm):
    mock_llm.invoke.return_value = MagicMock(
        content=(
            '{"agents": ["bigquery", "match_facts"], '
            '"response_mode": "multi", '
            '"reason": "Needs structured warehouse data", '
            '"primary_agent": "bigquery"}'
        )
    )

    from src.agents.planner_agent import plan_response

    plan = plan_response("how many days until the world cup?", [])
    assert plan["primary_agent"] == "bigquery"
    assert "bigquery" in plan["agents"]


@patch("src.agents.planner_agent._llm")
def test_planner_agent_falls_back_to_bigquery_for_temporal_queries(mock_llm):
    mock_llm.invoke.side_effect = Exception("planner unavailable")

    from src.agents.planner_agent import plan_response

    plan = plan_response("how many days until the world cup?", [])
    assert "bigquery" in plan["agents"]
    assert plan["primary_agent"] == "bigquery"


# ── Docs Agent ───────────────────────────────────────────────────────────────

def test_docs_agent(tmp_path):
    import src.agents.docs_agent as docs_agent
    docs_agent._DOCS_DIR = tmp_path
    docs_agent.log_session("whatsapp:+351912345678", "Who will win?", "Portugal!")
    files = list(tmp_path.glob("*.md"))
    assert len(files) == 1
    assert "Portugal!" in files[0].read_text()
