"""
Builds a canonical semantic model in BigQuery:
- Canonical fact + dimension tables
- Gold views for app use-cases
- Uniqueness checks for key tables

Usage:
    set -a && source .env && set +a
    poetry run python -m src.data.build_semantic_model
"""
from __future__ import annotations

import os

from src.tools.bigquery_tools import execute_sql, run_query


def _ref(dataset: str, name: str) -> str:
    project = os.environ["BIGQUERY_PROJECT_ID"]
    return f"`{project}.{dataset}.{name}`"


def _build_canonical_tables(dataset: str) -> None:
    fact_fixture_sql = f"""
    CREATE OR REPLACE TABLE {_ref(dataset, 'fact_fixture')}
    PARTITION BY fixture_date
    CLUSTER BY season, competition_id, home_team_id, away_team_id
    AS
    WITH src AS (
      SELECT
        fixture_id,
        season,
        DATE(date) AS fixture_date,
        TIMESTAMP(date) AS fixture_datetime,
        venue AS venue_name,
        venue_city,
        referee,
        status,
        home_team AS home_team_name,
        away_team AS away_team_name,
        SAFE_CAST(home_goals AS INT64) AS home_goals,
        SAFE_CAST(away_goals AS INT64) AS away_goals,
        CURRENT_TIMESTAMP() AS loaded_at
      FROM {_ref(dataset, 'fixtures_historical')}
    ),
    comp AS (
      SELECT
        fixture_id,
        ANY_VALUE(competition_id) AS competition_id,
        ANY_VALUE(competition_name) AS competition_name,
        ANY_VALUE(competition_country) AS competition_country,
        ANY_VALUE(competition_round) AS competition_round,
        ANY_VALUE(home_team_id) AS home_team_id,
        ANY_VALUE(away_team_id) AS away_team_id
      FROM {_ref(dataset, 'team_match_history')}
      GROUP BY fixture_id
    )
    SELECT
      s.fixture_id,
      s.season,
      s.fixture_date,
      s.fixture_datetime,
      c.competition_id,
      c.competition_name,
      c.competition_country,
      c.competition_round,
      c.home_team_id,
      s.home_team_name,
      c.away_team_id,
      s.away_team_name,
      s.venue_name,
      s.venue_city,
      s.referee,
      s.status,
      s.home_goals,
      s.away_goals,
      s.loaded_at
    FROM src s
    LEFT JOIN comp c USING (fixture_id)
    QUALIFY ROW_NUMBER() OVER (PARTITION BY s.fixture_id ORDER BY s.fixture_datetime DESC) = 1
    """

    fact_team_fixture_sql = f"""
    CREATE OR REPLACE TABLE {_ref(dataset, 'fact_team_fixture')}
    PARTITION BY match_date
    CLUSTER BY team_id, season, competition_id, opponent_id
    AS
    SELECT
      team_id,
      team_name,
      fixture_id,
      match_date,
      match_datetime,
      season,
      competition_id,
      competition_name,
      competition_country,
      competition_round,
      home_team_id,
      home_team_name,
      away_team_id,
      away_team_name,
      was_home,
      opponent_id,
      opponent_name,
      goals_scored,
      goals_conceded,
      result,
      home_goals,
      away_goals,
      home_goals_ht,
      away_goals_ht,
      home_goals_et,
      away_goals_et,
      home_goals_pen,
      away_goals_pen,
      venue_name,
      venue_city,
      referee,
      ingested_at,
      data_source
    FROM {_ref(dataset, 'team_match_history')}
    QUALIFY ROW_NUMBER() OVER (
      PARTITION BY team_id, fixture_id
      ORDER BY match_datetime DESC, ingested_at DESC
    ) = 1
    """

    fact_fixture_event_sql = f"""
    CREATE OR REPLACE TABLE {_ref(dataset, 'fact_fixture_event')}
    PARTITION BY event_date
    CLUSTER BY fixture_id, team_id, event_type
    AS
    WITH base AS (
      SELECT
        e.fixture_id,
        DATE(f.fixture_datetime) AS event_date,
        e.season,
        e.league_id AS competition_id,
        e.time_elapsed,
        e.time_extra,
        e.team_id,
        e.team_name,
        e.player_id,
        e.player_name,
        e.assist_id,
        e.assist_name,
        e.event_type,
        e.event_detail,
        e.event_comments,
        e.ingested_at,
        e.data_source
      FROM {_ref(dataset, 'fixture_events')} e
      LEFT JOIN {_ref(dataset, 'fact_fixture')} f ON f.fixture_id = e.fixture_id
    )
    SELECT
      CONCAT(
        CAST(fixture_id AS STRING), '|',
        IFNULL(CAST(team_id AS STRING), ''), '|',
        IFNULL(CAST(time_elapsed AS STRING), ''), '|',
        IFNULL(event_type, ''), '|',
        IFNULL(event_detail, ''), '|',
        IFNULL(CAST(player_id AS STRING), '')
      ) AS event_key,
      fixture_id,
      event_date,
      season,
      competition_id,
      time_elapsed,
      time_extra,
      team_id,
      team_name,
      player_id,
      player_name,
      assist_id,
      assist_name,
      event_type,
      event_detail,
      event_comments,
      ingested_at,
      data_source
    FROM base
    """

    fact_fixture_team_stat_sql = f"""
    CREATE OR REPLACE TABLE {_ref(dataset, 'fact_fixture_team_stat')}
    PARTITION BY match_date
    CLUSTER BY fixture_id, team_id, stat_type
    AS
    SELECT
      fixture_id,
      season,
      competition_id,
      competition_name,
      match_date,
      team_id,
      team_name,
      is_home,
      opponent_id,
      opponent_name,
      stat_type,
      stat_value_num,
      stat_value_text,
      stat_value_unit,
      ingested_at,
      data_source
    FROM {_ref(dataset, 'fixture_stats')}
    QUALIFY ROW_NUMBER() OVER (
      PARTITION BY fixture_id, team_id, stat_type
      ORDER BY ingested_at DESC
    ) = 1
    """

    dim_team_sql = f"""
    CREATE OR REPLACE TABLE {_ref(dataset, 'dim_team')}
    AS
    WITH all_teams AS (
      SELECT team_id, team_name FROM {_ref(dataset, 'fact_team_fixture')}
      UNION ALL
      SELECT home_team_id AS team_id, home_team_name AS team_name FROM {_ref(dataset, 'fact_fixture')}
      UNION ALL
      SELECT away_team_id AS team_id, away_team_name AS team_name FROM {_ref(dataset, 'fact_fixture')}
      UNION ALL
      SELECT team_id, team_name FROM {_ref(dataset, 'team_stats')}
      UNION ALL
      SELECT team_id, team_name FROM {_ref(dataset, 'standings')}
    )
    SELECT
      team_id,
      ARRAY_AGG(team_name IGNORE NULLS ORDER BY team_name LIMIT 1)[OFFSET(0)] AS team_name,
      CURRENT_TIMESTAMP() AS loaded_at
    FROM all_teams
    WHERE team_id IS NOT NULL
    GROUP BY team_id
    """

    dim_competition_sql = f"""
    CREATE OR REPLACE TABLE {_ref(dataset, 'dim_competition')}
    AS
    SELECT
      competition_id,
      ARRAY_AGG(competition_name IGNORE NULLS ORDER BY competition_name LIMIT 1)[OFFSET(0)] AS competition_name,
      ARRAY_AGG(competition_country IGNORE NULLS ORDER BY competition_country LIMIT 1)[OFFSET(0)] AS competition_country,
      CURRENT_TIMESTAMP() AS loaded_at
    FROM {_ref(dataset, 'fact_team_fixture')}
    WHERE competition_id IS NOT NULL
    GROUP BY competition_id
    """

    dim_venue_sql = f"""
    CREATE OR REPLACE TABLE {_ref(dataset, 'dim_venue')}
    AS
    SELECT
      TO_HEX(MD5(CONCAT(IFNULL(venue_name, ''), '|', IFNULL(venue_city, '')))) AS venue_id,
      venue_name,
      venue_city,
      CURRENT_TIMESTAMP() AS loaded_at
    FROM {_ref(dataset, 'fact_fixture')}
    WHERE venue_name IS NOT NULL OR venue_city IS NOT NULL
    GROUP BY venue_name, venue_city
    """

    dim_referee_sql = f"""
    CREATE OR REPLACE TABLE {_ref(dataset, 'dim_referee')}
    AS
    SELECT
      TO_HEX(MD5(referee)) AS referee_id,
      referee,
      CURRENT_TIMESTAMP() AS loaded_at
    FROM {_ref(dataset, 'fact_fixture')}
    WHERE referee IS NOT NULL
      AND TRIM(referee) != ''
      AND LOWER(referee) NOT IN ('none', 'tbd', 'unknown')
    GROUP BY referee
    """

    for sql in [
        fact_fixture_sql,
        fact_team_fixture_sql,
        fact_fixture_event_sql,
        fact_fixture_team_stat_sql,
        dim_team_sql,
        dim_competition_sql,
        dim_venue_sql,
        dim_referee_sql,
    ]:
      execute_sql(sql)


def _build_gold_views(dataset: str) -> None:
    v_team_recent_form_sql = f"""
    CREATE OR REPLACE VIEW {_ref(dataset, 'v_team_recent_form')} AS
    WITH ranked AS (
      SELECT
        team_id,
        team_name,
        fixture_id,
        match_datetime,
        result,
        goals_scored,
        goals_conceded,
        ROW_NUMBER() OVER (PARTITION BY team_id ORDER BY match_datetime DESC) AS rn
      FROM {_ref(dataset, 'fact_team_fixture')}
      WHERE result IN ('W', 'D', 'L')
    )
    SELECT
      team_id,
      ANY_VALUE(team_name) AS team_name,
      STRING_AGG(result ORDER BY rn) AS form_last_5,
      SUM(CASE WHEN result = 'W' THEN 1 ELSE 0 END) AS wins_last_5,
      SUM(CASE WHEN result = 'D' THEN 1 ELSE 0 END) AS draws_last_5,
      SUM(CASE WHEN result = 'L' THEN 1 ELSE 0 END) AS losses_last_5,
      SUM(goals_scored) AS goals_for_last_5,
      SUM(goals_conceded) AS goals_against_last_5,
      COUNT(*) AS matches_used
    FROM ranked
    WHERE rn <= 5
    GROUP BY team_id
    """

    v_head_to_head_sql = f"""
    CREATE OR REPLACE VIEW {_ref(dataset, 'v_head_to_head')} AS
    WITH pair_rows AS (
      SELECT
        LEAST(home_team_id, away_team_id) AS team_a_id,
        GREATEST(home_team_id, away_team_id) AS team_b_id,
        fixture_id,
        fixture_datetime,
        home_team_id,
        away_team_id,
        home_goals,
        away_goals
      FROM {_ref(dataset, 'fact_fixture')}
      WHERE home_team_id IS NOT NULL
        AND away_team_id IS NOT NULL
        AND home_goals IS NOT NULL
        AND away_goals IS NOT NULL
    )
    SELECT
      team_a_id,
      team_b_id,
      COUNT(*) AS h2h_matches,
      SUM(CASE
            WHEN (home_team_id = team_a_id AND home_goals > away_goals)
              OR (away_team_id = team_a_id AND away_goals > home_goals) THEN 1 ELSE 0
          END) AS team_a_wins,
      SUM(CASE
            WHEN (home_team_id = team_b_id AND home_goals > away_goals)
              OR (away_team_id = team_b_id AND away_goals > home_goals) THEN 1 ELSE 0
          END) AS team_b_wins,
      SUM(CASE WHEN home_goals = away_goals THEN 1 ELSE 0 END) AS draws,
      MAX(fixture_datetime) AS last_meeting_datetime
    FROM pair_rows
    GROUP BY team_a_id, team_b_id
    """

    v_next_fixtures_sql = f"""
    CREATE OR REPLACE VIEW {_ref(dataset, 'v_next_fixtures')} AS
    SELECT
      fixture_id,
      fixture_datetime,
      fixture_date,
      season,
      competition_id,
      competition_name,
      competition_round,
      home_team_id,
      home_team_name,
      away_team_id,
      away_team_name,
      venue_name,
      venue_city,
      referee,
      status
    FROM {_ref(dataset, 'fact_fixture')}
    WHERE status NOT IN ('FT', 'AET', 'PEN', 'AWD', 'WO')
      AND fixture_date >= CURRENT_DATE()
    ORDER BY fixture_datetime ASC
    """

    v_match_card_sql = f"""
    CREATE OR REPLACE VIEW {_ref(dataset, 'v_match_card')} AS
    SELECT
      f.fixture_id,
      f.fixture_datetime,
      f.fixture_date,
      f.season,
      f.competition_name,
      f.competition_round,
      f.home_team_id,
      f.home_team_name,
      f.away_team_id,
      f.away_team_name,
      f.venue_name,
      f.venue_city,
      f.referee,
      f.status,
      f.home_goals,
      f.away_goals,
      hr.form_last_5 AS home_form_last_5,
      ar.form_last_5 AS away_form_last_5,
      hs.points AS home_points,
      asd.points AS away_points,
      hs.rank AS home_rank,
      asd.rank AS away_rank,
      hs.group_name AS home_group,
      asd.group_name AS away_group
    FROM {_ref(dataset, 'fact_fixture')} f
    LEFT JOIN {_ref(dataset, 'v_team_recent_form')} hr ON hr.team_id = f.home_team_id
    LEFT JOIN {_ref(dataset, 'v_team_recent_form')} ar ON ar.team_id = f.away_team_id
    LEFT JOIN {_ref(dataset, 'standings')} hs
      ON hs.team_id = f.home_team_id AND hs.season = f.season
    LEFT JOIN {_ref(dataset, 'standings')} asd
      ON asd.team_id = f.away_team_id AND asd.season = f.season
    """

    v_prediction_features_sql = f"""
    CREATE OR REPLACE VIEW {_ref(dataset, 'v_prediction_features')} AS
    WITH recent AS (
      SELECT
        team_id,
        fixture_id,
        match_datetime,
        result,
        goals_scored,
        goals_conceded,
        ROW_NUMBER() OVER (PARTITION BY team_id ORDER BY match_datetime DESC) AS rn
      FROM {_ref(dataset, 'fact_team_fixture')}
      WHERE result IN ('W', 'D', 'L')
    ),
    agg AS (
      SELECT
        team_id,
        AVG(CASE result WHEN 'W' THEN 3 WHEN 'D' THEN 1 ELSE 0 END) AS points_per_match_last_10,
        AVG(goals_scored - goals_conceded) AS goal_diff_per_match_last_10,
        COUNT(*) AS samples_last_10
      FROM recent
      WHERE rn <= 10
      GROUP BY team_id
    )
    SELECT
      f.fixture_id,
      f.fixture_datetime,
      f.home_team_id,
      f.away_team_id,
      h.points_per_match_last_10 AS home_ppm_last_10,
      a.points_per_match_last_10 AS away_ppm_last_10,
      h.goal_diff_per_match_last_10 AS home_gdpm_last_10,
      a.goal_diff_per_match_last_10 AS away_gdpm_last_10,
      h.samples_last_10 AS home_samples_last_10,
      a.samples_last_10 AS away_samples_last_10
    FROM {_ref(dataset, 'v_next_fixtures')} f
    LEFT JOIN agg h ON h.team_id = f.home_team_id
    LEFT JOIN agg a ON a.team_id = f.away_team_id
    """

    v_data_contract_tables_sql = f"""
    CREATE OR REPLACE VIEW {_ref(dataset, 'v_data_contract_tables')} AS
    SELECT 'fixtures_historical' AS object_name, 'source' AS object_layer, 'source_only' AS app_usage, 'raw fixture snapshot' AS business_concept
    UNION ALL SELECT 'team_match_history', 'source', 'source_only', 'raw team-match history'
    UNION ALL SELECT 'fixture_events', 'source', 'source_only', 'raw fixture events'
    UNION ALL SELECT 'fixture_stats', 'source', 'source_only', 'raw fixture team stats'
    UNION ALL SELECT 'standings', 'source', 'source_only', 'raw standings'
    UNION ALL SELECT 'team_stats', 'source', 'source_only', 'raw team season stats'

    UNION ALL SELECT 'fact_fixture', 'canonical_fact', 'app_allowed', 'fixture'
    UNION ALL SELECT 'fact_team_fixture', 'canonical_fact', 'app_allowed', 'team fixture participation'
    UNION ALL SELECT 'fact_fixture_event', 'canonical_fact', 'app_allowed', 'fixture event'
    UNION ALL SELECT 'fact_fixture_team_stat', 'canonical_fact', 'app_allowed', 'fixture team statistic'
    UNION ALL SELECT 'dim_team', 'canonical_dim', 'app_allowed', 'team'
    UNION ALL SELECT 'dim_competition', 'canonical_dim', 'app_allowed', 'competition'
    UNION ALL SELECT 'dim_venue', 'canonical_dim', 'app_allowed', 'venue'
    UNION ALL SELECT 'dim_referee', 'canonical_dim', 'app_allowed', 'referee'

    UNION ALL SELECT 'v_team_recent_form', 'gold_view', 'app_allowed', 'team recent form'
    UNION ALL SELECT 'v_head_to_head', 'gold_view', 'app_allowed', 'head to head summary'
    UNION ALL SELECT 'v_next_fixtures', 'gold_view', 'app_allowed', 'upcoming fixture list'
    UNION ALL SELECT 'v_match_card', 'gold_view', 'app_allowed', 'match card'
    UNION ALL SELECT 'v_prediction_features', 'gold_view', 'app_allowed', 'prediction feature set'
    UNION ALL SELECT 'v_dq_uniqueness_checks', 'gold_view', 'app_allowed', 'uniqueness quality checks'
    """

    for sql in [
        v_team_recent_form_sql,
        v_head_to_head_sql,
        v_next_fixtures_sql,
        v_match_card_sql,
        v_prediction_features_sql,
      v_data_contract_tables_sql,
    ]:
      execute_sql(sql)


def _run_uniqueness_checks(dataset: str) -> None:
    checks = [
      ("fact_fixture", "fixture_id"),
      ("fact_team_fixture", "team_id, fixture_id"),
      ("fact_fixture_team_stat", "fixture_id, team_id, stat_type"),
      ("dim_team", "team_id"),
      ("dim_competition", "competition_id"),
    ]

    failures: list[str] = []
    for table_name, cols in checks:
      sql = f"""
      SELECT COUNT(*) AS duplicate_groups
      FROM (
        SELECT {cols}, COUNT(*) AS c
        FROM {_ref(dataset, table_name)}
        GROUP BY {cols}
        HAVING COUNT(*) > 1
      )
      """
      df = run_query(sql)
      dup_groups = int(df.iloc[0]["duplicate_groups"])
      if dup_groups > 0:
        failures.append(f"{table_name}({cols}) duplicate groups: {dup_groups}")

    check_view_sql = f"""
    CREATE OR REPLACE VIEW {_ref(dataset, 'v_dq_uniqueness_checks')} AS
    SELECT 'fact_fixture' AS table_name, 'fixture_id' AS key_columns,
      (SELECT COUNT(*) FROM (SELECT fixture_id FROM {_ref(dataset, 'fact_fixture')} GROUP BY fixture_id HAVING COUNT(*) > 1)) AS duplicate_groups
    UNION ALL
    SELECT 'fact_team_fixture', 'team_id,fixture_id',
      (SELECT COUNT(*) FROM (SELECT team_id, fixture_id FROM {_ref(dataset, 'fact_team_fixture')} GROUP BY team_id, fixture_id HAVING COUNT(*) > 1))
    UNION ALL
    SELECT 'fact_fixture_team_stat', 'fixture_id,team_id,stat_type',
      (SELECT COUNT(*) FROM (SELECT fixture_id, team_id, stat_type FROM {_ref(dataset, 'fact_fixture_team_stat')} GROUP BY fixture_id, team_id, stat_type HAVING COUNT(*) > 1))
    UNION ALL
    SELECT 'dim_team', 'team_id',
      (SELECT COUNT(*) FROM (SELECT team_id FROM {_ref(dataset, 'dim_team')} GROUP BY team_id HAVING COUNT(*) > 1))
    UNION ALL
    SELECT 'dim_competition', 'competition_id',
      (SELECT COUNT(*) FROM (SELECT competition_id FROM {_ref(dataset, 'dim_competition')} GROUP BY competition_id HAVING COUNT(*) > 1))
    """
    execute_sql(check_view_sql)

    if failures:
      raise RuntimeError("Uniqueness checks failed: " + "; ".join(failures))


def run() -> None:
    dataset = os.environ["BIGQUERY_DATASET_ID"]

    print("=" * 68)
    print("Building canonical semantic model")
    print(f"Dataset: {os.environ['BIGQUERY_PROJECT_ID']}.{dataset}")
    print("=" * 68)

    _build_canonical_tables(dataset)
    print("Canonical fact and dimension tables created.")

    _build_gold_views(dataset)
    print("Gold views created.")

    _run_uniqueness_checks(dataset)
    print("Uniqueness checks passed.")

    print("=" * 68)
    print("Semantic model build completed successfully.")
    print("=" * 68)


if __name__ == "__main__":
    run()
