"""fact_match_team — canonical team-side match fact.

Grain: one row per (match_id, team_id). Two rows per match (home + away).

Combines:
  - raw_fixtures (latest snapshot) for outcome columns
  - raw_fixture_statistics (latest snapshot per stat_type) PIVOTED to wide columns

Stat columns are canonicalized to snake_case with explicit suffix units. Unknown
API stat_type values are dropped silently (we only project well-known ones).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from google.cloud import bigquery

from src.tools import bigquery_tools as _bq_tools

logger = logging.getLogger(__name__)

TABLE_NAME = "fact_match_team"


# Map API stat_type -> (canonical_column, value_kind: 'num'|'pct')
# pct values: parsed % (0-100 in raw, kept as 0-100 here).
STAT_MAP: list[tuple[str, str, str]] = [
    ("Shots on Goal",          "shots_on_target_count", "num"),
    ("Shots off Goal",         "shots_off_target_count","num"),
    ("Total Shots",            "shots_total_count",     "num"),
    ("Blocked Shots",          "shots_blocked_count",   "num"),
    ("Shots insidebox",        "shots_inside_box_count","num"),
    ("Shots outsidebox",       "shots_outside_box_count","num"),
    ("Fouls",                  "fouls_count",           "num"),
    ("Corner Kicks",           "corners_count",         "num"),
    ("Offsides",               "offsides_count",        "num"),
    ("Ball Possession",        "possession_pct",        "pct"),
    ("Yellow Cards",           "yellow_cards_count",    "num"),
    ("Red Cards",              "red_cards_count",       "num"),
    ("Goalkeeper Saves",       "goalkeeper_saves_count","num"),
    ("Total passes",           "passes_total_count",    "num"),
    ("Passes accurate",        "passes_accurate_count", "num"),
    ("Passes %",               "passes_accuracy_pct",   "pct"),
    ("expected_goals",         "xg",                    "num"),
    ("goals_prevented",        "goals_prevented",       "num"),
]


def _stat_columns_sql() -> str:
    parts: list[str] = []
    for stat_type, col, kind in STAT_MAP:
        parts.append(
            f"MAX(IF(stat_type = '{stat_type}', stat_value_num, NULL)) AS {col}"
        )
    return ",\n        ".join(parts)


def _stat_schema_fields() -> list[bigquery.SchemaField]:
    fields: list[bigquery.SchemaField] = []
    for _, col, kind in STAT_MAP:
        desc = {
            "shots_on_target_count":  "Shots on goal (count).",
            "shots_off_target_count": "Shots off goal (count).",
            "shots_total_count":      "Total shots attempted (count).",
            "shots_blocked_count":    "Shots blocked (count).",
            "shots_inside_box_count": "Shots from inside the box (count).",
            "shots_outside_box_count":"Shots from outside the box (count).",
            "fouls_count":            "Fouls committed (count).",
            "corners_count":          "Corner kicks (count).",
            "offsides_count":         "Offsides (count).",
            "possession_pct":         "Ball possession percentage (0-100).",
            "yellow_cards_count":     "Yellow cards received (count).",
            "red_cards_count":        "Red cards received (count).",
            "goalkeeper_saves_count": "Goalkeeper saves (count).",
            "passes_total_count":     "Total passes attempted (count).",
            "passes_accurate_count":  "Accurate passes (count).",
            "passes_accuracy_pct":    "Pass accuracy percentage (0-100).",
            "xg":                     "Expected goals for this team in this match.",
            "goals_prevented":        "Expected goals prevented by the goalkeeper.",
        }[col]
        fields.append(bigquery.SchemaField(col, "FLOAT64", description=desc))
    return fields


FACT_MATCH_TEAM_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("match_id",         "INT64", mode="REQUIRED",
                         description="FK to fact_match.match_id."),
    bigquery.SchemaField("team_id",          "INT64", mode="REQUIRED",
                         description="FK to dim_team.team_id."),
    bigquery.SchemaField("match_date",       "DATE",  mode="REQUIRED",
                         description="UTC calendar date of the match. Partition key."),
    bigquery.SchemaField("kickoff_at",       "TIMESTAMP",
                         description="UTC kickoff timestamp."),
    bigquery.SchemaField("season_year",      "INT64",  description="Season year."),
    bigquery.SchemaField("competition_id",   "INT64",  description="FK to dim_competition.competition_id."),
    bigquery.SchemaField("competition_name", "STRING", description="Denormalized competition name."),
    bigquery.SchemaField("team_name",        "STRING", description="Team name at match time."),
    bigquery.SchemaField("opponent_team_id", "INT64",  description="FK to dim_team.team_id for the opponent."),
    bigquery.SchemaField("opponent_team_name","STRING",description="Opponent team name at match time."),
    bigquery.SchemaField("is_home",          "BOOL",   description="True if this team played as the home side."),
    bigquery.SchemaField("goals_for",        "INT64",  description="Goals scored by this team (full time, includes ET, excludes shootout)."),
    bigquery.SchemaField("goals_against",    "INT64",  description="Goals conceded by this team (same scope as goals_for)."),
    bigquery.SchemaField("goal_diff",        "INT64",  description="goals_for - goals_against."),
    bigquery.SchemaField("result",           "STRING", description="W / D / L from this team's perspective. NULL if not completed."),
    bigquery.SchemaField("points",           "INT64",  description="League-style points: 3 for W, 1 for D, 0 for L. NULL if not completed."),
    bigquery.SchemaField("is_clean_sheet",   "BOOL",   description="True if completed and goals_against = 0."),
    bigquery.SchemaField("did_score",        "BOOL",   description="True if completed and goals_for > 0."),
    *_stat_schema_fields(),
    bigquery.SchemaField("built_at",         "TIMESTAMP", mode="REQUIRED",
                         description="UTC timestamp this fact row was built."),
]


def _fqn(t: str = TABLE_NAME) -> str:
    p = os.environ["BIGQUERY_PROJECT_ID"]; d = os.environ["BIGQUERY_DATASET_ID"]
    return f"{p}.{d}.{t}"


def _ref(t: str = TABLE_NAME) -> str:
    return f"`{_fqn(t)}`"


def build() -> dict[str, object]:
    client = _bq_tools._client()
    fqn = _fqn()
    now_iso = datetime.now(timezone.utc).isoformat()
    stats_select = _stat_columns_sql()

    select_sql = f"""
    WITH
    -- One row per match (latest snapshot)
    match_latest AS (
      SELECT * EXCEPT(rn)
      FROM (
        SELECT *,
          ROW_NUMBER() OVER (PARTITION BY match_id ORDER BY ingested_at DESC) AS rn
        FROM {_ref('raw_fixtures')}
        WHERE match_id IS NOT NULL
      )
      WHERE rn = 1
    ),
    -- Expand to one row per (match, team_side)
    match_sides AS (
      SELECT
        match_id, match_date, kickoff_at, season_year,
        competition_id, competition_name,
        home_team_id AS team_id, home_team_name AS team_name,
        away_team_id AS opponent_team_id, away_team_name AS opponent_team_name,
        TRUE  AS is_home,
        home_goals AS goals_for, away_goals AS goals_against,
        is_completed
      FROM match_latest
      WHERE home_team_id IS NOT NULL
      UNION ALL
      SELECT
        match_id, match_date, kickoff_at, season_year,
        competition_id, competition_name,
        away_team_id AS team_id, away_team_name AS team_name,
        home_team_id AS opponent_team_id, home_team_name AS opponent_team_name,
        FALSE AS is_home,
        away_goals AS goals_for, home_goals AS goals_against,
        is_completed
      FROM match_latest
      WHERE away_team_id IS NOT NULL
    ),
    -- Latest stat snapshot per (match_id, team_id, stat_type)
    stats_latest AS (
      SELECT * EXCEPT(rn)
      FROM (
        SELECT *,
          ROW_NUMBER() OVER (
            PARTITION BY match_id, team_id, stat_type
            ORDER BY ingested_at DESC
          ) AS rn
        FROM {_ref('raw_fixture_statistics')}
        WHERE match_id IS NOT NULL AND team_id IS NOT NULL
      )
      WHERE rn = 1
    ),
    -- Pivot stats wide per (match_id, team_id)
    stats_wide AS (
      SELECT
        match_id, team_id,
        {stats_select}
      FROM stats_latest
      GROUP BY match_id, team_id
    )
    SELECT
      ms.match_id,
      ms.team_id,
      ms.match_date,
      ms.kickoff_at,
      ms.season_year,
      ms.competition_id,
      ms.competition_name,
      ms.team_name,
      ms.opponent_team_id,
      ms.opponent_team_name,
      ms.is_home,
      ms.goals_for,
      ms.goals_against,
      (ms.goals_for - ms.goals_against)                        AS goal_diff,
      CASE
        WHEN NOT ms.is_completed THEN NULL
        WHEN ms.goals_for > ms.goals_against THEN 'W'
        WHEN ms.goals_for < ms.goals_against THEN 'L'
        WHEN ms.goals_for = ms.goals_against THEN 'D'
      END                                                       AS result,
      CASE
        WHEN NOT ms.is_completed THEN NULL
        WHEN ms.goals_for > ms.goals_against THEN 3
        WHEN ms.goals_for = ms.goals_against THEN 1
        ELSE 0
      END                                                       AS points,
      (ms.is_completed AND ms.goals_against = 0)                AS is_clean_sheet,
      (ms.is_completed AND ms.goals_for > 0)                    AS did_score,
      sw.shots_on_target_count, sw.shots_off_target_count, sw.shots_total_count,
      sw.shots_blocked_count, sw.shots_inside_box_count, sw.shots_outside_box_count,
      sw.fouls_count, sw.corners_count, sw.offsides_count,
      sw.possession_pct, sw.yellow_cards_count, sw.red_cards_count,
      sw.goalkeeper_saves_count, sw.passes_total_count, sw.passes_accurate_count,
      sw.passes_accuracy_pct, sw.xg, sw.goals_prevented,
      TIMESTAMP('{now_iso}') AS built_at
    FROM match_sides ms
    LEFT JOIN stats_wide sw USING (match_id, team_id)
    """

    table = bigquery.Table(fqn, schema=FACT_MATCH_TEAM_SCHEMA)
    table.description = (
        "Canonical team-side match fact. Grain: one row per (match_id, team_id). "
        "Outcome + wide-pivoted stats. Built from raw_fixtures + raw_fixture_statistics."
    )
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="match_date",
    )
    table.clustering_fields = ["team_id", "competition_id"]
    client.delete_table(fqn, not_found_ok=True)
    client.create_table(table)

    job_config = bigquery.QueryJobConfig(
        destination=fqn,
        write_disposition="WRITE_TRUNCATE",
    )
    client.query(select_sql, job_config=job_config).result()
    rows = int(client.get_table(fqn).num_rows)
    logger.info("fact_match_team built: %d rows", rows)
    return {"built": True, "rows": rows}


def run() -> dict[str, object]:
    return build()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(run())
