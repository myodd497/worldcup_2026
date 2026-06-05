"""mart_match_history — agent-facing list of completed matches.

Grain: one row per match_id where is_completed = TRUE.
Wide view denormalizing competition, venue, both team-sides' stats.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from google.cloud import bigquery

from src.tools import bigquery_tools as _bq_tools

logger = logging.getLogger(__name__)

TABLE_NAME = "mart_match_history"


# Wide-pivot stat columns mirror fact_match_team (home_*/away_*).
_STAT_COLS = [
    ("possession_pct",         "Ball possession percentage."),
    ("shots_total_count",      "Total shots."),
    ("shots_on_target_count",  "Shots on target."),
    ("shots_off_target_count", "Shots off target."),
    ("shots_blocked_count",    "Shots blocked."),
    ("shots_inside_box_count", "Shots from inside the box."),
    ("shots_outside_box_count","Shots from outside the box."),
    ("corners_count",          "Corner kicks."),
    ("fouls_count",            "Fouls committed."),
    ("offsides_count",         "Offsides."),
    ("yellow_cards_count",     "Yellow cards."),
    ("red_cards_count",        "Red cards."),
    ("goalkeeper_saves_count", "Goalkeeper saves."),
    ("passes_total_count",     "Total passes."),
    ("passes_accurate_count",  "Accurate passes."),
    ("passes_accuracy_pct",    "Pass accuracy percentage."),
    ("xg",                     "Expected goals."),
    ("goals_prevented",        "Goals prevented by GK."),
]


def _wide_stat_fields() -> list[bigquery.SchemaField]:
    out: list[bigquery.SchemaField] = []
    for col, desc in _STAT_COLS:
        out.append(bigquery.SchemaField(f"home_{col}", "FLOAT64", description=f"Home: {desc}"))
        out.append(bigquery.SchemaField(f"away_{col}", "FLOAT64", description=f"Away: {desc}"))
    return out


SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("match_id",          "INT64",     mode="REQUIRED",
                         description="FK to fact_match.match_id."),
    bigquery.SchemaField("match_date",        "DATE",      mode="REQUIRED",
                         description="UTC calendar date. Partition key."),
    bigquery.SchemaField("kickoff_at",        "TIMESTAMP", description="UTC kickoff timestamp."),
    bigquery.SchemaField("season_year",       "INT64",     description="Season year."),
    bigquery.SchemaField("competition_id",    "INT64",     description="FK to dim_competition."),
    bigquery.SchemaField("competition_name",  "STRING",    description="Competition name."),
    bigquery.SchemaField("competition_round", "STRING",    description="Round label."),
    bigquery.SchemaField("home_team_id",      "INT64",     description="FK to dim_team."),
    bigquery.SchemaField("home_team_name",    "STRING",    description="Home team name."),
    bigquery.SchemaField("away_team_id",      "INT64",     description="FK to dim_team."),
    bigquery.SchemaField("away_team_name",    "STRING",    description="Away team name."),
    bigquery.SchemaField("venue_name",        "STRING",    description="Stadium."),
    bigquery.SchemaField("venue_city",        "STRING",    description="City."),
    bigquery.SchemaField("home_goals",        "INT64",     description="Final home goals."),
    bigquery.SchemaField("away_goals",        "INT64",     description="Final away goals."),
    bigquery.SchemaField("has_extra_time",    "BOOL",      description="Extra time played."),
    bigquery.SchemaField("has_penalty_shootout","BOOL",    description="Decided by penalty shootout."),
    bigquery.SchemaField("winner_team_id",    "INT64",     description="FK to dim_team. NULL on draw."),
    bigquery.SchemaField("is_draw",           "BOOL",      description="Match ended as a draw in regular time."),
    bigquery.SchemaField("match_result_label","STRING",    description="Human-readable result e.g. 'Home win', 'Away win', 'Draw', 'Home win on penalties'."),
    *_wide_stat_fields(),
    bigquery.SchemaField("built_at",          "TIMESTAMP", mode="REQUIRED",
                         description="UTC timestamp this mart row was built."),
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

    home_stats_sql = ",\n      ".join(f"h.{c} AS home_{c}" for c, _ in _STAT_COLS)
    away_stats_sql = ",\n      ".join(f"a.{c} AS away_{c}" for c, _ in _STAT_COLS)

    sql = f"""
    SELECT
      m.match_id,
      m.match_date,
      m.kickoff_at,
      m.season_year,
      m.competition_id,
      m.competition_name,
      m.competition_round,
      m.home_team_id,
      m.home_team_name,
      m.away_team_id,
      m.away_team_name,
      m.venue_name,
      m.venue_city,
      m.home_goals,
      m.away_goals,
      m.has_extra_time,
      m.has_penalty_shootout,
      m.winner_team_id,
      m.is_draw,
      CASE
        WHEN m.is_draw                                            THEN 'Draw'
        WHEN m.has_penalty_shootout AND m.winner_team_id = m.home_team_id THEN 'Home win on penalties'
        WHEN m.has_penalty_shootout AND m.winner_team_id = m.away_team_id THEN 'Away win on penalties'
        WHEN m.winner_team_id = m.home_team_id                    THEN 'Home win'
        WHEN m.winner_team_id = m.away_team_id                    THEN 'Away win'
        ELSE 'Unknown'
      END AS match_result_label,
      {home_stats_sql},
      {away_stats_sql},
      TIMESTAMP('{now_iso}') AS built_at
    FROM {_ref('fact_match')} m
    LEFT JOIN {_ref('fact_match_team')} h
      ON h.match_id = m.match_id AND h.team_id = m.home_team_id
    LEFT JOIN {_ref('fact_match_team')} a
      ON a.match_id = m.match_id AND a.team_id = m.away_team_id
    WHERE m.is_completed
    """

    table = bigquery.Table(fqn, schema=SCHEMA)
    table.description = "Agent-facing completed match history with wide-pivoted home/away stats."
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY, field="match_date")
    table.clustering_fields = ["competition_id", "home_team_id", "away_team_id"]
    client.delete_table(fqn, not_found_ok=True)
    client.create_table(table)
    client.query(sql, job_config=bigquery.QueryJobConfig(
        destination=fqn, write_disposition="WRITE_TRUNCATE")).result()
    rows = int(client.get_table(fqn).num_rows)
    logger.info("mart_match_history built: %d rows", rows)
    return {"built": True, "rows": rows}


def run() -> dict[str, object]:
    return build()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(run())
