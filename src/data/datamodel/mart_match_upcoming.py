"""mart_match_upcoming — agent-facing list of upcoming / live matches.

Grain: one row per match_id where match_status IN ('SCHEDULED','LIVE','POSTPONED')
       AND match_date >= CURRENT_DATE() - 1.
Wide view enriched with team form & h2h.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from google.cloud import bigquery

from src.tools import bigquery_tools as _bq_tools

logger = logging.getLogger(__name__)

TABLE_NAME = "mart_match_upcoming"


SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("match_id",            "INT64",     mode="REQUIRED",
                         description="FK to fact_match.match_id."),
    bigquery.SchemaField("match_date",          "DATE",      mode="REQUIRED",
                         description="UTC calendar date. Partition key."),
    bigquery.SchemaField("kickoff_at",          "TIMESTAMP", description="UTC kickoff timestamp."),
    bigquery.SchemaField("days_until_kickoff",  "INT64",     description="Days from CURRENT_DATE to match_date. Negative if past."),
    bigquery.SchemaField("competition_id",      "INT64",     description="FK to dim_competition."),
    bigquery.SchemaField("competition_name",    "STRING",    description="Competition name."),
    bigquery.SchemaField("competition_round",   "STRING",    description="Round label."),
    bigquery.SchemaField("match_status",        "STRING",    description="SCHEDULED, LIVE, POSTPONED."),
    bigquery.SchemaField("home_team_id",        "INT64",     description="FK to dim_team."),
    bigquery.SchemaField("home_team_name",      "STRING",    description="Home team name."),
    bigquery.SchemaField("home_is_wc2026",      "BOOL",      description="True if home team is WC2026 participant."),
    bigquery.SchemaField("away_team_id",        "INT64",     description="FK to dim_team."),
    bigquery.SchemaField("away_team_name",      "STRING",    description="Away team name."),
    bigquery.SchemaField("away_is_wc2026",      "BOOL",      description="True if away team is WC2026 participant."),
    bigquery.SchemaField("venue_name",          "STRING",    description="Stadium."),
    bigquery.SchemaField("venue_city",          "STRING",    description="City."),
    bigquery.SchemaField("home_form_last5",     "STRING",    description="Most recent 5 results for home team (newest first), e.g. 'WWDLW'."),
    bigquery.SchemaField("away_form_last5",     "STRING",    description="Most recent 5 results for away team (newest first)."),
    bigquery.SchemaField("h2h_matches",         "INT64",     description="Past completed meetings between these two teams."),
    bigquery.SchemaField("h2h_home_wins",       "INT64",     description="Past wins by today's home team across all past meetings."),
    bigquery.SchemaField("h2h_away_wins",       "INT64",     description="Past wins by today's away team across all past meetings."),
    bigquery.SchemaField("h2h_draws",           "INT64",     description="Past draws between these two teams."),
    bigquery.SchemaField("built_at",            "TIMESTAMP", mode="REQUIRED",
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

    sql = f"""
    WITH upc AS (
      SELECT *
      FROM {_ref('fact_match')}
      WHERE match_status IN ('SCHEDULED','LIVE','POSTPONED')
        AND match_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
    ),
    -- last 5 completed results per team (chronological newest first)
    team_form AS (
      SELECT team_id,
             STRING_AGG(result, '' ORDER BY match_date DESC, match_id DESC LIMIT 5) AS form5
      FROM (
        SELECT team_id, match_date, match_id, result
        FROM {_ref('fact_match_team')}
        WHERE result IS NOT NULL
      )
      GROUP BY team_id
    ),
    -- head-to-head over completed matches; canonical pair key
    h2h_pair AS (
      SELECT
        LEAST(home_team_id, away_team_id)    AS team_lo,
        GREATEST(home_team_id, away_team_id) AS team_hi,
        COUNTIF(is_completed)                                                          AS h2h_matches,
        COUNTIF(is_completed AND winner_team_id = LEAST(home_team_id, away_team_id))   AS lo_wins,
        COUNTIF(is_completed AND winner_team_id = GREATEST(home_team_id, away_team_id))AS hi_wins,
        COUNTIF(is_completed AND winner_team_id IS NULL AND NOT is_draw IS FALSE)      AS draws_raw,
        COUNTIF(is_completed AND is_draw)                                               AS draws
      FROM {_ref('fact_match')}
      WHERE home_team_id IS NOT NULL AND away_team_id IS NOT NULL AND is_completed
      GROUP BY team_lo, team_hi
    )
    SELECT
      u.match_id,
      u.match_date,
      u.kickoff_at,
      DATE_DIFF(u.match_date, CURRENT_DATE(), DAY) AS days_until_kickoff,
      u.competition_id,
      u.competition_name,
      u.competition_round,
      u.match_status,
      u.home_team_id,
      u.home_team_name,
      IFNULL(th.is_wc2026_participant, FALSE) AS home_is_wc2026,
      u.away_team_id,
      u.away_team_name,
      IFNULL(ta.is_wc2026_participant, FALSE) AS away_is_wc2026,
      u.venue_name,
      u.venue_city,
      fh.form5 AS home_form_last5,
      fa.form5 AS away_form_last5,
      IFNULL(h.h2h_matches, 0) AS h2h_matches,
      CASE
        WHEN h.team_lo = u.home_team_id THEN IFNULL(h.lo_wins, 0)
        ELSE                                  IFNULL(h.hi_wins, 0)
      END AS h2h_home_wins,
      CASE
        WHEN h.team_lo = u.away_team_id THEN IFNULL(h.lo_wins, 0)
        ELSE                                  IFNULL(h.hi_wins, 0)
      END AS h2h_away_wins,
      IFNULL(h.draws, 0) AS h2h_draws,
      TIMESTAMP('{now_iso}') AS built_at
    FROM upc u
    LEFT JOIN {_ref('dim_team')} th ON th.team_id = u.home_team_id
    LEFT JOIN {_ref('dim_team')} ta ON ta.team_id = u.away_team_id
    LEFT JOIN team_form fh          ON fh.team_id = u.home_team_id
    LEFT JOIN team_form fa          ON fa.team_id = u.away_team_id
    LEFT JOIN h2h_pair  h           ON h.team_lo  = LEAST(u.home_team_id,  u.away_team_id)
                                   AND h.team_hi  = GREATEST(u.home_team_id, u.away_team_id)
    """

    table = bigquery.Table(fqn, schema=SCHEMA)
    table.description = "Agent-facing upcoming/live matches with form and h2h context."
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY, field="match_date")
    table.clustering_fields = ["competition_id", "home_team_id", "away_team_id"]
    client.delete_table(fqn, not_found_ok=True)
    client.create_table(table)
    client.query(sql, job_config=bigquery.QueryJobConfig(
        destination=fqn, write_disposition="WRITE_TRUNCATE")).result()
    rows = int(client.get_table(fqn).num_rows)
    logger.info("mart_match_upcoming built: %d rows", rows)
    return {"built": True, "rows": rows}


def run() -> dict[str, object]:
    return build()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(run())
