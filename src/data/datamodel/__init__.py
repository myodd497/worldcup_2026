"""New canonical data model (raw / dim / fact / mart).

Each module here owns exactly one BigQuery object. Builders are idempotent and
prefer to source data from existing BQ tables before spending API budget.

Naming conventions enforced across this package:
  - snake_case columns
  - <entity>_id for keys; foreign keys keep the same name as the primary key
  - *_at  -> TIMESTAMP (UTC)
  - *_date -> DATE
  - *_count / *_pct / *_minutes suffixes for measures
  - is_* / has_* for booleans
  - *_status UPPERCASE STRING enums
  - home_* / away_* for sides (never team_a / local / visitor)
"""

# Re-export all module schemas for catalog introspection
from src.data.datamodel import (
    # RAW
    raw_fixtures,
    raw_fixture_events,
    raw_fixture_statistics,
    raw_standings,
    raw_player_stats,
    # DIM
    dim_team,
    dim_competition,
    dim_venue,
    dim_date,
    dim_player,
    # FACT
    fact_match,
    fact_match_team,
    fact_match_event,
    fact_standings_snapshot,
    fact_player_match_stat,
    # MART
    mart_team_profile,
    mart_team_form,
    mart_head_to_head,
    mart_match_history,
    mart_match_upcoming,
    mart_tournament_state,
)
