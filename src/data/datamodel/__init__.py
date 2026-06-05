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
