# Data Contract (Canonical Model)

This document defines the canonical BigQuery object per business concept and the app-query rules.

## App Query Rule

- Application/runtime queries must read only from canonical `fact_` / `dim_` tables and `v_` gold views.
- Source ingestion tables are source-only and must not be used directly by app features.

## Source-Only Tables

- `fixtures_historical`
- `team_match_history`
- `fixture_events`
- `fixture_stats`
- `standings`
- `team_stats`

## Canonical Object Per Business Concept

- Fixture: `fact_fixture`
- Team fixture participation: `fact_team_fixture`
- Fixture event timeline: `fact_fixture_event`
- Fixture team stat line: `fact_fixture_team_stat`
- Team master: `dim_team`
- Competition master: `dim_competition`
- Venue master: `dim_venue`
- Referee master: `dim_referee`

## Gold Views (App-Facing)

- Team form: `v_team_recent_form`
- Head-to-head summary: `v_head_to_head`
- Upcoming fixtures: `v_next_fixtures`
- Match card composite: `v_match_card`
- Prediction feature set: `v_prediction_features`
- Data quality checks: `v_dq_uniqueness_checks`
- Table usage contract: `v_data_contract_tables`

## Uniqueness Guarantees

- `fact_fixture`: unique by `fixture_id`
- `fact_team_fixture`: unique by (`team_id`, `fixture_id`)
- `fact_fixture_team_stat`: unique by (`fixture_id`, `team_id`, `stat_type`)
- `dim_team`: unique by `team_id`
- `dim_competition`: unique by `competition_id`
