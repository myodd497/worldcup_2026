"""SQL few-shot library — canonical question→SQL templates retrieved by similarity.

Each example has:
  - question (natural language)
  - tags (keywords for retrieval)
  - sql (the canonical pattern with <PLACEHOLDERS>)
  - notes (gotchas)

`select_few_shots(question, k=3)` returns the most relevant examples,
formatted as a block to drop into the SQL generator prompt.

This replaces the 11 recipes that were hard-coded into the system prompt.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class FewShot:
    question: str
    tags: tuple[str, ...]
    sql: str
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Canonical examples
# ─────────────────────────────────────────────────────────────────────────────

FEW_SHOTS: tuple[FewShot, ...] = (
    FewShot(
        question="Which teams will participate in the World Cup 2026?",
        tags=("participants", "world cup", "wc2026", "teams", "qualified"),
        sql="""
SELECT team_id, team_name, country_name
FROM dim_team
WHERE is_wc2026_participant = TRUE
ORDER BY team_name
""",
    ),

    FewShot(
        question="What is Mexico's current form? Show their last 10 games.",
        tags=("form", "last 10", "recent", "team", "results", "streak"),
        sql="""
-- Form summary
SELECT *
FROM mart_team_form
WHERE team_id = <TEAM_ID>;

-- Last 10 detailed games
SELECT
  fmt.match_date,
  dt.team_name AS opponent,
  fmt.goals_for,
  fmt.goals_against,
  fmt.result
FROM fact_match_team fmt
JOIN dim_team dt ON dt.team_id = fmt.opponent_team_id
WHERE fmt.team_id = <TEAM_ID>
  AND fmt.result IS NOT NULL
ORDER BY fmt.match_date DESC
LIMIT 10
""",
        notes="Resolve 'Mexico' → team_id via the entity resolver before running.",
    ),

    FewShot(
        question="Top 10 Portugal players by goal contributions (goals + assists) in the last 10 Portugal games.",
        tags=("top", "players", "goal contributions", "goals", "assists", "last n"),
        sql="""
WITH ranked AS (
  SELECT
    fp.player_id,
    fp.team_id,
    fp.goals,
    fp.assists,
    fp.goal_contributions,
    fp.match_date,
    ROW_NUMBER() OVER (PARTITION BY fp.team_id ORDER BY fp.match_date DESC) AS rn
  FROM fact_player_match_stat fp
  WHERE fp.team_id = <TEAM_ID>
),
last_n AS (
  SELECT * FROM ranked WHERE rn <= 10
)
SELECT
  dp.player_name,
  SUM(goals)              AS goals,
  SUM(assists)            AS assists,
  SUM(goal_contributions) AS goal_contributions,
  COUNT(*)                AS matches
FROM last_n
JOIN dim_player dp USING (player_id)
GROUP BY dp.player_name
ORDER BY goal_contributions DESC
LIMIT 10
""",
        notes="ROW_NUMBER ranks ALL of the team's matches by recency, then we filter <= 10. goal_contributions is pre-computed.",
    ),

    FewShot(
        question="Top 10 Spain players with worst discipline (yellow + red cards) in the last 10 Spain games.",
        tags=("discipline", "yellow", "red", "cards", "worst", "players", "last n"),
        sql="""
WITH ranked AS (
  SELECT
    fp.player_id,
    fp.team_id,
    fp.yellow_cards,
    fp.red_cards,
    fp.match_date,
    ROW_NUMBER() OVER (PARTITION BY fp.team_id ORDER BY fp.match_date DESC) AS rn
  FROM fact_player_match_stat fp
  WHERE fp.team_id = <TEAM_ID>
),
last_n AS (
  SELECT * FROM ranked WHERE rn <= 10
)
SELECT
  dp.player_name,
  SUM(yellow_cards)                          AS yellows,
  SUM(red_cards)                             AS reds,
  SUM(yellow_cards) + 2 * SUM(red_cards)     AS discipline_score
FROM last_n
JOIN dim_player dp USING (player_id)
GROUP BY dp.player_name
ORDER BY discipline_score DESC
LIMIT 10
""",
    ),

    FewShot(
        question="For all WC2026 teams, which player has the most minutes played in the last month?",
        tags=("minutes", "played", "all teams", "wc2026", "last month", "most"),
        sql="""
SELECT
  dp.player_name,
  dt.team_name,
  SUM(fp.minutes_played) AS total_minutes,
  COUNT(*)               AS matches
FROM fact_player_match_stat fp
JOIN dim_team   dt ON dt.team_id = fp.team_id
JOIN dim_player dp ON dp.player_id = fp.player_id
WHERE dt.is_wc2026_participant = TRUE
  AND fp.match_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH)
GROUP BY dp.player_name, dt.team_name
ORDER BY total_minutes DESC
LIMIT 25
""",
        notes="Use DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH) for 'last month'.",
    ),

    FewShot(
        question="Team with most shots on target and most shots conceded in the last 5 games.",
        tags=("shots", "shots on target", "conceded", "last 5", "team"),
        sql="""
WITH ranked AS (
  SELECT
    fmt.team_id,
    fmt.match_id,
    fmt.match_date,
    fmt.shots_on_target_count,
    ROW_NUMBER() OVER (PARTITION BY fmt.team_id ORDER BY fmt.match_date DESC) AS rn
  FROM fact_match_team fmt
  WHERE fmt.shots_on_target_count IS NOT NULL
),
last5 AS (SELECT * FROM ranked WHERE rn <= 5),
own AS (
  SELECT team_id, SUM(shots_on_target_count) AS shots_on_target
  FROM last5
  GROUP BY team_id
),
opp AS (
  SELECT
    a.team_id,
    SUM(b.shots_on_target_count) AS shots_on_target_conceded
  FROM last5 a
  JOIN fact_match_team b
    ON a.match_id = b.match_id AND b.team_id != a.team_id
  WHERE b.shots_on_target_count IS NOT NULL
  GROUP BY a.team_id
)
SELECT
  dt.team_name,
  own.shots_on_target,
  opp.shots_on_target_conceded
FROM own
JOIN opp USING (team_id)
JOIN dim_team dt USING (team_id)
WHERE dt.is_wc2026_participant = TRUE
ORDER BY own.shots_on_target DESC
LIMIT 25
""",
        notes="Self-join fact_match_team on match_id to derive 'conceded' from opponent's row.",
    ),

    FewShot(
        question="Best defence and best attack across WC2026 teams (goals scored / conceded).",
        tags=("best", "defense", "defence", "attack", "goals", "scored", "conceded"),
        sql="""
SELECT
  team_name,
  goals_for_total,
  goals_against_total,
  goals_for_per_match,
  goals_against_per_match,
  clean_sheets,
  matches_played
FROM mart_team_profile
WHERE is_wc2026_participant = TRUE AND matches_played > 0
ORDER BY goals_for_per_match DESC
LIMIT 25
""",
        notes="Sort ASC by goals_against_per_match for best defence. mart_team_profile is lifetime; use fact_match_team with date filters for a specific window.",
    ),

    FewShot(
        question="Teams with highest and lowest average ball possession percentage.",
        tags=("possession", "ball", "highest", "lowest", "team"),
        sql="""
SELECT
  dt.team_name,
  ROUND(AVG(fmt.possession_pct), 1) AS avg_possession_pct,
  COUNT(*) AS matches_with_data
FROM fact_match_team fmt
JOIN dim_team dt USING (team_id)
WHERE fmt.possession_pct IS NOT NULL
  AND fmt.result IS NOT NULL
  AND dt.is_wc2026_participant = TRUE
GROUP BY dt.team_name
HAVING matches_with_data >= 3
ORDER BY avg_possession_pct DESC
LIMIT 25
""",
        notes="possession_pct is often NULL in international fixtures — always filter NOT NULL and require a min sample size.",
    ),

    FewShot(
        question="Best attacking and defending team in the 2022 World Cup.",
        tags=("attack", "defence", "best", "world cup", "2022", "specific edition"),
        sql="""
SELECT
  dt.team_name,
  SUM(fmt.goals_for)     AS goals_scored,
  SUM(fmt.goals_against) AS goals_conceded,
  COUNT(*)               AS matches_played,
  ROUND(SUM(fmt.goals_for)     / COUNT(*), 2) AS gs_per_match,
  ROUND(SUM(fmt.goals_against) / COUNT(*), 2) AS gc_per_match,
  COUNTIF(fmt.is_clean_sheet)                 AS clean_sheets
FROM fact_match_team fmt
JOIN dim_team dt USING (team_id)
WHERE fmt.competition_id = 1
  AND fmt.season_year = 2022
  AND fmt.result IS NOT NULL
GROUP BY dt.team_name
ORDER BY goals_scored DESC
LIMIT 25
""",
    ),

    FewShot(
        question="Top 10 teams across all World Cup history.",
        tags=("top", "all-time", "world cup history", "all editions", "best ever"),
        sql="""
SELECT
  dt.team_name,
  COUNT(*)                                          AS matches_played,
  COUNTIF(fmt.result = 'W')                         AS wins,
  COUNTIF(fmt.result = 'D')                         AS draws,
  COUNTIF(fmt.result = 'L')                         AS losses,
  SUM(fmt.goals_for)                                AS goals_for,
  SUM(fmt.goals_against)                            AS goals_against,
  SUM(fmt.goals_for) - SUM(fmt.goals_against)       AS goal_diff,
  SUM(IF(fmt.result='W',3,IF(fmt.result='D',1,0)))  AS points,
  ROUND(SAFE_DIVIDE(SUM(IF(fmt.result='W',3,IF(fmt.result='D',1,0))), COUNT(*)), 2) AS pts_per_match,
  ROUND(SAFE_DIVIDE(COUNTIF(fmt.result='W'), COUNT(*)) * 100, 1) AS win_pct
FROM fact_match_team fmt
JOIN dim_team dt USING (team_id)
WHERE fmt.competition_id = 1
  AND fmt.result IS NOT NULL
GROUP BY dt.team_name
HAVING matches_played >= 5
ORDER BY pts_per_match DESC, win_pct DESC, goal_diff DESC
LIMIT 10
""",
        notes="Rank by points-per-match (3/1/0), tiebreak on win_pct then goal_diff.",
    ),

    FewShot(
        question="Head-to-head between Portugal and Spain.",
        tags=("head", "h2h", "between", "two teams", "vs"),
        sql="""
SELECT *
FROM mart_head_to_head
WHERE team_lo_id = LEAST(<TEAM_A_ID>, <TEAM_B_ID>)
  AND team_hi_id = GREATEST(<TEAM_A_ID>, <TEAM_B_ID>)
""",
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {"the","a","an","is","are","of","for","in","on","with","and","or","to","by","what","which","who","how","most","best","top","last","next","all","show","tell","give","me","that"}


def _tokens(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z][a-z0-9]+", (text or "").lower())
        if w not in _STOPWORDS and len(w) > 2
    }


def _score(shot: FewShot, q_tokens: set[str]) -> float:
    tag_terms = {t for tag in shot.tags for t in re.split(r"\s+", tag.lower())}
    q_terms = _tokens(shot.question)
    score = 0.0
    for t in q_tokens:
        if t in tag_terms:
            score += 2.0
        if t in q_terms:
            score += 1.0
    return score


def select_few_shots(question: str, k: int = 3) -> list[FewShot]:
    """Hybrid retrieval: semantic embeddings + keyword tag overlap as tiebreaker."""
    from src.agents.embeddings import cosine, embed, embed_one

    q_tokens = _tokens(question)
    kw = {i: _score(s, q_tokens) for i, s in enumerate(FEW_SHOTS)}

    q_vec = embed_one((question or "").strip())
    sem: dict[int, float] = {}
    if q_vec is not None:
        shot_vecs = embed([f"{s.question} | tags: {', '.join(s.tags)}" for s in FEW_SHOTS])
        for i, vec in enumerate(shot_vecs):
            sem[i] = cosine(q_vec, vec) if vec is not None else 0.0

    def _final(i: int) -> float:
        k_norm = min(1.0, kw[i] / 5.0)
        if sem:
            return 0.75 * sem.get(i, 0.0) + 0.25 * k_norm
        return k_norm

    ranked = sorted(range(len(FEW_SHOTS)), key=_final, reverse=True)
    out: list[FewShot] = []
    for i in ranked:
        if _final(i) > 0:
            out.append(FEW_SHOTS[i])
        if len(out) >= k:
            break
    return out


def format_few_shots(question: str, k: int = 3) -> str:
    """Drop-in block for the SQL generator prompt."""
    shots = select_few_shots(question, k=k)
    if not shots:
        return "_(no matching example queries found)_"
    parts = [f"## Similar example queries (top {len(shots)})"]
    for i, s in enumerate(shots, 1):
        parts.append(f"\n### Example {i}\n**Q:** {s.question}")
        parts.append(f"```sql\n{s.sql.strip()}\n```")
        if s.notes:
            parts.append(f"_Notes: {s.notes}_")
    return "\n".join(parts)
