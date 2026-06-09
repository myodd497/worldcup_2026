"""
Dashboard charts for the World Cup 2026 Streamlit app.

Fetches data from BigQuery and renders Plotly charts for:
- Top Scorers bar chart
- Team Attack vs Defense scatter plot
- Group standings lollipop chart
- Player comparison radar chart
"""
from __future__ import annotations

import os
from datetime import date
from typing import Any

import pandas as pd

# Lazy import plotly — only needed on the dashboard tab
try:
    import plotly.express as px
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False
    px = None  # type: ignore
    go = None  # type: ignore


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
_GOLD = "#f0c040"
_NAVY = "#0a1f2e"
_ACCENT_BLUE = "#1e90ff"
_QUALIFY_GREEN = "rgba(80, 200, 120, 0.6)"
_ELIMINATED_GRAY = "rgba(180, 180, 180, 0.4)"


def _wc_where_clause() -> str:
    """Return WHERE clause for WC2026 player queries.
    Before June 11, 2026: uses last 2 friendly matches for all WC2026 teams.
    On/after June 11, 2026: queries all WC2026 matches.
    """
    today = date.today()
    wc_start = date(2026, 6, 11)
    if today < wc_start:
        return """AND fps.match_id IN (
            WITH wc_teams AS (
                SELECT team_id FROM dim_team WHERE is_wc2026_participant = TRUE
            ),
            team_matches AS (
                SELECT fm.match_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY wc.team_id
                        ORDER BY fm.match_date DESC, fm.match_id DESC
                    ) AS rn
                FROM fact_match fm
                JOIN wc_teams wc ON wc.team_id = fm.home_team_id OR wc.team_id = fm.away_team_id
                WHERE fm.competition_id != 1
                    AND fm.match_status = 'FINISHED'
                    AND fm.match_date < '2026-06-11'
                    AND fm.home_goals IS NOT NULL
            )
            SELECT match_id FROM team_matches WHERE rn <= 2
        )"""
    else:
        return "AND fps.competition_id = 1 AND fps.season_year = 2026"


def _wc_where_clause_team() -> str:
    """Return WHERE clause for team-level WC2026 queries."""
    today = date.today()
    wc_start = date(2026, 6, 11)
    if today < wc_start:
        return """AND fmt.match_id IN (
            WITH wc_teams AS (
                SELECT team_id FROM dim_team WHERE is_wc2026_participant = TRUE
            ),
            team_matches AS (
                SELECT fm.match_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY wc.team_id
                        ORDER BY fm.match_date DESC, fm.match_id DESC
                    ) AS rn
                FROM fact_match fm
                JOIN wc_teams wc ON wc.team_id = fm.home_team_id OR wc.team_id = fm.away_team_id
                WHERE fm.competition_id != 1
                    AND fm.match_status = 'FINISHED'
                    AND fm.match_date < '2026-06-11'
                    AND fm.home_goals IS NOT NULL
            )
            SELECT match_id FROM team_matches WHERE rn <= 2
        )"""
    else:
        return "AND fmt.competition_id = 1 AND fmt.season_year = 2026"


def _run_bq(sql: str) -> pd.DataFrame:
    """Run a BigQuery query and return a DataFrame."""
    from src.tools.bigquery_tools import run_query
    return run_query(sql)


def _get_project_dataset() -> tuple[str, str] | None:
    p = os.environ.get("BIGQUERY_PROJECT_ID")
    d = os.environ.get("BIGQUERY_DATASET_ID")
    return (p, d) if p and d else None


# ---------------------------------------------------------------------------
# 1. Top Scorers Horizontal Bar Chart
# ---------------------------------------------------------------------------

def top_scorers_bar_chart(metric: str = "goals") -> go.Figure | None:
    """Returns a horizontal bar chart of top 100 players by the selected metric.

    Args:
        metric: Metric key (e.g. 'goals', 'assists', 'rating', 'key passes', etc.).
                Must be one of the keys in worldcup_style._TOP_SCORER_METRICS.
    """
    if not _PLOTLY_AVAILABLE:
        return None
    creds = _get_project_dataset()
    if not creds:
        return None
    project, dataset = creds

    # Resolve metric to db column and label
    from src.server.worldcup_style import _TOP_SCORER_METRICS
    metric = metric.lower()
    if metric not in _TOP_SCORER_METRICS:
        metric = "goals"
    metric_label, col_header, db_column = _TOP_SCORER_METRICS[metric]

    sql = f"""
        SELECT dp.player_name, dt.team_name, SUM({db_column}) AS metric_val
        FROM `{project}.{dataset}.fact_player_match_stat` fps
        JOIN `{project}.{dataset}.dim_player` dp USING (player_id)
        JOIN `{project}.{dataset}.dim_team` dt ON dt.team_id = fps.team_id
        WHERE 1=1
          {_wc_where_clause()}
          AND {db_column} > 0
        GROUP BY dp.player_name, dt.team_name
        ORDER BY metric_val DESC
        LIMIT 100
    """
    try:
        df = _run_bq(sql)
    except Exception:
        return None
    if df.empty:
        return None

    # Inject flag emojis
    from src.server.worldcup_style import get_flag
    df["label"] = df.apply(
        lambda r: f"{get_flag(r['team_name']) or ''} {r['player_name']}", axis=1
    )

    # Calculate chart height: ~28px per bar, min 400px, max 1200px
    n_rows = len(df)
    chart_height = max(400, min(1200, n_rows * 28))

    fig = px.bar(
        df.sort_values("metric_val", ascending=True),
        x="metric_val", y="label", orientation="h",
        title=f"⚽ Top Scorers — {metric_label}",
        text="metric_val",
        color="metric_val",
        color_continuous_scale=["#1e90ff", "#f0c040"],
    )
    fig.update_traces(
        textposition="outside",
        marker_line_width=0,
        hovertemplate=f"%{{y}}: <b>%{{x}}</b> {metric_label}<extra></extra>",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(10,31,46,0.6)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc",
        title_font_color="#f0c040",
        xaxis_title="",
        yaxis_title="",
        height=chart_height,
        margin=dict(l=10, r=10, t=46, b=10),
        coloraxis_showscale=False,
        # Enable vertical scroll when the chart is taller than the container
        dragmode="pan",
        yaxis=dict(
            automargin=True,
            fixedrange=False,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Team Attack vs Defense Scatter Plot
# ---------------------------------------------------------------------------

def team_attack_defense_scatter() -> go.Figure | None:
    """Returns a scatter plot: goals scored vs conceded per WC2026 team."""
    creds = _get_project_dataset()
    if not creds:
        return None
    project, dataset = creds

    sql = f"""
        SELECT
            dt.team_name,
            SUM(fmt.goals_for) AS goals_scored,
            SUM(fmt.goals_against) AS goals_conceded,
            COUNT(*) AS matches
        FROM `{project}.{dataset}.fact_match_team` fmt
        JOIN `{project}.{dataset}.dim_team` dt ON dt.team_id = fmt.team_id
        WHERE 1=1
          {_wc_where_clause_team()}
          AND fmt.result IS NOT NULL
        GROUP BY dt.team_name
        HAVING COUNT(*) > 0
        ORDER BY goals_scored DESC
    """
    try:
        df = _run_bq(sql)
    except Exception:
        return None
    if df.empty:
        return None

    from src.server.worldcup_style import get_flag
    df["flag"] = df["team_name"].apply(lambda t: get_flag(t) or "")
    df["label"] = df["flag"] + " " + df["team_name"]
    df["goals_per_game"] = df["goals_scored"] / df["matches"]
    df["conceded_per_game"] = df["goals_conceded"] / df["matches"]

    # Compute overall averages for quadrant lines
    avg_gf = df["goals_per_game"].mean()
    avg_ga = df["conceded_per_game"].mean()

    fig = go.Figure()

    # Quadrant backgrounds
    fig.add_shape(type="rect", x0=avg_gf, x1=999, y0=0, y1=avg_ga,
                  fillcolor="rgba(80,200,120,0.08)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=-999, x1=avg_gf, y0=avg_ga, y1=999,
                  fillcolor="rgba(255,100,100,0.06)", line_width=0, layer="below")

    # Average lines
    fig.add_hline(y=avg_ga, line_dash="dash", line_color="rgba(255,255,255,0.15)",
                  annotation_text=f"Avg GA: {avg_ga:.1f}", annotation_position="top right")
    fig.add_vline(x=avg_gf, line_dash="dash", line_color="rgba(255,255,255,0.15)",
                  annotation_text=f"Avg GF: {avg_gf:.1f}", annotation_position="top right")

    fig.add_trace(go.Scatter(
        x=df["goals_per_game"], y=df["conceded_per_game"],
        mode="markers+text",
        text=df["flag"],
        textposition="middle center",
        textfont_size=16,
        marker=dict(
            size=df["matches"] * 6 + 20,
            color=df["goals_per_game"],
            colorscale=["#1e90ff", "#f0c040"],
            showscale=False,
            line=dict(width=1, color="rgba(255,255,255,0.15)"),
        ),
        hovertemplate=(
            "%{customdata[0]}<br>"
            "GF/game: <b>%{x:.1f}</b><br>"
            "GA/game: <b>%{y:.1f}</b><br>"
            "Matches: <b>%{customdata[1]}</b><extra></extra>"
        ),
        customdata=df[["team_name", "matches"]],
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(10,31,46,0.6)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc",
        title=dict(text="🛡️ Attack vs Defense (per game)", font_color="#f0c040"),
        xaxis_title="Goals Scored / Game",
        yaxis_title="Goals Conceded / Game",
        height=380,
        margin=dict(l=10, r=10, t=36, b=10),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Group Standings Lollipop Chart
# ---------------------------------------------------------------------------

def group_standings_chart(group_name: str = "A") -> go.Figure | None:
    """Returns a lollipop chart of standings for a given group."""
    creds = _get_project_dataset()
    if not creds:
        return None
    project, dataset = creds

    sql = f"""
        SELECT standing_rank, team_name, points, played, wins, draws, losses,
               goals_for, goals_against, goal_diff
        FROM `{project}.{dataset}.mart_tournament_state`
        WHERE competition_id = 1 AND season_year = 2026
          AND LOWER(group_name) = LOWER('{group_name.replace("'", "''")}')
        ORDER BY standing_rank ASC
    """
    try:
        df = _run_bq(sql)
    except Exception:
        return None
    if df.empty:
        return None

    from src.server.worldcup_style import get_flag
    df["flag"] = df["team_name"].apply(lambda t: get_flag(t) or "")
    df["label"] = df["flag"] + " " + df["team_name"]
    df["color"] = df["standing_rank"].apply(
        lambda r: _GOLD if r <= 2 else "rgba(180,180,180,0.5)"
    )

    # Sort descending for horizontal display (rank 1 at top)
    df = df.sort_values("standing_rank", ascending=False)

    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["points"], row["points"]],
            y=[row["label"], row["label"]],
            mode="lines+markers",
            line=dict(color=row["color"], width=3),
            marker=dict(
                size=[8, 18],
                color=[row["color"], row["color"]],
                symbol=["circle", "circle"],
            ),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Add points as a separate trace for hover
    fig.add_trace(go.Bar(
        y=df["label"], x=df["points"],
        orientation="h",
        marker_color=df["color"],
        text=df["points"].astype(str) + " pts",
        textposition="outside",
        hovertemplate=(
            "%{y}<br>"
            "Points: <b>%{x}</b><br>"
            "W-D-L: %{customdata[0]}-%{customdata[1]}-%{customdata[2]}<br>"
            "GD: %{customdata[3]}<extra></extra>"
        ),
        customdata=df[["wins", "draws", "losses", "goal_diff"]],
        showlegend=False,
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(10,31,46,0.6)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc",
        title=dict(text=f"📊 Group {group_name} Standings", font_color="#f0c040"),
        xaxis_title="Points",
        yaxis_title="",
        height=280,
        margin=dict(l=10, r=10, t=36, b=10),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        bargap=0.3,
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Player Comparison Radar Chart
# ---------------------------------------------------------------------------

def player_comparison_radar(player1: str = "", player2: str = "") -> go.Figure | None:
    """Returns a radar chart comparing two players across multiple dimensions."""
    creds = _get_project_dataset()
    if not creds:
        return None
    project, dataset = creds

    if not player1 or not player2:
        # Default: top 2 scorers
        sql_default = f"""
            SELECT dp.player_name
            FROM `{project}.{dataset}.fact_player_match_stat` fps
            JOIN `{project}.{dataset}.dim_player` dp USING (player_id)
            WHERE 1=1
              {_wc_where_clause()}
            GROUP BY dp.player_name
            ORDER BY SUM(fps.goals) DESC
            LIMIT 2
        """
        try:
            default_df = _run_bq(sql_default)
            if not default_df.empty:
                names = default_df["player_name"].tolist()
                player1 = names[0] if len(names) > 0 else ""
                player2 = names[1] if len(names) > 1 else ""
        except Exception:
            return None

    if not player1 or not player2:
        return None

    safe1 = player1.replace("'", "''")
    safe2 = player2.replace("'", "''")

    sql = f"""
        SELECT
            dp.player_name,
            SUM(fps.goals) AS goals,
            SUM(fps.assists) AS assists,
            SAFE_DIVIDE(SUM(fps.passes_accurate), SUM(fps.passes_total)) * 100 AS pass_accuracy,
            SUM(fps.dribbles_success) AS dribbles,
            AVG(fps.rating) AS rating,
            SUM(fps.minutes_played) / 90.0 AS matches_90
        FROM `{project}.{dataset}.fact_player_match_stat` fps
        JOIN `{project}.{dataset}.dim_player` dp USING (player_id)
        WHERE 1=1
          {_wc_where_clause()}
          AND (LOWER(dp.player_name) = LOWER('{safe1}')
               OR LOWER(dp.player_name) = LOWER('{safe2}'))
        GROUP BY dp.player_name
        ORDER BY dp.player_name
    """
    try:
        df = _run_bq(sql)
    except Exception:
        return None
    if df.empty or len(df) < 2:
        return None

    dimensions = ["goals", "assists", "pass_accuracy", "dribbles", "rating", "matches_90"]
    labels = ["Goals", "Assists", "Pass Acc %", "Dribbles", "Rating (x10)", "90s Played"]

    # Normalize values to 0-100 scale for radar
    max_vals = {}
    for dim in dimensions:
        max_vals[dim] = max(df[dim].max(), 0.01)

    fig = go.Figure()
    colors = [_GOLD, _ACCENT_BLUE]

    for i, (_, row) in enumerate(df.iterrows()):
        values = []
        for dim in dimensions:
            val = row[dim] or 0
            if dim == "rating":
                val = val * 10  # scale 0-10 to 0-100
            values.append(min(val / max_vals[dim] * 100, 100))

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill="toself",
            name=row["player_name"],
            fillcolor=colors[i].replace(")", ",0.25)"),
            line=dict(color=colors[i], width=2),
            marker=dict(size=4),
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(10,31,46,0.6)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc",
        title=dict(text=f"⭐ {player1} vs {player2}", font_color="#f0c040"),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 105],
                showticklabels=False,
                gridcolor="rgba(255,255,255,0.08)",
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.08)",
            ),
        ),
        height=340,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5,
        ),
    )
    return fig


def get_available_players() -> list[str]:
    """Return top 20 WC2026 players for dropdown selection."""
    creds = _get_project_dataset()
    if not creds:
        return []
    project, dataset = creds
    sql = f"""
        SELECT dp.player_name
        FROM `{project}.{dataset}.fact_player_match_stat` fps
        JOIN `{project}.{dataset}.dim_player` dp USING (player_id)
        WHERE 1=1
          {_wc_where_clause()}
        GROUP BY dp.player_name
        ORDER BY SUM(fps.goal_contributions) DESC
        LIMIT 20
    """
    try:
        df = _run_bq(sql)
        return df["player_name"].tolist() if not df.empty else []
    except Exception:
        return []
