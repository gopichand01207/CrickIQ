"""
visualizations.py
-----------------
Plotly chart helper functions for the CrickIQ Streamlit dashboard.

Every function returns a plotly Figure object ready to be rendered via
st.plotly_chart(fig, use_container_width=True).
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Shared theme
# ---------------------------------------------------------------------------
GREEN = "#2ECC71"
DARK_BG = "#0E1117"
CARD_BG = "#1A1A2E"
GRID_COLOR = "#2D2D2D"
TEXT_COLOR = "#FFFFFF"

_LAYOUT_DEFAULTS = dict(
    paper_bgcolor=DARK_BG,
    plot_bgcolor=DARK_BG,
    font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
    margin=dict(l=40, r=40, t=50, b=40),
)


def _apply_dark_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**_LAYOUT_DEFAULTS)
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    return fig


# ---------------------------------------------------------------------------
# Dashboard charts
# ---------------------------------------------------------------------------

def team_wins_bar(wins_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of all-time wins per team."""
    df = wins_df.sort_values("wins")
    fig = go.Figure(
        go.Bar(
            x=df["wins"],
            y=df["winner"],
            orientation="h",
            marker_color=GREEN,
            text=df["wins"],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="All-Time Wins by Team",
        xaxis_title="Wins",
        yaxis_title="",
        **_LAYOUT_DEFAULTS,
    )
    fig.update_xaxes(gridcolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR)
    return fig


def matches_per_season_line(matches_df: pd.DataFrame) -> go.Figure:
    """Line chart of number of matches per season."""
    per_season = matches_df.groupby("season").size().reset_index(name="matches")
    fig = go.Figure(
        go.Scatter(
            x=per_season["season"],
            y=per_season["matches"],
            mode="lines+markers",
            line=dict(color=GREEN, width=2),
            marker=dict(size=8, color=GREEN),
            fill="tozeroy",
            fillcolor="rgba(46,204,113,0.1)",
        )
    )
    fig.update_layout(
        title="Matches Per Season",
        xaxis_title="Season",
        yaxis_title="Matches",
        **_LAYOUT_DEFAULTS,
    )
    fig.update_xaxes(gridcolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR)
    return fig


def toss_decision_pie(matches_df: pd.DataFrame) -> go.Figure:
    """Pie chart of toss decisions (bat vs field)."""
    counts = matches_df["toss_decision"].value_counts()
    fig = go.Figure(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.4,
            marker=dict(colors=[GREEN, "#3498DB"]),
            textfont=dict(color=TEXT_COLOR),
        )
    )
    fig.update_layout(
        title="Toss Decision Distribution",
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# Match predictor charts
# ---------------------------------------------------------------------------

def win_probability_gauge(team1: str, team2: str, prob_team1: float) -> go.Figure:
    """Dual gauge showing win probabilities for both teams."""
    prob_team2 = 1.0 - prob_team1

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}]],
    )

    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=round(prob_team1 * 100, 1),
            title={"text": team1, "font": {"color": TEXT_COLOR, "size": 14}},
            number={"suffix": "%", "font": {"color": GREEN}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": TEXT_COLOR},
                "bar": {"color": GREEN},
                "bgcolor": CARD_BG,
                "bordercolor": GREEN,
                "steps": [
                    {"range": [0, 50], "color": "#1A1A2E"},
                    {"range": [50, 100], "color": "#0D2E1A"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=round(prob_team2 * 100, 1),
            title={"text": team2, "font": {"color": TEXT_COLOR, "size": 14}},
            number={"suffix": "%", "font": {"color": "#3498DB"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": TEXT_COLOR},
                "bar": {"color": "#3498DB"},
                "bgcolor": CARD_BG,
                "bordercolor": "#3498DB",
                "steps": [
                    {"range": [0, 50], "color": "#1A1A2E"},
                    {"range": [50, 100], "color": "#0A1A2E"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ),
        row=1, col=2,
    )

    fig.update_layout(height=300, **_LAYOUT_DEFAULTS)
    return fig


def feature_importance_bar(importances: pd.Series) -> go.Figure:
    """Horizontal bar chart of feature importances."""
    df = importances.sort_values()
    colors = [GREEN if v > df.median() else "#3498DB" for v in df.values]
    fig = go.Figure(
        go.Bar(
            x=df.values,
            y=df.index,
            orientation="h",
            marker_color=colors,
        )
    )
    fig.update_layout(
        title="Feature Importances (Random Forest)",
        xaxis_title="Importance",
        yaxis_title="",
        **_LAYOUT_DEFAULTS,
    )
    fig.update_xaxes(gridcolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR)
    return fig


# ---------------------------------------------------------------------------
# Player analytics charts
# ---------------------------------------------------------------------------

def top_batsmen_bar(batting_df: pd.DataFrame, n: int = 10) -> go.Figure:
    """Bar chart of top-n batsmen by total runs."""
    df = batting_df.head(n).sort_values("runs")
    fig = go.Figure(
        go.Bar(
            x=df["runs"],
            y=df["batsman"],
            orientation="h",
            marker_color=GREEN,
            text=df["runs"],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Top {n} Batsmen by Runs",
        xaxis_title="Runs",
        **_LAYOUT_DEFAULTS,
    )
    fig.update_xaxes(gridcolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR)
    return fig


def top_bowlers_bar(bowling_df: pd.DataFrame, n: int = 10) -> go.Figure:
    """Bar chart of top-n bowlers by wickets."""
    df = bowling_df.head(n).sort_values("wickets")
    fig = go.Figure(
        go.Bar(
            x=df["wickets"],
            y=df["bowler"],
            orientation="h",
            marker_color="#E74C3C",
            text=df["wickets"],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Top {n} Bowlers by Wickets",
        xaxis_title="Wickets",
        **_LAYOUT_DEFAULTS,
    )
    fig.update_xaxes(gridcolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR)
    return fig


def player_of_match_bar(pom_df: pd.DataFrame) -> go.Figure:
    """Bar chart of Player of the Match awards."""
    df = pom_df.sort_values("awards")
    fig = go.Figure(
        go.Bar(
            x=df["awards"],
            y=df["player_of_match"],
            orientation="h",
            marker=dict(color=df["awards"], colorscale="Viridis"),
            text=df["awards"],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Player of the Match Awards",
        xaxis_title="Awards",
        **_LAYOUT_DEFAULTS,
    )
    fig.update_xaxes(gridcolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR)
    return fig


# ---------------------------------------------------------------------------
# Clustering charts
# ---------------------------------------------------------------------------

def cluster_scatter(cluster_df: pd.DataFrame, title: str = "Team Clusters") -> go.Figure:
    """2-D PCA scatter plot coloured by cluster."""
    palette = {0: GREEN, 1: "#3498DB", 2: "#E74C3C", 3: "#F39C12"}
    label_map = {0: "Dominant", 1: "Competitive", 2: "Rising", 3: "Rebuilding"}

    fig = go.Figure()
    for cluster_id, group in cluster_df.groupby("cluster"):
        fig.add_trace(
            go.Scatter(
                x=group["pca_x"],
                y=group["pca_y"],
                mode="markers+text",
                name=label_map.get(cluster_id, f"Cluster {cluster_id}"),
                text=group.get("label", group.index),
                textposition="top center",
                textfont=dict(color=TEXT_COLOR, size=11),
                marker=dict(
                    size=14,
                    color=palette.get(cluster_id, GREEN),
                    line=dict(color="white", width=1),
                ),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="PCA Component 1",
        yaxis_title="PCA Component 2",
        legend=dict(bgcolor=CARD_BG, bordercolor=GREEN),
        **_LAYOUT_DEFAULTS,
    )
    fig.update_xaxes(gridcolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR)
    return fig


def radar_chart(cluster_stats: pd.DataFrame, categories: list) -> go.Figure:
    """Radar chart comparing cluster-averaged stats."""
    palette = [GREEN, "#3498DB", "#E74C3C", "#F39C12"]
    label_map = {0: "Dominant", 1: "Competitive", 2: "Rising", 3: "Rebuilding"}

    fig = go.Figure()
    for i, row in cluster_stats.iterrows():
        values = [row[c] for c in categories] + [row[categories[0]]]
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill="toself",
                name=label_map.get(int(row.get("cluster", i)), f"Cluster {i}"),
                line=dict(color=palette[i % len(palette)]),
                fillcolor=palette[i % len(palette)].replace("#", "rgba(") + ",0.15)",
            )
        )
    fig.update_layout(
        polar=dict(
            bgcolor=CARD_BG,
            radialaxis=dict(visible=True, gridcolor=GRID_COLOR),
            angularaxis=dict(gridcolor=GRID_COLOR),
        ),
        title="Cluster Comparison (Radar)",
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# Historical analysis charts
# ---------------------------------------------------------------------------

def head_to_head_heatmap(h2h_matrix: pd.DataFrame) -> go.Figure:
    """Heatmap of head-to-head win rates between teams."""
    fig = go.Figure(
        go.Heatmap(
            z=h2h_matrix.values,
            x=h2h_matrix.columns.tolist(),
            y=h2h_matrix.index.tolist(),
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            text=h2h_matrix.round(2).astype(str).values,
            texttemplate="%{text}",
            colorbar=dict(title="Win Rate"),
        )
    )
    fig.update_layout(
        title="Head-to-Head Win Rate Matrix",
        xaxis=dict(tickangle=-30),
        **_LAYOUT_DEFAULTS,
    )
    return fig


def avg_score_trend(season_avg: pd.DataFrame) -> go.Figure:
    """Line chart of average team score per season."""
    fig = go.Figure(
        go.Scatter(
            x=season_avg["season"],
            y=season_avg["avg_score"],
            mode="lines+markers",
            line=dict(color=GREEN, width=2),
            marker=dict(size=7, color=GREEN),
            fill="tozeroy",
            fillcolor="rgba(46,204,113,0.1)",
        )
    )
    fig.update_layout(
        title="Average Team Score Per Season",
        xaxis_title="Season",
        yaxis_title="Avg Score",
        **_LAYOUT_DEFAULTS,
    )
    fig.update_xaxes(gridcolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR)
    return fig


def venue_matches_bar(venue_df: pd.DataFrame) -> go.Figure:
    """Bar chart of matches hosted per venue."""
    df = venue_df.sort_values("total_matches", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=df["total_matches"],
            y=df["venue"],
            orientation="h",
            marker=dict(color=df["total_matches"], colorscale="Greens"),
            text=df["total_matches"],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Matches Hosted by Venue",
        xaxis_title="Matches",
        **_LAYOUT_DEFAULTS,
    )
    fig.update_xaxes(gridcolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR)
    return fig
