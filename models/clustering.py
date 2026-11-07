"""
clustering.py
-------------
K-Means clustering for IPL teams and players using CrickIQ data.

Teams are clustered into 4 groups based on performance metrics.
Players are clustered into 3 groups based on batting metrics.
PCA is used to project the clusters into 2-D for visualisation.
"""

import logging
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Cluster label map
TEAM_CLUSTER_LABELS = {
    0: "Dominant",
    1: "Competitive",
    2: "Rising",
    3: "Rebuilding",
}

PLAYER_CLUSTER_LABELS = {
    0: "Elite",
    1: "Consistent",
    2: "Developing",
}


# ---------------------------------------------------------------------------
# Team clustering
# ---------------------------------------------------------------------------

def cluster_teams(matches_df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    Cluster IPL teams (k=4) based on:
      - total wins
      - win percentage
      - average win-by-runs margin
      - average win-by-wickets margin

    Returns
    -------
    DataFrame with columns: team, wins, win_pct, avg_win_runs,
                             avg_win_wkts, cluster, cluster_label,
                             pca_x, pca_y
    """
    teams = pd.concat([matches_df["team1"], matches_df["team2"]]).unique()
    rows = []

    for team in teams:
        played = matches_df[(matches_df["team1"] == team) | (matches_df["team2"] == team)]
        wins_df = played[played["winner"] == team]
        win_pct = len(wins_df) / max(len(played), 1)
        avg_runs = wins_df["win_by_runs"].mean() if len(wins_df) > 0 else 0
        avg_wkts = wins_df["win_by_wickets"].mean() if len(wins_df) > 0 else 0
        rows.append({
            "team": team,
            "wins": len(wins_df),
            "total_matches": len(played),
            "win_pct": win_pct,
            "avg_win_runs": avg_runs,
            "avg_win_wkts": avg_wkts,
        })

    team_df = pd.DataFrame(rows)

    feature_cols = ["wins", "win_pct", "avg_win_runs", "avg_win_wkts"]
    X = team_df[feature_cols].fillna(0).values

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    team_df["cluster"] = kmeans.fit_predict(X_sc)

    # Re-label clusters by win_pct descending
    cluster_win_pct = team_df.groupby("cluster")["win_pct"].mean().sort_values(ascending=False)
    rank_map = {old: new for new, old in enumerate(cluster_win_pct.index)}
    team_df["cluster"] = team_df["cluster"].map(rank_map)
    team_df["cluster_label"] = team_df["cluster"].map(TEAM_CLUSTER_LABELS)

    # PCA for 2-D vis
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_sc)
    team_df["pca_x"] = coords[:, 0]
    team_df["pca_y"] = coords[:, 1]
    team_df["label"] = team_df["team"]

    return team_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Player clustering
# ---------------------------------------------------------------------------

def cluster_players(batting_df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """
    Cluster batsmen (k=3) based on:
      - average runs per innings (average)
      - strike rate
      - innings count (consistency proxy)

    Parameters
    ----------
    batting_df : pd.DataFrame
        Output of aggregate_batting() — must have columns:
        batsman, average, strike_rate, innings.

    Returns
    -------
    DataFrame with columns: batsman, average, strike_rate, innings,
                             cluster, cluster_label, pca_x, pca_y
    """
    df = batting_df[["batsman", "average", "strike_rate", "innings"]].dropna().copy()
    df = df[df["innings"] >= 5]  # Minimum games played

    feature_cols = ["average", "strike_rate", "innings"]
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_sc)

    # Re-label by average descending
    cluster_avg = df.groupby("cluster")["average"].mean().sort_values(ascending=False)
    rank_map = {old: new for new, old in enumerate(cluster_avg.index)}
    df["cluster"] = df["cluster"].map(rank_map)
    df["cluster_label"] = df["cluster"].map(PLAYER_CLUSTER_LABELS)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_sc)
    df["pca_x"] = coords[:, 0]
    df["pca_y"] = coords[:, 1]
    df["label"] = df["batsman"]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cluster summary stats
# ---------------------------------------------------------------------------

def team_cluster_summary(team_cluster_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns per-cluster average stats for radar chart.

    Returns
    -------
    DataFrame indexed by cluster with averaged numeric columns.
    """
    numeric_cols = ["wins", "win_pct", "avg_win_runs", "avg_win_wkts"]
    summary = (
        team_cluster_df.groupby("cluster")[numeric_cols]
        .mean()
        .round(2)
        .reset_index()
    )
    summary["cluster_label"] = summary["cluster"].map(TEAM_CLUSTER_LABELS)
    return summary


def player_cluster_summary(player_cluster_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns per-cluster average stats for radar chart.

    Returns
    -------
    DataFrame indexed by cluster with averaged numeric columns.
    """
    numeric_cols = ["average", "strike_rate", "innings"]
    summary = (
        player_cluster_df.groupby("cluster")[numeric_cols]
        .mean()
        .round(2)
        .reset_index()
    )
    summary["cluster_label"] = summary["cluster"].map(PLAYER_CLUSTER_LABELS)
    return summary
