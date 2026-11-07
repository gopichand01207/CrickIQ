"""
preprocessor.py
---------------
Data cleaning and preprocessing utilities for CrickIQ.

Handles:
  - Loading CSVs (generating them if absent)
  - Dropping or imputing missing values
  - Type coercions
  - Merging match and delivery data
"""

import os
import logging
import sys

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "data")
MATCHES_CSV = os.path.join(DATA_DIR, "ipl_matches.csv")
DELIVERIES_CSV = os.path.join(DATA_DIR, "ipl_deliveries.csv")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _ensure_data():
    """Auto-generate CSV files if they don't exist."""
    if not os.path.exists(MATCHES_CSV) or not os.path.exists(DELIVERIES_CSV):
        logger.info("CSV data not found. Running data generator …")
        gen_script = os.path.join(DATA_DIR, "generate_data.py")
        import importlib.util
        spec = importlib.util.spec_from_file_location("generate_data", gen_script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()


def load_matches(regenerate: bool = False) -> pd.DataFrame:
    """
    Load the IPL matches CSV, auto-generating it if absent.

    Parameters
    ----------
    regenerate : bool
        Force re-generation of data even if CSVs exist.

    Returns
    -------
    pd.DataFrame
    """
    if regenerate or not os.path.exists(MATCHES_CSV):
        _ensure_data()
    df = pd.read_csv(MATCHES_CSV)
    return _clean_matches(df)


def load_deliveries(regenerate: bool = False) -> pd.DataFrame:
    """
    Load the IPL deliveries CSV, auto-generating it if absent.

    Returns
    -------
    pd.DataFrame
    """
    if regenerate or not os.path.exists(DELIVERIES_CSV):
        _ensure_data()
    df = pd.read_csv(DELIVERIES_CSV)
    return _clean_deliveries(df)


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def _clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types and fill nulls for the matches DataFrame."""
    df = df.copy()

    # Date column
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # String columns — fill NaN with empty string
    str_cols = [
        "city", "team1", "team2", "toss_winner", "toss_decision",
        "result", "winner", "player_of_match", "venue", "umpire1", "umpire2",
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    # Numeric columns
    int_cols = ["id", "season", "dl_applied", "win_by_runs", "win_by_wickets"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Drop rows with no match id
    df = df.dropna(subset=["id"])
    df = df[df["team1"] != ""]
    df = df[df["team2"] != ""]

    return df.reset_index(drop=True)


def _clean_deliveries(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types and fill nulls for the deliveries DataFrame."""
    df = df.copy()

    str_cols = [
        "batting_team", "bowling_team", "batsman", "non_striker",
        "bowler", "player_dismissed", "dismissal_kind", "fielder",
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    int_cols = [
        "match_id", "inning", "over", "ball", "is_super_over",
        "wide_runs", "bye_runs", "legbye_runs", "noball_runs", "penalty_runs",
        "batsman_runs", "extra_runs", "total_runs",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def compute_team_season_stats(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute wins / matches played per team per season.

    Returns
    -------
    DataFrame with columns: season, team, matches_played, wins, win_pct
    """
    rows = []
    teams = pd.concat([matches_df["team1"], matches_df["team2"]]).unique()

    for season in sorted(matches_df["season"].unique()):
        s_df = matches_df[matches_df["season"] == season]
        for team in teams:
            played = s_df[(s_df["team1"] == team) | (s_df["team2"] == team)]
            wins = played[played["winner"] == team]
            rows.append({
                "season": season,
                "team": team,
                "matches_played": len(played),
                "wins": len(wins),
                "win_pct": round(len(wins) / max(len(played), 1) * 100, 1),
            })

    return pd.DataFrame(rows)


def compute_batting_stats(deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate batting statistics per batsman.

    Returns
    -------
    DataFrame with columns: batsman, runs, balls_faced, innings,
                             strike_rate, average, fifties, centuries
    """
    df = deliveries_df[deliveries_df["wide_runs"] == 0].copy()

    agg = (
        df.groupby("batsman")
        .agg(
            runs=("batsman_runs", "sum"),
            balls_faced=("batsman_runs", "count"),
            innings=("match_id", "nunique"),
        )
        .reset_index()
    )
    agg["strike_rate"] = (agg["runs"] / agg["balls_faced"].replace(0, np.nan) * 100).round(1)
    agg["average"] = (agg["runs"] / agg["innings"].replace(0, np.nan)).round(1)

    # Innings scores (sum per match per batsman)
    innings_scores = df.groupby(["match_id", "batsman"])["batsman_runs"].sum().reset_index()
    innings_scores.columns = ["match_id", "batsman", "score"]

    fifties = innings_scores[(innings_scores["score"] >= 50) & (innings_scores["score"] < 100)]
    centuries = innings_scores[innings_scores["score"] >= 100]

    agg["fifties"] = agg["batsman"].map(
        fifties.groupby("batsman").size()
    ).fillna(0).astype(int)
    agg["centuries"] = agg["batsman"].map(
        centuries.groupby("batsman").size()
    ).fillna(0).astype(int)

    return agg.sort_values("runs", ascending=False).reset_index(drop=True)


def compute_bowling_stats(deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate bowling statistics per bowler.

    Returns
    -------
    DataFrame with columns: bowler, wickets, runs_conceded, overs_bowled,
                             economy, bowling_average
    """
    df = deliveries_df.copy()

    wickets_df = df[df["player_dismissed"] != ""]
    wickets_agg = wickets_df.groupby("bowler").size().reset_index(name="wickets")

    runs_agg = (
        df.groupby("bowler")["total_runs"]
        .sum()
        .reset_index()
        .rename(columns={"total_runs": "runs_conceded"})
    )
    balls_agg = (
        df[df["wide_runs"] == 0]
        .groupby("bowler")
        .size()
        .reset_index(name="balls_bowled")
    )

    agg = runs_agg.merge(balls_agg, on="bowler", how="left")
    agg = agg.merge(wickets_agg, on="bowler", how="left")
    agg["wickets"] = agg["wickets"].fillna(0).astype(int)
    agg["balls_bowled"] = agg["balls_bowled"].fillna(0).astype(int)
    agg["overs_bowled"] = (agg["balls_bowled"] / 6).round(1)
    agg["economy"] = (agg["runs_conceded"] / agg["overs_bowled"].replace(0, np.nan)).round(2)
    agg["bowling_average"] = (
        agg["runs_conceded"] / agg["wickets"].replace(0, np.nan)
    ).round(2)

    return agg.sort_values("wickets", ascending=False).reset_index(drop=True)
