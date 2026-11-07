"""
player_performance.py
---------------------
Player performance analysis and prediction model for CrickIQ.

Provides:
  - Batting and bowling stat aggregations
  - GradientBoostingRegressor to predict a batsman's expected innings score
  - get_top_batsmen() and get_top_bowlers() convenience functions
"""

import os
import logging
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
GB_MODEL_PATH = os.path.join(_HERE, "gb_player_model.joblib")
LE_PLAYER_PATH = os.path.join(_HERE, "le_player.joblib")


# ---------------------------------------------------------------------------
# Aggregation functions
# ---------------------------------------------------------------------------

def aggregate_batting(deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-batsman career aggregates from delivery data.

    Returns
    -------
    DataFrame with: batsman, total_runs, innings, balls_faced,
                    strike_rate, average, fifties, centuries
    """
    # Exclude wides from ball counts
    df = deliveries_df.copy()
    df_no_wide = df[df["wide_runs"] == 0]

    runs_agg = df.groupby("batsman")["batsman_runs"].sum().reset_index(name="total_runs")
    balls_agg = df_no_wide.groupby("batsman").size().reset_index(name="balls_faced")
    innings_agg = df.groupby("batsman")["match_id"].nunique().reset_index(name="innings")

    result = runs_agg.merge(balls_agg, on="batsman", how="left")
    result = result.merge(innings_agg, on="batsman", how="left")
    result["balls_faced"] = result["balls_faced"].fillna(1)
    result["innings"] = result["innings"].fillna(1)

    result["strike_rate"] = (result["total_runs"] / result["balls_faced"] * 100).round(2)
    result["average"] = (result["total_runs"] / result["innings"]).round(2)

    # Innings scores
    innings_scores = (
        df.groupby(["match_id", "batsman"])["batsman_runs"]
        .sum()
        .reset_index()
        .rename(columns={"batsman_runs": "score"})
    )
    fifty_counts = (
        innings_scores[(innings_scores["score"] >= 50) & (innings_scores["score"] < 100)]
        .groupby("batsman")
        .size()
        .reset_index(name="fifties")
    )
    century_counts = (
        innings_scores[innings_scores["score"] >= 100]
        .groupby("batsman")
        .size()
        .reset_index(name="centuries")
    )

    result = result.merge(fifty_counts, on="batsman", how="left")
    result = result.merge(century_counts, on="batsman", how="left")
    result["fifties"] = result["fifties"].fillna(0).astype(int)
    result["centuries"] = result["centuries"].fillna(0).astype(int)

    # Filter: at least 20 balls faced
    result = result[result["balls_faced"] >= 20]
    return result.sort_values("total_runs", ascending=False).reset_index(drop=True)


def aggregate_bowling(deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-bowler career aggregates from delivery data.

    Returns
    -------
    DataFrame with: bowler, wickets, runs_conceded, overs_bowled,
                    economy, bowling_average
    """
    df = deliveries_df.copy()

    wickets_df = df[df["player_dismissed"].notna() & (df["player_dismissed"] != "")]
    wickets_agg = wickets_df.groupby("bowler").size().reset_index(name="wickets")

    runs_agg = df.groupby("bowler")["total_runs"].sum().reset_index(name="runs_conceded")
    balls_agg = (
        df[df["wide_runs"] == 0]
        .groupby("bowler")
        .size()
        .reset_index(name="balls_bowled")
    )

    result = runs_agg.merge(balls_agg, on="bowler", how="left")
    result = result.merge(wickets_agg, on="bowler", how="left")
    result["wickets"] = result["wickets"].fillna(0).astype(int)
    result["balls_bowled"] = result["balls_bowled"].fillna(0)
    result["overs_bowled"] = (result["balls_bowled"] / 6).round(1)
    result["economy"] = (
        result["runs_conceded"] / result["overs_bowled"].replace(0, np.nan)
    ).round(2)
    result["bowling_average"] = (
        result["runs_conceded"] / result["wickets"].replace(0, np.nan)
    ).round(2)

    # Filter: at least 10 overs
    result = result[result["overs_bowled"] >= 10]
    return result.sort_values("wickets", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Top-N helpers
# ---------------------------------------------------------------------------

def get_top_batsmen(batting_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return top-n batsmen by total runs."""
    return batting_df.head(n)


def get_top_bowlers(bowling_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return top-n bowlers by wickets."""
    return bowling_df.head(n)


# ---------------------------------------------------------------------------
# GradientBoosting model: predict expected innings score
# ---------------------------------------------------------------------------

def train_score_predictor(deliveries_df: pd.DataFrame) -> dict:
    """
    Train a GradientBoostingRegressor to predict a batsman's innings score.

    Features used:
      - career_avg     : career batting average
      - career_sr      : career strike rate
      - career_innings : number of innings played
      - batsman_enc    : label-encoded batsman id

    Returns
    -------
    dict with keys: model, le_player, mae, r2
    """
    batting = aggregate_batting(deliveries_df)

    # Innings-level dataset
    innings_scores = (
        deliveries_df.groupby(["match_id", "batsman"])["batsman_runs"]
        .sum()
        .reset_index()
        .rename(columns={"batsman_runs": "score"})
    )

    merged = innings_scores.merge(
        batting[["batsman", "average", "strike_rate", "innings"]],
        on="batsman",
        how="left",
    ).dropna()

    le = LabelEncoder()
    merged["batsman_enc"] = le.fit_transform(merged["batsman"])

    X = merged[["average", "strike_rate", "innings", "batsman_enc"]].values
    y = merged["score"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    gb = GradientBoostingRegressor(
        n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42
    )
    gb.fit(X_train, y_train)
    preds = gb.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    joblib.dump(gb, GB_MODEL_PATH)
    joblib.dump(le, LE_PLAYER_PATH)

    logger.info("GradientBoosting score predictor — MAE: %.2f | R²: %.3f", mae, r2)
    return {"model": gb, "le_player": le, "mae": mae, "r2": r2}


def predict_player_score(batsman: str, batting_df: pd.DataFrame) -> Optional[float]:
    """
    Predict expected innings score for `batsman`.

    Returns None if the model is not trained or the player is unknown.
    """
    if not os.path.exists(GB_MODEL_PATH):
        return None

    gb: GradientBoostingRegressor = joblib.load(GB_MODEL_PATH)
    le: LabelEncoder = joblib.load(LE_PLAYER_PATH)

    player_stats = batting_df[batting_df["batsman"] == batsman]
    if player_stats.empty:
        return None

    row = player_stats.iloc[0]
    try:
        batsman_enc = int(le.transform([batsman])[0])
    except ValueError:
        return None

    X = np.array([[row["average"], row["strike_rate"], row["innings"], batsman_enc]])
    return float(gb.predict(X)[0])
