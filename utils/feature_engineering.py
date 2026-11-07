"""
feature_engineering.py
-----------------------
Feature engineering pipeline for the CrickIQ match-outcome predictor.

Derives higher-level features from raw IPL match data:
  - toss_win_match_win      : did the toss winner also win the match?
  - venue_win_rate          : team win rate at the given venue
  - season_form             : rolling win% over last 5 matches per team
  - head_to_head_win_rate   : historical win % of team1 against team2
  - batting_first_advantage : proxy for batting-first run premium at venue
  - label-encoded team and venue columns
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Main feature pipeline
# ---------------------------------------------------------------------------

def build_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw match DataFrame into a feature matrix suitable for ML.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Raw IPL matches data (as produced by generate_data.py).

    Returns
    -------
    pd.DataFrame
        Feature matrix.  Also returns label-encoders as module-level state.
    """
    df = matches_df.copy()
    df = df[df["result"] == "normal"].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Binary target: did team1 win?
    df["team1_won"] = (df["winner"] == df["team1"]).astype(int)

    # --- Feature 1: toss_win_match_win ---
    df["toss_win_match_win"] = (df["toss_winner"] == df["winner"]).astype(int)

    # --- Feature 2: toss_decision_bat (1=bat, 0=field) ---
    df["toss_decision_bat"] = (df["toss_decision"] == "bat").astype(int)

    # --- Feature 3: toss_winner_is_team1 ---
    df["toss_winner_is_team1"] = (df["toss_winner"] == df["team1"]).astype(int)

    # --- Feature 4: venue_win_rate for team1 ---
    df["venue_win_rate_team1"] = df.apply(
        lambda r: _venue_win_rate(df, r["team1"], r["venue"], r.name), axis=1
    )

    # --- Feature 5: venue_win_rate for team2 ---
    df["venue_win_rate_team2"] = df.apply(
        lambda r: _venue_win_rate(df, r["team2"], r["venue"], r.name), axis=1
    )

    # --- Feature 6: season_form team1 (rolling win % last 5 matches) ---
    df["season_form_team1"] = df.apply(
        lambda r: _season_form(df, r["team1"], r.name), axis=1
    )

    # --- Feature 7: season_form team2 ---
    df["season_form_team2"] = df.apply(
        lambda r: _season_form(df, r["team2"], r.name), axis=1
    )

    # --- Feature 8: head_to_head_win_rate (team1 vs team2) ---
    df["h2h_win_rate_team1"] = df.apply(
        lambda r: _head_to_head(df, r["team1"], r["team2"], r.name), axis=1
    )

    # --- Feature 9: batting_first_advantage proxy ---
    df["batting_first_advantage"] = df.apply(
        lambda r: _batting_first_advantage(df, r["venue"], r.name), axis=1
    )

    # --- Feature 10: win_by_runs proxy (from previous matches at venue) ---
    df["avg_win_margin_runs"] = df.groupby("venue")["win_by_runs"].transform(
        lambda x: x.expanding().mean().shift(1).fillna(x.mean())
    )

    # --- Label Encode teams and venue ---
    le_team = LabelEncoder()
    all_teams = pd.concat([df["team1"], df["team2"]]).unique()
    le_team.fit(all_teams)
    df["team1_enc"] = le_team.transform(df["team1"])
    df["team2_enc"] = le_team.transform(df["team2"])

    le_venue = LabelEncoder()
    df["venue_enc"] = le_venue.fit_transform(df["venue"])

    # Store encoders for later use (predict time)
    df.attrs["le_team"] = le_team
    df.attrs["le_venue"] = le_venue

    return df


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _venue_win_rate(df: pd.DataFrame, team: str, venue: str, current_idx: int) -> float:
    """Historical win rate of `team` at `venue` before `current_idx`."""
    past = df.iloc[:current_idx]
    at_venue = past[(past["venue"] == venue) & ((past["team1"] == team) | (past["team2"] == team))]
    if len(at_venue) == 0:
        return 0.5
    wins = (at_venue["winner"] == team).sum()
    return wins / len(at_venue)


def _season_form(df: pd.DataFrame, team: str, current_idx: int, window: int = 5) -> float:
    """Rolling win % of `team` in the last `window` matches before `current_idx`."""
    past = df.iloc[:current_idx]
    team_matches = past[(past["team1"] == team) | (past["team2"] == team)].tail(window)
    if len(team_matches) == 0:
        return 0.5
    wins = (team_matches["winner"] == team).sum()
    return wins / len(team_matches)


def _head_to_head(df: pd.DataFrame, team1: str, team2: str, current_idx: int) -> float:
    """Win rate of `team1` against `team2` before `current_idx`."""
    past = df.iloc[:current_idx]
    h2h = past[
        ((past["team1"] == team1) & (past["team2"] == team2))
        | ((past["team1"] == team2) & (past["team2"] == team1))
    ]
    if len(h2h) == 0:
        return 0.5
    wins = (h2h["winner"] == team1).sum()
    return wins / len(h2h)


def _batting_first_advantage(df: pd.DataFrame, venue: str, current_idx: int) -> float:
    """
    Proxy for batting-first advantage at a venue.

    Returns the fraction of matches at the venue (before current_idx) where
    the team batting first won.
    """
    past = df.iloc[:current_idx]
    at_venue = past[past["venue"] == venue]
    if len(at_venue) == 0:
        return 0.5
    bat_first_wins = at_venue[
        (at_venue["toss_decision"] == "bat") & (at_venue["toss_winner"] == at_venue["winner"])
        | (at_venue["toss_decision"] == "field") & (at_venue["toss_winner"] != at_venue["winner"])
        & (at_venue["winner"] != "")
    ]
    return len(bat_first_wins) / len(at_venue)


# ---------------------------------------------------------------------------
# Feature list used for modelling
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "toss_decision_bat",
    "toss_winner_is_team1",
    "venue_win_rate_team1",
    "venue_win_rate_team2",
    "season_form_team1",
    "season_form_team2",
    "h2h_win_rate_team1",
    "batting_first_advantage",
    "avg_win_margin_runs",
    "team1_enc",
    "team2_enc",
    "venue_enc",
]

FEATURE_DISPLAY_NAMES = {
    "toss_decision_bat": "Toss Decision (Bat)",
    "toss_winner_is_team1": "Toss Winner is Team 1",
    "venue_win_rate_team1": "Venue Win Rate (Team 1)",
    "venue_win_rate_team2": "Venue Win Rate (Team 2)",
    "season_form_team1": "Recent Form (Team 1)",
    "season_form_team2": "Recent Form (Team 2)",
    "h2h_win_rate_team1": "Head-to-Head Win Rate",
    "batting_first_advantage": "Batting First Advantage",
    "avg_win_margin_runs": "Avg Win Margin (Runs)",
    "team1_enc": "Team 1 Strength Index",
    "team2_enc": "Team 2 Strength Index",
    "venue_enc": "Venue Index",
}
