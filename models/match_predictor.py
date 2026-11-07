"""
match_predictor.py
------------------
Match outcome prediction using Random Forest and Logistic Regression.

Workflow
--------
1. Load and engineer features from match data.
2. Train RandomForestClassifier + LogisticRegression (80/20 split).
3. Persist trained models with joblib.
4. Expose predict_match() for the Streamlit UI.
"""

import os
import logging
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils.feature_engineering import (
    build_features,
    FEATURE_COLUMNS,
    FEATURE_DISPLAY_NAMES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = _HERE
RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.joblib")
LR_MODEL_PATH = os.path.join(MODEL_DIR, "lr_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
LE_TEAM_PATH = os.path.join(MODEL_DIR, "le_team.joblib")
LE_VENUE_PATH = os.path.join(MODEL_DIR, "le_venue.joblib")
FEAT_DF_PATH = os.path.join(MODEL_DIR, "feature_df.joblib")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_models(matches_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Engineer features, train RF + LR, persist models, return metrics dict.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Cleaned IPL matches DataFrame.

    Returns
    -------
    dict with keys: rf_accuracy, lr_accuracy, rf_report, lr_report,
                    rf_cm, lr_cm, feature_importances, feature_df
    """
    logger.info("Building features …")
    feat_df = build_features(matches_df)

    X = feat_df[FEATURE_COLUMNS].values
    y = feat_df["team1_won"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale for LR
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_report = classification_report(y_test, rf_preds, output_dict=True)
    rf_cm = confusion_matrix(y_test, rf_preds)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    lr_preds = lr.predict(X_test_sc)
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_report = classification_report(y_test, lr_preds, output_dict=True)
    lr_cm = confusion_matrix(y_test, lr_preds)

    # Feature importances
    importances = pd.Series(
        rf.feature_importances_,
        index=[FEATURE_DISPLAY_NAMES.get(c, c) for c in FEATURE_COLUMNS],
    )

    # Save models
    joblib.dump(rf, RF_MODEL_PATH)
    joblib.dump(lr, LR_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feat_df.attrs["le_team"], LE_TEAM_PATH)
    joblib.dump(feat_df.attrs["le_venue"], LE_VENUE_PATH)
    joblib.dump(feat_df, FEAT_DF_PATH)

    logger.info("RF Accuracy: %.3f | LR Accuracy: %.3f", rf_acc, lr_acc)

    return {
        "rf_accuracy": rf_acc,
        "lr_accuracy": lr_acc,
        "rf_report": rf_report,
        "lr_report": lr_report,
        "rf_cm": rf_cm,
        "lr_cm": lr_cm,
        "feature_importances": importances,
        "feature_df": feat_df,
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_match(
    team1: str,
    team2: str,
    toss_winner: str,
    toss_decision: str,
    venue: str,
) -> Dict[str, Any]:
    """
    Predict the winner of a match given match inputs.

    Loads persisted models if available; otherwise trains fresh ones.

    Returns
    -------
    dict with keys: predicted_winner, prob_team1, prob_team2, confidence,
                    feature_importances
    """
    # Load models
    if not os.path.exists(RF_MODEL_PATH):
        raise FileNotFoundError(
            "Trained model not found. Please train via the Streamlit app first."
        )

    rf: RandomForestClassifier = joblib.load(RF_MODEL_PATH)
    scaler: StandardScaler = joblib.load(SCALER_PATH)
    le_team: LabelEncoder = joblib.load(LE_TEAM_PATH)
    le_venue: LabelEncoder = joblib.load(LE_VENUE_PATH)
    feat_df: pd.DataFrame = joblib.load(FEAT_DF_PATH)

    # Encode inputs safely
    def safe_encode(le, val, default=0):
        try:
            return int(le.transform([val])[0])
        except ValueError:
            return default

    team1_enc = safe_encode(le_team, team1)
    team2_enc = safe_encode(le_team, team2)
    venue_enc = safe_encode(le_venue, venue)

    # Historical features (use dataset averages as proxy for new match)
    toss_decision_bat = int(toss_decision == "bat")
    toss_winner_is_team1 = int(toss_winner == team1)

    # Venue win rate (compute from feature_df)
    t1_venue = feat_df[(feat_df["team1"] == team1) & (feat_df["venue"] == venue)]
    venue_win_rate_team1 = (
        (t1_venue["winner"] == team1).mean() if len(t1_venue) > 0 else 0.5
    )
    t2_venue = feat_df[(feat_df["team2"] == team2) & (feat_df["venue"] == venue)]
    venue_win_rate_team2 = (
        (t2_venue["winner"] == team2).mean() if len(t2_venue) > 0 else 0.5
    )

    # Season form (last 10 from feat_df)
    t1_matches = feat_df[(feat_df["team1"] == team1) | (feat_df["team2"] == team1)].tail(10)
    season_form_team1 = (t1_matches["winner"] == team1).mean() if len(t1_matches) > 0 else 0.5

    t2_matches = feat_df[(feat_df["team1"] == team2) | (feat_df["team2"] == team2)].tail(10)
    season_form_team2 = (t2_matches["winner"] == team2).mean() if len(t2_matches) > 0 else 0.5

    # H2H
    h2h = feat_df[
        ((feat_df["team1"] == team1) & (feat_df["team2"] == team2))
        | ((feat_df["team1"] == team2) & (feat_df["team2"] == team1))
    ]
    h2h_win_rate_team1 = (h2h["winner"] == team1).mean() if len(h2h) > 0 else 0.5

    batting_first_advantage = feat_df[feat_df["venue"] == venue]["batting_first_advantage"].mean()
    if np.isnan(batting_first_advantage):
        batting_first_advantage = 0.5

    avg_win_margin_runs = feat_df["avg_win_margin_runs"].mean()

    feature_vector = np.array([[
        toss_decision_bat,
        toss_winner_is_team1,
        venue_win_rate_team1,
        venue_win_rate_team2,
        season_form_team1,
        season_form_team2,
        h2h_win_rate_team1,
        batting_first_advantage,
        avg_win_margin_runs,
        team1_enc,
        team2_enc,
        venue_enc,
    ]])

    prob = rf.predict_proba(feature_vector)[0]
    # prob[1] = team1 wins
    prob_team1 = prob[1]
    prob_team2 = 1.0 - prob_team1

    predicted_winner = team1 if prob_team1 >= 0.5 else team2
    confidence = max(prob_team1, prob_team2)

    importances = pd.Series(
        rf.feature_importances_,
        index=[FEATURE_DISPLAY_NAMES.get(c, c) for c in FEATURE_COLUMNS],
    )

    return {
        "predicted_winner": predicted_winner,
        "prob_team1": float(prob_team1),
        "prob_team2": float(prob_team2),
        "confidence": float(confidence),
        "feature_importances": importances,
    }
