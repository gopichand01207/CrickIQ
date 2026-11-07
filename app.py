"""
app.py
------
CrickIQ: Cricket Analytics & Match Outcome Predictor
Main Streamlit application entry-point.

Run with:  streamlit run app.py

Pages
-----
1. 🏠  Dashboard         — Key metrics and overview charts
2. 🔮  Match Predictor   — ML-based match outcome prediction
3. 📊  Player Analytics  — Batting/bowling stats and rankings
4. 🔵  Team Clustering   — K-Means cluster visualisation
5. 📈  Historical Analysis — Season/venue/H2H deep dives
"""

import os
import sys
import warnings
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# Add project root to sys.path so relative imports work regardless of CWD
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration (MUST be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CrickIQ — Cricket Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---- Global background ---- */
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1B2A 0%, #1A1A2E 100%);
        border-right: 1px solid #2ECC71;
    }
    [data-testid="stSidebar"] .css-1d391kg { padding-top: 1rem; }

    /* ---- Metric cards ---- */
    [data-testid="metric-container"] {
        background-color: #1A1A2E;
        border: 1px solid #2ECC71;
        border-radius: 10px;
        padding: 10px 15px;
    }
    [data-testid="metric-container"] label { color: #2ECC71 !important; font-size: 0.85rem; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #FFFFFF; font-size: 1.8rem; }

    /* ---- Headers ---- */
    h1, h2, h3 { color: #2ECC71 !important; }

    /* ---- Buttons ---- */
    .stButton > button {
        background: linear-gradient(90deg, #1DB954 0%, #2ECC71 100%);
        color: #000000;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1rem;
        padding: 0.5rem 2rem;
        transition: all 0.2s ease;
    }
    .stButton > button:hover { opacity: 0.85; transform: scale(1.02); }

    /* ---- Selectbox ---- */
    .stSelectbox label { color: #2ECC71 !important; }

    /* ---- DataFrame ---- */
    .stDataFrame { border: 1px solid #2D2D2D; border-radius: 8px; }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab"] {
        background-color: #1A1A2E;
        border-radius: 6px 6px 0 0;
        color: #AAAAAA;
    }
    .stTabs [aria-selected="true"] { color: #2ECC71 !important; border-bottom: 2px solid #2ECC71; }

    /* ---- Banner ---- */
    .crickiq-banner {
        background: linear-gradient(135deg, #0D2E1A 0%, #1A1A2E 50%, #0D2E1A 100%);
        border: 1px solid #2ECC71;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .crickiq-banner h1 { font-size: 2.8rem; letter-spacing: 3px; margin-bottom: 0.2rem; }
    .crickiq-banner p { color: #AAAAAA; font-size: 1rem; margin: 0; }

    /* ---- Spinner text ---- */
    .stSpinner > div { border-top-color: #2ECC71 !important; }

    /* ---- Success / info boxes ---- */
    .stSuccess { background-color: #0D2E1A; border-left: 4px solid #2ECC71; }
    .stInfo    { background-color: #0A1A2E; border-left: 4px solid #3498DB; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Lazy imports (after sys.path setup)
# ---------------------------------------------------------------------------
from utils.preprocessor import (
    load_matches,
    load_deliveries,
    compute_batting_stats,
    compute_bowling_stats,
)
from utils.visualizations import (
    team_wins_bar,
    matches_per_season_line,
    toss_decision_pie,
    win_probability_gauge,
    feature_importance_bar,
    top_batsmen_bar,
    top_bowlers_bar,
    player_of_match_bar,
    cluster_scatter,
    radar_chart,
    head_to_head_heatmap,
    avg_score_trend,
    venue_matches_bar,
)
from models.clustering import (
    cluster_teams,
    cluster_players,
    team_cluster_summary,
    player_cluster_summary,
    TEAM_CLUSTER_LABELS,
)

# ---------------------------------------------------------------------------
# Session-state data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_all_data():
    """Load and cache both DataFrames."""
    matches = load_matches()
    deliveries = load_deliveries()
    return matches, deliveries


@st.cache_data(show_spinner=False)
def _get_batting(_deliveries):
    return compute_batting_stats(_deliveries)


@st.cache_data(show_spinner=False)
def _get_bowling(_deliveries):
    return compute_bowling_stats(_deliveries)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

PAGES = [
    "🏠 Dashboard",
    "🔮 Match Predictor",
    "📊 Player Analytics",
    "🔵 Team Clustering",
    "📈 Historical Analysis",
]

with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:1rem 0;'>"
        "<span style='font-size:3rem;'>🏏</span><br/>"
        "<span style='color:#2ECC71;font-size:1.4rem;font-weight:bold;letter-spacing:2px;'>CrickIQ</span><br/>"
        "<span style='color:#888;font-size:0.75rem;'>Cricket Analytics Platform</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#555;font-size:0.7rem;'>"
        "Built with Streamlit · scikit-learn · Plotly<br/>Data: Synthetic IPL 2008–2023"
        "</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with st.spinner("Loading CrickIQ data …"):
    matches_df, deliveries_df = _load_all_data()

batting_df = _get_batting(deliveries_df)
bowling_df = _get_bowling(deliveries_df)

TEAMS = sorted(pd.concat([matches_df["team1"], matches_df["team2"]]).unique().tolist())
VENUES = sorted(matches_df["venue"].unique().tolist())
SEASONS = sorted(matches_df["season"].unique().tolist())


# ===========================================================================
# PAGE 1: Dashboard
# ===========================================================================

if page == "🏠 Dashboard":
    st.markdown(
        "<div class='crickiq-banner'>"
        "<h1>🏏 CrickIQ</h1>"
        "<p>Cricket Analytics & Match Outcome Predictor · IPL 2008–2023</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # --- KPI Metrics ---
    normal_matches = matches_df[matches_df["result"] == "normal"]
    total_matches = len(matches_df)
    total_teams = len(TEAMS)
    total_seasons = len(SEASONS)
    avg_win_runs = int(normal_matches["win_by_runs"][normal_matches["win_by_runs"] > 0].mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏟️ Total Matches", f"{total_matches:,}")
    c2.metric("🏏 Teams", total_teams)
    c3.metric("📅 Seasons", total_seasons)
    c4.metric("📊 Avg Win Margin (runs)", avg_win_runs)

    st.markdown("")

    # --- Row 1: Wins bar + Matches per season ---
    col1, col2 = st.columns([3, 2])
    with col1:
        wins_df = (
            matches_df[matches_df["winner"] != ""]
            .groupby("winner")
            .size()
            .reset_index(name="wins")
        )
        st.plotly_chart(team_wins_bar(wins_df), use_container_width=True)

    with col2:
        st.plotly_chart(matches_per_season_line(matches_df), use_container_width=True)

    # --- Row 2: Toss pie + Season wins table ---
    col3, col4 = st.columns([1, 2])
    with col3:
        st.plotly_chart(toss_decision_pie(matches_df), use_container_width=True)

    with col4:
        st.subheader("Season Overview")
        season_summary = (
            matches_df.groupby("season")
            .agg(
                total_matches=("id", "count"),
                unique_winners=("winner", "nunique"),
                toss_field_pct=("toss_decision", lambda x: round((x == "field").mean() * 100, 1)),
            )
            .reset_index()
        )
        season_summary.columns = ["Season", "Matches", "Unique Winners", "Field Toss %"]
        st.dataframe(
            season_summary.style.background_gradient(cmap="Greens", subset=["Matches"]),
            use_container_width=True,
            height=320,
        )


# ===========================================================================
# PAGE 2: Match Predictor
# ===========================================================================

elif page == "🔮 Match Predictor":
    st.markdown("## 🔮 Match Outcome Predictor")
    st.markdown("Select match conditions to predict the winner using **Random Forest + Logistic Regression**.")

    col_a, col_b = st.columns(2)
    with col_a:
        team1 = st.selectbox("🏏 Team 1", TEAMS, index=0, key="pred_t1")
        toss_winner = st.selectbox("🪙 Toss Winner", [team1, "Team 2"], key="pred_tw")
    with col_b:
        team2_opts = [t for t in TEAMS if t != team1]
        team2 = st.selectbox("🏏 Team 2", team2_opts, index=0, key="pred_t2")
        toss_decision = st.selectbox("🪙 Toss Decision", ["bat", "field"], key="pred_td")

    venue = st.selectbox("🏟️ Venue", VENUES, key="pred_venue")

    # Resolve toss_winner
    if toss_winner == "Team 2":
        toss_winner_resolved = team2
    else:
        toss_winner_resolved = team1

    if st.button("🔮 Predict Winner", key="predict_btn"):
        with st.spinner("Training models & predicting …"):
            # Train if needed
            from models.match_predictor import train_models, predict_match, RF_MODEL_PATH

            if not os.path.exists(RF_MODEL_PATH):
                train_result = train_models(matches_df)
                st.session_state["train_result"] = train_result
            else:
                if "train_result" not in st.session_state:
                    train_result = train_models(matches_df)
                    st.session_state["train_result"] = train_result

            result = predict_match(
                team1=team1,
                team2=team2,
                toss_winner=toss_winner_resolved,
                toss_decision=toss_decision,
                venue=venue,
            )

        winner = result["predicted_winner"]
        prob_t1 = result["prob_team1"]
        confidence = result["confidence"]

        # Banner
        st.markdown(
            f"<div style='background:#0D2E1A;border:2px solid #2ECC71;border-radius:10px;"
            f"padding:1.2rem;text-align:center;margin:1rem 0;'>"
            f"<span style='font-size:1.1rem;color:#888;'>Predicted Winner</span><br/>"
            f"<span style='font-size:2.2rem;font-weight:bold;color:#2ECC71;'>🏆 {winner}</span><br/>"
            f"<span style='color:#AAA;'>Confidence: <b style='color:#2ECC71;'>{confidence*100:.1f}%</b></span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Gauge
        st.plotly_chart(
            win_probability_gauge(team1, team2, prob_t1),
            use_container_width=True,
        )

        # Feature importance
        st.subheader("📊 Key Factors Driving the Prediction")
        fi = result["feature_importances"]
        st.plotly_chart(feature_importance_bar(fi), use_container_width=True)

        # Model metrics
        if "train_result" in st.session_state:
            tr = st.session_state["train_result"]
            mc1, mc2 = st.columns(2)
            mc1.metric("🌲 Random Forest Accuracy", f"{tr['rf_accuracy']*100:.1f}%")
            mc2.metric("📉 Logistic Regression Accuracy", f"{tr['lr_accuracy']*100:.1f}%")

    else:
        st.info("👆 Select match conditions above and click **Predict Winner** to see the ML prediction.")


# ===========================================================================
# PAGE 3: Player Analytics
# ===========================================================================

elif page == "📊 Player Analytics":
    st.markdown("## 📊 Player Analytics")

    search = st.text_input("🔍 Search player name", placeholder="e.g. Kohli, Bumrah …")

    tab_bat, tab_bowl, tab_pom = st.tabs(["🏏 Top Batsmen", "🎯 Top Bowlers", "🏆 Player of the Match"])

    # --- Batsmen tab ---
    with tab_bat:
        n_bat = st.slider("Show top N batsmen", 5, 30, 10, key="n_bat")
        bat_display = batting_df.copy()
        if search:
            bat_display = bat_display[bat_display["batsman"].str.contains(search, case=False, na=False)]
        bat_display = bat_display.rename(columns={
            "batsman": "Batsman", "runs": "Runs", "innings": "Innings",
            "balls_faced": "Balls", "strike_rate": "SR", "average": "Avg",
            "fifties": "50s", "centuries": "100s",
        })
        st.dataframe(
            bat_display[["Batsman", "Runs", "Innings", "Balls", "SR", "Avg", "50s", "100s"]]
            .head(n_bat)
            .reset_index(drop=True)
            .style.background_gradient(cmap="Greens", subset=["Runs"]),
            use_container_width=True,
        )
        top_bat = batting_df.copy()
        if search:
            top_bat = top_bat[top_bat["batsman"].str.contains(search, case=False, na=False)]
        st.plotly_chart(top_batsmen_bar(top_bat, n=n_bat), use_container_width=True)

    # --- Bowlers tab ---
    with tab_bowl:
        n_bowl = st.slider("Show top N bowlers", 5, 30, 10, key="n_bowl")
        bowl_display = bowling_df.copy()
        if search:
            bowl_display = bowl_display[bowl_display["bowler"].str.contains(search, case=False, na=False)]
        bowl_display = bowl_display.rename(columns={
            "bowler": "Bowler", "wickets": "Wickets", "runs_conceded": "Runs",
            "overs_bowled": "Overs", "economy": "Economy", "bowling_average": "Avg",
        })
        st.dataframe(
            bowl_display[["Bowler", "Wickets", "Runs", "Overs", "Economy", "Avg"]]
            .head(n_bowl)
            .reset_index(drop=True)
            .style.background_gradient(cmap="Reds", subset=["Wickets"]),
            use_container_width=True,
        )
        top_bowl = bowling_df.copy()
        if search:
            top_bowl = top_bowl[top_bowl["bowler"].str.contains(search, case=False, na=False)]
        st.plotly_chart(top_bowlers_bar(top_bowl, n=n_bowl), use_container_width=True)

    # --- Player of Match tab ---
    with tab_pom:
        pom_df = (
            matches_df[matches_df["player_of_match"] != ""]
            .groupby("player_of_match")
            .size()
            .reset_index(name="awards")
            .sort_values("awards", ascending=False)
        )
        if search:
            pom_df = pom_df[pom_df["player_of_match"].str.contains(search, case=False, na=False)]
        st.plotly_chart(player_of_match_bar(pom_df.head(20)), use_container_width=True)
        st.dataframe(pom_df.head(20).reset_index(drop=True), use_container_width=True)


# ===========================================================================
# PAGE 4: Team Clustering
# ===========================================================================

elif page == "🔵 Team Clustering":
    st.markdown("## 🔵 Team Clustering (K-Means)")
    st.markdown(
        "Teams are grouped into **4 clusters** based on wins, win %, "
        "and average win margins using K-Means + PCA."
    )

    with st.spinner("Running K-Means clustering …"):
        team_clusters = cluster_teams(matches_df, n_clusters=4)
        cluster_summary = team_cluster_summary(team_clusters)

    # Scatter
    st.plotly_chart(cluster_scatter(team_clusters, title="IPL Team Clusters (PCA 2-D)"), use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Cluster Assignments")
        display_df = team_clusters[["team", "cluster_label", "wins", "total_matches", "win_pct"]].copy()
        display_df["win_pct"] = (display_df["win_pct"] * 100).round(1).astype(str) + "%"
        display_df.columns = ["Team", "Cluster", "Wins", "Played", "Win %"]
        st.dataframe(
            display_df.sort_values("Cluster").reset_index(drop=True)
            .style.background_gradient(cmap="Greens", subset=["Wins"]),
            use_container_width=True,
        )

    with col_right:
        st.subheader("Cluster Summary")
        radar_cats = ["wins", "win_pct", "avg_win_runs", "avg_win_wkts"]
        # Normalise for radar
        radar_data = cluster_summary.copy()
        for col in radar_cats:
            col_range = radar_data[col].max() - radar_data[col].min()
            radar_data[col] = (
                (radar_data[col] - radar_data[col].min()) / (col_range if col_range > 0 else 1)
            ).round(3)
        radar_data["cluster"] = cluster_summary["cluster"]
        cats_display = ["wins", "win_pct", "avg_win_runs", "avg_win_wkts"]
        try:
            st.plotly_chart(radar_chart(radar_data, cats_display), use_container_width=True)
        except Exception:
            st.dataframe(cluster_summary, use_container_width=True)

    # Cluster descriptions
    st.markdown("### 📋 Cluster Descriptions")
    desc_cols = st.columns(4)
    descriptions = {
        "Dominant":    ("🟢", "Consistently high win rate, large margins. The giants of IPL."),
        "Competitive": ("🔵", "Above-average performance. Strong contenders each season."),
        "Rising":      ("🟡", "Mid-table teams showing improvement over recent seasons."),
        "Rebuilding":  ("🔴", "Lower win rates; rebuilding squads for future campaigns."),
    }
    for i, (label, (icon, desc)) in enumerate(descriptions.items()):
        with desc_cols[i]:
            st.markdown(
                f"<div style='background:#1A1A2E;border:1px solid #333;border-radius:8px;"
                f"padding:0.8rem;text-align:center;'>"
                f"<span style='font-size:1.5rem;'>{icon}</span><br/>"
                f"<b style='color:#2ECC71;'>{label}</b><br/>"
                f"<span style='color:#AAA;font-size:0.8rem;'>{desc}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ===========================================================================
# PAGE 5: Historical Analysis
# ===========================================================================

elif page == "📈 Historical Analysis":
    st.markdown("## 📈 Historical Analysis")

    tab_season, tab_venue, tab_h2h, tab_trend = st.tabs([
        "📅 Season Stats", "🏟️ Venue Analysis", "⚔️ Head-to-Head", "📈 Score Trends",
    ])

    # --- Season Stats ---
    with tab_season:
        season_df = (
            matches_df.groupby("season")
            .agg(
                total_matches=("id", "count"),
                unique_winners=("winner", "nunique"),
                toss_field_pct=("toss_decision", lambda x: round((x == "field").mean() * 100, 1)),
                avg_win_runs=("win_by_runs", lambda x: round(x[x > 0].mean(), 1)),
                avg_win_wkts=("win_by_wickets", lambda x: round(x[x > 0].mean(), 1)),
            )
            .reset_index()
        )
        season_df.columns = ["Season", "Matches", "Unique Winners", "Field Toss %", "Avg Win (runs)", "Avg Win (wkts)"]
        st.dataframe(
            season_df.style.background_gradient(cmap="Greens", subset=["Matches"]),
            use_container_width=True,
        )

    # --- Venue Analysis ---
    with tab_venue:
        venue_stats = (
            matches_df.groupby("venue")
            .agg(
                total_matches=("id", "count"),
                bat_first_wins=("toss_decision", lambda x: 0),  # placeholder
            )
            .reset_index()
        )
        # Proper bat-first wins calculation
        def _bat_first_wins(df, venue):
            sub = df[df["venue"] == venue]
            return int(
                ((sub["toss_decision"] == "bat") & (sub["toss_winner"] == sub["winner"])).sum()
                + ((sub["toss_decision"] == "field") & (sub["toss_winner"] != sub["winner"]) & (sub["winner"] != "")).sum()
            )

        venue_stats["bat_first_wins"] = venue_stats["venue"].apply(
            lambda v: _bat_first_wins(matches_df, v)
        )
        venue_stats["bat_first_win_rate"] = (
            venue_stats["bat_first_wins"] / venue_stats["total_matches"] * 100
        ).round(1)
        venue_stats = venue_stats.sort_values("total_matches", ascending=False)

        st.plotly_chart(venue_matches_bar(venue_stats), use_container_width=True)
        display_v = venue_stats.copy()
        display_v.columns = ["Venue", "Matches", "Bat-First Wins", "Bat-First Win %"]
        st.dataframe(
            display_v.reset_index(drop=True)
            .style.background_gradient(cmap="Greens", subset=["Matches"]),
            use_container_width=True,
        )

    # --- Head-to-Head Heatmap ---
    with tab_h2h:
        st.subheader("Head-to-Head Win Rate Matrix")
        st.caption("Cell value = row-team win rate against column-team")

        h2h_matches = matches_df[matches_df["winner"] != ""]
        teams_set = sorted(pd.concat([h2h_matches["team1"], h2h_matches["team2"]]).unique())
        matrix = pd.DataFrame(np.nan, index=teams_set, columns=teams_set)

        for t1 in teams_set:
            for t2 in teams_set:
                if t1 == t2:
                    matrix.loc[t1, t2] = np.nan
                    continue
                sub = h2h_matches[
                    ((h2h_matches["team1"] == t1) & (h2h_matches["team2"] == t2))
                    | ((h2h_matches["team1"] == t2) & (h2h_matches["team2"] == t1))
                ]
                if len(sub) > 0:
                    matrix.loc[t1, t2] = round((sub["winner"] == t1).mean(), 2)

        st.plotly_chart(head_to_head_heatmap(matrix.fillna(0.5)), use_container_width=True)

    # --- Score Trends ---
    with tab_trend:
        # Compute avg score per innings per season from deliveries
        innings_scores = (
            deliveries_df.groupby(["match_id", "inning"])["total_runs"]
            .sum()
            .reset_index()
            .rename(columns={"total_runs": "innings_total"})
        )
        innings_scores = innings_scores.merge(
            matches_df[["id", "season"]],
            left_on="match_id",
            right_on="id",
            how="left",
        )
        season_avg = (
            innings_scores.groupby("season")["innings_total"]
            .mean()
            .reset_index()
            .rename(columns={"innings_total": "avg_score"})
        )
        season_avg["avg_score"] = season_avg["avg_score"].round(1)

        st.plotly_chart(avg_score_trend(season_avg), use_container_width=True)

        # Wins by toss decision per season
        toss_season = (
            matches_df[matches_df["winner"] != ""]
            .assign(toss_match_win=lambda d: (d["toss_winner"] == d["winner"]).astype(int))
            .groupby("season")["toss_match_win"]
            .mean()
            .reset_index()
        )
        toss_season["toss_match_win"] = (toss_season["toss_match_win"] * 100).round(1)

        fig_toss = go.Figure(
            go.Scatter(
                x=toss_season["season"],
                y=toss_season["toss_match_win"],
                mode="lines+markers",
                name="Toss-Win converts to Match-Win %",
                line=dict(color="#F39C12", width=2),
                marker=dict(size=7),
            )
        )
        fig_toss.update_layout(
            title="Toss-to-Match Win Conversion % Per Season",
            xaxis_title="Season",
            yaxis_title="% Matches won by Toss Winner",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(color="#FFFFFF"),
        )
        fig_toss.update_xaxes(gridcolor="#2D2D2D")
        fig_toss.update_yaxes(gridcolor="#2D2D2D")
        st.plotly_chart(fig_toss, use_container_width=True)
