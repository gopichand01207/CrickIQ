"""
queries.py
----------
SQL query helpers and ORM-style accessor functions for CrickIQ.

All functions accept a SQLAlchemy engine and return pandas DataFrames.
They work with both MySQL and SQLite backends.
"""

import pandas as pd
from sqlalchemy import text


# ---------------------------------------------------------------------------
# Table bootstrap
# ---------------------------------------------------------------------------

CREATE_MATCHES_TABLE = """
CREATE TABLE IF NOT EXISTS matches (
    id               INTEGER PRIMARY KEY,
    season           INTEGER,
    city             TEXT,
    date             TEXT,
    team1            TEXT,
    team2            TEXT,
    toss_winner      TEXT,
    toss_decision    TEXT,
    result           TEXT,
    dl_applied       INTEGER,
    winner           TEXT,
    win_by_runs      INTEGER,
    win_by_wickets   INTEGER,
    player_of_match  TEXT,
    venue            TEXT,
    umpire1          TEXT,
    umpire2          TEXT
)
"""

CREATE_DELIVERIES_TABLE = """
CREATE TABLE IF NOT EXISTS deliveries (
    match_id         INTEGER,
    inning           INTEGER,
    batting_team     TEXT,
    bowling_team     TEXT,
    over             INTEGER,
    ball             INTEGER,
    batsman          TEXT,
    non_striker      TEXT,
    bowler           TEXT,
    is_super_over    INTEGER,
    wide_runs        INTEGER,
    bye_runs         INTEGER,
    legbye_runs      INTEGER,
    noball_runs      INTEGER,
    penalty_runs     INTEGER,
    batsman_runs     INTEGER,
    extra_runs       INTEGER,
    total_runs       INTEGER,
    player_dismissed TEXT,
    dismissal_kind   TEXT,
    fielder          TEXT
)
"""


def bootstrap_tables(engine):
    """Create tables if they do not exist."""
    with engine.begin() as conn:
        conn.execute(text(CREATE_MATCHES_TABLE))
        conn.execute(text(CREATE_DELIVERIES_TABLE))


def load_csv_to_db(engine, matches_df: pd.DataFrame, deliveries_df: pd.DataFrame):
    """Write DataFrames to the database, replacing existing data."""
    bootstrap_tables(engine)
    matches_df.to_sql("matches", engine, if_exists="replace", index=False)
    deliveries_df.to_sql("deliveries", engine, if_exists="replace", index=False)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def fetch_all_matches(engine) -> pd.DataFrame:
    """Return all rows from the matches table."""
    with engine.connect() as conn:
        return pd.read_sql(text("SELECT * FROM matches"), conn)


def fetch_team_wins(engine) -> pd.DataFrame:
    """
    Return total wins per team.

    Returns
    -------
    DataFrame with columns: winner, wins
    """
    sql = """
        SELECT winner, COUNT(*) AS wins
        FROM matches
        WHERE winner != '' AND winner IS NOT NULL
        GROUP BY winner
        ORDER BY wins DESC
    """
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


def fetch_player_of_match_counts(engine, top_n: int = 20) -> pd.DataFrame:
    """
    Return top `top_n` players by Player-of-the-Match awards.

    Returns
    -------
    DataFrame with columns: player_of_match, awards
    """
    sql = f"""
        SELECT player_of_match, COUNT(*) AS awards
        FROM matches
        WHERE player_of_match != '' AND player_of_match IS NOT NULL
        GROUP BY player_of_match
        ORDER BY awards DESC
        LIMIT {top_n}
    """
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


def fetch_venue_stats(engine) -> pd.DataFrame:
    """
    Return per-venue match counts and batting-first win rate.

    Returns
    -------
    DataFrame with columns: venue, total_matches, bat_first_wins, bat_first_win_rate
    """
    sql = """
        SELECT
            venue,
            COUNT(*) AS total_matches,
            SUM(CASE
                    WHEN toss_decision = 'bat' AND toss_winner = winner THEN 1
                    WHEN toss_decision = 'field' AND toss_winner != winner AND winner != '' THEN 1
                    ELSE 0
                END) AS bat_first_wins
        FROM matches
        GROUP BY venue
        ORDER BY total_matches DESC
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    df["bat_first_win_rate"] = (df["bat_first_wins"] / df["total_matches"].replace(0, 1) * 100).round(1)
    return df


def fetch_season_stats(engine) -> pd.DataFrame:
    """
    Return per-season summary statistics.

    Returns
    -------
    DataFrame with columns: season, total_matches, unique_winners
    """
    sql = """
        SELECT
            season,
            COUNT(*) AS total_matches,
            COUNT(DISTINCT winner) AS unique_winners
        FROM matches
        GROUP BY season
        ORDER BY season
    """
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


def fetch_head_to_head(engine) -> pd.DataFrame:
    """
    Return raw match data needed for head-to-head matrix computation.

    Returns
    -------
    DataFrame with columns: team1, team2, winner
    """
    with engine.connect() as conn:
        return pd.read_sql(
            text("SELECT team1, team2, winner FROM matches WHERE winner != ''"),
            conn,
        )
