"""
generate_data.py
----------------
Generates realistic synthetic IPL match data for CrickIQ.

Produces two CSV files:
  - ipl_matches.csv   : 500+ match-level records (2008-2023)
  - ipl_deliveries.csv: Ball-by-ball delivery records for each match

Run directly:  python data/generate_data.py
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEAMS = [
    "Mumbai Indians",
    "Chennai Super Kings",
    "Royal Challengers Bangalore",
    "Kolkata Knight Riders",
    "Sunrisers Hyderabad",
    "Delhi Capitals",
    "Rajasthan Royals",
    "Punjab Kings",
]

VENUES = {
    "Mumbai Indians": ["Wankhede Stadium, Mumbai", "Brabourne Stadium, Mumbai"],
    "Chennai Super Kings": ["MA Chidambaram Stadium, Chennai", "Punjab Cricket Association Stadium, Mohali"],
    "Royal Challengers Bangalore": ["M. Chinnaswamy Stadium, Bangalore"],
    "Kolkata Knight Riders": ["Eden Gardens, Kolkata"],
    "Sunrisers Hyderabad": ["Rajiv Gandhi International Cricket Stadium, Hyderabad"],
    "Delhi Capitals": ["Arun Jaitley Stadium, Delhi", "Feroz Shah Kotla, Delhi"],
    "Rajasthan Royals": ["Sawai Mansingh Stadium, Jaipur", "Sharjah Cricket Stadium"],
    "Punjab Kings": ["Punjab Cricket Association IS Bindra Stadium, Mohali"],
}

ALL_VENUES = [v for venues in VENUES.values() for v in venues]

CITIES = {
    "Wankhede Stadium, Mumbai": "Mumbai",
    "Brabourne Stadium, Mumbai": "Mumbai",
    "MA Chidambaram Stadium, Chennai": "Chennai",
    "Punjab Cricket Association Stadium, Mohali": "Mohali",
    "M. Chinnaswamy Stadium, Bangalore": "Bangalore",
    "Eden Gardens, Kolkata": "Kolkata",
    "Rajiv Gandhi International Cricket Stadium, Hyderabad": "Hyderabad",
    "Arun Jaitley Stadium, Delhi": "Delhi",
    "Feroz Shah Kotla, Delhi": "Delhi",
    "Sawai Mansingh Stadium, Jaipur": "Jaipur",
    "Sharjah Cricket Stadium": "Sharjah",
    "Punjab Cricket Association IS Bindra Stadium, Mohali": "Mohali",
}

UMPIRES = [
    "Nitin Menon", "Anil Chaudhary", "S. Ravi", "C.K. Nandan",
    "K.N. Ananthapadmanabhan", "Virender Sharma", "Paul Reiffel",
    "Rod Tucker", "Sundaram Ravi", "Bruce Oxenford",
]

# Win-rate priors (higher → better team historically)
TEAM_STRENGTH = {
    "Mumbai Indians": 0.62,
    "Chennai Super Kings": 0.60,
    "Kolkata Knight Riders": 0.55,
    "Royal Challengers Bangalore": 0.50,
    "Sunrisers Hyderabad": 0.52,
    "Rajasthan Royals": 0.48,
    "Delhi Capitals": 0.47,
    "Punjab Kings": 0.44,
}

BATSMEN = [
    "V Kohli", "RG Sharma", "SK Raina", "MS Dhoni", "DA Warner",
    "AB de Villiers", "KL Rahul", "SR Watson", "G Gambhir", "RV Uthappa",
    "AM Rahane", "CH Gayle", "S Dhawan", "Q de Kock", "JP Faulkner",
    "DJ Bravo", "AT Rayudu", "F du Plessis", "SV Samson", "MM Ali",
    "PP Shaw", "SA Yadav", "DP Conway", "N Pooran", "JM Bairstow",
    "RD Gaikwad", "B Stokes", "M Vohra", "MW Stoinis", "HH Pandya",
]

BOWLERS = [
    "JJ Bumrah", "SL Malinga", "DJ Bravo", "B Kumar", "P Kumar",
    "RA Jadeja", "R Ashwin", "YS Chahal", "Harbhajan Singh", "SP Narine",
    "AD Russell", "MM Patel", "KA Pollard", "MJ McClenaghan", "UT Yadav",
    "MK Pandey", "Mohammed Siraj", "T Natarajan", "A Nortje", "K Rabada",
]

DISMISSAL_KINDS = [
    "caught", "bowled", "lbw", "run out", "stumped",
    "caught and bowled", "hit wicket", "",
]

SEASONS = list(range(2008, 2024))


# ---------------------------------------------------------------------------
# Match generation
# ---------------------------------------------------------------------------

def generate_matches(n: int = 560) -> pd.DataFrame:
    """Generate `n` synthetic IPL match records."""
    records = []
    match_id = 1

    matches_per_season = n // len(SEASONS)
    extra = n % len(SEASONS)

    for season_idx, season in enumerate(SEASONS):
        count = matches_per_season + (1 if season_idx < extra else 0)
        # Generate start date for the season (IPL typically Apr-May)
        season_start = pd.Timestamp(f"{season}-04-01")

        for i in range(count):
            match_date = season_start + pd.Timedelta(days=int(np.random.randint(0, 60)))

            # Pick two different teams
            team1, team2 = np.random.choice(TEAMS, size=2, replace=False).tolist()

            # Toss
            toss_winner = np.random.choice([team1, team2])
            toss_decision = np.random.choice(["bat", "field"], p=[0.4, 0.6])

            # Determine winner using team strength + toss advantage
            s1 = TEAM_STRENGTH[team1]
            s2 = TEAM_STRENGTH[team2]
            toss_boost = 0.05 if toss_winner == team1 else -0.05
            p_team1_wins = (s1 / (s1 + s2)) + toss_boost
            p_team1_wins = np.clip(p_team1_wins, 0.1, 0.9)

            if np.random.random() < p_team1_wins:
                winner = team1
            else:
                winner = team2

            # Result type
            result = np.random.choice(["normal", "tie", "no result"], p=[0.95, 0.03, 0.02])
            if result != "normal":
                winner = ""
                win_by_runs = 0
                win_by_wickets = 0
            else:
                if toss_decision == "bat":
                    if winner == toss_winner:
                        win_by_runs = int(np.random.randint(1, 100))
                        win_by_wickets = 0
                    else:
                        win_by_runs = 0
                        win_by_wickets = int(np.random.randint(1, 10))
                else:
                    if winner == toss_winner:
                        win_by_runs = 0
                        win_by_wickets = int(np.random.randint(1, 10))
                    else:
                        win_by_runs = int(np.random.randint(1, 100))
                        win_by_wickets = 0

            venue = np.random.choice(ALL_VENUES)
            city = CITIES[venue]

            player_of_match = np.random.choice(BATSMEN + BOWLERS)

            dl_applied = int(np.random.random() < 0.05)
            u1, u2 = np.random.choice(UMPIRES, size=2, replace=False).tolist()

            records.append({
                "id": match_id,
                "season": season,
                "city": city,
                "date": match_date.strftime("%Y-%m-%d"),
                "team1": team1,
                "team2": team2,
                "toss_winner": toss_winner,
                "toss_decision": toss_decision,
                "result": result,
                "dl_applied": dl_applied,
                "winner": winner,
                "win_by_runs": win_by_runs,
                "win_by_wickets": win_by_wickets,
                "player_of_match": player_of_match,
                "venue": venue,
                "umpire1": u1,
                "umpire2": u2,
            })
            match_id += 1

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Delivery generation
# ---------------------------------------------------------------------------

def generate_deliveries(matches_df: pd.DataFrame, deliveries_per_match: int = 240) -> pd.DataFrame:
    """Generate ball-by-ball delivery data for every match."""
    all_deliveries = []

    for _, match in matches_df.iterrows():
        mid = match["id"]
        t1 = match["team1"]
        t2 = match["team2"]

        for inning in [1, 2]:
            batting_team = t1 if inning == 1 else t2
            bowling_team = t2 if inning == 1 else t1

            # Pick random batsmen and bowlers for this innings
            innings_batsmen = np.random.choice(BATSMEN, size=11, replace=False).tolist()
            innings_bowlers = np.random.choice(BOWLERS, size=5, replace=False).tolist()

            batsman_idx = 0
            striker = innings_batsmen[batsman_idx]
            batsman_idx += 1
            non_striker = innings_batsmen[batsman_idx]
            batsman_idx += 1

            wickets = 0
            ball_count = 0
            target_balls = int(np.random.randint(100, 130))  # ~17-21 overs worth

            for over in range(20):
                bowler = innings_bowlers[over % 5]
                for ball in range(1, 7):
                    if wickets >= 10 or ball_count >= target_balls:
                        break

                    wide = int(np.random.random() < 0.04)
                    noball = int(np.random.random() < 0.02)
                    bye = int(np.random.random() < 0.01)
                    legbye = int(np.random.random() < 0.02)
                    batsman_runs = 0 if wide else int(np.random.choice([0, 1, 2, 3, 4, 6], p=[0.42, 0.25, 0.08, 0.02, 0.15, 0.08]))
                    extra_runs = wide + noball + bye + legbye
                    total_runs = batsman_runs + extra_runs

                    # Dismissal
                    dismissed = ""
                    dismissal_kind = ""
                    fielder = ""
                    if not wide and not noball and np.random.random() < 0.065:
                        dismissed = striker
                        dismissal_kind = np.random.choice(DISMISSAL_KINDS[:-1])
                        if dismissal_kind in ("caught", "run out", "stumped"):
                            fielder = np.random.choice(innings_batsmen)
                        wickets += 1
                        if batsman_idx < 11:
                            striker = innings_batsmen[batsman_idx]
                            batsman_idx += 1

                    all_deliveries.append({
                        "match_id": mid,
                        "inning": inning,
                        "batting_team": batting_team,
                        "bowling_team": bowling_team,
                        "over": over,
                        "ball": ball,
                        "batsman": striker,
                        "non_striker": non_striker,
                        "bowler": bowler,
                        "is_super_over": 0,
                        "wide_runs": wide,
                        "bye_runs": bye,
                        "legbye_runs": legbye,
                        "noball_runs": noball,
                        "penalty_runs": 0,
                        "batsman_runs": batsman_runs,
                        "extra_runs": extra_runs,
                        "total_runs": total_runs,
                        "player_dismissed": dismissed,
                        "dismissal_kind": dismissal_kind,
                        "fielder": fielder,
                    })

                    # Rotate strike on odd runs
                    if batsman_runs % 2 == 1:
                        striker, non_striker = non_striker, striker

                    ball_count += 1

                if wickets >= 10 or ball_count >= target_balls:
                    break

    return pd.DataFrame(all_deliveries)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("Generating IPL match data …")
    matches = generate_matches(560)
    matches_path = os.path.join(out_dir, "ipl_matches.csv")
    matches.to_csv(matches_path, index=False)
    print(f"  ✓ Saved {len(matches)} matches → {matches_path}")

    print("Generating delivery-level data (this may take ~30 s) …")
    deliveries = generate_deliveries(matches)
    deliveries_path = os.path.join(out_dir, "ipl_deliveries.csv")
    deliveries.to_csv(deliveries_path, index=False)
    print(f"  ✓ Saved {len(deliveries)} deliveries → {deliveries_path}")

    print("\nData generation complete.")


if __name__ == "__main__":
    main()
