# CrickIQ — Cricket Analytics & Match Outcome Predictor

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-8.0-4479A1?style=flat-square&logo=mysql&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.1-150458?style=flat-square&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.19-3F4F75?style=flat-square&logo=plotly&logoColor=white)

---

## 🏏 Project Overview

**CrickIQ** is a production-quality, end-to-end Cricket Analytics platform focused on the Indian Premier League (IPL). It combines machine learning, interactive visualisations, and a MySQL-backed data layer to deliver:

- **Match Outcome Prediction** using Random Forest and Logistic Regression
- **Player Performance Analysis** with batting, bowling, and award aggregations
- **Team Clustering** via K-Means + PCA
- **Historical Analysis** including season trends, venue stats, and head-to-head matrices

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔮 Match Predictor | Predict winner from team, toss, venue inputs |
| 📊 Player Analytics | Top batsmen / bowlers with sortable tables and charts |
| 🔵 Team Clustering | K-Means (k=4) clusters with 2-D PCA scatter and radar chart |
| 📈 Historical Analysis | Season stats, venue analysis, H2H heatmap, score trends |
| 🏠 Dashboard | KPI metrics, wins bar chart, season line chart, toss pie |
| 💾 MySQL + SQLite | MySQL primary store with automatic SQLite fallback |
| 🎨 Dark Theme | Custom CSS with cricket-inspired green accents (`#2ECC71`) |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (app.py)                    │
│   Dashboard  │  Predictor  │  Players  │  Clusters  │  History  │
└──────┬───────────────┬─────────────────┬────────────────────────┘
       │               │                 │
       ▼               ▼                 ▼
┌────────────┐  ┌────────────┐  ┌────────────────┐
│  utils/    │  │  models/   │  │  database/     │
│  preprocessor│  │  match_   │  │  db_config.py  │
│  feature_  │  │  predictor │  │  queries.py    │
│  engineering│  │  player_  │  └───────┬────────┘
│  visuali-  │  │  performance│          │
│  zations   │  │  clustering│    MySQL / SQLite
└────────────┘  └────────────┘
       │
       ▼
┌──────────────┐
│  data/       │
│  ipl_matches │
│  ipl_deliver │
│  generate_   │
│  data.py     │
└──────────────┘
```

---

## 🗂️ Project Structure

```
CrickIQ/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── setup_db.py                 # Database initialisation script
├── README.md                   # This file
├── data/
│   ├── generate_data.py        # Synthetic IPL data generator
│   ├── ipl_matches.csv         # 560 match records (auto-generated)
│   └── ipl_deliveries.csv      # Ball-by-ball data (auto-generated)
├── database/
│   ├── __init__.py
│   ├── db_config.py            # MySQL / SQLite engine factory
│   └── queries.py              # SQL query helpers
├── models/
│   ├── __init__.py
│   ├── match_predictor.py      # Random Forest + Logistic Regression
│   ├── player_performance.py   # GradientBoosting score predictor
│   └── clustering.py           # K-Means team & player clustering
├── utils/
│   ├── __init__.py
│   ├── feature_engineering.py  # Feature pipeline
│   ├── visualizations.py       # Plotly chart helpers
│   └── preprocessor.py         # Data loading and cleaning
└── notebooks/
    └── analysis.ipynb          # Exploratory Data Analysis
```

---

## 🚀 Installation & Running

### 1. Clone / download the project

```bash
cd CrickIQ
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate data (optional — app auto-generates on first run)

```bash
python data/generate_data.py
```

### 5. (Optional) Set up MySQL

```bash
# Create the database
mysql -u root -p -e "CREATE DATABASE crickiq;"

# Set connection env vars (or edit database/db_config.py)
export CRICKIQ_DB_HOST=localhost
export CRICKIQ_DB_USER=root
export CRICKIQ_DB_PASSWORD=yourpassword

# Load data into MySQL
python setup_db.py
```

> **Note:** MySQL is **optional**. The app automatically falls back to SQLite if MySQL is not available.

### 6. Run the Streamlit app

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** in your browser.

---

## 🤖 Machine Learning Models

### Match Predictor
| Model | Algorithm | Features | Target |
|-------|-----------|----------|--------|
| Random Forest | `RandomForestClassifier(n_estimators=200)` | 12 engineered features | Team 1 wins (binary) |
| Logistic Regression | `LogisticRegression(max_iter=1000)` | Same 12 features (scaled) | Team 1 wins (binary) |

**Engineered features:** toss decision, toss winner, venue win rate, season form (last 5 games), head-to-head win rate, batting-first advantage, average win margin.

### Player Score Predictor
- **Model:** `GradientBoostingRegressor(n_estimators=150, max_depth=4)`
- **Features:** career average, strike rate, innings count, label-encoded player ID
- **Target:** innings score (runs)

### Team Clustering
- **Algorithm:** K-Means (k=4)
- **Features:** total wins, win %, average win-by-runs, average win-by-wickets
- **Visualisation:** PCA → 2-D scatter + radar chart

---

## 📸 Screenshots

> _Run `streamlit run app.py` to see the live dashboard._

| Page | Description |
|------|-------------|
| 🏠 Dashboard | Team wins bar chart, season trends, toss distribution |
| 🔮 Predictor | Win probability gauges, feature importance chart |
| 📊 Players | Tabbed batting/bowling leaderboards with search |
| 🔵 Clusters | PCA scatter plot with radar chart comparison |
| 📈 History | H2H heatmap, venue stats, score trend lines |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend / UI | Streamlit 1.32 |
| ML & Stats | scikit-learn 1.4, scipy 1.12 |
| Data Wrangling | pandas 2.1, numpy 1.26 |
| Visualisation | Plotly 5.19, matplotlib, seaborn |
| Database | MySQL 8 (primary), SQLite (fallback) |
| ORM / Connector | SQLAlchemy 2.0, mysql-connector-python |
| Model Persistence | joblib 1.3 |

---

## 📄 License

MIT © 2024 — Built as a resume portfolio project demonstrating full-stack data science and ML engineering.
