# March Madness Prediction System

A multi-year NCAA Tournament bracket prediction pipeline built on the hypothesis that **team and player inconsistency metrics predict tournament underperformance better than raw efficiency ratings alone**.

Correctly predicted Michigan as the 2026 NCAA Champion (39.7% champion probability) — finishing in the **96.9th percentile** on ESPN Tournament Challenge.

---

## How It Works

The system collects game-by-game data going back to 2011 and engineers features in five groups:

- **Baseline efficiency** — Adjusted O/D ratings, Barthag, WAB, four factors (via Torvik)
- **Team variance** — Offensive/defensive CV, EFG% volatility, performance distribution width
- **Opponent-adjusted dropoff** — How efficiency shifts from Tier 3 to Tier 1 opponents
- **Star player concentration** — Herfindahl scoring index, top scorer usage and consistency
- **Location/home splits** — Performance gaps at neutral sites vs. home court

Six models are trained using walk-forward cross-validation (train on years 1–N, test on N+1): Linear Regression, Ridge, Random Forest, XGBoost, LightGBM, and MLP. An ensemble combines them using inverse-MAE weighting.

Four prediction modes are generated per year:

| Mode | Description |
|------|-------------|
| **Deterministic** | Full bracket simulation through actual seedings and regions |
| **Monte Carlo** | 10,000 bracket simulations → win probability distributions |
| **Scaled** | Raw predictions linearly stretched to preserve relative order |
| **Ranked** | Team rankings mapped to win totals via historical tournament structure |

The deterministic model is the primary output. A tuned K parameter controls how sharply the model favors higher-rated teams in each matchup.

---

## Results

| Year | ESPN Score | vs. Chalk | Round Accuracy | Champion Called? |
|------|-----------|-----------|----------------|-----------------|
| 2023 | 430 | +30 | 54.7% | ✗ (Houston favored) |
| 2024 | 1,300 | +630 | 50.0% | ✓ UConn |
| 2025 | 1,030 | +180 | 64.1% | ✗ (Auburn favored) |
| **2026** | **1,310 (96.9th pct)** | **+470** | **59.4%** | **✓ Michigan** |

---

## Project Structure

```
march_madness/
├── config.py                 # Paths, seasons, bracket regions, constants
├── pipeline.py               # Master orchestrator (run this first)
├── torvik_collector.py       # Torvik data: team stats, player stats, gamelogs, timemachine
├── tournament_collector.py   # Sports Reference: bracket results (target variable)
├── sportsref_collector.py    # Sports Reference: team and player gamelogs
├── feature_engineering.py    # Builds the full feature matrix from raw data
├── model.py                  # Ensemble model, bracket simulation, Monte Carlo
├── data/
│   ├── raw/
│   │   ├── torvik/           # Cached Torvik CSVs
│   │   ├── sportsref/        # Cached Sports Reference gamelogs
│   │   └── tournament/       # Cached tournament results
│   └── processed/            # Feature matrix outputs
└── visualizations/
    └── bracket_all_years.jsx # Interactive bracket dashboard (React)
```

---

## Setup

```bash
git clone https://github.com/yourusername/march-madness
cd march-madness
pip install -r requirements.txt
```

### Run the pipeline

```bash
# Verify data source connectivity first
python test_connections.py

# Step 1: Torvik data (fast)
python pipeline.py --steps torvik

# Step 2: Tournament results
python pipeline.py --steps tournament

# Step 3: Sports Reference gamelogs (slow — ~3s per request, safe to interrupt)
python pipeline.py --steps sportsref --year 2026

# Or run everything
python pipeline.py --steps all
```

### Generate predictions

```bash
python feature_engineering.py
python model.py --year 2026
```

---

## Data Sources

- **[Barttorvik.com](https://barttorvik.com)** — Adjusted efficiency ratings, four factors, player stats, game logs (2011–present)
- **[Sports Reference](https://www.sports-reference.com/cbb)** — Tournament bracket results, team and player game logs

> Sports Reference is scraped directly. A 3-second delay between requests is enforced to avoid rate limiting. All results are cached locally.

---

## Key Design Decisions

**Mode over mean for Monte Carlo** — The champion and runner-up predictions use the *mode* of the win distribution across 10,000 simulations, not the mean. Mean-based rounding systematically suppresses high-outcome predictions.

**Walk-forward cross-validation** — Models are always trained on historical years and tested on the next year. No future data leaks into training.

**Opponent tier assignment** — Each game in a team's log is labeled using Torvik T-Rank ratings frozen at Selection Sunday, ensuring opponent quality reflects end-of-season context rather than preseason or current rankings.

**CV over raw standard deviation** — Coefficient of variation (std/mean) is used for variance features rather than raw standard deviation, making comparisons meaningful across teams with different offensive outputs.

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
lightgbm
requests
beautifulsoup4
lxml
```
