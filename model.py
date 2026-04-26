"""
model.py
─────────
March Madness prediction model.

Trains multiple models to predict tournament wins (0-6) using variance-based features.
Also flags upset probability as a secondary target.

Models:
  - Linear regression (baseline)
  - Ridge regression
  - Random Forest
  - XGBoost
  - LightGBM
  - Neural network (MLP)
  - Ensemble (weighted average of best models)

Validation:
  - Walk-forward CV: train on years 1..N, test on year N+1
  - Also reports leave-one-year-out results

Metrics (all reported for every model):
  - MAE on wins (0-6 scale)
  - Exact round accuracy (correct wins predicted)
  - Upset detection F1 (seed >= 9 beats seed <= 4, or any double-digit seed wins)

Usage:
  python model.py                    # train and evaluate all models
  python model.py --year 2024        # predict a specific year (train on all prior)
  python model.py --feature-importance  # show top features
"""

import argparse
from email import errors
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, f1_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from config import BRACKET_REGIONS
from scipy.special import expit  # sigmoid
from scipy.stats import mode as scipy_mode

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logging.warning("xgboost not installed — skipping XGBoost model")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logging.warning("lightgbm not installed — skipping LightGBM model")

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent
DATA_PROC  = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = DATA_PROC / "features_all.csv"

# ── Constants ──────────────────────────────────────────────────────────────────
META_COLS = {"team_abbrev", "team_name", "year", "seed", "wins", "exit_round",  "is_upset_team"}
TARGET       = "wins"
MIN_TRAIN_YEARS = 3   # need at least 3 years before first test year

# Upset definition: team with seed >= 10 wins at least 1 game
# OR any team with seed > opponent seed wins (we use seed >= 9 as proxy for "underdog")
UPSET_SEED_THRESHOLD = 9


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Features not found at {FEATURES_PATH}. Run feature_engineering.py first.")
    df = pd.read_csv(FEATURES_PATH)
    log.info(f"Loaded features: {df.shape} — {df['year'].nunique()} years, {len(df)} team-seasons")
    return df


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return all non-metadata columns to use as features."""
    return [c for c in df.columns if c not in META_COLS]


def add_upset_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary upset flag:
      1 = team with seed >= UPSET_SEED_THRESHOLD won at least 1 game
      0 = otherwise
    This captures double-digit seeds and high seeds that outperform.
    """
    df = df.copy()
    return df


# ---- Helper Functions for improved rankings--------------------------------------------------

def apply_calibration_scaling(df: pd.DataFrame, raw_col: str, new_col: str) -> pd.DataFrame:
    """
    Option A — Calibration scaling.
    Stretches raw predictions so the top team reaches ~5.5 wins.
    Preserves relative ordering, just expands the range.
    """
    raw = df[raw_col]
    min_val = raw.min()
    max_val = raw.max()
    TARGET_MAX = 5.5
    TARGET_MIN = 0.0
    scaled = (raw - min_val) / (max_val - min_val)
    stretched = scaled * (TARGET_MAX - TARGET_MIN) + TARGET_MIN
    df[new_col] = stretched.round().clip(0, 6).astype(int)
    return df


def apply_rank_mapping(df: pd.DataFrame, raw_col: str, new_col: str) -> pd.DataFrame:
    """
    Option B — Rank-based mapping.
    Maps predicted rank to historical win totals based on tournament structure:
    1st->6, 2nd->5, 3rd/4th->4, 5th-8th->3, 9th-16th->2, 17th-32nd->1, rest->0
    """
    ranks = df[raw_col].rank(ascending=False, method='first').astype(int)
    def rank_to_wins(rank):
        if rank == 1:   return 6
        if rank == 2:   return 5
        if rank <= 4:   return 4
        if rank <= 8:   return 3
        if rank <= 16:  return 2
        if rank <= 32:  return 1
        return 0
    df[new_col] = ranks.map(rank_to_wins)
    return df

HISTORICAL_UPSET_RATES = {
    (1,16): 0.01, (2,15): 0.06, (3,14): 0.15, (4,13): 0.20,
    (5,12): 0.35, (6,11): 0.38, (7,10): 0.40, (8,9):  0.50,
}

def add_upset_rate_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add historical R64 upset rate based on seed matchup."""
    def get_rate(seed):
        s = int(seed)
        opponent_seed = 17 - s
        pair = (min(s, opponent_seed), max(s, opponent_seed))
        return HISTORICAL_UPSET_RATES.get(pair, 0.0)
    df = df.copy()
    df["historical_upset_rate"] = df["seed"].apply(get_rate)
    return df

# Deterministic and Monte Carlo simulations -------------------------------------------------

BRACKET_SEEDS = {
    # Each region: list of (slot, seed) in first-round matchup order
    # Slots pair as (1,16), (8,9), (5,12), (4,13), (6,11), (3,14), (7,10), (2,15)
    "matchups": [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]
}

K = 1.0  # default, will be overridden by tune_k() result

def win_prob(raw_a: float, raw_b: float, k: float = K, noise: float = 0.0) -> float:
    """Probability that team A beats team B. noise pulls toward 50/50."""
    base = float(expit(k * (raw_a - raw_b)))
    return base * (1 - noise) + 0.5 * noise


def build_region_bracket(region_teams: pd.DataFrame) -> list:
    """
    Given a DataFrame of teams in one region (with seed and raw_Ensemble),
    return list of first-round matchups as (team_a, team_b) tuples.
    Each element is a dict with team_name, seed, raw_Ensemble.
    """
    by_seed = {int(row["seed"]): row for _, row in region_teams.iterrows()}
    matchups = []
    for s1, s2 in BRACKET_SEEDS["matchups"]:
        if s1 in by_seed and s2 in by_seed:
            matchups.append((by_seed[s1], by_seed[s2]))
    return matchups


def simulate_region(matchups: list, k: float = K, deterministic: bool = False) -> tuple:
    wins = {m[i]["team_name"]: 0 for m in matchups for i in [0,1]}
    current_round = [list(m) for m in matchups]
    last_winner = None

    while len(current_round) >= 1:
        next_round_teams = []
        for matchup in current_round:
            a, b = matchup[0], matchup[1]
            p = win_prob(float(a["raw_Ensemble"]), float(b["raw_Ensemble"]), k)
            if deterministic:
                winner = a if p >= 0.5 else b
            else:
                winner = a if np.random.random() < p else b
            wins[winner["team_name"]] += 1
            next_round_teams.append(winner)
            last_winner = winner

        if len(next_round_teams) == 1:
            break  # region champion found, stop

        current_round = [[next_round_teams[i], next_round_teams[i+1]]
                         for i in range(0, len(next_round_teams)-1, 2)]

    return wins, last_winner


def simulate_final_four(region_winners: list, k: float = K, deterministic: bool = False) -> dict:
    """
    Simulate Final Four + Championship from 4 region winners.
    Returns dict of team_name -> additional wins (0-2).
    """
    wins = {t["team_name"]: 0 for t in region_winners}
    # Semifinal 1: region 0 vs region 1, Semifinal 2: region 2 vs region 3
    semifinal_winners = []
    for a, b in [(region_winners[0], region_winners[1]),
                 (region_winners[2], region_winners[3])]:
        p = win_prob(float(a["raw_Ensemble"]), float(b["raw_Ensemble"]), k)
        if deterministic:
            winner = a if p >= 0.5 else b
        else:
            winner = a if np.random.random() < p else b
        wins[winner["team_name"]] += 1
        semifinal_winners.append(winner)
    # Championship
    a, b = semifinal_winners[0], semifinal_winners[1]
    p = win_prob(float(a["raw_Ensemble"]), float(b["raw_Ensemble"]), k)
    if deterministic:
        winner = a if p >= 0.5 else b
    else:
        winner = a if np.random.random() < p else b
    wins[winner["team_name"]] += 1
    return wins


def run_bracket_simulation(df: pd.DataFrame, regions: list,
                            n_sims: int = 10000,
                            k: float = K) -> pd.DataFrame:
    """
    Run full bracket simulation.
    Adds pred_deterministic, pred_monte_carlo, and prob_* columns to df.
    """
    df = df.copy()
    region_col = "region"  # must exist in df

    # Deterministic
    det_wins = {name: 0 for name in df["team_name"]}
    region_winners_det = []
    for region in regions:
        rdf = df[df[region_col] == region]
        matchups = build_region_bracket(rdf)
        region_wins, region_winner = simulate_region(matchups, k=k, deterministic=True)
        for name, w in region_wins.items():
            det_wins[name] += w
        region_winners_det.append(region_winner)

    ff_wins = simulate_final_four(region_winners_det, k=k, deterministic=True)
    for name, w in ff_wins.items():
        det_wins[name] += w
    df["pred_deterministic"] = df["team_name"].map(det_wins).fillna(0).astype(int)


    # ── Monte Carlo ──────────────────────────────────────────────────────────
    mc_wins  = {name: [] for name in df["team_name"]}
    round_probs = {name: {r: 0 for r in ["r32","s16","e8","f4","final","champion"]}
                   for name in df["team_name"]}
    round_thresholds = {"r32":1,"s16":2,"e8":3,"f4":4,"final":5,"champion":6}

    for _ in range(n_sims):
        sim_wins = {name: 0 for name in df["team_name"]}
        region_winners = []
        for region in regions:
            rdf = df[df[region_col] == region]
            matchups = build_region_bracket(rdf)
            region_wins, region_winner = simulate_region(matchups, k=k, deterministic=False)
            for name, w in region_wins.items():
                sim_wins[name] += w
            region_winners.append(region_winner)

        ff_wins = simulate_final_four(region_winners, k=k, deterministic=False)
        for name, w in ff_wins.items():
            sim_wins[name] += w

        for name, w in sim_wins.items():
            mc_wins[name].append(w)
            for round_name, threshold in round_thresholds.items():
                if w >= threshold:
                    round_probs[name][round_name] += 1

    df["pred_monte_carlo"] = df["team_name"].map(
    lambda n: round(np.mean(mc_wins[n]), 2))

    df["pred_mc_rounded"] = df["team_name"].map(
    lambda n: int(np.bincount(mc_wins[n]).argmax()))
    for round_name in ["r32","s16","e8","f4","final","champion"]:
        df[f"prob_{round_name}"] = df["team_name"].map(
            lambda n, r=round_name: round(round_probs[n][r] / n_sims, 3))
        
    return df

def resolve_mc_conflicts(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    For each R64 matchup, if both teams have pred_mc_rounded >= 1,
    keep the higher raw score team at their rounded value and
    set the loser to 0 (using deterministic bracket as tiebreaker).
    """
    from config import BRACKET_REGIONS
    df = df.copy()

    region_map = BRACKET_REGIONS.get(year, {})
    df["region"] = df["team_name"].map(region_map).fillna("Unknown")

    # Standard bracket seeding: 1v16, 2v15, 3v14, 4v13, 5v12, 6v11, 7v10, 8v9
    SEED_PAIRS = [(1,16),(2,15),(3,14),(4,13),(5,12),(6,11),(7,10),(8,9)]

    for region in ["South","East","West","Midwest"]:
        region_df = df[df["region"]==region]
        for s_high, s_low in SEED_PAIRS:
            high = region_df[region_df["seed"]==s_high]
            low  = region_df[region_df["seed"]==s_low]
            if len(high)==0 or len(low)==0:
                continue
            high_idx = high.index[0]
            low_idx  = low.index[0]
            h_pred = df.loc[high_idx, "pred_mc_rounded"]
            l_pred = df.loc[low_idx,  "pred_mc_rounded"]
            # Conflict: both predict at least 1 win
            if h_pred >= 1 and l_pred >= 1:
                # Higher raw score wins, loser gets 0
                if df.loc[high_idx,"raw_Ensemble"] >= df.loc[low_idx,"raw_Ensemble"]:
                    df.loc[low_idx, "pred_mc_rounded"] = 0
                else:
                    df.loc[high_idx, "pred_mc_rounded"] = 0
    return df

def tune_k(df: pd.DataFrame, years: list,
           raw_col: str = "barthag",
           k_bounds: tuple = (0.01, 5.0),
           noise: float = 0.0) -> float:
    from scipy.optimize import minimize_scalar

    # Historical R64 advancement rates by seed (2011-2024, excl 2020)
    TARGET_ADV_RATES = {
        1:0.964, 2:0.875, 3:0.875, 4:0.804,
        5:0.607, 6:0.482, 7:0.643, 8:0.518,
        9:0.482, 10:0.357, 11:0.518, 12:0.393,
        13:0.196, 14:0.125, 15:0.125, 16:0.036,
    }

    def bracket_mae(k):
        errors = []
        for year in years:
            year_df = df[df["year"] == year].copy()
            if year not in BRACKET_REGIONS:
                continue
            region_map = BRACKET_REGIONS[year]
            year_df["region"] = year_df["team_name"].map(region_map).fillna("Unknown")
            year_df = year_df[year_df["region"] != "Unknown"]
            if len(year_df) < 60:
                continue

            mu = year_df[raw_col].mean()
            sd = year_df[raw_col].std()
            if sd < 1e-6:
                continue
            year_df["raw_Ensemble"] = (year_df[raw_col] - mu) / sd

            sim_df = run_bracket_simulation(
                year_df, regions=["South","East","West","Midwest"],
                n_sims=300, k=k
            )

            for seed, target in TARGET_ADV_RATES.items():
                seed_rows = sim_df[sim_df["seed"] == seed]
                if len(seed_rows) == 0:
                    continue
                simulated = seed_rows["prob_r32"].mean()
                errors.append((simulated - target) ** 2)

        return np.mean(errors) if errors else 99.0

    log.info("Tuning K across historical years...")
    result = minimize_scalar(bracket_mae, bounds=k_bounds, method="bounded",
                             options={"xatol": 0.02})
    best_k = round(result.x, 3)
    log.info(f"Best K = {best_k} (upset rate MSE = {result.fun:.6f})")
    return best_k
# ── Model Definitions ──────────────────────────────────────────────────────────


def get_models() -> Dict:
    """
    Returns dict of model_name -> sklearn-compatible pipeline.
    All models use median imputation for missing values.
    Tree-based models don't need scaling; linear/MLP do.
    """
    imputer = SimpleImputer(strategy="median")

    models = {}

    # Linear baseline
    models["Linear"] = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
        ("model",  LinearRegression()),
    ])

    # Ridge (regularized linear)
    models["Ridge"] = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
        ("model",  Ridge(alpha=1.0)),
    ])

    # Random Forest
    models["RandomForest"] = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model",  RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    # XGBoost
    if HAS_XGB:
        models["XGBoost"] = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model",  xgb.XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
            )),
        ])

    # LightGBM
    if HAS_LGB:
        models["LightGBM"] = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model",  lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )),
        ])

    # Neural network
    models["MLP"] = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
        ("model",  MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )),
    ])

    return models


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred_raw: np.ndarray,
                    seeds: np.ndarray) -> Dict:
    """
    Compute all three evaluation metrics.

    y_pred_raw: continuous regression output, rounded to nearest int for
                classification-style metrics.
    """
    # Clip and round predictions to valid range 0-6
    y_pred = np.clip(np.round(y_pred_raw), 0, 6).astype(int)
    y_true_int = y_true.astype(int)

    # 1. MAE on continuous predictions
    mae = mean_absolute_error(y_true, y_pred_raw)

    # 2. Exact round accuracy
    exact_acc = (y_pred == y_true_int).mean()

    # 3. Upset detection F1
    # Actual upsets: high-seed teams that won games
    actual_upsets   = ((seeds >= UPSET_SEED_THRESHOLD) & (y_true_int >= 1)).astype(int)
    # Predicted upsets: high-seed teams predicted to win games
    predicted_upsets = ((seeds >= UPSET_SEED_THRESHOLD) & (y_pred >= 1)).astype(int)

    if actual_upsets.sum() == 0:
        upset_f1 = np.nan
    else:
        upset_f1 = f1_score(actual_upsets, predicted_upsets, zero_division=0)

    return {
        "mae":          round(mae, 4),
        "exact_acc":    round(exact_acc, 4),
        "upset_f1":     round(upset_f1, 4) if not np.isnan(upset_f1) else np.nan,
    }


# ── Walk-Forward Cross Validation ─────────────────────────────────────────────

def walk_forward_cv(df: pd.DataFrame, models: Dict) -> pd.DataFrame:
    """
    Walk-forward validation: for each year Y (starting from MIN_TRAIN_YEARS+1),
    train on all years < Y, test on year Y.

    Returns DataFrame with one row per (model, test_year) with all metrics.
    """
    years = sorted(df["year"].unique())
    feature_cols = get_feature_cols(df)

    results = []

    for i, test_year in enumerate(years):
        train_years = [y for y in years if y < test_year]
        if len(train_years) < MIN_TRAIN_YEARS:
            continue

        train = df[df["year"].isin(train_years)]
        test  = df[df["year"] == test_year]

        X_train = train[feature_cols].values
        y_train = train[TARGET].values
        X_test  = test[feature_cols].values
        y_test  = test[TARGET].values
        seeds   = test["seed"].values

        log.info(f"  Test year {test_year}: train={len(train)} | test={len(test)}")

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics = compute_metrics(y_test, y_pred, seeds)
                results.append({
                    "model":     name,
                    "test_year": test_year,
                    "n_train":   len(train),
                    **metrics,
                })
            except Exception as e:
                log.warning(f"    {name} failed for {test_year}: {e}")

    return pd.DataFrame(results)


def leave_one_year_out_cv(df: pd.DataFrame, models: Dict) -> pd.DataFrame:
    """
    Leave-one-year-out: for each year Y, train on all OTHER years, test on Y.
    More data per fold than walk-forward, but uses future data.
    """
    years = sorted(df["year"].unique())
    feature_cols = get_feature_cols(df)

    results = []

    for test_year in years:
        train = df[df["year"] != test_year]
        test  = df[df["year"] == test_year]

        X_train = train[feature_cols].values
        y_train = train[TARGET].values
        X_test  = test[feature_cols].values
        y_test  = test[TARGET].values
        seeds   = test["seed"].values

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics = compute_metrics(y_test, y_pred, seeds)
                results.append({
                    "model":     name,
                    "test_year": test_year,
                    **metrics,
                })
            except Exception as e:
                log.warning(f"    {name} failed for {test_year}: {e}")

    return pd.DataFrame(results)


# ── Results Summary ────────────────────────────────────────────────────────────

def summarize_results(results: pd.DataFrame, cv_name: str) -> pd.DataFrame:
    """Aggregate per-year results into mean metrics per model."""
    summary = (
        results.groupby("model")[["mae", "exact_acc", "upset_f1"]]
        .mean()
        .round(4)
        .sort_values("mae")
        .reset_index()
    )
    summary.columns = ["Model", "MAE (↓)", "Exact Acc (↑)", "Upset F1 (↑)"]

    log.info(f"\n{'═'*60}")
    log.info(f"{cv_name} Results (averaged across test years)")
    log.info(f"{'═'*60}")
    log.info("\n" + summary.to_string(index=False))
    log.info(f"{'═'*60}\n")

    return summary


# ── Feature Importance ─────────────────────────────────────────────────────────

def show_feature_importance(df: pd.DataFrame, models: Dict, top_n: int = 25):
    """Train on all data and show feature importance for tree-based models."""
    feature_cols = get_feature_cols(df)
    X = df[feature_cols].values
    y = df[TARGET].values

    tree_models = {k: v for k, v in models.items()
                   if k in ("RandomForest", "XGBoost", "LightGBM")}

    for name, model in tree_models.items():
        try:
            model.fit(X, y)
            # Get the actual model from the pipeline
            m = model.named_steps["model"]
            importances = m.feature_importances_

            # Map back to feature names (after imputation, cols are the same)
            imp_df = pd.DataFrame({
                "feature":    feature_cols,
                "importance": importances,
            }).sort_values("importance", ascending=False).head(top_n)

            log.info(f"\nTop {top_n} features — {name}:")
            log.info(imp_df.to_string(index=False))
        except Exception as e:
            log.warning(f"Feature importance failed for {name}: {e}")


# ── Ensemble ───────────────────────────────────────────────────────────────────

def build_ensemble(df: pd.DataFrame, models: Dict,
                   wf_results: pd.DataFrame) -> Dict:
    """
    Build a simple weighted ensemble based on walk-forward MAE.
    Lower MAE = higher weight.
    """
    # Get mean MAE per model
    mae_by_model = wf_results.groupby("model")["mae"].mean()

    # Weight = 1/MAE, normalized
    weights = (1.0 / mae_by_model)
    weights = weights / weights.sum()

    log.info("\nEnsemble weights (based on walk-forward MAE):")
    for model, w in weights.sort_values(ascending=False).items():
        log.info(f"  {model:15s}: {w:.3f}")

    return weights.to_dict()


def ensemble_predict(df_train: pd.DataFrame, df_test: pd.DataFrame,
                     models: Dict, weights: Dict) -> np.ndarray:
    """Generate weighted ensemble predictions."""
    feature_cols = get_feature_cols(df_train)
    X_train = df_train[feature_cols].values
    y_train = df_train[TARGET].values
    X_test  = df_test[feature_cols].values

    preds = np.zeros(len(X_test))
    total_w = 0.0

    for name, model in models.items():
        w = weights.get(name, 0)
        if w <= 0:
            continue
        try:
            model.fit(X_train, y_train)
            p = model.predict(X_test)
            preds += w * p
            total_w += w
        except Exception as e:
            log.warning(f"Ensemble: {name} failed: {e}")

    if total_w > 0:
        preds /= total_w

    return preds


# ── Predict Specific Year ──────────────────────────────────────────────────────

def predict_year(df: pd.DataFrame, models: Dict, year: int,
                 weights: Dict = None) -> pd.DataFrame:
    """
    Train on all years < target year, predict target year.
    Returns predictions alongside actual results for comparison.
    """
    train = df[df["year"] < year]
    test  = df[df["year"] == year]

    if train.empty:
        log.error(f"No training data available before {year}")
        return pd.DataFrame()
    if test.empty:
        log.error(f"No test data for {year}")
        return pd.DataFrame()

    feature_cols = get_feature_cols(df)
    X_train = train[feature_cols].values
    y_train = train[TARGET].values
    X_test  = test[feature_cols].values

    results = test[["team_name", "seed", "wins", "exit_round"]].copy()

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            results[f"pred_{name}"] = np.clip(np.round(pred), 0, 6).astype(int)
            results[f"raw_{name}"]  = pred.round(2)
        except Exception as e:
            log.warning(f"  {name} prediction failed: {e}")

    # Ensemble
    if weights:
        ens_pred = ensemble_predict(train, test, models, weights)
        results["pred_Ensemble"] = np.clip(np.round(ens_pred), 0, 6).astype(int)
        results["raw_Ensemble"]  = ens_pred.round(2)

    results = results.sort_values("seed").reset_index(drop=True)

    log.info(f"\nPredictions for {year}:")
    log.info(results.to_string(index=False))

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="March Madness prediction model")
    parser.add_argument("--year",               type=int, default=None,
                        help="Predict a specific year (train on all prior years)")
    parser.add_argument("--feature-importance", action="store_true",
                        help="Show feature importance for tree models")
    parser.add_argument("--loyo",               action="store_true",
                        help="Also run leave-one-year-out CV (slower)")
    args = parser.parse_args()

    # Load data
    df = load_features()
    df = add_upset_flag(df)

    log.info(f"Feature columns: {len(get_feature_cols(df))}")
    log.info(f"Years: {sorted(df['year'].unique())}")
    log.info(f"Target distribution:\n{df['wins'].value_counts().sort_index().to_string()}")

    # Build models
    models = get_models()
    log.info(f"\nModels to evaluate: {list(models.keys())}")

    # Walk-forward CV
    log.info("\n" + "═"*60)
    log.info("Walk-Forward Cross Validation")
    log.info("═"*60)
    wf_results = walk_forward_cv(df, models)
    wf_summary = summarize_results(wf_results, "Walk-Forward CV")

    # Save per-year results
    wf_results.to_csv(MODELS_DIR / "wf_cv_results.csv", index=False)
    wf_summary.to_csv(MODELS_DIR / "wf_cv_summary.csv", index=False)

    # Leave-one-year-out CV (optional)
    if args.loyo:
        log.info("\n" + "═"*60)
        log.info("Leave-One-Year-Out CV")
        log.info("═"*60)
        loyo_results = leave_one_year_out_cv(df, models)
        loyo_summary = summarize_results(loyo_results, "Leave-One-Year-Out CV")
        loyo_results.to_csv(MODELS_DIR / "loyo_cv_results.csv", index=False)
        loyo_summary.to_csv(MODELS_DIR / "loyo_cv_summary.csv", index=False)

    # Build ensemble weights
    ensemble_weights = build_ensemble(df, models, wf_results)

    # Feature importance
    if args.feature_importance:
        show_feature_importance(df, models)

    # Predict specific year
    if args.year:
        predictions = predict_year(df, models, args.year, ensemble_weights)
        predictions = apply_calibration_scaling(predictions, "raw_Ensemble", "pred_Ensemble_scaled")
        predictions = apply_rank_mapping(predictions, "raw_Ensemble", "pred_Ensemble_ranked")
        predictions = add_upset_rate_feature(predictions)

        if args.year in BRACKET_REGIONS:
            region_map = BRACKET_REGIONS[args.year]
            predictions["region"] = predictions["team_name"].map(region_map).fillna("Unknown")

            tune_years = [y for y in range(2011, args.year) if y != 2020 and y in BRACKET_REGIONS]
            if tune_years:
                hist_df = df[df["year"].isin(tune_years)].copy()
                best_k = tune_k(hist_df, tune_years, raw_col="barthag")
            else:
                best_k = 1.0

            # Normalize raw_Ensemble for the prediction year too
            mu = predictions["raw_Ensemble"].mean()
            sd = predictions["raw_Ensemble"].std()
            predictions["raw_Ensemble_norm"] = (predictions["raw_Ensemble"] - mu) / sd
            # Temporarily swap in normalized scores for simulation
            predictions["raw_Ensemble_orig"] = predictions["raw_Ensemble"]
            predictions["raw_Ensemble"] = predictions["raw_Ensemble_norm"]

            predictions = run_bracket_simulation(
                predictions, regions=["South","East","West","Midwest"],
                n_sims=10000, k=best_k
            )
            predictions = resolve_mc_conflicts(predictions, args.year)

            # Restore original raw scores
            predictions["raw_Ensemble"] = predictions["raw_Ensemble_orig"]
            predictions.drop(columns=["raw_Ensemble_norm","raw_Ensemble_orig"], inplace=True)
            predictions["tuned_k"] = best_k
        else:
            log.warning(f"No bracket regions for {args.year}, skipping simulation")

        predictions.to_csv(MODELS_DIR / f"predictions_{args.year}.csv", index=False)
        log.info(f"\nSaved predictions to models/predictions_{args.year}.csv")

    log.info("\nDone. Results saved to models/")