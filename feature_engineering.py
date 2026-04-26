"""
feature_engineering.py
───────────────────────
Builds the feature matrix for the March Madness prediction model.

Feature groups:
  1. Baseline efficiency      — AdjEM, barthag, WAB, seed, four factors
  2. Team variance            — game-to-game CV for offense, EFG, turnovers
  3. Opponent-adjusted dropoff — how stats change vs tier 1 vs tier 3 opponents
  4. Star player features     — top scorer concentration, usage, efficiency
  5. Location splits          — home/away/neutral performance gaps

Output:
  data/processed/features_{year}.csv   — one file per year (tournament teams only)
  data/processed/features_all.csv      — all years combined, ready for modeling

Usage:
  python feature_engineering.py
  python feature_engineering.py --year 2024   # single year
  python feature_engineering.py --refresh     # recompute all
"""

import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    DATA_TORVIK,
    DATA_TOURNAMENT,
    DATA_PROCESSED,
    DATA_SPORTSREF,
    ALL_YEARS,
    TIER_1_MAX_RANK,
    TIER_2_MAX_RANK,
    TOP_N_SCORERS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Data Loaders ───────────────────────────────────────────────────────────────

def load_tournament_teams(year: int) -> pd.DataFrame:
    path = DATA_TOURNAMENT / f"tournament_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Tournament data missing for {year}: {path}")
    df = pd.read_csv(path)
    # Normalize team abbreviations to lowercase for joining
    df["team_abbrev"] = df["team_abbrev"].str.lower().str.strip()
    return df


def load_four_factors(year: int) -> pd.DataFrame:
    path = DATA_TORVIK / f"four_factors_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Four factors missing for {year}")
    df = pd.read_csv(path)
    # If first column isn't 'team', the file was saved without headers
    # Fall back to the combined all_four_factors.csv filtered by year
    if "team" not in df.columns:
        all_path = DATA_TORVIK / "all_four_factors.csv"
        if all_path.exists():
            df = pd.read_csv(all_path)
            df = df[df["year"] == year].reset_index(drop=True)
        else:
            raise FileNotFoundError(f"Four factors missing headers for {year} and no combined file found")
    return df


def load_team_results(year: int) -> pd.DataFrame:
    path = DATA_TORVIK / f"team_results_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Team results missing for {year}")
    return pd.read_csv(path)


def load_gamelogs(year: int) -> pd.DataFrame:
    path = DATA_TORVIK / f"gamelogs_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Game logs missing for {year}")
    df = pd.read_csv(path)
    df["team"] = df["team"].str.strip()
    df["opponent"] = df["opponent"].str.strip()
    return df


def load_timemachine(year: int) -> pd.DataFrame:
    path = DATA_TORVIK / f"timemachine_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Timemachine missing for {year}")
    return pd.read_csv(path)


def load_player_stats(year: int) -> pd.DataFrame:
    path = DATA_TORVIK / f"player_stats_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Player stats missing for {year}")
    return pd.read_csv(path)


# ── Helpers ────────────────────────────────────────────────────────────────────

def cv(series: pd.Series) -> float:
    """Coefficient of variation (std / mean). Returns NaN if mean is 0."""
    mean = series.mean()
    if pd.isna(mean) or mean == 0:
        return np.nan
    return series.std() / mean


def _regular_season_only(gamelogs: pd.DataFrame) -> pd.DataFrame:
    """Keep only regular season games (game_type 0 = non-conf, 1 = conf)."""
    if "game_type" in gamelogs.columns:
        return gamelogs[gamelogs["game_type"].isin([0, 1])].copy()
    return gamelogs.copy()


def _assign_tier(rank) -> int:
    try:
        rank = int(rank)
        if rank <= TIER_1_MAX_RANK:
            return 1
        elif rank <= TIER_2_MAX_RANK:
            return 2
        else:
            return 3
    except (ValueError, TypeError):
        return 3


# ── Feature Group 1: Baseline Efficiency ──────────────────────────────────────

def build_baseline_features(four_factors: pd.DataFrame, seed: int, team_name: str) -> dict:
    """
    Season-level efficiency metrics from Torvik four factors.
    These are the strongest predictors and serve as controls.
    """
    # Match team name — Torvik uses full names, tournament uses abbrevs
    # We match on the four_factors 'team' column
    row = _match_team(four_factors, team_name)
    if row is None:
        log.warning(f"    No four factors match for '{team_name}'")
        return {}

    features = {
        "seed":        seed,
        "adj_o":       row.get("adj_o"),
        "adj_d":       row.get("adj_d"),
        "adj_em":      row.get("adj_em"),
        "barthag":     row.get("barthag"),
        "wab":         row.get("wab"),
        "off_efg":     row.get("off_efg"),
        "def_efg":     row.get("def_efg"),
        "off_to":      row.get("off_to"),
        "def_to":      row.get("def_to"),
        "off_or":      row.get("off_or"),
        "def_or":      row.get("def_or"),
        "off_ftr":     row.get("off_ftr"),
        "def_ftr":     row.get("def_ftr"),
        "fg3_pct":     row.get("fg3_pct"),
        "fg3_d_pct":   row.get("fg3_d_pct"),
        "fg2_pct":     row.get("fg2_pct"),
        "fg2_d_pct":   row.get("fg2_d_pct"),
        "adj_t":       row.get("adj_t"),
        "ft_pct":      row.get("ft_pct"),
    }
    return {k: v for k, v in features.items() if v is not None}


# ── Feature Group 2: Team Variance ────────────────────────────────────────────

# ── Feature Group 2: Team Variance ────────────────────────────────────────────

def build_variance_features(team_games: pd.DataFrame) -> dict:
    """
    Game-to-game consistency metrics from Torvik game logs.
    High variance = risky tournament team.
    Uses regular season games only.
    """
    rs = _regular_season_only(team_games)
    if len(rs) < 5:
        return {}

    features = {}

    # Offensive efficiency variance
    if "team_raw_o" in rs.columns:
        features["off_cv"]           = cv(rs["team_raw_o"].dropna())
        #features["off_std"]          = rs["team_raw_o"].std()
        features["off_mean"]         = rs["team_raw_o"].mean()
        features["off_20th_pct"]     = rs["team_raw_o"].quantile(0.20)
        features["off_80th_pct"]     = rs["team_raw_o"].quantile(0.80)
        features["off_range_80_20"]  = features["off_80th_pct"] - features["off_20th_pct"]
        features["pct_games_above_mean_o"] = (rs["team_raw_o"] > features["off_mean"]).mean()

    # Defensive efficiency variance
    if "team_adj_d" in rs.columns:
        features["def_cv"]  = cv(rs["team_adj_d"].dropna())
        #features["def_std"] = rs["team_adj_d"].std()

    # EFG% variance
    if "team_efg" in rs.columns:
        features["efg_cv"]  = cv(rs["team_efg"].dropna())
        #features["efg_std"] = rs["team_efg"].std()

    # Turnover rate variance
    if "team_to" in rs.columns:
        features["to_cv"]  = cv(rs["team_to"].dropna())
        features["to_std"] = rs["team_to"].std()

    # Offensive rebound rate variance
    #if "team_or" in rs.columns:
        #features["or_cv"] = cv(rs["team_or"].dropna())

    # Tempo variance — do they get forced out of their pace?
    #if "tempo" in rs.columns:
        #features["tempo_cv"]  = cv(rs["tempo"].dropna())
        #features["tempo_std"] = rs["tempo"].std()

    # Margin variance — close game rate
    if "margin" in rs.columns:
        features["margin_std"]        = rs["margin"].std()
        features["margin_mean"]       = rs["margin"].mean()
        features["pct_games_close"]   = (rs["margin"].abs() <= 5).mean()
        features["pct_games_blown"]   = (rs["margin"] < -10).mean()

    # Win/loss offensive output gap
    if "team_raw_o" in rs.columns and "result" in rs.columns:
        wins   = rs[rs["result"].str.startswith("W", na=False)]["team_raw_o"]
        losses = rs[rs["result"].str.startswith("L", na=False)]["team_raw_o"]
        if len(wins) > 0 and len(losses) > 0:
            features["off_win_loss_gap"] = wins.mean() - losses.mean()

    # Rolling late-season consistency: std of last 10 regular season games
    if "team_raw_o" in rs.columns and "date" in rs.columns:
        rs_dated = rs.dropna(subset=["date"]).copy()
        try:
            rs_dated["date"] = pd.to_datetime(rs_dated["date"], format="mixed")
            rs_dated = rs_dated.sort_values("date")
            last10 = rs_dated.tail(10)["team_raw_o"].dropna()
            if len(last10) >= 5:
                features["off_last10_std"]  = last10.std()
                features["off_last10_mean"] = last10.mean()
        except Exception:
            pass

    return features

# ── Feature Group 2a: Team Box Score Variance (Sports Reference) ──────────────

def build_team_boxscore_variance_features(sref_abbrev: str, year: int) -> dict:
    """
    Game-to-game variance in box score stats from Sports Reference team gamelogs.
    Captures what Torvik gamelogs can't: shooting splits, blocks, steals,
    fouls, rebounds, assists.
    """
    year_dir = DATA_SPORTSREF / "team_gamelogs" / str(year)
    if not year_dir.exists():
        return {}

    f = year_dir / f"{sref_abbrev}_{year}.csv"
    if not f.exists():
        return {}

    df = pd.read_csv(f)
    if df.empty:
        return {}

    # Regular season only
    if "is_regular_season" in df.columns:
        df = df[df["is_regular_season"] == 1]
    if len(df) < 5:
        return {}

    features = {}

    stat_map = {
        # Offense
        "fg2_pct":     "team_fg2_pct_cv",      # 2-point shooting consistency
        "fg3_pct":     "team_fg3_pct_cv",       # 3-point shooting consistency
        "ft_pct":      "team_ft_pct_cv",        # Free throw consistency
        "pts":         "team_pts_cv",           # PPG consistency
        "orb":         "team_orb_cv",           # Offensive rebound consistency
        "drb":         "team_drb_cv",           # Defensive rebound consistency
        "trb":         "team_trb_cv",           # Total rebound consistency
        "ast":         "team_ast_cv",           # Assist consistency
        "stl":         "team_stl_cv",           # Steal consistency
        "blk":         "team_blk_cv",           # Block consistency
        "tov":         "team_tov_cv",           # Turnover consistency
        "pf":          "team_pf_cv",            # Foul consistency
        # Defense (opponent stats)
        "pts_opp":     "team_pts_allowed_cv",   # PPG allowed consistency
        "opp_fg3_pct": "team_opp_fg3_pct_cv",  # Opponent 3pt% allowed consistency
        "opp_ft_pct":  "team_opp_ft_pct_cv",   # Opponent FT% allowed consistency
        "opp_orb":     "team_opp_orb_cv",       # Opponent offensive rebound consistency
        "opp_tov":     "team_opp_tov_cv",       # Turnovers forced consistency
        "opp_blk":     "team_opp_blk_cv",       # Blocks against consistency
    }

    for col, feat_name in stat_map.items():
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) >= 5 and vals.mean() != 0:
            features[feat_name] = vals.std() / vals.mean()
        else:
            features[feat_name] = np.nan

    # Assist-to-turnover ratio consistency
    if "ast" in df.columns and "tov" in df.columns:
        ast = pd.to_numeric(df["ast"], errors="coerce")
        tov = pd.to_numeric(df["tov"], errors="coerce")
        ratio = ast / tov.replace(0, np.nan)
        ratio = ratio.dropna()
        if len(ratio) >= 5 and ratio.mean() != 0:
            features["team_ast_to_ratio_cv"] = ratio.std() / ratio.mean()

    # Effective FG% allowed consistency (defensive shooting variance)
    if "opp_efg_pct" in df.columns:
        vals = pd.to_numeric(df["opp_efg_pct"], errors="coerce").dropna()
        if len(vals) >= 5 and vals.mean() != 0:
            features["team_opp_efg_cv"] = vals.std() / vals.mean()

    return features

# ── Feature Group 2b: Player Variance ─────────────────────────────────────────

def build_player_variance_features(sref_abbrev: str, year: int) -> dict:
    """
    Per-player game-to-game variance for the top 8 scorers.

    For each stat (pts, fg_pct, fg3_pct, ft_pct, orb, blk, stl):
      - Mean and CV across all top-8 players (team-level summary)
      - Star player (top scorer) CV specifically
      - Spread between star and rest of top 8

    High per-player variance = star is streaky = risky in tournament.
    """
    year_dir = DATA_SPORTSREF / "player_gamelogs" / str(year)
    if not year_dir.exists():
        return {}

    # Load all cached player gamelogs for this team/year
    player_files = list(year_dir.glob(f"*_{year}.csv"))
    if not player_files:
        return {}

    # Load player ID list to know which players belong to this team
    player_id_file = DATA_SPORTSREF / "player_ids" / str(year) / f"{sref_abbrev}_{year}_players.csv"
    if not player_id_file.exists():
        return {}

    player_ids_df = pd.read_csv(player_id_file)
    if player_ids_df.empty:
        return {}

    top_players = player_ids_df.dropna(subset=["ppg"]).head(TOP_N_SCORERS)
    if top_players.empty:
        return {}

    # Load each player's gamelog
    player_frames = []
    for _, player in top_players.iterrows():
        pid = player["player_id"]
        f = year_dir / f"{pid}_{year}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if df.empty:
            continue
        # Regular season only
        if "is_regular_season" in df.columns:
            df = df[df["is_regular_season"] == 1]
        df["player_rank"] = _ + 1  # 1 = top scorer
        df["ppg_season"]  = player["ppg"]
        player_frames.append(df)

    if not player_frames:
        return {}

    features = {}

    # Stats to compute variance for
    stat_cols = {
        "pts":     "pts",
        "fg_pct":  "fg_pct",
        "fg3_pct": "fg3_pct",
        "ft_pct":  "ft_pct",
        "orb":     "orb",
        "blk":     "blk",
        "stl":     "stl",
    }

    star_df = player_frames[0] if player_frames else None

    # Per-stat variance across all top-8 players combined
    all_players_df = pd.concat(player_frames, ignore_index=True)

    for col, label in stat_cols.items():
        if col not in all_players_df.columns:
            continue

        # Star player variance
        if star_df is not None and col in star_df.columns:
            star_vals = pd.to_numeric(star_df[col], errors="coerce").dropna()
            if len(star_vals) >= 5:
                features[f"star_{label}_cv"]   = cv(star_vals)
                features[f"star_{label}_std"]  = star_vals.std()
                features[f"star_{label}_mean"] = star_vals.mean()
                # Floor: worst 20% game for star
                features[f"star_{label}_20th"] = star_vals.quantile(0.20)

        # Team-wide variance: average CV across all top-8 players
        player_cvs = []
        for pf in player_frames:
            if col not in pf.columns:
                continue
            vals = pd.to_numeric(pf[col], errors="coerce").dropna()
            if len(vals) >= 5:
                player_cvs.append(cv(vals))
        if player_cvs:
            features[f"top8_{label}_cv_mean"] = np.nanmean(player_cvs)
            features[f"top8_{label}_cv_max"]  = np.nanmax(player_cvs)

    # Scoring consistency across top-8: how much does each player's output vary?
    # High = unreliable contributors
    pts_cvs = []
    pts_means = []
    for pf in player_frames:
        if "pts" not in pf.columns:
            continue
        vals = pd.to_numeric(pf["pts"], errors="coerce").dropna()
        if len(vals) >= 5:
            pts_cvs.append(cv(vals))
            pts_means.append(vals.mean())

    if pts_cvs:
        features["top8_pts_cv_mean"]   = np.nanmean(pts_cvs)
        features["top8_pts_cv_std"]    = np.nanstd(pts_cvs)  # spread of consistency across roster
        features["top8_pts_mean_std"]  = np.nanstd(pts_means)  # spread of scoring load

    # Star reliability: fraction of games star scored within 5pts of their average
    if star_df is not None and "pts" in star_df.columns:
        star_pts = pd.to_numeric(star_df["pts"], errors="coerce").dropna()
        if len(star_pts) >= 5:
            star_mean = star_pts.mean()
            features["star_pts_reliable_pct"] = (
                (star_pts - star_mean).abs() <= 5
            ).mean()
            features["star_pts_bad_game_pct"] = (
                star_pts < (star_mean * 0.6)
            ).mean()

    return features

# ── Feature Group 3: Opponent-Adjusted Dropoff ────────────────────────────────

def build_dropoff_features(team_games: pd.DataFrame, timemachine: pd.DataFrame) -> dict:
    """
    Core hypothesis features: how does team performance change vs strong opponents?

    Methodology:
      1. Join each game's opponent to their timemachine T-Rank
      2. Assign tier (1=top50, 2=51-100, 3=100+)
      3. Compute mean offensive stats per tier
      4. Dropoff = tier3_mean - tier1_mean (positive = worse vs good teams)
    """
    rs = _regular_season_only(team_games)
    if len(rs) < 5:
        return {}

    # Build opponent -> tier mapping from timemachine
    if timemachine.empty:
        return {}

    # Timemachine has team names and torvik_rank
    # Normalize for joining
    tm = timemachine.copy()
    name_col = _find_name_col(tm)
    if name_col is None:
        return {}

    tm["team_norm"] = tm[name_col].astype(str).str.lower().str.strip()
    tm["tier"] = tm["torvik_rank"].apply(_assign_tier)

    rs = rs.copy()
    rs["opp_norm"] = rs["opponent"].str.lower().str.strip()
    rs = rs.merge(tm[["team_norm", "tier"]], left_on="opp_norm", right_on="team_norm", how="left")

    # Fill unmatched opponents as tier 3
    rs["tier"] = rs["tier"].fillna(3).astype(int)

    features = {}

    stat_cols = {
        "team_raw_o": "off_rating",
        "team_efg":   "efg",
        "team_to":    "to",
        "team_or":    "or",
    }

    tier_means = {}
    for col, label in stat_cols.items():
        if col not in rs.columns:
            continue
        for tier in [1, 2, 3]:
            tier_games = rs[rs["tier"] == tier][col].dropna()
            if len(tier_games) >= 3:
                tier_means[f"{label}_tier{tier}"] = tier_games.mean()
                features[f"{label}_vs_tier{tier}_mean"] = tier_games.mean()
                features[f"{label}_vs_tier{tier}_n"] = len(tier_games)

    # Dropoff: how much worse vs tier 1 than tier 3
    for col, label in stat_cols.items():
        t1_key = f"{label}_tier1"
        t3_key = f"{label}_tier3"
        if t1_key in tier_means and t3_key in tier_means:
            # Positive = better vs weak, worse vs strong (bad sign for tourney)
            features[f"{label}_dropoff_t1_vs_t3"] = tier_means[t3_key] - tier_means[t1_key]

    # Tier 1 game count (exposure to good opponents)
    features["n_tier1_games"] = int((rs["tier"] == 1).sum())
    features["n_tier2_games"] = int((rs["tier"] == 2).sum())
    features["n_tier3_games"] = int((rs["tier"] == 3).sum())

    return features


# ── Feature Group 4: Star Player Features ─────────────────────────────────────

def build_star_features(player_stats: pd.DataFrame, team_name: str) -> dict:
    """
    Top scorer concentration and efficiency.
    High concentration = single point of failure in tournament.
    """

    if "team" not in player_stats.columns:
        return {}

    # Match players to this team
    team_players = player_stats[
        player_stats["team"].str.strip().str.lower() == team_name.lower()
    ].copy()

    if team_players.empty:
        return {}

    for col in ["ppg", "usg", "efg_pct", "ts_pct", "min_pct"]:
        if col in team_players.columns:
            team_players[col] = pd.to_numeric(team_players[col], errors="coerce")

    if "ppg" not in team_players.columns:
        return {}

    team_players = team_players.dropna(subset=["ppg"]).sort_values("ppg", ascending=False)
    if team_players.empty:
        return {}

    total_ppg = team_players["ppg"].sum()
    if total_ppg == 0:
        return {}

    top_n = team_players.head(TOP_N_SCORERS)
    top1 = team_players.iloc[0]

    features = {}

    # Star concentration: top scorer share of team points
    features["star_ppg_share"] = top1["ppg"] / total_ppg if total_ppg > 0 else np.nan

    # Top N scorers' combined share
    features["top3_ppg_share"] = team_players.head(3)["ppg"].sum() / total_ppg

    # Star player efficiency
    if "efg_pct" in team_players.columns:
        features["star_efg"] = top1.get("efg_pct", np.nan)

    if "ts_pct" in team_players.columns:
        features["star_ts"] = top1.get("ts_pct", np.nan)

    # Star usage rate
    if "usg" in team_players.columns:
        features["star_usg"] = top1.get("usg", np.nan)

    # Scoring balance: Herfindahl index across top N scorers
    # Lower = more balanced (more resilient), higher = more star-dependent
    top_n_shares = top_n["ppg"] / total_ppg
    features["scoring_herfindahl"] = (top_n_shares ** 2).sum()

    # Depth: how many players average 8+ ppg
    features["scorers_8ppg"] = int((team_players["ppg"] >= 8).sum())
    features["scorers_5ppg"] = int((team_players["ppg"] >= 5).sum())

    return features


# ── Feature Group 5: Location Splits ──────────────────────────────────────────

def build_location_features(team_games: pd.DataFrame) -> dict:
    """
    Home/away/neutral performance gaps.
    Tournament games are neutral — teams that struggle away from home are risky.
    """
    rs = _regular_season_only(team_games)
    if len(rs) < 5 or "location" not in rs.columns:
        return {}

    features = {}

    if "team_raw_o" in rs.columns:
        home    = rs[rs["location"] == "H"]
        away    = rs[rs["location"] == "A"]
        neutral = rs[rs["location"] == "N"]


        if len(home) >= 3:
            features["off_home_mean"]    = home["team_raw_o"].mean()
        if len(away) >= 3:
            features["off_away_mean"]    = away["team_raw_o"].mean()
        if len(neutral) >= 2:
            features["off_neutral_mean"] = neutral["team_raw_o"].mean()
        if len(home) >= 3 and len(away) >= 3:
            features["home_away_gap"] = home["team_raw_o"].mean() - away["team_raw_o"].mean()
        if len(home) >= 3 and len(neutral) >= 2:
            features["home_neutral_gap"] = home["team_raw_o"].mean() - neutral["team_raw_o"].mean()

    # Win rate by location
    if "result" in rs.columns:
        rs = rs.copy()
        rs["win"] = rs["result"].str.startswith("W", na=False).astype(int)
        for loc, label in [("H", "home"), ("A", "away"), ("N", "neutral")]:
            loc_games = rs[rs["location"] == loc]
            if len(loc_games) >= 3:
                features[f"win_pct_{label}"] = loc_games["win"].mean()

    return features


# ── Team Name Matching ─────────────────────────────────────────────────────────

TEAM_NAME_MAP = {
    # UConn
    "uconn":                        "connecticut",
    # State -> St. (Sports Reference uses full "State", Torvik uses "St.")
    "arizona state":                "arizona st.",
    "boise state":                  "boise st.",
    "cal state bakersfield":        "cal st. bakersfield",
    "cal state fullerton":          "cal st. fullerton",
    "cleveland state":              "cleveland st.",
    "colorado state":               "colorado st.",
    "florida state":                "florida st.",
    "fresno state":                 "fresno st.",
    "georgia state":                "georgia st.",
    "indiana state":                "indiana st.",
    "iowa state":                   "iowa st.",
    "jacksonville state":           "jacksonville st.",
    "kansas state":                 "kansas st.",
    "kennesaw state":               "kennesaw st.",
    "kent state":                   "kent st.",
    "long beach state":             "long beach st.",
    "mcneese state":                "mcneese st.",
    "michigan state":               "michigan st.",
    "mississippi state":            "mississippi st.",
    "montana state":                "montana st.",
    "morehead state":               "morehead st.",
    "murray state":                 "murray st.",
    "new mexico state":             "new mexico st.",
    "norfolk state":                "norfolk st.",
    "north dakota state":           "north dakota st.",
    "northwestern state":           "northwestern st.",
    "ohio state":                   "ohio st.",
    "oklahoma state":               "oklahoma st.",
    "oregon state":                 "oregon st.",
    "penn state":                   "penn st.",
    "san diego state":              "san diego st.",
    "south dakota state":           "south dakota st.",
    "utah state":                   "utah st.",
    "washington state":             "washington st.",
    "weber state":                  "weber st.",
    "wichita state":                "wichita st.",
    "wright state":                 "wright st.",
    # Parenthetical disambiguators
    "albany (ny)":                  "albany",
    "loyola (il)":                  "loyola chicago",
    "loyola (md)":                  "loyola md",
    "st. john's (ny)":              "st. john's",
    "miami (fl)":                   "miami fl",
    "miami (ohio)":                 "miami oh",
    "texas a&m-corpus christi":     "texas a&m corpus chris",
    # Hyphen/punctuation variants
    "gardner-webb":                 "gardner webb",
    "uc-davis":                     "uc davis",
    "uc-irvine":                    "uc irvine",
    "ucsb":                         "uc santa barbara",
    # Other mismatches
    "college of charleston":        "charleston",
    "nc state":                     "n.c. state",
    "n.c. state":                   "n.c. state",
    "ole miss":                     "mississippi",
    "unc":                          "north carolina",
    "pitt":                         "pittsburgh",
    "st. joseph's":                 "saint joseph's",
    "st. peter's":                  "saint peter's",
    "st. mary's":                   "saint mary's",
    "fdu":                          "fairleigh dickinson",
    "etsu":                         "east tennessee st.",
    "vcu":                          "vcu",
    "tcu":                          "tcu",
    "byu":                          "byu",
    "lsu":                          "lsu",
    "liu":                          "liu",
}

def _normalize_team_name(name: str) -> str:
    n = name.lower().strip()
    return TEAM_NAME_MAP.get(n, n)

def _find_name_col(df: pd.DataFrame) -> str:
    """Find the team name column in a dataframe."""
    for col in ["team", "Team", "team_name", "name"]:
        if col in df.columns:
            return col
    return None


def _match_team(df: pd.DataFrame, team_name: str):
    """
    Fuzzy match a team name against a dataframe's team column.
    Returns the matched row as a dict, or None.
    """
    name_col = _find_name_col(df)
    if name_col is None:
        return None

    df = df.copy()
    df["_norm"] = df[name_col].astype(str).str.lower().str.strip().apply(_normalize_team_name)
    target = _normalize_team_name(team_name)


    # Exact match first
    exact = df[df["_norm"] == target]
    if not exact.empty:
        return exact.iloc[0].to_dict()

    # Partial match — target contains or is contained by
    partial = df[df["_norm"].str.contains(target, regex=False) |
                 pd.Series([target] * len(df)).str.contains(df["_norm"], regex=False).values]
    if not partial.empty:
        return partial.iloc[0].to_dict()

    return None


def _match_team_gamelogs(gamelogs: pd.DataFrame, team_name: str) -> pd.DataFrame:
    """Filter gamelogs to a specific team with fuzzy name matching."""
    target = _normalize_team_name(team_name)
    gamelogs = gamelogs.copy()
    gamelogs["_norm"] = gamelogs["team"].str.lower().str.strip().apply(_normalize_team_name)

    exact = gamelogs[gamelogs["_norm"] == target]
    if not exact.empty:
        return exact

    # Try partial
    partial = gamelogs[gamelogs["_norm"].str.contains(target, regex=False)]
    if not partial.empty:
        return partial

    return pd.DataFrame()


# ── Main Feature Builder ───────────────────────────────────────────────────────

def build_features_for_year(year: int, force_refresh: bool = False) -> pd.DataFrame:
    out_path = DATA_PROCESSED / f"features_{year}.csv"
    if out_path.exists() and not force_refresh:
        log.info(f"  Cache hit: features_{year}.csv")
        return pd.read_csv(out_path)

    log.info(f"Building features for {year}...")

    # Load all data sources
    try:
        tournament  = load_tournament_teams(year)
        four_factors = load_four_factors(year)
        gamelogs    = load_gamelogs(year)
        timemachine = load_timemachine(year)
        player_stats = load_player_stats(year)
    except FileNotFoundError as e:
        log.error(f"  Missing data: {e}")
        return pd.DataFrame()

    rows = []
    for _, team_row in tournament.iterrows():
        team_abbrev = team_row["team_abbrev"]
        team_name   = team_row["team_name"]
        seed        = team_row.get("seed")
        wins        = team_row["wins"]
        exit_round  = team_row["exit_round"]

        log.info(f"  {team_name} (seed {seed}, {wins} wins)")

        # Get this team's game logs
        team_games = _match_team_gamelogs(gamelogs, team_name)
        if team_games.empty:
            log.warning(f"    No game logs found for '{team_name}'")

        # Build each feature group
        features = {
            "team_abbrev": team_abbrev,
            "team_name":   team_name,
            "year":        year,
            "seed":        seed,
            "wins":        wins,
            "exit_round":  exit_round,
        }

        features.update(build_baseline_features(four_factors, seed, team_name))

        if not team_games.empty:
            features.update(build_variance_features(team_games))
            features.update(build_dropoff_features(team_games, timemachine))
            features.update(build_location_features(team_games))

        features.update(build_star_features(player_stats, team_name))
        sref_abbrev = team_row["team_abbrev"]  # already the sref slug
        features.update(build_player_variance_features(sref_abbrev, year))
        features.update(build_team_boxscore_variance_features(sref_abbrev, year))

        DROP_FEATURES = {
        # Redundant std versions (keeping CV)
        "off_std", "def_std", "efg_std", "star_pts_std", "star_fg_pct_std",
        # Percentiles (keeping range only)
        "off_20th_pct", "off_80th_pct",
        # Tempo
        "tempo_cv", "tempo_std",
        # Low signal
        "or_cv", "adj_t", "fg2_pct", "fg2_d_pct",
        # Raw home/away (gap captures what matters)
        "off_home_mean", "off_away_mean",
        # Win% confounded by schedule
        "win_pct_home", "win_pct_away",
        # Sample size counters
        "off_rating_vs_tier1_n", "off_rating_vs_tier2_n", "off_rating_vs_tier3_n",
        "efg_vs_tier1_n", "efg_vs_tier2_n", "efg_vs_tier3_n",
        "to_vs_tier1_n", "to_vs_tier2_n",
        "or_vs_tier1_n", "or_vs_tier2_n",
        "or_vs_tier3_n", "to_vs_tier3_n",
        "n_tier2_games", "n_tier3_games",
        # Star player drops
        "scorers_5ppg", "star_efg", "star_ts",
        "star_fg_pct_20th", "star_fg3_pct_mean", "star_fg3_pct_cv",
        "star_ft_pct_cv", "star_ft_pct_mean", "star_ft_pct_20th",
        "star_orb_cv", "star_orb_mean",
        "star_stl_cv", "star_stl_mean", "star_blk_cv",
        "star_blk_20th", "star_blk_mean", "star_blk_std",
        "star_orb_20th", "star_orb_std",
        "star_stl_20th", "star_stl_std",
        "star_fg3_pct_20th", "star_fg3_pct_std",
        "star_ft_pct_std", "star_pts_bad_game_pct", "star_pts_20th",
        # Top-8 low signal
        "top8_ft_pct_cv_max", "top8_blk_mean_std", "top8_stl_mean_std",
        "is_upset_team", "is_upset_team",
        }

        features = {k: v for k, v in features.items() if k not in DROP_FEATURES}
        
        rows.append(features)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    log.info(f"  Saved {len(df)} teams to features_{year}.csv ({len(df.columns)} features)")
    return df


def build_all_features(years: list = ALL_YEARS, force_refresh: bool = False) -> pd.DataFrame:
    log.info("═" * 50)
    log.info("Building feature matrix for all years")
    log.info("═" * 50)

    frames = []
    for year in years:
        df = build_features_for_year(year, force_refresh)
        if not df.empty:
            frames.append(df)

    if not frames:
        log.error("No features built.")
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    out = DATA_PROCESSED / "features_all.csv"
    result.to_csv(out, index=False)

    log.info("═" * 50)
    log.info(f"Feature matrix complete: {result.shape}")
    log.info(f"  Rows: {len(result)} team-seasons")
    log.info(f"  Features: {len(result.columns) - 6} predictors")  # minus metadata cols
    log.info(f"  Target: 'wins' (0-6 scale)")
    log.info(f"  Saved: {out}")
    log.info("═" * 50)

    # Print feature group summary
    _summarize_features(result)

    return result


def _summarize_features(df: pd.DataFrame) -> None:
    """Print coverage stats for each feature group."""
    meta_cols = {"team_abbrev", "team_name", "year", "seed", "wins", "exit_round", "is_upset_team"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    baseline  = [c for c in feature_cols if c in {"adj_o","adj_d","adj_em","barthag","wab",
                  "off_efg","def_efg","off_to","def_to","off_or","def_or",
                  "off_ftr","def_ftr","fg3_pct","fg3_d_pct","fg2_pct","fg2_d_pct","adj_t","ft_pct"}]
    variance  = [c for c in feature_cols if any(x in c for x in ["_cv","_std","_mean","_pct","_gap","_range","20th","80th"])]
    dropoff   = [c for c in feature_cols if "tier" in c or "dropoff" in c or c.startswith("n_tier")]
    star      = [c for c in feature_cols if "star" in c or "herfindahl" in c or "scorer" in c or c.startswith("top")]
    location  = [c for c in feature_cols if any(x in c for x in ["home","away","neutral","win_pct"])]

    log.info("Feature group coverage (% non-null):")
    for label, cols in [("Baseline", baseline), ("Variance", variance),
                         ("Dropoff",  dropoff),  ("Star",     star),
                         ("Location", location)]:
        if cols:
            coverage = df[cols].notna().mean().mean() * 100
            log.info(f"  {label:10s}: {len(cols):2d} features | {coverage:.0f}% coverage")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build March Madness feature matrix")
    parser.add_argument("--year",    type=int, default=None)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    if args.year:
        df = build_features_for_year(args.year, args.refresh)
        print(df.to_string())
    else:
        build_all_features(force_refresh=args.refresh)