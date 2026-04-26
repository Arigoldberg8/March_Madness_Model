"""
torvik_collector.py
───────────────────
Pulls all Bart Torvik data needed for the pipeline:

  1. Team season results  — YEAR_team_results.csv
     Has headers, confirmed columns. Contains AdjO, AdjD, tempo, barthag, WAB.
     Pre-2020: 44 cols with "Fun Rk, adjt" merged last col (parsing quirk handled).
     2020+:    45 cols, clean.

  2. Team four factors    — trank.php export
     Provides labeled CSV: off_efg, off_to, off_or, off_ftr + defensive equivalents.
     We use this instead of teamslicejson which has no column documentation.

  3. Player season stats  — getadvstats.php
     No header row, 67 cols. Column positions confirmed from 2024 diagnosis.

  4. Timemachine rankings — for end-of-season opponent tier assignment.
"""

import gzip
import json
import time
import logging
import requests
import pandas as pd
from io import StringIO
from pathlib import Path
from typing import Optional

from config import (
    DATA_TORVIK,
    ALL_YEARS,
    TORVIK_TEAM_RESULTS_URL,
    TORVIK_PLAYER_STATS_URL,
    TORVIK_TIMEMACHINE_URL,
    SELECTION_SUNDAY_PLUS_ONE,
    TIER_1_MAX_RANK,
    TIER_2_MAX_RANK,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get(url: str, delay: float = 1.0) -> requests.Response:
    time.sleep(delay)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r


def _cache_path(filename: str) -> Path:
    return DATA_TORVIK / filename


# ── 1. Team Season Results ─────────────────────────────────────────────────────
#
# Pre-2020: 44 cols. Last col is "Fun Rk, adjt" — the comma inside the column
# name causes pandas to misparse the header row as data. Fix: read with
# header=None and manually promote row 0 to column names.
# 2020+: 45 cols, clean headers, reads normally.

TEAM_RESULTS_RENAME = {
    "adjoe":        "adj_o",
    "oe Rank":      "adj_o_rank",
    "adjde":        "adj_d",
    "de Rank":      "adj_d_rank",
    "rank.1":       "barthag_rank",
    "proj. W":      "proj_w",
    "Proj. L":      "proj_l",
    "WAB":          "wab",
    "WAB Rk":       "wab_rank",
    "Conf Win%":    "conf_win_pct",
    "adjt":         "adj_t",
    "Fun Rk, adjt": "adj_t",
    " adjt":        "adj_t",
}

_PRE_2020_YEARS = set(range(2008, 2020))


def fetch_team_results(year: int, force_refresh: bool = False) -> pd.DataFrame:
    url = TORVIK_TEAM_RESULTS_URL.format(year=year)
    cache_file = f"team_results_{year}.csv"
    path = _cache_path(cache_file)

    if not path.exists() or force_refresh:
        log.info(f"  Fetching: {url}")
        r = _get(url)
        path.write_text(r.text)

    if year in _PRE_2020_YEARS:
        raw_text = path.read_text()
        lines = raw_text.strip().split("\n")

        df = pd.read_csv(
            path,
            header=None,
            engine="python",
            on_bad_lines="skip",
            names=range(45),  # force 45 columns for all rows
        )

        header = df.iloc[0].astype(str).str.strip().tolist()
        try:
            fun_rk_idx = header.index("Fun Rk")
            header[fun_rk_idx] = "fun_rk"
            header[fun_rk_idx + 1] = "adj_t"
        except ValueError:
            pass

        # Deduplicate column names
        seen = {}
        deduped = []
        for col in header:
            if col in seen:
                seen[col] += 1
                deduped.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                deduped.append(col)

        df.columns = deduped
        df = df.iloc[1:].reset_index(drop=True)

    else:
        df = pd.read_csv(path, engine="python", on_bad_lines="warn")

    df = df.rename(columns=TEAM_RESULTS_RENAME)
    df["year"] = year

    for col in ["adj_o", "adj_d", "adj_t", "barthag", "wab"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "adj_o" in df.columns and "adj_d" in df.columns:
        df["adj_em"] = df["adj_o"] - df["adj_d"]
    else:
        log.warning(f"  adj_o/adj_d missing for {year}. Cols: {df.columns.tolist()}")

    return df

def fetch_all_team_results(years: list = ALL_YEARS, force_refresh: bool = False) -> pd.DataFrame:
    log.info("Fetching Torvik team results...")
    frames = []
    for year in years:
        log.info(f"  Year {year}")
        try:
            frames.append(fetch_team_results(year, force_refresh))
        except Exception as e:
            log.error(f"  Failed for {year}: {e}")
    if not frames:
        log.error("No team results collected.")
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    out = DATA_TORVIK / "all_team_results.csv"
    df.to_csv(out, index=False)
    log.info(f"Saved: {out} ({len(df)} rows)")
    return df


# ── 2. Team Four Factors ───────────────────────────────────────────────────────
#
# trank.php is now behind a browser verification wall, so we use teamslicejson
# instead. No header row — columns mapped by position from diagnosis.

TEAMSLICE_URL = "https://barttorvik.com/teamslicejson.php?year={year}&csv=1&type=R"

TEAMSLICE_COLS = [
    "team",         # 0
    "adj_o",        # 1
    "adj_d",        # 2
    "barthag",      # 3
    "record",       # 4
    "conf_wins",    # 5
    "g",            # 6
    "off_efg",      # 7
    "def_efg",      # 8
    "off_ftr",      # 9
    "def_ftr",      # 10
    "off_to",       # 11
    "def_to",       # 12
    "off_or",       # 13
    "def_or",       # 14
    "col_15",       # 15 — unknown
    "fg2_pct",      # 16
    "fg2_d_pct",    # 17
    "fg3_pct",      # 18
    "fg3_d_pct",    # 19
    "blk_d_pct",    # 20
    "blk_pct",      # 21
    "ast_rate",     # 22
    "ast_d_rate",   # 23
    "fg3_rate",     # 24
    "fg3_d_rate",   # 25
    "adj_t",        # 26
    "col_27",       # 27
    "col_28",       # 28
    "col_29",       # 29
    "year_col",     # 30
    "col_31",       # 31
    "col_32",       # 32
    "col_33",       # 33
    "wab",          # 34
    "ft_pct",       # 35
    "ft_d_pct",     # 36
]

_TEAMSLICE_NUM_COLS = [
    "adj_o", "adj_d", "adj_t", "barthag",
    "off_efg", "def_efg", "off_to", "def_to",
    "off_or", "def_or", "off_ftr", "def_ftr",
    "fg2_pct", "fg2_d_pct", "fg3_pct", "fg3_d_pct",
    "blk_pct", "blk_d_pct", "ast_rate", "ast_d_rate",
    "fg3_rate", "fg3_d_rate", "adj_t", "wab",
    "ft_pct", "ft_d_pct",
]


def fetch_four_factors(year: int, force_refresh: bool = False) -> pd.DataFrame:
    url = TEAMSLICE_URL.format(year=year)
    cache_file = f"four_factors_{year}.csv"
    path = _cache_path(cache_file)

    if not path.exists() or force_refresh:
        log.info(f"  Fetching: {url}")
        r = _get(url)
        path.write_text(r.text)

    df = pd.read_csv(path, header=None, engine="python", on_bad_lines="warn")

    if len(df.columns) == len(TEAMSLICE_COLS):
        df.columns = TEAMSLICE_COLS
    else:
        log.warning(f"  Col mismatch {year}: got {len(df.columns)}, expected {len(TEAMSLICE_COLS)}")
        df.columns = [f"col_{i}" for i in range(len(df.columns))]
        known = {0:"team", 1:"adj_o", 2:"adj_d", 3:"barthag",
                 7:"off_efg", 8:"def_efg", 9:"off_ftr", 10:"def_ftr",
                 11:"off_to", 12:"def_to", 13:"off_or", 14:"def_or",
                 16:"fg2_pct", 17:"fg2_d_pct", 18:"fg3_pct", 19:"fg3_d_pct",
                 26:"adj_t", 34:"wab", 35:"ft_pct", 36:"ft_d_pct"}
        df = df.rename(columns={f"col_{k}": v for k, v in known.items()})

    df["year"] = year
    df["adj_em"] = df["adj_o"] - df["adj_d"]

    for col in _TEAMSLICE_NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info(f"  {year}: {len(df)} teams")
    return df


def fetch_all_four_factors(years: list = ALL_YEARS, force_refresh: bool = False) -> pd.DataFrame:
    log.info("Fetching Torvik four factors...")
    frames = []
    for year in years:
        log.info(f"  Year {year}")
        try:
            frames.append(fetch_four_factors(year, force_refresh))
        except Exception as e:
            log.error(f"  Failed for {year}: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    out = DATA_TORVIK / "all_four_factors.csv"
    df.to_csv(out, index=False)
    log.info(f"Saved: {out} ({len(df)} rows)")
    return df

# ── 3. Player Season Stats ─────────────────────────────────────────────────────
#
# No header row, 67 columns.
# Key columns confirmed from 2024 diagnostic cross-referenced with toRvik docs.

PLAYER_STATS_COLS = [
    "player",       # 0
    "team",         # 1
    "conf",         # 2
    "g",            # 3
    "min_pct",      # 4  % of team minutes
    "o_rtg",        # 5  offensive rating
    "usg",          # 6  usage rate
    "efg_pct",      # 7  effective FG%
    "ts_pct",       # 8  true shooting %
    "fg3_rate",     # 9  3pt attempt rate (% of FGA)
    "col_10",       # 10
    "assist_rate",  # 11
    "to_rate",      # 12
    "ft_made",      # 13
    "ft_att",       # 14
    "ft_pct",       # 15
    "fg2_made",     # 16
    "fg2_att",      # 17
    "fg2_pct",      # 18
    "fg3_made",     # 19
    "fg3_att",      # 20
    "fg3_pct",      # 21
    "blk_rate",     # 22
    "stl_rate",     # 23
    "ftr",          # 24
    "yr",           # 25  class year: Fr/So/Jr/Sr/Grad
    "ht",           # 26
    "col_27",       # 27
    "col_28",       # 28
    "col_29",       # 29
    "col_30",       # 30
    "year_col",     # 31
    "player_id",    # 32
    "hometown",     # 33
    "col_34",       # 34
    "col_35",       # 35
    "col_36",       # 36
    "ppg",          # 37
    "rpg",          # 38
    "apg",          # 39
    "spg",          # 40
    "col_41",       # 41
    "col_42",       # 42
    "col_43",       # 43
    "col_44",       # 44
    "col_45",       # 45
    "col_46",       # 46
    "col_47",       # 47
    "col_48",       # 48
    "col_49",       # 49
    "col_50",       # 50
    "col_51",       # 51
    "col_52",       # 52
    "col_53",       # 53
    "col_54",       # 54
    "col_55",       # 55
    "col_56",       # 56
    "col_57",       # 57
    "col_58",       # 58
    "col_59",       # 59
    "col_60",       # 60
    "col_61",       # 61
    "col_62",       # 62
    "col_63",       # 63
    "position",     # 64
    "col_65",       # 65
    "dob",          # 66
]


def fetch_player_stats(year: int, force_refresh: bool = False) -> pd.DataFrame:
    url = TORVIK_PLAYER_STATS_URL.format(year=year)
    cache_file = f"player_stats_{year}.csv"
    path = _cache_path(cache_file)

    if not path.exists() or force_refresh:
        log.info(f"  Fetching: {url}")
        r = _get(url)
        path.write_text(r.text)

    df = pd.read_csv(path, header=None, engine="python", on_bad_lines="skip")

    if len(df.columns) == len(PLAYER_STATS_COLS):
        df.columns = PLAYER_STATS_COLS
    else:
        log.warning(f"  Col mismatch {year}: got {len(df.columns)}, expected {len(PLAYER_STATS_COLS)}")
        df.columns = [f"col_{i}" for i in range(len(df.columns))]
        known = {0:"player", 1:"team", 2:"conf", 3:"g", 4:"min_pct",
                 5:"o_rtg", 6:"usg", 7:"efg_pct", 8:"ts_pct",
                 15:"ft_pct", 18:"fg2_pct", 21:"fg3_pct",
                 25:"yr", 37:"ppg", 38:"rpg", 39:"apg", 40:"spg",
                 64:"position", 66:"dob"}
        df = df.rename(columns=known)

    df["year"] = year
    return df


def fetch_all_player_stats(years: list = ALL_YEARS, force_refresh: bool = False) -> pd.DataFrame:
    log.info("Fetching Torvik player stats...")
    frames = []
    for year in years:
        log.info(f"  Year {year}")
        try:
            frames.append(fetch_player_stats(year, force_refresh))
        except Exception as e:
            log.error(f"  Failed for {year}: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    out = DATA_TORVIK / "all_player_stats.csv"
    df.to_csv(out, index=False)
    log.info(f"Saved: {out} ({len(df)} rows)")
    return df


# ── 4. Timemachine Rankings ────────────────────────────────────────────────────

def fetch_timemachine_rankings(year: int, force_refresh: bool = False) -> pd.DataFrame:
    date = SELECTION_SUNDAY_PLUS_ONE.get(year)
    if not date:
        log.warning(f"  No timemachine date for {year}, skipping")
        return pd.DataFrame()

    cache_file = f"timemachine_{year}.csv"
    path = _cache_path(cache_file)

    if path.exists() and not force_refresh:
        log.info(f"  Cache hit: {cache_file}")
        return pd.read_csv(path)

    url = TORVIK_TIMEMACHINE_URL.format(date=date)
    log.info(f"  Fetching timemachine {year} ({date})")

    try:
        r = _get(url, delay=1.5)
        try:
            raw = json.loads(gzip.decompress(r.content))
        except OSError:
            raw = json.loads(r.content)

        # Format is array of arrays: [[team1_data...], [team2_data...], ...]
        # No header row — each element is one team's stats in positional order.
        # Confirmed column positions (from 2010 data diagnosis):
        #   0  = rank
        #   1  = team name
        #   2  = conference
        #   3  = record
        #   4  = adj_o
        #   5  = adj_o_rank
        #   6  = adj_d
        #   7  = adj_d_rank
        #   8  = barthag
        #   9  = barthag_rank
        #  10  = adj_em (= adj_o - adj_d)
        #  (remaining columns are not needed for tier assignment)
        if not isinstance(raw, list) or len(raw) < 1:
            log.warning(f"  Unexpected timemachine format for {year}")
            return pd.DataFrame()

        records = []
        for row in raw:
            if not isinstance(row, list) or len(row) < 9:
                continue
            try:
                records.append({
                    "torvik_rank": row[0],
                    "team":        row[1],
                    "conf":        row[2],
                    "record":      row[3],
                    "adj_o":       row[4],
                    "adj_d":       row[6],
                    "barthag":     row[8],
                })
            except (IndexError, TypeError):
                continue

        if not records:
            log.warning(f"  No valid rows parsed for timemachine {year}")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["year"] = year
        df["torvik_rank"] = pd.to_numeric(df["torvik_rank"], errors="coerce")
        df["opponent_tier"] = df["torvik_rank"].apply(_assign_tier)

        df.to_csv(path, index=False)
        return df

    except Exception as e:
        log.error(f"  Timemachine failed for {year}: {e}")
        return pd.DataFrame()

def _assign_tier(rank) -> int:
    try:
        rank = int(rank)
        if rank <= TIER_1_MAX_RANK:   return 1
        elif rank <= TIER_2_MAX_RANK: return 2
        else:                          return 3
    except (ValueError, TypeError):
        return 3


def fetch_all_timemachine_rankings(years: list = ALL_YEARS, force_refresh: bool = False) -> pd.DataFrame:
    log.info("Fetching Torvik timemachine rankings...")
    frames = []
    for year in years:
        log.info(f"  Year {year}")
        df = fetch_timemachine_rankings(year, force_refresh)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    out = DATA_TORVIK / "all_timemachine_rankings.csv"
    df.to_csv(out, index=False)
    log.info(f"Saved: {out} ({len(df)} rows)")
    return df


# ── 5. Team Game Logs ──────────────────────────────────────────────────────────
#
# Per-game stats for every team, every game.
# Used for opponent-adjusted dropoff features and variance calculations.
# One file per year covering all teams (~11k rows for 2024).

GAMELOG_URL = "https://barttorvik.com/getgamestats.php?year={year}&csv=1"

GAMELOG_COLS = [
    "date",         # 0
    "game_type",    # 1  (1=regular, 2=conf tourney, 3=ncaa tourney)
    "team",         # 2
    "conf",         # 3
    "opponent",     # 4
    "location",     # 5  (H/A/N)
    "result",       # 6  (W/L, score)
    "team_adj_o",   # 7
    "team_adj_d",   # 8
    "team_raw_o",   # 9
    "team_efg",     # 10
    "team_to",      # 11
    "team_or",      # 12
    "col_13",       # 13 — unknown
    "opp_raw_o",    # 14
    "opp_efg",      # 15
    "opp_to",       # 16
    "opp_or",       # 17
    "col_18",       # 18 — unknown
    "col_19",       # 19 — unknown
    "opp_conf",     # 20
    "col_21",       # 21
    "year",         # 22
    "tempo",        # 23
    "game_id",      # 24
    "team_coach",   # 25
    "opp_coach",    # 26
    "margin",       # 27
    "win_prob",     # 28
    "raw_json",     # 29
    "col_30",       # 30
]

_GAMELOG_NUM_COLS = [
    "team_adj_o", "team_adj_d", "team_raw_o",
    "team_efg", "team_to", "team_or",
    "opp_raw_o", "opp_efg", "opp_to", "opp_or",
    "tempo", "margin", "win_prob",
]


def fetch_gamelogs(year: int, force_refresh: bool = False) -> pd.DataFrame:
    url = GAMELOG_URL.format(year=year)
    cache_file = f"gamelogs_{year}.csv"
    path = _cache_path(cache_file)

    if path.exists() and not force_refresh:
        log.info(f"  Cache hit: gamelogs_{year}.csv")
        return pd.read_csv(path)

    log.info(f"  Fetching: {url}")
    r = _get(url)

    df = pd.read_csv(StringIO(r.text), header=None, engine="python", on_bad_lines="skip")

    if len(df.columns) == len(GAMELOG_COLS):
        df.columns = GAMELOG_COLS
    else:
        log.warning(f"  Col mismatch {year}: got {len(df.columns)}, expected {len(GAMELOG_COLS)}")
        df.columns = [f"col_{i}" for i in range(len(df.columns))]
        known = {0:"date", 1:"game_type", 2:"team", 3:"conf", 4:"opponent",
                 5:"location", 6:"result", 7:"team_adj_o", 8:"team_adj_d",
                 9:"team_raw_o", 10:"team_efg", 11:"team_to", 12:"team_or",
                 14:"opp_raw_o", 15:"opp_efg", 16:"opp_to", 17:"opp_or",
                 22:"year", 23:"tempo", 24:"game_id", 27:"margin", 28:"win_prob"}
        df = df.rename(columns={f"col_{k}": v for k, v in known.items()})

    for col in _GAMELOG_NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop the raw JSON column to save space
    if "raw_json" in df.columns:
        df = df.drop(columns=["raw_json"])

    path.write_text(df.to_csv(index=False))
    log.info(f"  {year}: {len(df)} games")
    return df


def fetch_all_gamelogs(years: list = ALL_YEARS, force_refresh: bool = False) -> pd.DataFrame:
    log.info("Fetching Torvik game logs...")
    frames = []
    for year in years:
        log.info(f"  Year {year}")
        try:
            frames.append(fetch_gamelogs(year, force_refresh))
        except Exception as e:
            log.error(f"  Failed for {year}: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    out = DATA_TORVIK / "all_gamelogs.csv"
    df.to_csv(out, index=False)
    log.info(f"Saved: {out} ({len(df)} rows)")
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch all Torvik data")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    fetch_all_team_results(force_refresh=args.refresh)
    fetch_all_four_factors(force_refresh=args.refresh)
    fetch_all_player_stats(force_refresh=args.refresh)
    fetch_all_timemachine_rankings(force_refresh=args.refresh)

    log.info("Torvik collection complete.")