"""
sportsref_collector.py
──────────────────────
Scrapes Sports Reference for team and player game logs.

Collects:
  1. Team game logs    — per-game box score stats (FG%, 3P%, FT%, ORB, BLK, STL, TOV)
  2. Player game logs  — per-game stats for top 8 scorers per tournament team

All pages follow the pattern:
  Team gamelog:   /cbb/schools/{abbrev}/men/{year}-gamelogs.html
  Season page:    /cbb/schools/{abbrev}/men/{year}.html  (to get player IDs)
  Player gamelog: /cbb/players/{player-id}/gamelog/{year}/

Rate limiting: 3s between requests. All data cached locally.
Safe to interrupt and resume — cached files are skipped.

Usage:
  # Collect for all tournament teams, all years
  python sportsref_collector.py

  # Single year
  python sportsref_collector.py --year 2024

  # Force refresh
  python sportsref_collector.py --year 2024 --refresh
"""

import re
import time
import logging
import requests
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO
from pathlib import Path
from typing import Optional

from config import (
    DATA_SPORTSREF,
    DATA_TOURNAMENT,
    ALL_YEARS,
    SPORTSREF_REQUEST_DELAY,
    TOP_N_SCORERS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://www.sports-reference.com"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sleep():
    time.sleep(SPORTSREF_REQUEST_DELAY)


def _cache_path(*parts) -> Path:
    p = DATA_SPORTSREF.joinpath(*parts[:-1])
    p.mkdir(parents=True, exist_ok=True)
    return p / parts[-1]


def _fetch(url: str) -> Optional[requests.Response]:
    _sleep()
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return r
    except Exception as e:
        log.warning(f"  Fetch failed: {url} — {e}")
        return None


def _uncomment(html: str) -> str:
    """Sports Reference hides tables in HTML comments — uncomment them."""
    return re.sub(r'<!--(.*?)-->', r'\1', html, flags=re.DOTALL)


def _parse_table(html: str, table_id: str) -> Optional[pd.DataFrame]:
    soup = BeautifulSoup(_uncomment(html), "lxml")
    table = soup.find("table", id=table_id)
    if not table:
        return None
    df = pd.read_html(StringIO(str(table)))[0]
    # Flatten multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            f"{b}" if "Unnamed" in str(a) else f"{a}_{b}" if b and str(b) != str(a) else f"{a}"
            for a, b in df.columns
        ]
    return df


def _load_tournament_teams(year: int) -> pd.DataFrame:
    path = DATA_TOURNAMENT / f"tournament_{year}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# ── Torvik name -> Sports Reference abbrev mapping ────────────────────────────
# Sports Reference uses URL-safe school abbreviations that differ from Torvik names.
# We build this mapping by checking the tournament page links we already scraped.

def _torvik_to_sref_abbrev(team_abbrev: str) -> str:
    """
    The tournament collector already stores team_abbrev as the Sports Reference
    URL slug (e.g. 'connecticut', 'north-carolina'). Use it directly.
    """
    return team_abbrev.lower().strip()


# ── 1. Team Game Logs ──────────────────────────────────────────────────────────

def fetch_team_gamelog(sref_abbrev: str, year: int, force_refresh: bool = False) -> pd.DataFrame:
    """
    Scrape a team's full game log for a season from Sports Reference.

    Returns one row per game with columns:
      date, opponent, location, result, pts_for, pts_against,
      fg, fga, fg_pct, fg3, fg3a, fg3_pct, fg2, fg2a, fg2_pct,
      efg_pct, ft, fta, ft_pct, orb, drb, trb, ast, stl, blk, tov, pf
      (plus opponent equivalents prefixed with opp_)
    """
    cache_file = _cache_path("team_gamelogs", str(year), f"{sref_abbrev}_{year}.csv")

    if cache_file.exists() and not force_refresh:
        return pd.read_csv(cache_file)

    url = f"{BASE_URL}/cbb/schools/{sref_abbrev}/men/{year}-gamelogs.html"
    log.info(f"    Fetching team gamelog: {sref_abbrev} {year}")
    r = _fetch(url)
    if r is None:
        return pd.DataFrame()

    df = _parse_table(r.text, "team_game_log")
    if df is None or df.empty:
        log.warning(f"    No team_game_log table for {sref_abbrev} {year}")
        return pd.DataFrame()

    df = _clean_team_gamelog(df, sref_abbrev, year)
    df.to_csv(cache_file, index=False)
    return df


def _clean_team_gamelog(df: pd.DataFrame, abbrev: str, year: int) -> pd.DataFrame:
    """Standardize column names and types for team game log."""
    # Drop header repeat rows (Sports Reference inserts header rows mid-table)
    df = df[df.iloc[:, 0].astype(str).str.match(r'^\d+$')].copy()

    # Rename columns to standard names
    rename = {}
    for col in df.columns:
        c = col.strip()
        # Map multi-level flattened names
        if c == "Rk":            rename[col] = "rk"
        elif c == "Gtm":         rename[col] = "game_num"
        elif c == "Date":        rename[col] = "date"
        elif c == "Unnamed: 3":  rename[col] = "location_flag"  # @ for away
        elif c == "Opp":         rename[col] = "opponent"
        elif c == "Type":        rename[col] = "game_type"
        elif c in ("Rslt", "Score_Rslt"): rename[col] = "result"
        elif c in ("Tm", "Score_Tm"):     rename[col] = "pts"
        elif c in ("Opp.1", "Score_Opp"): rename[col] = "pts_opp"
        elif c == "OT":          rename[col] = "ot"
        # Team stats
        elif c == "Team_FG":     rename[col] = "fg"
        elif c == "Team_FGA":    rename[col] = "fga"
        elif c == "Team_FG%":    rename[col] = "fg_pct"
        elif c == "Team_3P":     rename[col] = "fg3"
        elif c == "Team_3PA":    rename[col] = "fg3a"
        elif c == "Team_3P%":    rename[col] = "fg3_pct"
        elif c == "Team_2P":     rename[col] = "fg2"
        elif c == "Team_2PA":    rename[col] = "fg2a"
        elif c == "Team_2P%":    rename[col] = "fg2_pct"
        elif c == "Team_eFG%":   rename[col] = "efg_pct"
        elif c == "Team_FT":     rename[col] = "ft"
        elif c == "Team_FTA":    rename[col] = "fta"
        elif c == "Team_FT%":    rename[col] = "ft_pct"
        elif c == "Team_ORB":    rename[col] = "orb"
        elif c == "Team_DRB":    rename[col] = "drb"
        elif c == "Team_TRB":    rename[col] = "trb"
        elif c == "Team_AST":    rename[col] = "ast"
        elif c == "Team_STL":    rename[col] = "stl"
        elif c == "Team_BLK":    rename[col] = "blk"
        elif c == "Team_TOV":    rename[col] = "tov"
        elif c == "Team_PF":     rename[col] = "pf"
        # Opponent stats
        elif c == "Opponent_FG":   rename[col] = "opp_fg"
        elif c == "Opponent_FGA":  rename[col] = "opp_fga"
        elif c == "Opponent_FG%":  rename[col] = "opp_fg_pct"
        elif c == "Opponent_3P":   rename[col] = "opp_fg3"
        elif c == "Opponent_3PA":  rename[col] = "opp_fg3a"
        elif c == "Opponent_3P%":  rename[col] = "opp_fg3_pct"
        elif c == "Opponent_2P":   rename[col] = "opp_fg2"
        elif c == "Opponent_2PA":  rename[col] = "opp_fg2a"
        elif c == "Opponent_2P%":  rename[col] = "opp_fg2_pct"
        elif c == "Opponent_eFG%": rename[col] = "opp_efg_pct"
        elif c == "Opponent_FT":   rename[col] = "opp_ft"
        elif c == "Opponent_FTA":  rename[col] = "opp_fta"
        elif c == "Opponent_FT%":  rename[col] = "opp_ft_pct"
        elif c == "Opponent_ORB":  rename[col] = "opp_orb"
        elif c == "Opponent_DRB":  rename[col] = "opp_drb"
        elif c == "Opponent_TRB":  rename[col] = "opp_trb"
        elif c == "Opponent_AST":  rename[col] = "opp_ast"
        elif c == "Opponent_STL":  rename[col] = "opp_stl"
        elif c == "Opponent_BLK":  rename[col] = "opp_blk"
        elif c == "Opponent_TOV":  rename[col] = "opp_tov"
        elif c == "Opponent_PF":   rename[col] = "opp_pf"

    df = df.rename(columns=rename)

    # Parse location from location_flag column (@ = away, blank = home, neutral handled in game_type)
    if "location_flag" in df.columns:
        df["location"] = df["location_flag"].apply(
            lambda x: "A" if str(x).strip() == "@" else "N" if str(x).strip() == "N" else "H"
        )

    # Parse win/loss from result
    if "result" in df.columns:
        df["win"] = (df["result"].str.startswith("W") == True).astype(int)

    # Convert numeric columns
    num_cols = ["pts", "pts_opp", "fg", "fga", "fg_pct", "fg3", "fg3a", "fg3_pct",
                "fg2", "fg2a", "fg2_pct", "efg_pct", "ft", "fta", "ft_pct",
                "orb", "drb", "trb", "ast", "stl", "blk", "tov", "pf",
                "opp_fg", "opp_fga", "opp_fg_pct", "opp_fg3", "opp_fg3a", "opp_fg3_pct",
                "opp_efg_pct", "opp_ft", "opp_fta", "opp_ft_pct",
                "opp_orb", "opp_stl", "opp_blk", "opp_tov"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse game type: REG = regular season, conf tourney, NCAA tourney
    if "game_type" in df.columns:
        df["is_regular_season"] = df["game_type"].str.contains("REG", na=False).astype(int)
        df["is_ncaa_tourney"] = df["game_type"].str.contains("NCAA", na=False).astype(int)

    df["team_abbrev"] = abbrev
    df["year"] = year
    return df


def fetch_all_team_gamelogs(year: int, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch team gamelogs for all tournament teams in a given year."""
    tournament = _load_tournament_teams(year)
    if tournament.empty:
        log.warning(f"  No tournament teams found for {year}")
        return pd.DataFrame()

    frames = []
    for _, row in tournament.iterrows():
        abbrev = _torvik_to_sref_abbrev(row["team_abbrev"])
        df = fetch_team_gamelog(abbrev, year, force_refresh)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    out = _cache_path("team_gamelogs", f"all_team_gamelogs_{year}.csv")
    result.to_csv(out, index=False)
    log.info(f"  Saved team gamelogs for {year}: {len(result)} games across {len(frames)} teams")
    return result


# ── 2. Player IDs from Season Page ────────────────────────────────────────────

def fetch_player_ids(sref_abbrev: str, year: int, force_refresh: bool = False) -> pd.DataFrame:
    """
    Scrape the team season page to get player IDs and season stats.
    Returns DataFrame with player_id, player_name, ppg columns.
    """
    cache_file = _cache_path("player_ids", str(year), f"{sref_abbrev}_{year}_players.csv")

    if cache_file.exists() and not force_refresh:
        return pd.read_csv(cache_file)

    url = f"{BASE_URL}/cbb/schools/{sref_abbrev}/men/{year}.html"
    log.info(f"    Fetching player IDs: {sref_abbrev} {year}")
    r = _fetch(url)
    if r is None:
        return pd.DataFrame()

    # Get per-game stats table for PPG sorting
    df = _parse_table(r.text, "players_per_game")
    if df is None or df.empty:
        log.warning(f"    No players_per_game table for {sref_abbrev} {year}")
        return pd.DataFrame()

    # Extract player IDs from links in the raw HTML
    html = _uncomment(r.text)
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id="players_per_game")
    if not table:
        return pd.DataFrame()

    player_rows = []
    for row in table.find_all("tr"):
        a = row.find("a", href=re.compile(r"/cbb/players/"))
        if not a:
            continue
        href = a.get("href", "")
        # Extract player_id from /cbb/players/donovan-clingan-1.html
        match = re.search(r"/cbb/players/([^/]+)\.html", href)
        if not match:
            continue
        player_id = match.group(1)
        player_name = a.get_text(strip=True)

        # Get PTS from this row
        cells = row.find_all("td")
        pts = None
        for td in cells:
            if td.get("data-stat") == "pts_per_g":
                try:
                    pts = float(td.get_text(strip=True))
                except (ValueError, TypeError):
                    pass

        player_rows.append({
            "player_id":   player_id,
            "player_name": player_name,
            "ppg":         pts,
            "team_abbrev": sref_abbrev,
            "year":        year,
        })

    if not player_rows:
        return pd.DataFrame()

    df = pd.DataFrame(player_rows)
    df = df.sort_values("ppg", ascending=False).reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    return df


# ── 3. Player Game Logs ────────────────────────────────────────────────────────

def fetch_player_gamelog(player_id: str, player_name: str, year: int,
                         force_refresh: bool = False) -> pd.DataFrame:
    """
    Scrape per-game stats for a single player.

    Returns one row per game with:
      date, opponent, result, gs, mp, fg, fga, fg_pct, fg3, fg3a, fg3_pct,
      ft, fta, ft_pct, orb, drb, trb, ast, stl, blk, tov, pf, pts, gmsc
    """
    cache_file = _cache_path("player_gamelogs", str(year), f"{player_id}_{year}.csv")

    if cache_file.exists() and not force_refresh:
        return pd.read_csv(cache_file)

    url = f"{BASE_URL}/cbb/players/{player_id}/gamelog/{year}/"
    r = _fetch(url)
    if r is None:
        return pd.DataFrame()

    df = _parse_table(r.text, "player_game_log")
    if df is None or df.empty:
        log.warning(f"    No gamelog for {player_name} {year}")
        return pd.DataFrame()

    df = _clean_player_gamelog(df, player_id, player_name, year)
    df.to_csv(cache_file, index=False)
    return df


def _clean_player_gamelog(df: pd.DataFrame, player_id: str,
                           player_name: str, year: int) -> pd.DataFrame:
    """Standardize player gamelog columns."""
    # Drop non-game rows (header repeats, "Did Not Play", etc.)
    df = df[df["Rk"].astype(str).str.match(r'^\d+$')].copy()

    rename = {
        "Rk": "rk", "Gcar": "career_game", "Gtm": "game_num",
        "Date": "date", "Team": "team", "Unnamed: 5": "location_flag",
        "Opp": "opponent", "Type": "game_type", "Result": "result",
        "GS": "gs", "MP": "mp",
        "FG": "fg", "FGA": "fga", "FG%": "fg_pct",
        "3P": "fg3", "3PA": "fg3a", "3P%": "fg3_pct",
        "2P": "fg2", "2PA": "fg2a", "2P%": "fg2_pct",
        "eFG%": "efg_pct",
        "FT": "ft", "FTA": "fta", "FT%": "ft_pct",
        "ORB": "orb", "DRB": "drb", "TRB": "trb",
        "AST": "ast", "STL": "stl", "BLK": "blk",
        "TOV": "tov", "PF": "pf", "PTS": "pts", "GmSc": "gmsc",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "location_flag" in df.columns:
        df["location"] = df["location_flag"].apply(
            lambda x: "A" if str(x).strip() == "@" else "N" if str(x).strip() == "N" else "H"
        )

    if "result" in df.columns:
        df["win"] = df["result"].str.startswith("W").astype(int)

    if "game_type" in df.columns:
        df["is_regular_season"] = df["game_type"].str.contains("REG", na=False).astype(int)

    num_cols = ["fg", "fga", "fg_pct", "fg3", "fg3a", "fg3_pct",
                "fg2", "fg2a", "fg2_pct", "efg_pct",
                "ft", "fta", "ft_pct", "orb", "drb", "trb",
                "ast", "stl", "blk", "tov", "pf", "pts", "gmsc"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["player_id"]   = player_id
    df["player_name"] = player_name
    df["year"]        = year
    return df


def fetch_top_player_gamelogs(sref_abbrev: str, year: int,
                               force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch gamelogs for the top N scorers on a team.
    Gets player IDs from season page first, then fetches each player's gamelog.
    """
    players = fetch_player_ids(sref_abbrev, year, force_refresh)
    if players.empty:
        log.warning(f"  No players found for {sref_abbrev} {year}")
        return pd.DataFrame()

    # Take top N scorers with valid PPG
    top_players = players.dropna(subset=["ppg"]).head(TOP_N_SCORERS)
    log.info(f"  Fetching gamelogs for top {len(top_players)} scorers: {sref_abbrev} {year}")

    frames = []
    for _, player in top_players.iterrows():
        pid  = player["player_id"]
        name = player["player_name"]
        log.info(f"    {name} ({player['ppg']:.1f} ppg)")
        df = fetch_player_gamelog(pid, name, year, force_refresh)
        if not df.empty:
            df["team_abbrev"] = sref_abbrev
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_all_player_gamelogs(year: int, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch top-N player gamelogs for all tournament teams in a given year."""
    tournament = _load_tournament_teams(year)
    if tournament.empty:
        log.warning(f"  No tournament teams for {year}")
        return pd.DataFrame()

    frames = []
    for _, row in tournament.iterrows():
        abbrev = _torvik_to_sref_abbrev(row["team_abbrev"])
        log.info(f"\n  Team: {row['team_name']} ({abbrev})")
        df = fetch_top_player_gamelogs(abbrev, year, force_refresh)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    out = _cache_path("player_gamelogs", f"all_player_gamelogs_{year}.csv")
    result.to_csv(out, index=False)
    log.info(f"\n  Saved player gamelogs for {year}: {len(result)} game entries")
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scrape Sports Reference team and player gamelogs")
    parser.add_argument("--year",    type=int, default=None, help="Single year to collect")
    parser.add_argument("--refresh", action="store_true",    help="Force re-download")
    parser.add_argument(
        "--collect",
        nargs="+",
        choices=["team", "players", "all"],
        default=["all"],
    )
    args = parser.parse_args()

    years = [args.year] if args.year else ALL_YEARS
    collect = set(args.collect)

    for year in years:
        log.info(f"\n{'═'*50}")
        log.info(f"Year {year}")
        log.info(f"{'═'*50}")

        if "all" in collect or "team" in collect:
            log.info("Collecting team gamelogs...")
            fetch_all_team_gamelogs(year, args.refresh)

        if "all" in collect or "players" in collect:
            log.info("Collecting player gamelogs...")
            fetch_all_player_gamelogs(year, args.refresh)

    log.info("\nSports Reference collection complete.")