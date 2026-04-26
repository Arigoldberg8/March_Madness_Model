"""
tournament_collector.py
────────────────────────
Collects tournament seeds and results (our target variable) from Sports Reference.
Maps each tournament team to their number of wins (0–6 scale).

Sports Reference tournament page:
  https://www.sports-reference.com/cbb/postseason/{year}-ncaa.html

We scrape the bracket results directly from these pages.
Results are cached to data/raw/tournament/.
"""

import time
import logging
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Optional

from config import (
    DATA_TOURNAMENT,
    ALL_YEARS,
    ROUND_TO_WINS,
    SPORTSREF_REQUEST_DELAY,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

TOURNAMENT_URL = "https://www.sports-reference.com/cbb/postseason/{year}-ncaa.html"

# Round names as they appear in Sports Reference HTML
# Maps Sports Reference round labels to our standard labels
ROUND_LABEL_MAP = {
    # Round of 64
    "First Round":          "R64",
    "First Round (South)":  "R64",
    "First Round (East)":   "R64",
    "First Round (West)":   "R64",
    "First Round (Midwest)":"R64",
    # Round of 32
    "Second Round":         "R32",
    # Sweet 16
    "Sweet Sixteen":        "S16",
    "Sweet 16":             "S16",
    # Elite 8
    "Elite Eight":          "E8",
    "Elite 8":              "E8",
    "Regional Finals":      "E8",
    # Final Four
    "Final Four":           "F4",
    "National Semifinals":  "F4",
    # Championship
    "National Championship":"Champion",
    "Championship":         "Champion",
}


def _sleep():
    time.sleep(SPORTSREF_REQUEST_DELAY)


def _cache_path(filename: str) -> Path:
    return DATA_TOURNAMENT / filename


def fetch_tournament_results(year: int, force_refresh: bool = False) -> pd.DataFrame:
    cache_file = _cache_path(f"tournament_{year}.csv")

    if cache_file.exists() and not force_refresh:
        log.info(f"  Cache hit: tournament_{year}.csv")
        return pd.read_csv(cache_file)

    url = TOURNAMENT_URL.format(year=year)
    log.info(f"  Fetching tournament results for {year}: {url}")
    _sleep()

    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
    except Exception as e:
        log.error(f"  Failed to fetch tournament page for {year}: {e}")
        return pd.DataFrame()

    import re
    soup = BeautifulSoup(r.text, "lxml")

    all_brackets = soup.find_all("div", id="bracket", class_="team16")
    if not all_brackets:
        log.error(f"  No bracket divs found for {year}")
        return pd.DataFrame()

    # Round index -> exit round for losers in that round
    # Winners advance and will appear in next round as losers
    LOSER_EXIT = {
       0: ("R64", 0),   # lost in R64 -> 0 wins
       1: ("R32", 1),   # lost in R32 -> 1 win
        2: ("S16", 2),   # lost in S16 -> 2 wins
       3: ("E8",  3),   # lost in E8  -> 3 wins
       # round 4 is skipped — just shows regional winner moving to F4
    }

    team_best = {}

    for bracket in all_brackets:
        round_divs = bracket.find_all("div", class_="round", recursive=False)

        for round_idx, round_div in enumerate(round_divs):
            if round_idx not in LOSER_EXIT:
               continue  # skip round 4

            exit_label, exit_wins = LOSER_EXIT[round_idx]

            for a in round_div.find_all("a", href=re.compile(r'/cbb/schools/')):
                href = a.get("href", "")
                parts = href.strip("/").split("/")
                try:
                    school_idx = parts.index("schools")
                    abbrev = parts[school_idx + 1]
                    team_name = a.get_text(strip=True)
                    is_winner = "winner" in (a.parent.get("class") or [])

                    # Get seed from sibling span
                    seed = None
                    seed_span = a.parent.find("span")
                    if seed_span:
                        try:
                            seed = int(seed_span.get_text(strip=True))
                        except ValueError:
                            pass

                    if is_winner:
                        # Winners advance — only record if not seen deeper yet
                        if abbrev not in team_best or exit_wins > team_best[abbrev]["wins"]:
                            team_best[abbrev] = {
                                "team_abbrev": abbrev,
                                "team_name":   team_name,
                                "seed":        seed,
                                "wins":        exit_wins,
                                "exit_round":  exit_label,
                            }
                        if abbrev not in team_best:
                            team_best[abbrev] = {
                                "team_abbrev": abbrev,
                                "team_name":   team_name,
                                "seed":        seed,
                                "wins":        exit_wins,
                                "exit_round":  exit_label,
                            }
                    else:
                        # Losers exit here — this is their final record
                        if abbrev not in team_best or exit_wins > team_best[abbrev]["wins"]:
                            team_best[abbrev] = {
                                "team_abbrev": abbrev,
                                "team_name":   team_name,
                                "seed":        seed,
                                "wins":        exit_wins,
                                "exit_round":  exit_label,
                            }

                except (ValueError, IndexError):
                    continue

    # Now handle Final Four and Champion from team4 bracket
    team4 = soup.find("div", id="bracket", class_="team4")
    if team4:
        f4_rounds = team4.find_all("div", class_="round", recursive=False)
        F4_EXIT = {
            0: ("F4", 4),           # lost in F4 -> 4 wins
            1: ("Runner-up", 5),    # lost in championship -> 5 wins
            2: ("Champion", 6),     # won it all -> 6 wins
        }
        for round_idx, round_div in enumerate(f4_rounds):
            if round_idx not in F4_EXIT:
                continue
            exit_label, exit_wins = F4_EXIT[round_idx]
            for a in round_div.find_all("a", href=re.compile(r'/cbb/schools/')):
                href = a.get("href", "")
                parts = href.strip("/").split("/")
                try:
                    school_idx = parts.index("schools")
                    abbrev = parts[school_idx + 1]
                    team_name = a.get_text(strip=True)
                    is_winner = "winner" in (a.parent.get("class") or [])

                    if is_winner and exit_wins < 6:
                        # Will be updated in next round
                        if abbrev not in team_best or exit_wins > team_best[abbrev]["wins"]:
                            team_best[abbrev] = {
                                "team_abbrev": abbrev,
                                "team_name":   team_name,
                                "seed":        team_best.get(abbrev, {}).get("seed"),
                                "wins":        exit_wins,
                                "exit_round":  exit_label,
                            }  
                    else:
                        if abbrev not in team_best or exit_wins > team_best[abbrev]["wins"]:
                            team_best[abbrev] = {
                                "team_abbrev": abbrev,
                                "team_name":   team_name,
                                "seed":        team_best.get(abbrev, {}).get("seed"),
                                "wins":        exit_wins,
                                "exit_round":  exit_label,
                            }
                except (ValueError, IndexError):
                    continue
    if not team_best:
        log.error(f"  No teams parsed from bracket for {year}")
        return pd.DataFrame()

    df = pd.DataFrame(list(team_best.values()))
    df["year"] = year
    df = df.sort_values("wins", ascending=False).reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    log.info(f"  Saved {len(df)} tournament teams for {year}")
    return df


def _extract_seed(element) -> Optional[int]:
    """Try to extract seed from text near the team link."""
    # Seeds often appear as a preceding text node or sibling span
    parent = element.parent
    if parent:
        text = parent.get_text()
        import re
        match = re.search(r'\b([1-9]|1[0-6])\b', text)
        if match:
            return int(match.group(1))
    return None


def _fix_champion_runnerup(df: pd.DataFrame) -> pd.DataFrame:
    """
    In bracket parsing, both finalist teams appear in the championship game.
    The winner gets 6 wins, the loser gets 5.
    This is usually handled correctly if we track wins properly,
    but we validate here just in case.
    """
    champions = df[df["exit_round"] == "Champion"]
    if len(champions) > 1:
        # Keep the one that played the most games (should already be correct)
        # or just take the one labeled Champion from the winning side
        log.warning("  Multiple champions detected — check bracket parse logic")
    return df


def _parse_tournament_from_table(html_text: str, year: int, cache_file: Path) -> pd.DataFrame:
    from io import StringIO
    log.info(f"  Using table fallback for {year}")
    try:
        tables = pd.read_html(StringIO(html_text))
    except Exception as e:
        log.error(f"  No tables found for {year}: {e}")
        return pd.DataFrame()

    if not tables:
        log.error(f"  No tables found for {year}")
        return pd.DataFrame()

    df = max(tables, key=len)
    df["year"] = year
    df.to_csv(cache_file, index=False)
    log.warning(f"  Fallback table saved for {year} — manual review recommended")
    return df


def fetch_all_tournament_results(years: list = ALL_YEARS, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch tournament results for all seasons and combine."""
    log.info("Fetching tournament results...")
    frames = []
    for year in years:
        log.info(f"  Year {year}")
        df = fetch_tournament_results(year, force_refresh)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    out = DATA_TOURNAMENT / "all_tournament_results.csv"
    result.to_csv(out, index=False)
    log.info(f"Saved: {out} ({len(result)} rows)")
    return result


def validate_tournament_results(df: pd.DataFrame) -> None:
    """
    Basic sanity checks on tournament results.
    Each year should have exactly 64 teams, wins summing to 63.
    """
    log.info("Validating tournament results...")
    for year, grp in df.groupby("year"):
        n_teams = len(grp)
        total_wins = grp["wins"].sum()
        expected_wins = 63  # single elimination: 64 teams, 63 games

        issues = []
        if n_teams != 64:
            issues.append(f"expected 64 teams, got {n_teams}")
        if total_wins != expected_wins:
            issues.append(f"expected {expected_wins} total wins, got {total_wins}")

        if issues:
            log.warning(f"  {year}: " + "; ".join(issues))
        else:
            log.info(f"  {year}: OK ({n_teams} teams, {total_wins} wins)")


# ── Optional: manual override CSV ─────────────────────────────────────────────
# If scraping fails for certain years, you can create a manual override CSV at
# data/raw/tournament/manual_overrides.csv with columns:
# year, team_abbrev, team_name, seed, wins, exit_round
# These will be merged in during the pipeline step.

def load_manual_overrides() -> pd.DataFrame:
    path = DATA_TOURNAMENT / "manual_overrides.csv"
    if path.exists():
        log.info("Loading manual tournament overrides...")
        return pd.read_csv(path)
    return pd.DataFrame()


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch tournament seeds and results")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    df = fetch_all_tournament_results(force_refresh=args.refresh)
    if args.validate and not df.empty:
        validate_tournament_results(df)
