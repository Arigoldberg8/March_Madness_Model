"""
pipeline.py
───────────
Master orchestrator for the March Madness data pipeline.

Run order:
  Step 1: Torvik data (fast, no rate limiting)
  Step 2: Tournament results (moderate scraping)
  Step 3: Sports Reference — one year at a time due to rate limits

Usage:
  # Full pipeline for all years (slow — run overnight for SportsRef steps)
  python pipeline.py --steps all

  # Just Torvik (fast, good starting point)
  python pipeline.py --steps torvik

  # Sports Reference for a specific year
  python pipeline.py --steps sportsref --year 2024

  # Tournament results only
  python pipeline.py --steps tournament

  # Refresh specific data
  python pipeline.py --steps torvik --refresh
"""

import logging
import argparse
from pathlib import Path

from config import ALL_YEARS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run_torvik(refresh: bool = False):
    log.info("═" * 50)
    log.info("STEP 1: Torvik Data Collection")
    log.info("═" * 50)
    from torvik_collector import (
        fetch_all_team_results,
        fetch_all_four_factors,
        fetch_all_player_stats,
        fetch_all_timemachine_rankings,
        fetch_all_gamelogs,
    )
    fetch_all_team_results(force_refresh=refresh)
    fetch_all_four_factors(force_refresh=refresh)
    fetch_all_player_stats(force_refresh=refresh)
    fetch_all_timemachine_rankings(force_refresh=refresh)
    fetch_all_gamelogs(force_refresh=refresh)
    log.info("Torvik collection complete.\n")


def run_tournament(refresh: bool = False):
    log.info("═" * 50)
    log.info("STEP 2: Tournament Results")
    log.info("═" * 50)
    from tournament_collector import fetch_all_tournament_results, validate_tournament_results
    df = fetch_all_tournament_results(force_refresh=refresh)
    if not df.empty:
        validate_tournament_results(df)
    log.info("Tournament collection complete.\n")


def run_sportsref(year: int = None, refresh: bool = False):
    log.info("═" * 50)
    log.info("STEP 3: Sports Reference Data Collection")
    log.info("  NOTE: This is slow. ~3s per request, 100s of teams.")
    log.info("  All results are cached — safe to interrupt and resume.")
    log.info("═" * 50)

    from sportsref_collector import (
        fetch_all_team_gamelogs,
        fetch_all_player_gamelogs,
    )

    years_to_run = [year] if year else ALL_YEARS

    for yr in years_to_run:
        log.info(f"\n── Year {yr} ──")
        log.info("  Team gamelogs...")
        fetch_all_team_gamelogs(yr, refresh)
        log.info("  Player gamelogs...")
        fetch_all_player_gamelogs(yr, refresh)

    log.info("Sports Reference collection complete.\n")


def run_all(year: int = None, refresh: bool = False):
    run_torvik(refresh)
    run_tournament(refresh)
    run_sportsref(year, refresh)

    log.info("═" * 50)
    log.info("ALL COLLECTION STEPS COMPLETE")
    log.info("Next step: run feature_engineering.py")
    log.info("═" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="March Madness data pipeline")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["torvik", "tournament", "sportsref", "all"],
        default=["all"],
        help="Which pipeline steps to run"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Specific year for Sports Reference collection (omit to run all years)"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-download of cached data"
    )

    args = parser.parse_args()
    steps = set(args.steps)

    if "all" in steps:
        run_all(args.year, args.refresh)
    else:
        if "torvik" in steps:
            run_torvik(args.refresh)
        if "tournament" in steps:
            run_tournament(args.refresh)
        if "sportsref" in steps:
            run_sportsref(args.year, args.refresh)