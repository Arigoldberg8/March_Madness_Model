"""
Central config for the March Madness prediction pipeline.
All paths, seasons, and constants live here.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_RAW         = ROOT / "data" / "raw"
DATA_TORVIK      = DATA_RAW / "torvik"
DATA_SPORTSREF   = DATA_RAW / "sportsref"
DATA_TOURNAMENT  = DATA_RAW / "tournament"
DATA_PROCESSED   = ROOT / "data" / "processed"

for p in [DATA_TORVIK, DATA_SPORTSREF, DATA_TOURNAMENT, DATA_PROCESSED]:
    p.mkdir(parents=True, exist_ok=True)


# ── Seasons ────────────────────────────────────────────────────────────────────
# Sports Reference uses the end year of the season (2024 = 2023-24 season)
# Torvik uses the same convention
ALL_YEARS = [y for y in range(2011, 2027) if y != 2020]  # exclude 2020 (no tournament)


# ── Torvik URL Templates ───────────────────────────────────────────────────────
# Team season results (AdjO, AdjD, AdjEM, tempo, etc.)
TORVIK_TEAM_RESULTS_URL     = "https://barttorvik.com/{year}_team_results.csv"

# Regular-season-only team slice (pre-tournament stats, for cleaner features)
# type=R = regular season only; excludes conference tournaments
TORVIK_FOUR_FACTORS_URL     = "https://barttorvik.com/trank.php?year={year}&json=0&csv=1"

# Player season stats
TORVIK_PLAYER_STATS_URL     = "https://barttorvik.com/getadvstats.php?year={year}&csv=1"

# Time machine: T-Rank ratings frozen at a specific date (post-Selection Sunday)
# Used to get end-of-season opponent rankings for tier assignment
# Format: YYYYMMDD — use the day after Selection Sunday each year
TORVIK_TIMEMACHINE_URL      = "https://barttorvik.com/timemachine/team_results/{date}_team_results.json.gz"


# ── Selection Sunday Dates (day after, for timemachine) ───────────────────────
# These are the dates we freeze Torvik rankings at for opponent tier assignment
# Each team's opponents get labeled with their rank as of Selection Sunday
SELECTION_SUNDAY_PLUS_ONE = {
    2010: "20100315",
    2011: "20110314",
    2012: "20120312",
    2013: "20130318",
    2014: "20140317",
    2015: "20150316",
    2016: "20160314",
    2017: "20170313",
    2018: "20180312",
    2019: "20190318",
    # 2020: no tournament
    2021: "20210315",
    2022: "20220314",
    2023: "20230313",
    2024: "20240318",
    2025: "20250317",
    2026: "20260316",

}

# regions
BRACKET_REGIONS = {
    2025: {
        # South
        "Auburn": "South", "Alabama State": "South",
        "Louisville": "South", "Creighton": "South",
        "Michigan": "South", "UC-San Diego": "South",
        "Texas A&M": "South", "Yale": "South",
        "Ole Miss": "South", "UNC": "South",
        "Iowa State": "South", "Lipscomb": "South",
        "Marquette": "South", "New Mexico": "South",
        "Michigan State": "South", "Bryant": "South",
        # East
        "Duke": "East", "Mount St. Mary's": "East",
        "Mississippi State": "East", "Baylor": "East",
        "Oregon": "East", "Liberty": "East",
        "Arizona": "East", "Akron": "East",
        "BYU": "East", "VCU": "East",
        "Wisconsin": "East", "Montana": "East",
        "Saint Mary's": "East", "Vanderbilt": "East",
        "Alabama": "East", "Robert Morris": "East",
        # West
        "Florida": "West", "Norfolk State": "West",
        "UConn": "West", "Oklahoma": "West",
        "Memphis": "West", "Colorado State": "West",
        "Maryland": "West", "Grand Canyon": "West",
        "Missouri": "West", "Drake": "West",
        "Texas Tech": "West", "UNC Wilmington": "West",
        "Kansas": "West", "Arkansas": "West",
        "St. John's (NY)": "West", "Omaha": "West",
        # Midwest
        "Houston": "Midwest", "SIU-Edwardsville": "Midwest",
        "Gonzaga": "Midwest", "Georgia": "Midwest",
        "Clemson": "Midwest", "McNeese State": "Midwest",
        "Purdue": "Midwest", "High Point": "Midwest",
        "Illinois": "Midwest", "Xavier": "Midwest",
        "Kentucky": "Midwest", "Troy": "Midwest",
        "UCLA": "Midwest", "Utah State": "Midwest",
        "Tennessee": "Midwest", "Wofford": "Midwest",
    },
    2024: {
        # East
        "UConn": "East", "Stetson": "East",
        "Florida Atlantic": "East", "Northwestern": "East",
        "San Diego State": "East", "UAB": "East",
        "Auburn": "East", "Yale": "East",
        "BYU": "East", "Duquesne": "East",
        "Illinois": "East", "Morehead State": "East",
        "Washington State": "East", "Drake": "East",
        "Iowa State": "East", "South Dakota State": "East",

        # West
        "UNC": "West", "Wagner": "West",
        "Mississippi State": "West", "Michigan State": "West",
        "Saint Mary's": "West", "Grand Canyon": "West",
        "Alabama": "West", "College of Charleston": "West",
        "Clemson": "West", "New Mexico": "West",
        "Baylor": "West", "Colgate": "West",
        "Dayton": "West", "Nevada": "West",
        "Arizona": "West", "Long Beach State": "West",

        # South
        "Houston": "South", "Longwood": "South",
        "Nebraska": "South", "Texas A&M": "South",
        "Wisconsin": "South", "James Madison": "South",
        "Duke": "South", "Vermont": "South",
        "NC State": "South", "Texas Tech": "South",
        "Kentucky": "South", "Oakland": "South",
        "Florida": "South", "Colorado": "South",
        "Marquette": "South", "Western Kentucky": "South",

        # Midwest
        "Purdue": "Midwest", "Grambling": "Midwest",
        "Utah State": "Midwest", "TCU": "Midwest",
        "Gonzaga": "Midwest", "McNeese State": "Midwest",
        "Kansas": "Midwest", "Samford": "Midwest",
        "South Carolina": "Midwest", "Oregon": "Midwest",
        "Creighton": "Midwest", "Akron": "Midwest",
        "Texas": "Midwest", "Colorado State": "Midwest",
        "Tennessee": "Midwest", "St. Peter's": "Midwest",
    },
    2023: {
        # South
        "Alabama": "South", "Texas A&M-Corpus Christi": "South",
        "Maryland": "South", "West Virginia": "South",
        "San Diego State": "South", "College of Charleston": "South",
        "Virginia": "South", "Furman": "South",
        "Creighton": "South", "NC State": "South",
        "Baylor": "South", "UCSB": "South",
        "Missouri": "South", "Utah State": "South",
        "Arizona": "South", "Princeton": "South",

        # East
        "Purdue": "East", "FDU": "East",
        "Memphis": "East", "Florida Atlantic": "East",
        "Duke": "East", "Oral Roberts": "East",
        "Tennessee": "East", "Louisiana": "East",
        "Kentucky": "East", "Providence": "East",
        "Kansas State": "East", "Montana State": "East",
        "Michigan State": "East", "USC": "East",
        "Marquette": "East", "Vermont": "East",

        # West
        "Kansas": "West", "Howard": "West",
        "Arkansas": "West", "Illinois": "West",
        "Saint Mary's": "West", "VCU": "West",
        "UConn": "West", "Iona": "West",
        "TCU": "West", "Arizona State": "West",
        "Gonzaga": "West", "Grand Canyon": "West",
        "Northwestern": "West", "Boise State": "West",
        "UCLA": "West", "UNC Asheville": "West",

        # Midwest
        "Houston": "Midwest", "Northern Kentucky": "Midwest",
        "Iowa": "Midwest", "Auburn": "Midwest",
        "Miami (FL)": "Midwest", "Drake": "Midwest",
        "Indiana": "Midwest", "Kent State": "Midwest",
        "Iowa State": "Midwest", "Pitt": "Midwest",
        "Xavier": "Midwest", "Kennesaw State": "Midwest",
        "Texas A&M": "Midwest", "Penn State": "Midwest",
        "Texas": "Midwest", "Colgate": "Midwest",
    },
    2026: {
        # East (1-seed: Duke)
        "Duke": "East", "Siena": "East",
        "Ohio State": "East", "TCU": "East",
        "St. John's (NY)": "East", "Northern Iowa": "East",
        "Kansas": "East", "Cal Baptist": "East",
        "Louisville": "East", "South Florida": "East",
        "Michigan State": "East", "North Dakota State": "East",
        "UCLA": "East", "UCF": "East",
        "UConn": "East", "Furman": "East",

        # West (1-seed: Arizona)
        "Arizona": "West", "LIU": "West",
        "Villanova": "West", "Utah State": "West",
        "Wisconsin": "West", "High Point": "West",
        "Arkansas": "West", "Hawaii": "West",
        "BYU": "West", "Texas": "West",
        "Gonzaga": "West", "Kennesaw State": "West",
        "Miami (FL)": "West", "Missouri": "West",
        "Purdue": "West", "Queens": "West",

        # Midwest (1-seed: Michigan)
        "Michigan": "Midwest", "Howard": "Midwest",
        "Georgia": "Midwest", "Saint Louis": "Midwest",
        "Texas Tech": "Midwest", "Akron": "Midwest",
        "Alabama": "Midwest", "Hofstra": "Midwest",
        "Tennessee": "Midwest", "Miami (OH)": "Midwest",
        "Virginia": "Midwest", "Wright State": "Midwest",
        "Kentucky": "Midwest", "Santa Clara": "Midwest",
        "Iowa State": "Midwest", "Tennessee State": "Midwest",

        # South (1-seed: Florida)
        "Florida": "South", "Prairie View A&M": "South",
        "Clemson": "South", "Iowa": "South",
        "Vanderbilt": "South", "McNeese": "South",
        "Nebraska": "South", "Troy": "South",
        "North Carolina": "South", "VCU": "South",
        "Illinois": "South", "Penn": "South",
        "Saint Mary's": "South", "Texas A&M": "South",
        "Houston": "South", "Idaho": "South",
    },
}

# ── Opponent Quality Tiers (based on end-of-season T-Rank) ────────────────────
TIER_1_MAX_RANK = 50    # Top 50 teams
TIER_2_MAX_RANK = 100   # Ranks 51–100
# Tier 3 = everyone else (100+)


# ── Top N scorers to use for individual variance features ─────────────────────
TOP_N_SCORERS = 8


# ── Sports Reference request delay (seconds) to avoid rate limiting ───────────
SPORTSREF_REQUEST_DELAY = 3.0


# ── Tournament result mapping ─────────────────────────────────────────────────
# Maps round of exit to number of wins (our target variable)
ROUND_TO_WINS = {
    "R64":        0,   # Lost in Round of 64
    "R32":        1,   # Lost in Round of 32
    "S16":        2,   # Lost in Sweet 16
    "E8":         3,   # Lost in Elite 8
    "F4":         4,   # Lost in Final Four
    "Runner-up":  5,   # Lost in Championship
    "Champion":   6,   # Won it all
}