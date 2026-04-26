"""
diagnose_torvik.py
──────────────────
Inspects the raw structure of Torvik CSVs AND verifies our column assignments.
Run this after any changes to torvik_collector.py to confirm columns map correctly.
"""

import sys
import requests
import pandas as pd
from io import StringIO

YEARS_TO_CHECK = [2024, 2019, 2015]

def inspect(label, url):
    print(f"\n{'─'*60}")
    print(f"{label}")
    print(f"URL: {url}")
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        text = r.text
        lines = text.strip().split("\n")
        print(f"\nRaw first 2 lines:")
        for i, line in enumerate(lines[:2]):
            print(f"  [{i}] {line[:120]}")

        df = pd.read_csv(StringIO(text), engine="python", on_bad_lines="skip")
        print(f"\nWith header=0: {df.shape} — cols: {df.columns.tolist()[:10]}")

        df2 = pd.read_csv(StringIO(text), header=None, engine="python", on_bad_lines="skip")
        print(f"With header=None: {df2.shape}")
    except Exception as e:
        print(f"ERROR: {e}")


def verify_collector():
    """Run collector functions and verify key columns come through."""
    print(f"\n{'═'*60}")
    print("VERIFYING COLLECTOR COLUMN ASSIGNMENTS")
    print(f"{'═'*60}")

    sys.path.insert(0, ".")
    from torvik_collector import fetch_team_results, fetch_team_slice, fetch_player_stats

    for year in [2024, 2019]:
        print(f"\n── Team Results {year} ──")
        try:
            df = fetch_team_results(year, force_refresh=False)
            key_cols = ["team", "adj_o", "adj_d", "adj_em", "barthag", "adj_t"]
            for col in key_cols:
                val = df[col].iloc[0] if col in df.columns else "MISSING"
                print(f"  {col:15s}: {val}")
        except Exception as e:
            print(f"  ERROR: {e}")

        print(f"\n── Team Slice {year} ──")
        try:
            df = fetch_team_slice(year, force_refresh=False)
            key_cols = ["team", "adj_o", "adj_d", "barthag", "efg_pct", "fg3_pct", "orb", "adj_t"]
            for col in key_cols:
                val = df[col].iloc[0] if col in df.columns else "MISSING"
                print(f"  {col:15s}: {val}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n── Player Stats 2024 (top 5 by ppg) ──")
    try:
        df = fetch_player_stats(2024, force_refresh=False)
        key_cols = ["player", "team", "yr", "g", "fg3_pct", "ft_pct", "usg"]
        for col in key_cols:
            val = df[col].iloc[0] if col in df.columns else "MISSING"
            print(f"  {col:15s}: {val}")
        print(f"\n  All columns present: {df.columns.tolist()}")
    except Exception as e:
        print(f"  ERROR: {e}")


if __name__ == "__main__":
    # Raw inspection
    for year in YEARS_TO_CHECK:
        inspect(f"TEAM RESULTS {year}", f"https://barttorvik.com/{year}_team_results.csv")
    inspect("TEAM SLICE 2024", "https://barttorvik.com/teamslicejson.php?year=2024&csv=1&type=R")
    inspect("PLAYER STATS 2024", "https://barttorvik.com/getadvstats.php?year=2024&csv=1")

    # Collector verification
    verify_collector()