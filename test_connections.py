"""
test_connections.py
────────────────────
Run this first to verify both Torvik and Sports Reference are accessible
and that the data looks as expected before running the full pipeline.

Usage:
  python test_connections.py
"""

import sys
import requests
import pandas as pd
from io import StringIO

def test_torvik():
    print("\n── Torvik ──────────────────────────────────────")
    try:
        # Team results
        url = "https://barttorvik.com/2024_team_results.csv"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text), header=None)
        print(f"✓ Team results: {len(df)} rows, {len(df.columns)} columns")
        print(f"  First row: {df.iloc[0].tolist()[:6]}")

        # Teamslice
        url2 = "https://barttorvik.com/teamslicejson.php?year=2024&csv=1&type=R"
        r2 = requests.get(url2, timeout=15)
        r2.raise_for_status()
        df2 = pd.read_csv(StringIO(r2.text), header=None)
        print(f"✓ Team slice (regular season): {len(df2)} rows, {len(df2.columns)} columns")

        # Player stats
        url3 = "https://barttorvik.com/getadvstats.php?year=2024&csv=1"
        r3 = requests.get(url3, timeout=15)
        r3.raise_for_status()
        df3 = pd.read_csv(StringIO(r3.text), header=None)
        print(f"✓ Player stats: {len(df3)} rows, {len(df3.columns)} columns")
        print(f"  Columns: {df3.columns.tolist()[:8]}")

    except Exception as e:
        print(f"✗ Torvik error: {e}")
        return False
    return True


def test_sportsref():
    print("\n── Sports Reference Python Library ─────────────")
    try:
        from sportsreference.ncaab.teams import Teams
        teams = Teams(2024)
        team_list = [(t.name, t.abbreviation) for t in teams]
        print(f"✓ Teams loaded: {len(team_list)} teams for 2024")
        print(f"  Sample: {team_list[:5]}")
    except ImportError:
        print("✗ sportsreference not installed — run: pip install sportsreference")
        return False
    except Exception as e:
        print(f"✗ Sports Reference error: {e}")
        return False
    return True


def test_torvik_player_columns():
    """Print actual Torvik player stat columns so we can map them correctly."""
    print("\n── Torvik Player Stat Columns ───────────────────")
    try:
        url = "https://barttorvik.com/getadvstats.php?year=2024&csv=1"
        r = requests.get(url, timeout=15)
        df = pd.read_csv(StringIO(r.text))
        print(f"  Columns ({len(df.columns)}): {df.columns.tolist()}")
        print(f"  Sample row:\n{df.iloc[0]}")
    except Exception as e:
        print(f"✗ Error: {e}")


def test_torvik_team_columns():
    """Print actual Torvik team result columns."""
    print("\n── Torvik Team Result Columns ───────────────────")
    try:
        url = "https://barttorvik.com/2024_team_results.csv"
        r = requests.get(url, timeout=15)
        df = pd.read_csv(StringIO(r.text))
        print(f"  Columns ({len(df.columns)}): {df.columns.tolist()}")
        print(f"  Sample row:\n{df.iloc[0]}")
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    print("Testing data source connectivity...")

    torvik_ok  = test_torvik()
    sportsref_ok = test_sportsref()

    # Print actual column info to help us validate our column name assumptions
    test_torvik_team_columns()
    test_torvik_player_columns()

    print("\n── Summary ──────────────────────────────────────")
    print(f"  Torvik:          {'✓' if torvik_ok  else '✗'}")
    print(f"  Sports Reference:{'✓' if sportsref_ok else '✗'}")

    if torvik_ok and sportsref_ok:
        print("\nAll sources OK. Ready to run the pipeline.")
        print("Start with: python pipeline.py --steps torvik")
    else:
        print("\nFix connection issues before running the pipeline.")
        sys.exit(1)
