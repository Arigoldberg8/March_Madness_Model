"""
diagnose_columns.py
────────────────────
Pinpoints exact column positions across years for team results and team slice.
Run this and paste output to fix column mappings.
"""

import requests
import pandas as pd
from io import StringIO

def print_all_cols(label, url, header_row=0):
    print(f"\n{'─'*60}")
    print(f"{label}")
    r = requests.get(url, timeout=15)
    text = r.text

    # Try quoting to handle embedded commas
    df = pd.read_csv(StringIO(text), header=header_row, quotechar='"',
                     engine="python", on_bad_lines="warn")
    print(f"Shape: {df.shape}")
    print(f"\nAll columns with index:")
    for i, col in enumerate(df.columns):
        sample = df.iloc[0, i] if len(df) > 0 else "N/A"
        print(f"  [{i:2d}] {str(col):35s} | {sample}")


# Team results — all years with different column counts
for year in [2024, 2023, 2019, 2015, 2012, 2010]:
    print_all_cols(
        f"TEAM RESULTS {year} (header=0)",
        f"https://barttorvik.com/{year}_team_results.csv",
        header_row=0
    )

# Team slice — no header
print(f"\n{'═'*60}")
print("TEAM SLICE — checking column positions vs known stats")
print(f"{'═'*60}")
for year in [2024, 2019, 2015]:
    r = requests.get(f"https://barttorvik.com/teamslicejson.php?year={year}&csv=1&type=R", timeout=15)
    df = pd.read_csv(StringIO(r.text), header=None, quotechar='"', engine="python", on_bad_lines="warn")
    print(f"\nTeam Slice {year}: {df.shape}")
    print(f"First row (all values):")
    for i, val in enumerate(df.iloc[0]):
        print(f"  [{i:2d}] {val}")
