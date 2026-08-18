"""
Remove scanner CSVs that contain no eligible/setup data.

Rules:
- Empty tabular CSVs are removed.
- MIB CSVs are kept only when at least one ACTIONABLE=YES row exists.
- Engulfing CSVs are kept only when at least one bullish or bearish setup exists.
- Other scanners are kept only when they contain at least one data row.

This keeps both email and GitHub Actions artifacts focused on actual scanner results.
"""

import glob
import os
import re
import pandas as pd

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")


def remove_file(path, reason):
    try:
        os.remove(path)
        print(f"REMOVED: {os.path.basename(path)} -> {reason}")
    except FileNotFoundError:
        pass


def has_meaningful_data(path):
    name = os.path.basename(path).lower()

    # Engulfing scanners write a human-readable CSV with section headings.
    if "engulfing" in name:
        text = open(path, "r", encoding="utf-8", errors="ignore").read()
        if not text.strip():
            return False
        counts = [int(x) for x in re.findall(r"-\s*(\d+)\s+found", text, flags=re.I)]
        if counts:
            return sum(counts) > 0
        return not re.search(r"no bullish.*?no bearish", text, flags=re.I | re.S)

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"KEEP (could not parse): {os.path.basename(path)} -> {exc}")
        return True

    if df.empty:
        return False

    # MIB output contains a row for every checked stock, but only actionable
    # setups are trading candidates. Do not email/store a MIB file with zero setups.
    if "ACTIONABLE" in df.columns:
        values = df["ACTIONABLE"].astype(str).str.strip().str.upper()
        return (values == "YES").any()

    # Some scanners may use setup_type instead of ACTIONABLE.
    if "setup_type" in df.columns:
        values = df["setup_type"].astype(str).str.strip().str.upper()
        return values.isin({"LONG", "SHORT"}).any()

    return len(df.index) > 0


def main():
    if not os.path.isdir(OUTPUT_DIR):
        print(f"Output directory not found: {OUTPUT_DIR}")
        return

    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.csv")))
    print(f"Checking {len(files)} CSV file(s) in {OUTPUT_DIR}")

    for path in files:
        if not has_meaningful_data(path):
            remove_file(path, "no eligible scanner data")
        else:
            print(f"KEEP: {os.path.basename(path)}")


if __name__ == "__main__":
    main()
