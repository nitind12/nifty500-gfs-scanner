"""
Remove scanner CSVs that contain no eligible/setup data.

Rules:
- Empty CSVs are removed.
- CSVs containing only a status/message such as "No matches" or "0 matches"
  are removed.
- MIB CSVs are kept only when at least one ACTIONABLE=YES row exists.
- Engulfing CSVs are kept only when at least one bullish or bearish setup exists.
- Other scanners are kept only when at least one genuine data row exists.

This keeps both email and GitHub Actions artifacts focused on actual scanner results.
"""

import glob
import os
import re
import pandas as pd

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")

NO_DATA_PATTERNS = [
    r"^no\s+matches?$",
    r"^no\s+stocks?$",
    r"^no\s+eligible(?:\s+stocks?)?(?:\s+found)?$",
    r"^no\s+(?:eligible\s+)?data$",
    r"^no\s+(?:valid\s+)?setups?$",
    r"^no\s+(?:bullish|bearish)\s+setups?$",
    r"^0\s+matches?$",
    r"^0\s+stocks?$",
    r"^0\s+setups?$",
]


def remove_file(path, reason):
    try:
        os.remove(path)
        print(f"REMOVED: {os.path.basename(path)} -> {reason}")
    except FileNotFoundError:
        pass


def is_no_data_text(value):
    if pd.isna(value):
        return True
    text = str(value).strip()
    if not text:
        return True
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return any(re.match(pattern, normalized, flags=re.I) for pattern in NO_DATA_PATTERNS)


def has_meaningful_data(path):
    name = os.path.basename(path).lower()

    # Engulfing scanners write human-readable CSVs with section headings.
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

    # No rows at all.
    if df.empty:
        return False

    # Remove rows that are entirely blank.
    df = df.dropna(how="all")
    if df.empty:
        return False

    # MIB output contains a row for every checked stock, but only actionable
    # setups are trading candidates.
    if "ACTIONABLE" in df.columns:
        values = df["ACTIONABLE"].astype(str).str.strip().str.upper()
        return (values == "YES").any()

    # Some scanners may use setup_type instead of ACTIONABLE.
    if "setup_type" in df.columns:
        values = df["setup_type"].astype(str).str.strip().str.upper()
        return values.isin({"LONG", "SHORT", "BULLISH", "BEARISH"}).any()

    # A scanner may write a one-row status message such as "No matches".
    # Treat such a file as having no scanner result.
    for _, row in df.iterrows():
        non_empty = [v for v in row.tolist() if not pd.isna(v) and str(v).strip()]
        if not non_empty:
            continue
        if all(is_no_data_text(v) for v in non_empty):
            continue
        return True

    return False


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
