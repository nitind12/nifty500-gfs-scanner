"""
Send ONE daily email containing only scanners with real eligible data.

Rules:
- Only *_latest.csv files are candidates.
- Empty/header-only CSVs are ignored.
- MIB is emailed only when ACTIONABLE=YES exists.
- Engulfing is emailed only when at least one bullish/bearish setup exists.
- If no scanner has eligible data, NO email is sent.
- Historical dated CSVs are never emailed.
"""

import os
import glob
import re
import datetime as dt
import smtplib
import pandas as pd
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
EMAIL_APP_PASSWORD = os.environ.get("EMAIL_APP_PASSWORD")
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL")


def has_real_data(path):
    """Return True only when the scanner output contains an eligible result."""
    name = os.path.basename(path).lower()

    # Engulfing output is a human-readable multi-section CSV/text file.
    if "engulfing" in name:
        text = open(path, "r", encoding="utf-8", errors="ignore").read()
        counts = [int(x) for x in re.findall(r"-\s*(\d+)\s+found", text, flags=re.I)]
        if counts:
            return sum(counts) > 0
        return bool(re.search(r"\b(BULLISH|BEARISH).*?(SETUP|ENGULFING)", text, flags=re.I)) and not re.search(
            r"No bullish.*No bearish", text, flags=re.I | re.S
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"SKIP (cannot read CSV): {os.path.basename(path)} -> {exc}")
        return False

    if df.empty:
        return False

    # MIB contains rows for scanned stocks; only actionable setups count.
    if "ACTIONABLE" in df.columns:
        values = df["ACTIONABLE"].astype(str).str.strip().str.upper()
        return (values == "YES").any()

    if "setup_type" in df.columns:
        values = df["setup_type"].astype(str).str.strip().str.upper()
        return values.isin({"LONG", "SHORT", "BULLISH", "BEARISH"}).any()

    meaningful = df.dropna(how="all").copy()
    if meaningful.empty:
        return False
    for col in meaningful.columns:
        meaningful[col] = meaningful[col].astype(str).str.strip()
    meaningful = meaningful.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return not meaningful.dropna(how="all").empty


def main():
    if not (EMAIL_ADDRESS and EMAIL_APP_PASSWORD and RECIPIENT_EMAIL):
        print("Email env vars not set - skipping combined email.")
        return

    candidates = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*_latest.csv")))
    csv_files = []
    for path in candidates:
        if has_real_data(path):
            csv_files.append(path)
            print(f"KEEP EMAIL: {os.path.basename(path)}")
        else:
            print(f"SKIP EMAIL: {os.path.basename(path)} -> no eligible data")

    # Critical rule: if nothing actionable was found anywhere, do not send any email.
    if not csv_files:
        print(f"No scanner has eligible data in {OUTPUT_DIR} - NO EMAIL WILL BE SENT.")
        return

    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = RECIPIENT_EMAIL
    msg["Subject"] = f"Daily Scanner Results - {dt.date.today().isoformat()} ({len(csv_files)} scanners)"

    body_lines = [
        "Daily scan run complete. Only scanners with eligible results are attached:\n",
    ]
    body_lines.extend(f"  - {os.path.basename(f)}" for f in csv_files)
    msg.attach(MIMEText("\n".join(body_lines), "plain"))

    for csv_path in csv_files:
        with open(csv_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(csv_path)}")
        msg.attach(part)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_APP_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, RECIPIENT_EMAIL, msg.as_string())
        print(f"Daily email sent to {RECIPIENT_EMAIL} with {len(csv_files)} attachment(s).")
    except Exception as e:
        print(f"Email failed to send: {e}")
        raise


if __name__ == "__main__":
    main()
