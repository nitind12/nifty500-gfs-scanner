"""
Sends ONE email with ALL CSV files found in OUTPUT_DIR attached.
Run this as the LAST step in the combined workflow, after all scanner
scripts have finished writing their CSVs to the same output folder.
"""

import os
import glob
import datetime as dt
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
EMAIL_APP_PASSWORD = os.environ.get("EMAIL_APP_PASSWORD")
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL")


def main():
    if not (EMAIL_ADDRESS and EMAIL_APP_PASSWORD and RECIPIENT_EMAIL):
        print("Email env vars not set - skipping combined email.")
        return

    csv_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in {OUTPUT_DIR} - nothing to email.")
        return

    print(f"Found {len(csv_files)} CSV file(s) to attach:")
    for f in csv_files:
        print(f"  - {f}")

    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = RECIPIENT_EMAIL
    msg["Subject"] = f"Weekly Scanner Results - {dt.date.today().isoformat()} ({len(csv_files)} files)"

    body_lines = [
        "Weekly scan run complete. All scanner outputs attached:\n",
    ]
    for f in csv_files:
        body_lines.append(f"  - {os.path.basename(f)}")
    body_lines.append("\nScanners run (Weekly TF): NIFTY 500 RSI (Quarterly/Monthly/Weekly), "
                       "Breakout (Nifty 500, Weekly, 104-week lookback), "
                       "MIB Watchlist (Weekly bias + Daily mother candle), "
                       "MIB Market (Weekly bias + Daily mother candle), "
                       "Engulfing @ S/R (Nifty 500, Weekly).")
    msg.attach(MIMEText("\n".join(body_lines), "plain"))

    for csv_path in csv_files:
        with open(csv_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(csv_path)}")
        msg.attach(part)

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_APP_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, RECIPIENT_EMAIL, msg.as_string())
        server.quit()
        print(f"Combined email sent to {RECIPIENT_EMAIL} with {len(csv_files)} attachment(s).")
    except Exception as e:
        print(f"Email failed to send: {e}")


if __name__ == "__main__":
    main()
