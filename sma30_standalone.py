#!/usr/bin/env python3
"""
SMA30 Cross-Above Scanner - Standalone (Nifty 500 / custom list)

Kahin se bhi chalaiye, kisi folder ya env variable ki zarurat nahi.

Install:   pip install pandas yfinance
Run:       python sma30_standalone.py
Examples:  python sma30_standalone.py --lookback 5 --min-vol-ratio 1.2
           python sma30_standalone.py --symbols TCS INFY RELIANCE
           python sma30_standalone.py --file mylist.csv        (column 'Symbol' ya 1 symbol per line)
           python sma30_standalone.py --interval 1wk --period 3y   (weekly)
"""

import argparse
import datetime as dt
import os
import sys

import pandas as pd
import yfinance as yf

NIFTY500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
SMA_PERIOD = 30
CHUNK = 100


def load_symbols(args):
    if args.symbols:
        return [s.upper().replace(".NS", "") for s in args.symbols]

    if args.file:
        df = pd.read_csv(args.file)
        col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        return [str(s).strip().upper().replace(".NS", "") for s in df[col].dropna()]

    try:
        df = pd.read_csv(NIFTY500_URL, storage_options={"User-Agent": "Mozilla/5.0"})
        syms = [s.strip() for s in df["Symbol"]]
        print(f"Nifty 500 list NSE se mili: {len(syms)} symbols")
        return syms
    except Exception as e:
        print(f"NSE list nahi mili ({e}); local file try kar raha hoon...")

    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(here, "ind_nifty500list.csv"), "ind_nifty500list.csv"):
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Local list mili: {path}")
            return [s.strip() for s in df["Symbol"]]

    sys.exit("Koi symbol list nahi mili. --symbols ya --file use kijiye.")


def analyze(sym, df, lookback, slope_bars, min_vol_ratio):
    df = df.dropna(subset=["Close"]).copy()
    if len(df) < SMA_PERIOD + max(lookback, slope_bars) + 2:
        return None

    df["SMA30"] = df["Close"].rolling(SMA_PERIOD).mean()
    df["VolAvg20"] = df["Volume"].rolling(20).mean()
    df["Above"] = df["Close"] > df["SMA30"]
    df["CrossUp"] = df["Above"] & ~df["Above"].shift(1, fill_value=False)

    last = df.iloc[-1]
    if not last["Above"]:
        return None
    recent = df.tail(lookback)
    if not recent["CrossUp"].any():
        return None

    sma_now, sma_prev = last["SMA30"], df["SMA30"].iloc[-1 - slope_bars]
    if not sma_now > sma_prev:
        return None

    vol_ratio = float(last["Volume"] / last["VolAvg20"]) if last["VolAvg20"] else 0.0
    if vol_ratio < min_vol_ratio:
        return None

    return {
        "Symbol": sym,
        "Close": round(float(last["Close"]), 2),
        "SMA30": round(float(sma_now), 2),
        "Pct_Above_SMA": round(float(last["Close"] / sma_now - 1) * 100, 2),
        "SMA30_Slope_Pct": round(float(sma_now / sma_prev - 1) * 100, 2),
        "Vol_Ratio": round(vol_ratio, 2),
        "Cross_Date": recent[recent["CrossUp"]].index[-1].strftime("%Y-%m-%d"),
    }


def main():
    p = argparse.ArgumentParser(description="SMA30 cross-above scanner")
    p.add_argument("--symbols", nargs="+", help="NSE symbols (bina .NS)")
    p.add_argument("--file", help="CSV/text file with symbols")
    p.add_argument("--lookback", type=int, default=3, help="Cross last N candles me (default 3)")
    p.add_argument("--slope-bars", type=int, default=5, help="SMA30 N candles pehle se upar ho (default 5)")
    p.add_argument("--min-vol-ratio", type=float, default=0.0, help="Min volume / 20-bar avg (default off)")
    p.add_argument("--period", default="1y", help="History period (default 1y)")
    p.add_argument("--interval", default="1d", help="1d, 1wk, 1h ... (default 1d)")
    p.add_argument("--out", default=None, help="Output CSV path (default: sma30_cross_YYYYMMDD.csv)")
    args = p.parse_args()

    symbols = load_symbols(args)
    results = []

    for i in range(0, len(symbols), CHUNK):
        batch = symbols[i:i + CHUNK]
        tickers = [s + ".NS" for s in batch]
        print(f"Downloading {i + 1}-{i + len(batch)} / {len(symbols)} ...")
        try:
            data = yf.download(tickers, period=args.period, interval=args.interval,
                               group_by="ticker", auto_adjust=True,
                               progress=False, threads=True)
        except Exception as e:
            print(f"  batch fail: {e}")
            continue

        for sym, tk in zip(batch, tickers):
            try:
                df = data[tk] if len(tickers) > 1 else data
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(-1)
                res = analyze(sym, df, args.lookback, args.slope_bars, args.min_vol_ratio)
                if res:
                    results.append(res)
            except Exception:
                continue

    if not results:
        print("\nKoi stock criteria match nahi hua.")
        return

    out = pd.DataFrame(results).sort_values(["Cross_Date", "Vol_Ratio"], ascending=[False, False])
    path = args.out or f"sma30_cross_{dt.date.today():%Y%m%d}.csv"
    out.to_csv(path, index=False)
    print("\n" + out.to_string(index=False))
    print(f"\n{len(out)} stocks mile. Saved -> {path}")


if __name__ == "__main__":
    main()
