#!/usr/bin/env python3
"""
NIFTY hourly scanner for Supertrend + EMA(20) regime
- Bullish signal  = Close > Supertrend AND Close > EMA20
- Bearish signal  = Close < Supertrend AND Close < EMA20
Runs safely on GitHub Actions. Optional Telegram alerts via secrets.

Signals tell you WHAT to set up:
  Bullish  -> Sell 1 lot ATM Put, Buy OTM Put as hedge
  Bearish  -> Sell 1 lot ATM Call, Buy OTM Call as hedge

Author: you + ChatGPT
"""
from __future__ import annotations

import os, sys, json, math
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
import requests
import yfinance as yf

from indicators import ema, supertrend

# ---------- Config (override via ENV in GitHub Secrets if you want) ----------
TICKER            = os.getenv("NIFTY_TICKER", "^NSEI")   # Yahoo: NIFTY 50 index
INTERVAL          = "60m"                                # hourly
LOOKBACK_DAYS     = int(os.getenv("LOOKBACK_DAYS", "60"))
EMA_PERIOD        = int(os.getenv("EMA_PERIOD", "20"))
ST_ATR_PERIOD     = int(os.getenv("ST_ATR_PERIOD", "7"))
ST_MULTIPLIER     = float(os.getenv("ST_MULTIPLIER", "3.5"))
OTM_GAP_POINTS    = int(os.getenv("OTM_GAP_POINTS", "100"))  # hedge gap suggestion
LOG_LEVEL         = os.getenv("LOG_LEVEL", "INFO")

# Telegram (optional). Put these in GitHub repo -> Settings -> Secrets -> Actions
TG_TOKEN          = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID        = os.getenv("TELEGRAM_CHAT_ID", "")
SEND_TELEGRAM     = bool(TG_TOKEN and TG_CHAT_ID)

IST = ZoneInfo("Asia/Kolkata")

def log(msg: str):
    if LOG_LEVEL.upper() == "DEBUG":
        print(f"[{datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S %Z')}] {msg}", flush=True)
    else:
        print(msg, flush=True)

def round_to_50(x: float) -> int:
    return int(round(x / 50.0) * 50)

def suggested_option_legs(side: str, spot: float) -> dict:
    """
    For a bullish signal, we suggest: SELL ATM PUT, BUY OTM PUT
    For a bearish signal, we suggest: SELL ATM CALL, BUY OTM CALL
    """
    atm = round_to_50(spot)
    if side == "BULL":
        return {
            "sell": f"{atm} PE (ATM)",
            "buy":  f"{max(50, atm - OTM_GAP_POINTS)} PE (hedge ~{OTM_GAP_POINTS})"
        }
    else:
        return {
            "sell": f"{atm} CE (ATM)",
            "buy":  f"{atm + OTM_GAP_POINTS} CE (hedge ~{OTM_GAP_POINTS})"
        }

def fetch_hourly() -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)
    df = yf.download(
        TICKER, start=start, end=end, interval=INTERVAL, auto_adjust=False, progress=False
    )
    if df.empty:
        raise RuntimeError("No data returned from yfinance. Try again or change ticker.")
    # Ensure tz-aware; Yahoo returns tz-aware for intraday. Convert to IST for convenience.
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone.utc)
    df = df.tz_convert(IST)
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["EMA20"] = ema(out["Close"], EMA_PERIOD)
    st = supertrend(out, period=ST_ATR_PERIOD, multiplier=ST_MULTIPLIER)
    out["ST"] = st["ST"]
    out["ST_DIR"] = st["ST_DIR"]  # +1 uptrend, -1 downtrend
    return out

def latest_hourly_close_row(df: pd.DataFrame) -> pd.Series | None:
    """
    NSE hourly bars typically close at minute == 15 (10:15, 11:15, ... 15:15 IST).
    We'll use the last row, but only act if it's a same-day hourly close bar.
    """
    last = df.iloc[-1]
    ts: datetime = df.index[-1]
    now_ist = datetime.now(IST)
    # Only act on same-day bars & when the last bar is not stale.
    if ts.date() != now_ist.date():
        log(f"Last bar {ts} not from today; skipping.")
        return None
    # Accept the bar even if we're checking a bit later (workflow is scheduled).
    if now_ist - ts > timedelta(hours=6):
        log(f"Last bar {ts} looks stale (>6h); skipping.")
        return None
    # Optional strict check for minute==15
    if ts.minute != 15:
        log(f"Last bar minute != 15 (ts={ts}); likely mid-bar; skipping.")
        return None
    return last

def evaluate_signal(df: pd.DataFrame) -> dict:
    last = latest_hourly_close_row(df)
    if last is None:
        return {"status": "NO_ACTION"}

    prev = df.iloc[-2] if len(df) >= 2 else None

    bull = (last["Close"] > last["ST"]) and (last["Close"] > last["EMA20"])
    bear = (last["Close"] < last["ST"]) and (last["Close"] < last["EMA20"])

    # Cross/flip detection (only alert on a NEW regime)
    bull_prev = prev is not None and (prev["Close"] > prev["ST"]) and (prev["Close"] > prev["EMA20"])
    bear_prev = prev is not None and (prev["Close"] < prev["ST"]) and (prev["Close"] < prev["EMA20"])

    crossed_to_bull = bull and not bull_prev
    crossed_to_bear = bear and not bear_prev

    spot = float(last["Close"])
    ts = df.index[-1]
    if crossed_to_bull:
        legs = suggested_option_legs("BULL", spot)
        return {
            "status": "SIGNAL",
            "side": "BULLISH",
            "timestamp": ts.isoformat(),
            "close": spot,
            "ema20": float(last["EMA20"]),
            "supertrend": float(last["ST"]),
            "option_legs": legs
        }
    if crossed_to_bear:
        legs = suggested_option_legs("BEAR", spot)
        return {
            "status": "SIGNAL",
            "side": "BEARISH",
            "timestamp": ts.isoformat(),
            "close": spot,
            "ema20": float(last["EMA20"]),
            "supertrend": float(last["ST"]),
            "option_legs": legs
        }

    return {
        "status": "NO_SIGNAL",
        "timestamp": ts.isoformat(),
        "close": spot,
        "ema20": float(last["EMA20"]),
        "supertrend": float(last["ST"])
    }

def send_telegram(msg: str) -> None:
    if not SEND_TELEGRAM:
        log("Telegram disabled (missing TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID).")
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": TG_CHAT_ID, "text": msg})
        if r.status_code != 200:
            log(f"Telegram error: {r.status_code} {r.text}")
        else:
            log("Telegram sent.")
    except Exception as e:
        log(f"Telegram send failed: {e}")

def format_signal_text(sig: dict) -> str:
    if sig["status"] != "SIGNAL":
        return ""
    side = sig["side"]
    ts_ist = datetime.fromisoformat(sig["timestamp"]).astimezone(IST).strftime("%Y-%m-%d %H:%M IST")
    spot = sig["close"]
    st = sig["supertrend"]
    ema20 = sig["ema20"]
    legs = sig["option_legs"]
    return (
        f"📈 NIFTY Hourly Signal ({ts_ist})\n"
        f"Side: {side}\n"
        f"Close: {spot:.2f} | ST: {st:.2f} | EMA20: {ema20:.2f}\n"
        f"Suggested (weekly):\n"
        f" • SELL {legs['sell']}\n"
        f" • BUY  {legs['buy']}\n"
        f"Exit if Supertrend flips or by Wed EOD/Thu AM."
    )

def main():
    log("Fetching data...")
    df = fetch_hourly()
    log(f"Rows: {len(df)} | Last bar: {df.index[-1]}")
    df = compute_indicators(df)
    sig = evaluate_signal(df)

    # Always print a JSON summary to logs
    print(json.dumps(sig, indent=2))

    # Optional Telegram alert only when a new signal appears
    if sig.get("status") == "SIGNAL":
        msg = format_signal_text(sig)
        log(msg)
        send_telegram(msg)
    else:
        log(f"No new signal. Status: {sig['status']}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"ERROR: {e}")
        sys.exit(1)
